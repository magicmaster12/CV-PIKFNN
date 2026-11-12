"""Refined KM-PIKFNN solver for a square plate with a central elliptical hole.

The approximation uses KM point-force fundamental-solution kernels.  For
moderate aspect ratios, fictitious sources are placed on an inner curve inside
the excluded hole and on an artificial square outside the physical square.  For
highly flattened holes, the squeezed physical inner curve is replaced by a
fixed source circle in the Joukowski plane and two geometry-derived
near-tip enrichment poles, supplemented by inverse-power
analytic enrichment functions.

For fixed source locations, all boundary residuals are linear in the kernel
coefficients.  The coefficients are therefore identified by a column-scaled,
regularized boundary objective. By default Adam and L-BFGS optimize this
objective from an SVD warm start and print their actual iteration losses.
Use --no-warm-start for zero initialization, or --optimizer svd for a direct solve.
"""

import argparse
import csv
import time
from dataclasses import dataclass, asdict
import json
from scipy.linalg import lstsq, null_space
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle


# -----------------------------------------------------------------------------
# Physical parameters and nondimensionalization (plane stress)
# -----------------------------------------------------------------------------
L = 2.0
E_REAL = 1.0e6
NU = 0.3
U0_REAL = 1.0e-3
A_MAJOR = 0.5

U_SCALE = 1.0e3
E = E_REAL / 1.0e6
U0 = U0_REAL * U_SCALE
KAPPA = (3.0 - NU) / (1.0 + NU)
G = E / (2.0 * (1.0 + NU))
STRESS_TO_PA = E_REAL / U_SCALE
C0 = 2.0 * np.pi * (KAPPA + 1.0)


@dataclass
class SolverConfig:
    n_side: int = 400
    n_hole: int = 2880
    n_inner_sources: int = 96
    n_outer_sources_per_side: int = 32
    inner_source_scale: float = 0.70
    outer_source_scale: float = 1.25
    mapped_lambda_threshold: float = 0.20
    n_mapped_sources: int = 64
    mapped_source_radius: float = 0.65
    w_displacement: float = 20.0
    w_side_traction: float = 3.0
    w_hole_traction: float = 15.0
    ridge_relative: float = 1.0e-16
    tip_points_per_branch: int = 400
    use_warm_start: bool = True
    optimizer: str = 'adam_lbfgs'
    adam_steps: int = 1000
    adam_lr: float = 1e-3
    lbfgs_epochs: int = 20
    lbfgs_max_iter: int = 20
    print_every: int = 100


def pole_order_for_lambda(lam: float) -> int:
    """Analytic enrichment used only when the ellipse becomes crack-like."""
    return 0 if lam >= 0.20 else 32


def build_source_points(lam: float, cfg: SolverConfig) -> Tuple[np.ndarray, int, bool]:
    """Return physical-plane sources and select the mapped-circle representation."""
    b_minor = A_MAJOR * lam
    use_mapped_inner = lam <= cfg.mapped_lambda_threshold

    if use_mapped_inner:
        # The inner representation is constructed on a fixed auxiliary circle;
        # no physical source curve is squeezed as lambda tends to zero.
        inner = np.empty(0, dtype=np.complex128)
        n_inner = 0
    else:
        n_inner = cfg.n_inner_sources
        # For lambda=1 this is exactly a concentric circle of radius 0.7a.
        theta = (np.arange(n_inner, dtype=float) + 0.5) * (2.0 * np.pi / n_inner)
        inner = (
            cfg.inner_source_scale * A_MAJOR * np.cos(theta)
            + 1j * cfg.inner_source_scale * b_minor * np.sin(theta)
        )

    # Artificial square: x,y = +/- alpha*L/2, alpha>1.  These are not physical
    # boundary points.  Endpoints are trimmed so no source is duplicated at a corner.
    n_outer = cfg.n_outer_sources_per_side
    extent = cfg.outer_source_scale * (L / 2.0)
    values = np.linspace(-extent, extent, n_outer + 2, dtype=float)[1:-1]
    outer = np.concatenate(
        [
            values - 1j * extent,
            extent + 1j * values,
            values[::-1] + 1j * extent,
            -extent + 1j * values[::-1],
        ]
    )

    sources = np.concatenate([inner, outer]).astype(np.complex128)
    assert np.all(np.maximum(np.abs(outer.real), np.abs(outer.imag)) > L / 2.0)
    if n_inner:
        assert np.all((inner.real / A_MAJOR) ** 2 + (inner.imag / b_minor) ** 2 < 1.0)
    return sources, n_inner, use_mapped_inner


def joukowski_inverse(z: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exterior inverse map and its first two derivatives with respect to z."""
    z = np.asarray(z, dtype=np.complex128).reshape(-1, 1)
    c1 = 0.5 * A_MAJOR * (1.0 + lam)
    c2 = 0.5 * A_MAJOR * (1.0 - lam)
    discriminant = np.sqrt(z**2 - 4.0 * c1 * c2)
    root_1 = (z + discriminant) / (2.0 * c1)
    root_2 = (z - discriminant) / (2.0 * c1)
    zeta = np.where(np.abs(root_1) >= np.abs(root_2), root_1, root_2)

    map_zeta = c1 - c2 / zeta**2
    map_zeta_zeta = 2.0 * c2 / zeta**3
    zeta_z = 1.0 / map_zeta
    zeta_zz = -map_zeta_zeta / map_zeta**3
    return zeta, zeta_z, zeta_zz


def point_force_influence(z: np.ndarray, sources: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Influence matrices for real kernel coefficients and four far-field terms."""
    z = np.asarray(z, dtype=np.complex128).reshape(-1, 1)
    source_row = sources.reshape(1, -1)
    q = z - source_row
    log_q = np.log(q)
    n_points = z.shape[0]
    n_sources = source_row.shape[1]
    n_coefficients = 2 * n_sources + 4

    phi_kernel = -log_q / C0
    phi_z_kernel = -1.0 / (C0 * q)
    phi_zz_kernel = 1.0 / (C0 * q**2)
    psi_conjugate_kernel = KAPPA * log_q / C0
    psi_kernel = np.conj(source_row) / (C0 * q)
    psi_z_conjugate_kernel = KAPPA / (C0 * q)
    psi_z_kernel = -np.conj(source_row) / (C0 * q**2)

    phi = np.zeros((n_points, n_coefficients), dtype=np.complex128)
    phi_z = np.zeros_like(phi)
    phi_zz = np.zeros_like(phi)
    psi = np.zeros_like(phi)
    psi_z = np.zeros_like(phi)

    # P_m = a_m + i b_m.
    phi[:, :n_sources] = phi_kernel
    phi[:, n_sources : 2 * n_sources] = 1j * phi_kernel
    phi_z[:, :n_sources] = phi_z_kernel
    phi_z[:, n_sources : 2 * n_sources] = 1j * phi_z_kernel
    phi_zz[:, :n_sources] = phi_zz_kernel
    phi_zz[:, n_sources : 2 * n_sources] = 1j * phi_zz_kernel

    psi[:, :n_sources] = psi_conjugate_kernel + psi_kernel
    psi[:, n_sources : 2 * n_sources] = 1j * (psi_kernel - psi_conjugate_kernel)
    psi_z[:, :n_sources] = psi_z_conjugate_kernel + psi_z_kernel
    psi_z[:, n_sources : 2 * n_sources] = 1j * (
        psi_z_kernel - psi_z_conjugate_kernel
    )

    # Linear far-field potentials phi_inf*z and psi_inf*z.
    phi[:, 2 * n_sources] = z[:, 0]
    phi_z[:, 2 * n_sources] = 1.0
    phi[:, 2 * n_sources + 1] = 1j * z[:, 0]
    phi_z[:, 2 * n_sources + 1] = 1j
    psi[:, 2 * n_sources + 2] = z[:, 0]
    psi_z[:, 2 * n_sources + 2] = 1.0
    psi[:, 2 * n_sources + 3] = 1j * z[:, 0]
    psi_z[:, 2 * n_sources + 3] = 1j

    displacement = (
        KAPPA * phi - z * np.conj(phi_z) - np.conj(psi)
    ) / (2.0 * G)
    stress_sum = 4.0 * phi_z.real
    stress_difference = 2.0 * (np.conj(z) * phi_zz + psi_z)
    sigma_xx = 0.5 * (stress_sum - stress_difference.real)
    sigma_yy = 0.5 * (stress_sum + stress_difference.real)
    sigma_xy = 0.5 * stress_difference.imag
    return displacement.real, displacement.imag, sigma_xx, sigma_yy, sigma_xy


def point_force_resultant_influence(z: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Influence matrix of phi + z*conj(phi') + conj(psi)."""
    z = np.asarray(z, dtype=np.complex128).reshape(-1, 1)
    source_row = sources.reshape(1, -1)
    q = z - source_row
    log_q = np.log(q)
    n_points = z.shape[0]
    n_sources = source_row.shape[1]
    n_coefficients = 2 * n_sources + 4

    phi_kernel = -log_q / C0
    phi_z_kernel = -1.0 / (C0 * q)
    psi_conjugate_kernel = KAPPA * log_q / C0
    psi_kernel = np.conj(source_row) / (C0 * q)

    phi = np.zeros((n_points, n_coefficients), dtype=np.complex128)
    phi_z = np.zeros_like(phi)
    psi = np.zeros_like(phi)
    phi[:, :n_sources] = phi_kernel
    phi[:, n_sources : 2 * n_sources] = 1j * phi_kernel
    phi_z[:, :n_sources] = phi_z_kernel
    phi_z[:, n_sources : 2 * n_sources] = 1j * phi_z_kernel
    psi[:, :n_sources] = psi_conjugate_kernel + psi_kernel
    psi[:, n_sources : 2 * n_sources] = 1j * (psi_kernel - psi_conjugate_kernel)
    phi[:, 2 * n_sources] = z[:, 0]
    phi_z[:, 2 * n_sources] = 1.0
    phi[:, 2 * n_sources + 1] = 1j * z[:, 0]
    phi_z[:, 2 * n_sources + 1] = 1j
    psi[:, 2 * n_sources + 2] = z[:, 0]
    psi[:, 2 * n_sources + 3] = 1j * z[:, 0]
    return phi + z * np.conj(phi_z) + np.conj(psi)


def pole_influence(z: np.ndarray, lam: float, order: int) -> Tuple[np.ndarray, ...]:
    """Joukowski inverse-power enrichment, analytic in the material domain."""
    z = np.asarray(z, dtype=np.complex128).reshape(-1, 1)
    zeta, zeta_z, zeta_zz = joukowski_inverse(z, lam)
    powers = np.arange(1, order + 1, dtype=float).reshape(1, -1)

    basis = zeta ** (-powers)
    basis_zeta = -powers * zeta ** (-powers - 1.0)
    basis_zeta_zeta = powers * (powers + 1.0) * zeta ** (-powers - 2.0)
    basis_z = basis_zeta * zeta_z
    basis_zz = basis_zeta_zeta * zeta_z**2 + basis_zeta * zeta_zz

    zero = np.zeros_like(basis)
    phi = np.concatenate([basis, 1j * basis, zero, zero], axis=1)
    phi_z = np.concatenate([basis_z, 1j * basis_z, zero, zero], axis=1)
    phi_zz = np.concatenate([basis_zz, 1j * basis_zz, zero, zero], axis=1)
    psi = np.concatenate([zero, zero, basis, 1j * basis], axis=1)
    psi_z = np.concatenate([zero, zero, basis_z, 1j * basis_z], axis=1)

    displacement = (
        KAPPA * phi - z * np.conj(phi_z) - np.conj(psi)
    ) / (2.0 * G)
    stress_sum = 4.0 * phi_z.real
    stress_difference = 2.0 * (np.conj(z) * phi_zz + psi_z)
    sigma_xx = 0.5 * (stress_sum - stress_difference.real)
    sigma_yy = 0.5 * (stress_sum + stress_difference.real)
    sigma_xy = 0.5 * stress_difference.imag
    return displacement.real, displacement.imag, sigma_xx, sigma_yy, sigma_xy


def pole_resultant_influence(z: np.ndarray, lam: float, order: int) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128).reshape(-1, 1)
    zeta, zeta_z, _ = joukowski_inverse(z, lam)
    powers = np.arange(1, order + 1, dtype=float).reshape(1, -1)
    basis = zeta ** (-powers)
    basis_z = (-powers * zeta ** (-powers - 1.0)) * zeta_z
    zero = np.zeros_like(basis)
    phi = np.concatenate([basis, 1j * basis, zero, zero], axis=1)
    phi_z = np.concatenate([basis_z, 1j * basis_z, zero, zero], axis=1)
    psi = np.concatenate([zero, zero, basis, 1j * basis], axis=1)
    return phi + z * np.conj(phi_z) + np.conj(psi)


def mapped_circle_influence(
    z: np.ndarray,
    lam: float,
    n_kernels: int,
    source_radius: float,
) -> Tuple[np.ndarray, ...]:
    """Analytic rational kernels with uniformly spaced centers in the zeta-plane.

    The physical ellipse is mapped to |zeta|=1.  Kernel centers remain on the
    fixed circle |eta|=source_radius<1, so they do not collapse when lambda is
    reduced.  The chain-rule factors naturally amplify the near-tip gradients.
    """
    z = np.asarray(z, dtype=np.complex128).reshape(-1, 1)
    zeta, zeta_z, zeta_zz = joukowski_inverse(z, lam)
    theta = (np.arange(n_kernels, dtype=float) + 0.5) * (2.0 * np.pi / n_kernels)
    # Geometry-derived interior poles at the zeros of omega'(zeta).
    # |eta_tip| < 1 for every positive aspect ratio: no material-domain pole.
    eta_tip = np.sqrt((1.0 - lam) / (1.0 + lam))
    eta = np.r_[source_radius * np.exp(1j * theta), eta_tip, -eta_tip].reshape(1, -1)

    q = zeta - eta
    basis = 1.0 / q
    basis_zeta = -1.0 / q**2
    basis_zeta_zeta = 2.0 / q**3
    basis_z = basis_zeta * zeta_z
    basis_zz = basis_zeta_zeta * zeta_z**2 + basis_zeta * zeta_zz

    zero = np.zeros_like(basis)
    phi = np.concatenate([basis, 1j * basis, zero, zero], axis=1)
    phi_z = np.concatenate([basis_z, 1j * basis_z, zero, zero], axis=1)
    phi_zz = np.concatenate([basis_zz, 1j * basis_zz, zero, zero], axis=1)
    psi = np.concatenate([zero, zero, basis, 1j * basis], axis=1)
    psi_z = np.concatenate([zero, zero, basis_z, 1j * basis_z], axis=1)

    displacement = (
        KAPPA * phi - z * np.conj(phi_z) - np.conj(psi)
    ) / (2.0 * G)
    stress_sum = 4.0 * phi_z.real
    stress_difference = 2.0 * (np.conj(z) * phi_zz + psi_z)
    sigma_xx = 0.5 * (stress_sum - stress_difference.real)
    sigma_yy = 0.5 * (stress_sum + stress_difference.real)
    sigma_xy = 0.5 * stress_difference.imag
    return displacement.real, displacement.imag, sigma_xx, sigma_yy, sigma_xy


def mapped_circle_resultant_influence(
    z: np.ndarray,
    lam: float,
    n_kernels: int,
    source_radius: float,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128).reshape(-1, 1)
    zeta, zeta_z, _ = joukowski_inverse(z, lam)
    theta = (np.arange(n_kernels, dtype=float) + 0.5) * (2.0 * np.pi / n_kernels)
    # Geometry-derived interior poles at the zeros of omega'(zeta).
    # |eta_tip| < 1 for every positive aspect ratio: no material-domain pole.
    eta_tip = np.sqrt((1.0 - lam) / (1.0 + lam))
    eta = np.r_[source_radius * np.exp(1j * theta), eta_tip, -eta_tip].reshape(1, -1)
    q = zeta - eta
    basis = 1.0 / q
    basis_z = (-1.0 / q**2) * zeta_z
    zero = np.zeros_like(basis)
    phi = np.concatenate([basis, 1j * basis, zero, zero], axis=1)
    phi_z = np.concatenate([basis_z, 1j * basis_z, zero, zero], axis=1)
    psi = np.concatenate([zero, zero, basis, 1j * basis], axis=1)
    return phi + z * np.conj(phi_z) + np.conj(psi)


def field_influence(
    z: np.ndarray,
    sources: np.ndarray,
    lam: float,
    pole_order: int,
    use_mapped_inner: bool,
    cfg: SolverConfig,
) -> Tuple[np.ndarray, ...]:
    components = [point_force_influence(z, sources)]
    if use_mapped_inner:
        components.append(
            mapped_circle_influence(
                z, lam, cfg.n_mapped_sources, cfg.mapped_source_radius
            )
        )
    if pole_order > 0:
        components.append(pole_influence(z, lam, pole_order))
    if len(components) == 1:
        return components[0]
    return tuple(np.hstack(fields) for fields in zip(*components))


def resultant_influence(
    z: np.ndarray,
    sources: np.ndarray,
    lam: float,
    pole_order: int,
    cfg: SolverConfig,
) -> np.ndarray:
    components = [point_force_resultant_influence(z, sources)]
    components.append(
        mapped_circle_resultant_influence(
            z, lam, cfg.n_mapped_sources, cfg.mapped_source_radius
        )
    )
    if pole_order > 0:
        components.append(pole_resultant_influence(z, lam, pole_order))
    return np.hstack(components)


def boundary_points(lam: float, cfg: SolverConfig) -> Dict[str, np.ndarray]:
    dx = L / cfg.n_side
    edge = np.linspace(-L / 2.0 + 0.5 * dx, L / 2.0 - 0.5 * dx, cfg.n_side)
    theta = (np.arange(cfg.n_hole, dtype=float) + 0.5) * (2.0 * np.pi / cfg.n_hole)
    b_minor = A_MAJOR * lam
    x_hole = A_MAJOR * np.cos(theta)
    y_hole = b_minor * np.sin(theta)
    nx_hole = b_minor * np.cos(theta)
    ny_hole = A_MAJOR * np.sin(theta)
    normal_length = np.hypot(nx_hole, ny_hole)
    return {
        "edge": edge,
        "hole_z": x_hole + 1j * y_hole,
        "hole_nx": nx_hole / normal_length,
        "hole_ny": ny_hole / normal_length,
    }


def append_residual(
    matrix_rows: List[np.ndarray],
    rhs_rows: List[np.ndarray],
    influence: np.ndarray,
    target: float,
    weight: float,
    reference_scale: float,
) -> None:
    n_rows = influence.shape[0]
    factor = np.sqrt(weight / n_rows) / reference_scale
    matrix_rows.append(factor * np.asarray(influence, dtype=float))
    rhs_rows.append(factor * np.full(n_rows, target, dtype=float))


def assemble_boundary_system(
    lam: float,
    cfg: SolverConfig,
    sources: np.ndarray,
    pole_order: int,
    use_mapped_inner: bool,
    n_inner: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the exact boundary loss used to identify the kernel weights."""
    points = boundary_points(lam, cfg)
    edge = points["edge"]
    matrix_rows: List[np.ndarray] = []
    rhs_rows: List[np.ndarray] = []

    # Match the BEM problem exactly: ux=0 and uy=+/-u0 on both loaded edges.
    for y_value, uy_target in [(L / 2.0, U0), (-L / 2.0, -U0)]:
        ux, uy, _, _, _ = field_influence(
            edge + 1j * y_value, sources, lam, pole_order, use_mapped_inner, cfg
        )
        append_residual(matrix_rows, rhs_rows, ux, 0.0, cfg.w_displacement, U0)
        append_residual(matrix_rows, rhs_rows, uy, uy_target, cfg.w_displacement, U0)

    # Traction-free left and right edges.
    for x_value, normal_x in [(-L / 2.0, -1.0), (L / 2.0, 1.0)]:
        _, _, sxx, syy, sxy = field_influence(
            x_value + 1j * edge, sources, lam, pole_order, use_mapped_inner, cfg
        )
        traction_x = normal_x * sxx
        traction_y = normal_x * sxy
        append_residual(
            matrix_rows, rhs_rows, traction_x, 0.0, cfg.w_side_traction, 1.0
        )
        append_residual(
            matrix_rows, rhs_rows, traction_y, 0.0, cfg.w_side_traction, 1.0
        )

    hole_weight = cfg.w_hole_traction * (2.0 if lam < 0.20 else 1.0)
    if use_mapped_inner:
        # On the regularized unit circle, use the integrated KM traction relation.
        # It avoids multiplying the residual by the near-singular map derivative.
        boundary_resultant = resultant_influence(
            points["hole_z"], sources, lam, pole_order, cfg
        )
        boundary_resultant -= np.mean(boundary_resultant, axis=0, keepdims=True)
        append_residual(
            matrix_rows, rhs_rows, boundary_resultant.real, 0.0, hole_weight, 1.0
        )
        append_residual(
            matrix_rows, rhs_rows, boundary_resultant.imag, 0.0, hole_weight, 1.0
        )
    else:
        # For ordinary holes, direct pointwise traction gives tighter local control.
        _, _, sxx, syy, sxy = field_influence(
            points["hole_z"], sources, lam, pole_order, use_mapped_inner, cfg
        )
        nx = points["hole_nx"].reshape(-1, 1)
        ny = points["hole_ny"].reshape(-1, 1)
        traction_x = nx * sxx + ny * sxy
        traction_y = nx * sxy + ny * syy
        append_residual(matrix_rows, rhs_rows, traction_x, 0.0, hole_weight, 1.0)
        append_residual(matrix_rows, rhs_rows, traction_y, 0.0, hole_weight, 1.0)

    if use_mapped_inner:
        local = np.geomspace(max(lam * 1e-3, 1e-8), 0.3, cfg.tip_points_per_branch)
        theta = np.r_[(np.arange(cfg.n_hole)+0.5)*2*np.pi/cfg.n_hole,
                      0.0, np.pi, local, 2*np.pi-local, np.pi-local, np.pi+local]
        z_tip = A_MAJOR*np.cos(theta)+1j*A_MAJOR*lam*np.sin(theta)
        nx, ny = lam*np.cos(theta), np.sin(theta)
        length = np.hypot(nx, ny)
        nx, ny = (nx/length)[:,None], (ny/length)[:,None]
        _, _, sxx, syy, sxy = field_influence(z_tip, sources, lam, pole_order, True, cfg)
        append_residual(matrix_rows, rhs_rows, nx*sxx+ny*sxy, 0.0, cfg.w_hole_traction, 1.0)
        append_residual(matrix_rows, rhs_rows, nx*sxy+ny*syy, 0.0, cfg.w_hole_traction, 1.0)

    matrix = np.vstack(matrix_rows)
    rhs = np.concatenate(rhs_rows)

    # Logarithmic point-force terms enclosed by a hole must have zero resultant
    # coefficient for a single-valued displacement.  Enforce the two real parts
    # exactly through equality constraints in the ridge solve.
    n_coefficients = matrix.shape[1]
    n_sources = sources.size
    constraints = np.zeros((2 if n_inner else 0, n_coefficients), dtype=float)
    if n_inner:
        constraints[0, :n_inner] = 1.0
        constraints[1, n_sources : n_sources + n_inner] = 1.0
    return matrix, rhs, constraints


def solve_scaled_constrained_ridge(matrix, rhs, constraints, ridge_relative,
                                   initial_coefficients=None, training_cfg=None):
    """Constrained augmented SVD solve; optionally solve a correction to a warm start.

    Warm/cold starts minimize exactly the same regularized boundary objective.
    COMSOL values are never used here.  Avoid forming A.T@A, which squares the
    conditioning of this high-gradient approximation.
    """
    column_scale = np.maximum(np.linalg.norm(matrix,axis=0),1e-14)
    scaled = matrix/column_scale
    constraint_scaled = constraints/column_scale
    basis = null_space(constraint_scaled) if len(constraints) else np.eye(scaled.shape[1])
    reduced = scaled@basis
    ridge = ridge_relative*np.sum(scaled**2)/scaled.shape[1]
    use_warm = initial_coefficients is not None
    initial = np.zeros(scaled.shape[1]) if not use_warm else np.asarray(initial_coefficients)*column_scale
    # Keep the initial state in the feasible subspace.
    initial = basis@(basis.T@initial)
    augmented = np.vstack([reduced,np.sqrt(ridge)*basis])
    target = np.r_[rhs-scaled@initial, -np.sqrt(ridge)*initial]
    correction, _, rank, singular_values = lstsq(augmented,target,cond=1e-13,lapack_driver='gelsd')
    coefficients = (initial+basis@correction)/column_scale
    history = []
    origin = 'SVD'
    if training_cfg is not None and training_cfg.optimizer == 'adam_lbfgs':
        try:
            import torch
        except ImportError as exc:
            raise ImportError('Adam/L-BFGS需要PyTorch：请在当前Python环境安装torch，或使用 --optimizer svd。') from exc
        if training_cfg.adam_steps < 0 or training_cfg.lbfgs_epochs < 0 or training_cfg.print_every < 1:
            raise ValueError('迭代次数须非负，print-every须为正整数。')
        # Optimize in the null space: force constraints hold throughout training.
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float64
        At = torch.as_tensor(augmented, dtype=dtype, device=dev)
        bt = torch.as_tensor(np.r_[rhs,np.zeros(basis.shape[0])], dtype=dtype, device=dev)
        q0 = basis.T@(coefficients*column_scale) if training_cfg.use_warm_start else np.zeros(basis.shape[1])
        q = torch.nn.Parameter(torch.as_tensor(q0,dtype=dtype,device=dev).clone())
        boundary_rows = len(rhs)
        def losses():
            r = At@q-bt
            boundary = torch.sum(r[:boundary_rows]**2)
            regularization = torch.sum(r[boundary_rows:]**2)
            return boundary+regularization,boundary,regularization
        best = float('inf'); best_q = q.detach().clone()
        start = time.perf_counter()
        def record(stage,iteration,display=False):
            nonlocal best,best_q,origin
            with torch.no_grad():
                total,boundary,regularization = [float(v) for v in losses()]
            history.append((stage,iteration,total,boundary,regularization))
            if total < best:
                best = total; best_q = q.detach().clone(); origin = f'{stage} {iteration}'
            if display:
                print(f'{stage:12s} {iteration:5d} | total={total:.8e} | boundary={boundary:.8e} | reg={regularization:.3e} | time={time.perf_counter()-start:.2f}s',flush=True)
        print(f'Optimizer: Adam -> L-BFGS | float64 | {dev} | initialization={"SVD/cache" if training_cfg.use_warm_start else "zero"}',flush=True)
        record('initial',0,True)
        adam = torch.optim.Adam([q],lr=training_cfg.adam_lr)
        for epoch in range(1,training_cfg.adam_steps+1):
            adam.zero_grad(); total,_,_ = losses(); total.backward(); adam.step()
            record('Adam',epoch,epoch==1 or epoch%training_cfg.print_every==0 or epoch==training_cfg.adam_steps)
        with torch.no_grad(): q.copy_(best_q)
        lbfgs = torch.optim.LBFGS([q],lr=1.,max_iter=training_cfg.lbfgs_max_iter,
                                history_size=100,tolerance_grad=1e-12,tolerance_change=1e-14,
                                line_search_fn='strong_wolfe')
        for epoch in range(1,training_cfg.lbfgs_epochs+1):
            def closure():
                lbfgs.zero_grad(); total,_,_ = losses(); total.backward(); return total
            lbfgs.step(closure); record('L-BFGS',epoch,True)
        coefficients = (basis@best_q.cpu().numpy())/column_scale
        print(f'Lowest-loss coefficients: {origin}; objective={best:.8e}',flush=True)
    residual = matrix@coefficients-rhs
    diagnostics = {
        'ridge':float(ridge),
        'relative_boundary_residual':float(np.linalg.norm(residual)/np.linalg.norm(rhs)),
        'coefficient_l2_norm':float(np.linalg.norm(coefficients)),
        'constraint_residual':float(np.linalg.norm(constraints@coefficients)),
        'condition_number_regularized':float(singular_values[0]/singular_values[-1]),
        'used_warm_start':bool(training_cfg.use_warm_start) if training_cfg is not None and training_cfg.optimizer=='adam_lbfgs' else bool(use_warm),
        'optimizer':training_cfg.optimizer if training_cfg is not None else 'svd',
        'best_origin':origin, 'training_history':history, 'effective_rank':int(rank)}
    return coefficients, diagnostics


def evaluate_on_grid(
    lam: float,
    cfg: SolverConfig,
    sources: np.ndarray,
    pole_order: int,
    use_mapped_inner: bool,
    coefficients: np.ndarray,
    grid_size: int,
) -> Dict[str, np.ndarray]:
    coordinates = np.linspace(-0.99 * L / 2.0, 0.99 * L / 2.0, grid_size)
    xx, yy = np.meshgrid(coordinates, coordinates)
    b_minor = A_MAJOR * lam
    valid = xx**2 / A_MAJOR**2 + yy**2 / b_minor**2 > 1.02
    z_valid = xx[valid] + 1j * yy[valid]

    maps = {key: np.full_like(xx, np.nan, dtype=float) for key in ["Sxx", "Syy", "Sxy", "VM"]}
    collected = {key: [] for key in maps}
    chunk_size = 1200
    for start in range(0, z_valid.size, chunk_size):
        influence = field_influence(
            z_valid[start : start + chunk_size], sources, lam, pole_order,
            use_mapped_inner, cfg
        )
        sxx = influence[2] @ coefficients * STRESS_TO_PA
        syy = influence[3] @ coefficients * STRESS_TO_PA
        sxy = influence[4] @ coefficients * STRESS_TO_PA
        vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)
        for key, values in zip(["Sxx", "Syy", "Sxy", "VM"], [sxx, syy, sxy, vm]):
            collected[key].append(values)

    for key in maps:
        maps[key][valid] = np.concatenate(collected[key])
    maps["XX"] = xx
    maps["YY"] = yy
    return maps


def plot_source_distribution(
    lam: float,
    cfg: SolverConfig,
    sources: np.ndarray,
    n_inner: int,
    use_mapped_inner: bool,
    output_dir: Path,
) -> None:
    b_minor = A_MAJOR * lam
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    inner = sources[:n_inner]
    outer = sources[n_inner:]
    source_extent = cfg.outer_source_scale * L / 2.0

    if use_mapped_inner:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), constrained_layout=True)
        ax, ax_zeta = axes
    else:
        fig, ax = plt.subplots(figsize=(7.0, 6.6), constrained_layout=True)
        ax_zeta = None
    ax.add_patch(
        Rectangle((-L / 2.0, -L / 2.0), L, L, fill=False, color="#222222", lw=1.6, label="Physical boundary")
    )
    ax.add_patch(
        Rectangle((-source_extent, -source_extent), 2 * source_extent, 2 * source_extent,
                  fill=False, color="#7A7A7A", lw=1.0, ls="--", label="Artificial source boundary")
    )
    ax.plot(A_MAJOR * np.cos(theta), b_minor * np.sin(theta), color="#222222", lw=1.6)
    ax.scatter(outer.real, outer.imag, s=25, color="#C43C39", marker="s", label="Outer sources")
    if n_inner:
        ax.scatter(inner.real, inner.imag, s=22, color="#2F6B9A", label="Inner sources")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08 * source_extent, 1.08 * source_extent)
    ax.set_ylim(-1.08 * source_extent, 1.08 * source_extent)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(rf"Physical boundary and fictitious sources ($\lambda={lam:g}$)")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.97, 0.92), fontsize=8)
    if ax_zeta is not None:
        unit_circle = np.exp(1j * theta)
        source_circle = cfg.mapped_source_radius * np.exp(1j * theta)
        source_angles = (
            np.arange(cfg.n_mapped_sources, dtype=float) + 0.5
        ) * (2.0 * np.pi / cfg.n_mapped_sources)
        mapped_sources = cfg.mapped_source_radius * np.exp(1j * source_angles)
        ax_zeta.plot(unit_circle.real, unit_circle.imag, color="#222222", lw=1.6,
                     label=r"Mapped hole boundary $|\zeta|=1$")
        ax_zeta.plot(source_circle.real, source_circle.imag, color="#7A7A7A", lw=1.0,
                     ls="--", label=rf"Source circle $|\eta|={cfg.mapped_source_radius:g}$")
        eta_tip = np.sqrt((1.0-lam)/(1.0+lam))
        ax_zeta.scatter([eta_tip,-eta_tip],[0,0],s=50,marker="*",color="#252525",
                        label="Near-tip enrichment poles")
        ax_zeta.scatter(mapped_sources.real, mapped_sources.imag, s=22,
                        color="#2F6B9A", label="Mapped analytic kernels")
        ax_zeta.set_aspect("equal", adjustable="box")
        ax_zeta.set_xlim(-1.12, 1.12)
        ax_zeta.set_ylim(-1.12, 1.12)
        ax_zeta.set_xlabel(r"$\mathrm{Re}(\zeta)$")
        ax_zeta.set_ylabel(r"$\mathrm{Im}(\zeta)$")
        ax_zeta.set_title("Auxiliary Joukowski plane")
        ax_zeta.legend(frameon=False, loc="center", fontsize=8)
    fig.savefig(output_dir / f"source_distribution_lambda_{lam:g}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stress_maps(
    lam: float,
    result: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    fields = [result["Sxx"], result["Syy"], result["Sxy"], result["VM"]]
    titles = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$", r"$\sigma_{\mathrm{vM}}$"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), constrained_layout=True)
    extent = [-L / 2.0, L / 2.0, -L / 2.0, L / 2.0]
    for index, (ax, field, title) in enumerate(zip(axes.flat, fields, titles)):
        if index == 3:
            vmin = 0.0
            vmax = np.nanpercentile(field, 99.5)
            cmap = "turbo"
        else:
            vmax = np.nanpercentile(np.abs(field), 99.5)
            vmin = -vmax
            cmap = "RdBu_r"
        image = ax.imshow(field, origin="lower", extent=extent, cmap=cmap,
                          vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="equal")
        ax.add_patch(Ellipse((0.0, 0.0), 2.0 * A_MAJOR, 2.0 * A_MAJOR * lam,
                             facecolor="white", edgecolor="#333333", lw=0.5, zorder=10))
        ax.set_title(title + rf" (Pa), $\lambda={lam:g}$")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.formatter.set_powerlimits((0, 0))
    fig.savefig(output_dir / f"stress_maps_lambda_{lam:g}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_tip_profile(
    lam: float,
    cfg: SolverConfig,
    sources: np.ndarray,
    pole_order: int,
    use_mapped_inner: bool,
    coefficients: np.ndarray,
    output_dir: Path,
    make_plot: bool = True,
) -> float:
    """Resolve the localized stress ahead of both tips on a logarithmic grid."""
    tip_radius = A_MAJOR * lam**2
    distance = np.geomspace(max(0.1 * tip_radius, 1.0e-6), 0.25, 360)
    z_right = A_MAJOR + distance + 0j
    z_left = -A_MAJOR - distance + 0j

    vm_values = []
    for z_points in [z_right, z_left]:
        influence = field_influence(
            z_points, sources, lam, pole_order, use_mapped_inner, cfg
        )
        sxx = influence[2] @ coefficients * STRESS_TO_PA
        syy = influence[3] @ coefficients * STRESS_TO_PA
        sxy = influence[4] @ coefficients * STRESS_TO_PA
        vm_values.append(np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2))
    vm_right, vm_left = vm_values
    vm_average = 0.5 * (vm_right + vm_left)

    profile = np.column_stack([distance, vm_left, vm_right, vm_average])
    np.savetxt(
        output_dir / f"tip_profile_lambda_{lam:g}.csv",
        profile,
        delimiter=",",
        header="distance_from_tip_m,vm_left_pa,vm_right_pa,vm_average_pa",
        comments="",
    )

    if not make_plot:
        return float(np.max(np.concatenate([vm_left, vm_right])))
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.loglog(distance / A_MAJOR, vm_left, color="#3A6EA5", lw=1.6, label="Left tip")
    ax.loglog(distance / A_MAJOR, vm_right, color="#C44E52", lw=1.6, ls="--", label="Right tip")
    ax.set_xlabel(r"Normalized distance from tip, $r/a$")
    ax.set_ylabel(r"$\sigma_{\mathrm{vM}}$ (Pa)")
    ax.set_title(rf"Near-tip stress profile ($\lambda={lam:g}$)")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.45)
    ax.legend(frameon=False)
    fig.savefig(output_dir / f"tip_profile_lambda_{lam:g}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return float(np.max(np.concatenate([vm_left, vm_right])))


def solve_case(
    lam: float,
    cfg: SolverConfig,
    grid_size: int,
    output_dir: Path,
    individual_plots: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    start_time = time.perf_counter()
    sources, n_inner, use_mapped_inner, pole_order, coefficients, diagnostics = solve_boundary_coefficients(
        lam, cfg, output_dir)
    result = evaluate_on_grid(
        lam, cfg, sources, pole_order, use_mapped_inner, coefficients, grid_size
    )
    max_tip_vm = save_tip_profile(
        lam, cfg, sources, pole_order, use_mapped_inner, coefficients, output_dir,
        make_plot=individual_plots
    )
    diagnostics.update(
        {
            "lambda": float(lam),
            "sources": float(sources.size),
            "mapped_sources": float(cfg.n_mapped_sources if use_mapped_inner else 0),
            "pole_order": float(pole_order),
            "trainable_coefficients": float(coefficients.size),
            "max_von_mises_pa": float(np.nanmax(result["VM"])),
            "max_tip_profile_von_mises_pa": max_tip_vm,
            "elapsed_seconds": float(time.perf_counter() - start_time),
        }
    )

    if individual_plots:
        plot_source_distribution(lam, cfg, sources, n_inner, use_mapped_inner, output_dir)
        plot_stress_maps(lam, result, output_dir)
    np.savez_compressed(
        output_dir / f"km_pikfnn_fields_lambda_{lam:g}.npz",
        **result,
        coefficients=coefficients,
        sources=sources,
        mapped_source_radius=np.array(cfg.mapped_source_radius if use_mapped_inner else np.nan),
    )
    print(
        f"lambda={lam:g} | physical sources={sources.size} | "
        f"mapped sources={cfg.n_mapped_sources if use_mapped_inner else 0} | "
        f"Laurent order={pole_order} | "
        f"relative boundary residual={diagnostics['relative_boundary_residual']:.3e} | "
        f"time={diagnostics['elapsed_seconds']:.2f} s"
    )
    return result, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected point-force KM-PIKFNN solver")
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[1.0, 0.5, 0.2, 0.1, 0.05, 0.02])
    parser.add_argument("--grid-size", type=int, default=420)
    parser.add_argument("--inner-source-scale", type=float, default=0.70)
    parser.add_argument("--outer-source-scale", type=float, default=1.25)
    parser.add_argument("--mapped-source-radius", type=float, default=0.65)
    parser.add_argument("--mapped-sources", type=int, default=64)
    parser.add_argument("--mapped-threshold", type=float, default=0.20)
    parser.add_argument("--ridge", type=float, default=1.0e-16)
    parser.add_argument("--output-dir", type=Path,
                        default=Path.cwd() / "KM_PIKFNN_Six_Cases_Results")
    parser.add_argument('--hole-data',type=Path,default=Path(__file__).resolve().parent/'hole stress.txt')
    parser.add_argument('--tip-data',type=Path,default=Path(__file__).resolve().parent/'tip_stress.txt')
    parser.add_argument('--individual-plots',action='store_true')
    parser.add_argument('--plot-only',action='store_true',
                        help='Redraw the hole-boundary comparison from saved CSV files, without solving.')
    parser.add_argument('--optimizer',choices=['adam_lbfgs','svd'],default='adam_lbfgs')
    parser.add_argument('--no-warm-start',action='store_true')
    parser.add_argument('--adam-steps',type=int,default=1000)
    parser.add_argument('--adam-lr',type=float,default=1e-3)
    parser.add_argument('--lbfgs-epochs',type=int,default=20)
    parser.add_argument('--lbfgs-max-iter',type=int,default=20)
    parser.add_argument('--print-every',type=int,default=100)
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    if not (0.0 < args.inner_source_scale < 1.0):
        raise ValueError("inner-source-scale must satisfy 0 < scale < 1")
    if args.outer_source_scale <= 1.0:
        raise ValueError("outer-source-scale must be greater than 1 so sources remain outside the plate")
    if not (0.0 < args.mapped_source_radius < 1.0):
        raise ValueError("mapped-source-radius must satisfy 0 < rho < 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = SolverConfig(
        inner_source_scale=args.inner_source_scale,
        outer_source_scale=args.outer_source_scale,
        mapped_source_radius=args.mapped_source_radius,
        n_mapped_sources=args.mapped_sources,
        mapped_lambda_threshold=args.mapped_threshold,
        ridge_relative=args.ridge,
        optimizer=args.optimizer,use_warm_start=not args.no_warm_start,
        adam_steps=args.adam_steps,adam_lr=args.adam_lr,
        lbfgs_epochs=args.lbfgs_epochs,lbfgs_max_iter=args.lbfgs_max_iter,
        print_every=args.print_every,
    )

    diagnostics_rows: List[Dict[str, float]] = []
    field_cases = []
    for lam in args.lambdas:
        result, diagnostics = solve_case(float(lam), cfg, args.grid_size, output_dir,
                                         individual_plots=args.individual_plots)
        field_cases.append((float(lam),result))
        diagnostics_rows.append(diagnostics)
    plot_grouped_stress_fields(field_cases,output_dir)

    diagnostics_path = output_dir / "solver_diagnostics.csv"
    fieldnames = list(diagnostics_rows[0].keys())
    with diagnostics_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics_rows)
    print(f"Results saved to: {output_dir}")



# ============================================================================
# 孔边 COMSOL / KM-PIKFNN 对比（新增末尾代码）
# 把 hole stress.txt 放在本脚本旁，或修改 COMSOL_HOLE_FILE。
# 默认运行原求解后绘图；仅绘图：python 本脚本.py --compare-only
# 有已保存的 NPZ 时读取系数；没有时仅组装并求解边界，不生成全域云图。
# ============================================================================
import sys
import warnings

SCRIPT_DIRECTORY = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
COMSOL_HOLE_FILE = SCRIPT_DIRECTORY / 'hole stress.txt'
SHOW_COMPARISON = True
SAVE_COMPARISON = True


REFINED_SOLVER_VERSION = 5
_BOUNDARY_SOLUTIONS = {}


def solve_boundary_coefficients(lam, cfg, output_dir=None):
    sources,n_inner,mapped=build_source_points(lam,cfg)
    order=pole_order_for_lambda(lam)
    matrix,rhs,constraints=assemble_boundary_system(lam,cfg,sources,order,mapped,n_inner)
    signature=json.dumps({'version':REFINED_SOLVER_VERSION,'lambda':float(lam),
                          'cfg':asdict(cfg),'L':L,'a':A_MAJOR,'E':E_REAL,'nu':NU,'u0':U0_REAL},
                         sort_keys=True)
    if signature in _BOUNDARY_SOLUTIONS:
        print(f'lambda={lam:g}: 复用本次运行的最终系数用于绘图，不重复训练。',flush=True)
        return _BOUNDARY_SOLUTIONS[signature]
    cache_path=Path(output_dir)/f'boundary_coefficients_lambda_{lam:g}.npz' if output_dir else None
    initial=None
    if cfg.use_warm_start and cache_path and cache_path.exists():
        with np.load(cache_path) as cached:
            if str(cached['configuration_signature'])==signature:
                initial=cached['coefficients'].copy()
                if initial.shape != (matrix.shape[1],) or not np.all(np.isfinite(initial)):
                    initial=None
            else:
                print(f'lambda={lam:g}: 配置变化，重新求解，不复用旧系数。')
    coefficients,diagnostics=solve_scaled_constrained_ridge(
        matrix,rhs,constraints,cfg.ridge_relative,initial_coefficients=initial,training_cfg=cfg)
    history=diagnostics.pop('training_history',[])
    if output_dir and history:
        import csv
        Path(output_dir).mkdir(parents=True,exist_ok=True)
        with (Path(output_dir)/f'training_history_lambda_{lam:g}.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.writer(stream)
            writer.writerow(['stage','iteration','total_loss','weighted_boundary_loss','regularization'])
            writer.writerows(history)
    # Independent validation points differ from both training and COMSOL samples.
    theta=(np.arange(1907)+0.371)*2*np.pi/1907
    if mapped:
        offsets=np.geomspace(max(lam*1.31e-3,1e-8),0.291,223)
        theta=np.r_[theta,offsets,2*np.pi-offsets,np.pi-offsets,np.pi+offsets]
    z=A_MAJOR*np.cos(theta)+1j*A_MAJOR*lam*np.sin(theta)
    fields=field_influence(z,sources,lam,order,mapped,cfg)
    sxx,syy,sxy=[fields[i]@coefficients*STRESS_TO_PA for i in (2,3,4)]
    nx,ny=lam*np.cos(theta),np.sin(theta)
    length=np.hypot(nx,ny);nx/=length;ny/=length
    traction=np.hypot(nx*sxx+ny*sxy,nx*sxy+ny*syy)
    diagnostics['holdout_max_hole_traction_pa']=float(np.max(traction))
    if cache_path:
        cache_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache_path,coefficients=coefficients,configuration_signature=signature)
    result=(sources,n_inner,mapped,order,coefficients,diagnostics)
    _BOUNDARY_SOLUTIONS[signature]=result
    return result


def read_comsol_hole_data(path):
    """读取当前导出格式 x,y,lamda,sxx,syy,sxy,vm；所有坐标 m，应力 Pa。"""
    data = np.loadtxt(path, comments='%', ndmin=2, encoding='utf-8-sig')
    if data.shape[1] != 7:
        raise ValueError('需要7列：x,y,lamda,solid.sx,solid.sy,solid.sxy,solid.mises')
    if not np.all(np.isfinite(data)):
        warnings.warn('发现非有限值：相关行已剔除，请检查原始导出文件。')
        data = data[np.all(np.isfinite(data), axis=1)]
    if not len(data):
        raise ValueError('没有有效孔边数据')
    if np.any(data[:, 2] <= 0):
        raise ValueError('孔径比必须大于0')
    return data


def split_at_angular_gaps(theta_deg, values):
    """缺失较长角度区间时断开曲线，避免跨缺口连线。"""
    differences = np.diff(theta_deg)
    positive = differences[differences > 1e-8]
    threshold = 3.0 * np.median(positive) if positive.size else np.inf
    xx, yy = [], []
    for i, (angle, value) in enumerate(zip(theta_deg, values)):
        if i and theta_deg[i] - theta_deg[i - 1] > threshold:
            xx.append(np.nan); yy.append(np.nan)
        xx.append(angle); yy.append(value)
    return np.asarray(xx), np.asarray(yy)


def sampled_relative_l2(prediction, reference):
    """在实际有效导出点上的离散相对误差；不是连续孔周积分误差。"""
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(prediction-reference)/denominator) if denominator > 0 else float('nan')


def plot_hole_boundary_comparison(comparisons, output_dir, show=True, save=True,
                                  filename='hole_boundary_four_stresses_comparison'):
    """Compact 4-by-N boundary-stress comparison for a two-column paper figure."""
    if not comparisons:
        raise ValueError('No hole-boundary curves are available for plotting.')
    output_dir = Path(output_dir)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
    style = {'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
             'mathtext.fontset':'stix', 'font.size':15, 'pdf.fonttype':42,
             'axes.labelsize':17, 'axes.titlesize':18,
             'xtick.labelsize':14, 'ytick.labelsize':14}
    with plt.rc_context(style):
        labels = [r'$\sigma_{xx}$',r'$\sigma_{yy}$',
                  r'$\sigma_{xy}$',r'$\sigma_{\mathrm{vM}}$']
        ncols = len(comparisons)
        fig, axes = plt.subplots(4,ncols,figsize=(2.55*ncols+1.2,9.2),
                                 squeeze=False,sharex='col')
        fig.subplots_adjust(left=.065,right=.995,bottom=.105,top=.945,
                            wspace=.19,hspace=.20)
        for col,(lam,degrees,km_fields,fem_fields) in enumerate(comparisons):
            for row in range(4):
                ax=axes[row,col]
                fx,fy=split_at_angular_gaps(degrees,fem_fields[:,row])
                kx,ky=split_at_angular_gaps(degrees,km_fields[:,row])
                ax.plot(fx,fy,color='#252525',lw=1.45,label='COMSOL FEM')
                ax.plot(kx,ky,color='#BE4540',lw=1.45,ls='--',dashes=(5,2.5),
                        label='CV-PIKFNN')
                ax.set_xlim(0,360)
                # Interior ticks avoid the 360/0 label collision between columns.
                ax.set_xticks([90,180,270])
                ax.tick_params(axis='both',direction='in',length=3.5,pad=2)
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),
                                    useMathText=True)
                ax.yaxis.get_offset_text().set_size(12)
                ax.grid(alpha=.15,lw=.45)
                if col==0:
                    ax.set_ylabel(labels[row],labelpad=7)
                if row==0:
                    ax.set_title(rf'$\lambda={lam:g}$',pad=14)
                if row==3:
                    ax.set_ylim(bottom=0)
                else:
                    ax.tick_params(axis='x',labelbottom=False)
        # Each panel retains its own y ticks: the stress scale changes with lambda.
        # Row labels, x-axis title, unit and method legend appear only once.
        fig.text(.012,.52,'Stress (Pa)',ha='center',va='center',rotation=90,
                 fontsize=15)
        fig.supxlabel(r'$\theta$ (deg)',y=.055,fontsize=16)
        handles,legend_labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='lower center',
                   bbox_to_anchor=(.5,.008),ncol=2,frameon=False,fontsize=15,
                   handlelength=2.4,columnspacing=2.8)
        if save:
            for ext in ('png','pdf'):
                fig.savefig(output_dir/f'{filename}.{ext}',
                            dpi=400,bbox_inches='tight',facecolor='white')
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_hole_boundary_layouts(comparisons, output_dir, show=True, save=True):
    """Keep the six-column overview and export two legible three-column halves."""
    plot_hole_boundary_comparison(comparisons,output_dir,show=show,save=save)
    if len(comparisons)>3 and save:
        for part, start in enumerate(range(0,len(comparisons),3),1):
            plot_hole_boundary_comparison(
                comparisons[start:start+3],output_dir,show=False,save=True,
                filename=f'hole_boundary_four_stresses_part{part}')


def redraw_hole_boundary_from_csv(output_dir, lambdas, show=True):
    """Replot saved sampled stresses without invoking the elasticity solver."""
    output_dir=Path(output_dir)
    comparisons=[]
    for lam in lambdas:
        path=output_dir/f'hole_boundary_comparison_lambda_{lam:g}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing saved boundary curves: {path}')
        values=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2)
        if values.shape[1]!=11:
            raise ValueError(f'Expected 11 columns in {path}, found {values.shape[1]}')
        order=np.argsort(values[:,0])
        values=values[order]
        fem=np.column_stack((values[:,3:6],values[:,9]))
        km=np.column_stack((values[:,6:9],values[:,10]))
        comparisons.append((float(lam),values[:,0],km,fem))
    plot_hole_boundary_layouts(comparisons,output_dir,show=show)


def compare_comsol_hole_stress(data_file, cfg, output_dir, show=True, save=True, lambdas=None):
    data = read_comsol_hole_data(data_file)
    output_dir = Path(output_dir)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
    comparisons, metrics = [], []
    selected = np.unique(data[:,2])[::-1] if lambdas is None else np.asarray(lambdas,dtype=float)
    missing = [float(lam) for lam in selected
               if not np.any(np.isclose(data[:,2],lam,rtol=1e-10,atol=1e-12))]
    if missing:
        raise ValueError(f'孔边文件缺少工况 {missing}；请使用重新导出的数据，已有工况为 {np.unique(data[:,2]).tolist()}')
    for lam in selected:
        block = data[np.isclose(data[:, 2], lam, rtol=1e-10, atol=1e-12)]
        x, y = block[:, 0], block[:, 1]
        boundary_error = np.max(np.abs((x/A_MAJOR)**2+(y/(A_MAJOR*lam))**2-1))
        if boundary_error > 1e-5:
            raise ValueError(f'lambda={lam:g}: 坐标不在对应孔边，最大椭圆残差={boundary_error:.3e}')
        theta = np.mod(np.arctan2(y/(A_MAJOR*lam), x/A_MAJOR), 2*np.pi)
        theta[np.isclose(theta, 2*np.pi, atol=1e-10, rtol=0)] = 0.0
        ids = np.argsort(theta)
        block, theta = block[ids], theta[ids]
        # 合并0/360度等重复点，避免误差中重复计数。
        keep = np.r_[True, np.diff(theta) > 1e-10]
        block, theta = block[keep], theta[keep]
        z = block[:, 0] + 1j*block[:, 1]

        sources, n_inner, mapped, order, coefficients, diagnostics = solve_boundary_coefficients(
            float(lam), cfg, output_dir if save else None)
        origin = 'warm-start SVD correction' if diagnostics['used_warm_start'] else 'cold-start augmented SVD'
        influence = field_influence(z, sources, float(lam), order, mapped, cfg)
        km = np.column_stack([influence[i] @ coefficients * STRESS_TO_PA for i in (2,3,4)])
        fem = block[:, 3:6]
        km_vm = np.sqrt(km[:,0]**2-km[:,0]*km[:,1]+km[:,1]**2+3*km[:,2]**2)
        # 使用导出的COMSOL von Mises；不替换为另行计算的值。
        fem_vm = block[:, 6]
        degrees = np.degrees(theta)
        maximum_gap = np.degrees(np.max(np.diff(np.r_[theta, theta[0]+2*np.pi])))
        row = {'lambda':float(lam), 'valid_unique_points':len(theta),
               'max_angle_gap_deg':float(maximum_gap)}
        for j,name in enumerate(('sxx','syy','sxy')):
            row[f'sampled_relative_L2_{name}'] = sampled_relative_l2(km[:,j],fem[:,j])
        row['sampled_relative_L2_vm'] = sampled_relative_l2(km_vm,fem_vm)
        row['relative_boundary_residual'] = diagnostics['relative_boundary_residual']
        row['used_warm_start'] = diagnostics['used_warm_start']
        nx,ny=lam*np.cos(theta),np.sin(theta)
        length=np.hypot(nx,ny);nx/=length;ny/=length
        fem_traction=np.hypot(nx*fem[:,0]+ny*fem[:,2],nx*fem[:,2]+ny*fem[:,1])
        km_traction=np.hypot(nx*km[:,0]+ny*km[:,2],nx*km[:,2]+ny*km[:,1])
        row['fem_max_hole_traction_pa']=float(np.max(fem_traction))
        row['km_max_hole_traction_pa']=float(np.max(km_traction))
        row['km_holdout_max_hole_traction_pa']=diagnostics['holdout_max_hole_traction_pa']

        metrics.append(row)
        comparisons.append((float(lam), degrees,
                            np.column_stack([km,km_vm]),np.column_stack([fem,fem_vm])))
        print(f'lambda={lam:g}: N={len(theta)}, sampled L2(syy)={row["sampled_relative_L2_syy"]:.3e}, '
              f'max angle gap={maximum_gap:.3f} deg; {origin}')
        if save:
            columns = np.column_stack([degrees, block[:,:2], fem, km, fem_vm, km_vm])
            np.savetxt(output_dir/f'hole_boundary_comparison_lambda_{lam:g}.csv', columns,
                       delimiter=',', comments='',
                       header='theta_deg,x_m,y_m,fem_sxx_pa,fem_syy_pa,fem_sxy_pa,km_sxx_pa,km_syy_pa,km_sxy_pa,fem_vm_pa,km_vm_pa')

    plot_hole_boundary_layouts(comparisons,output_dir,show=show,save=save)
    if save:
        with (output_dir/'hole_boundary_error_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metrics[0]))
            writer.writeheader();writer.writerows(metrics)
    return metrics



# 六个孔径比在同一距离网格上比较右孔端前方的 von Mises 应力。
TIP_DECAY_MIN_DISTANCE = 1.0e-6     # m；不取孔端 d=0，便于双对数绘图
TIP_DECAY_MAX_DISTANCE = 0.25       # m；右孔端前方，仍在物理板内
TIP_DECAY_POINT_COUNT = 600


def plot_six_tip_decay_curves(lambdas, cfg, output_dir, show=True, save=True):
    """Evaluate at x=a+d, y=0; reuse matching coefficients, never grid interpolation.

    This figure contains KM-PIKFNN predictions only. COMSOL hole-boundary values
    cannot be used as reference values along this material-domain path.
    """
    lambdas=np.asarray(lambdas,dtype=float)
    if not len(lambdas) or np.any(lambdas<=0) or np.unique(lambdas).size != lambdas.size:
        raise ValueError('孔径比列表必须非空，且每项为不同的正数')
    dmin,dmax=TIP_DECAY_MIN_DISTANCE,TIP_DECAY_MAX_DISTANCE
    if not 0<dmin<dmax<L/2-A_MAJOR:
        raise ValueError('距离范围必须满足 0<dmin<dmax<L/2-a')
    if TIP_DECAY_POINT_COUNT<2:
        raise ValueError('距离采样点数必须至少为2')
    distance=np.geomspace(dmin,dmax,TIP_DECAY_POINT_COUNT)
    z=A_MAJOR+distance+0j
    profiles=[]
    output_dir=Path(output_dir)
    for lam in lambdas:
        sources,_,mapped,order,coefficients,_=solve_boundary_coefficients(
            float(lam),cfg,output_dir if save else None)
        influence=field_influence(z,sources,float(lam),order,mapped,cfg)
        sxx,syy,sxy=[influence[i]@coefficients*STRESS_TO_PA for i in (2,3,4)]
        vm=np.sqrt(sxx**2-sxx*syy+syy**2+3*sxy**2)
        if not np.all(np.isfinite(vm)) or np.any(vm<=0):
            raise ValueError(f'lambda={lam:g}: 路径应力存在非正值或非有限值，不能直接画双对数曲线')
        profiles.append(vm)
    colors=['#326A93','#BE4540','#558657','#9A74A8','#D28C36','#4C9B9A']
    line_styles=['-','--','-.',':','-','--']
    style={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
           'mathtext.fontset':'stix','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        fig,ax=plt.subplots(figsize=(7.2,5.3),dpi=160)
        for i,(lam,vm) in enumerate(zip(lambdas,profiles)):
            ax.loglog(distance/A_MAJOR,vm,lw=1.8,color=colors[i%len(colors)],
                      ls=line_styles[i%len(line_styles)],label=rf'$\lambda={lam:g}$')
        ax.set_xlabel(r'Normalized distance ahead of the right hole end, $d/a$')
        ax.set_ylabel(r'$\sigma_{\mathrm{vM}}$ (Pa)')
        ax.set_xlim(distance[0]/A_MAJOR,distance[-1]/A_MAJOR)
        ax.grid(which='major',color='#D6D6D6',lw=.6,alpha=.65)
        ax.grid(which='minor',color='#E5E5E5',lw=.4,alpha=.3)
        ax.legend(loc='upper right',frameon=False,ncol=2,fontsize=11,
                  handlelength=2.6,columnspacing=1.1)
        fig.tight_layout()
        paths=[]
        if save:
            output_dir.mkdir(parents=True,exist_ok=True)
            for ext in ('png','pdf','svg'):
                path=output_dir/f'six_tip_decay_curves.{ext}'
                fig.savefig(path,dpi=400,bbox_inches='tight',facecolor='white')
                paths.append(path)
            table=np.column_stack([distance,distance/A_MAJOR,*profiles])
            header='distance_from_right_hole_end_m,d_over_a,'+','.join(
                f'vm_lambda_{lam:g}_pa' for lam in lambdas)
            np.savetxt(output_dir/'six_tip_decay_curves.csv',table,delimiter=',',header=header,comments='')
            print(f'Six tip-decay curves saved to: {output_dir}')
        if show: plt.show()
        else: plt.close(fig)
    return distance,np.column_stack(profiles),paths



COMSOL_TIP_FILE = SCRIPT_DIRECTORY / 'tip_stress.txt'
TIP_COMPARISON_MARKERS = 26       # 每条COMSOL曲线显示的标记数；误差使用全部点
TIP_COMPARISON_LEGEND_SIZE = 11


def read_comsol_tip_data(path):
    """Accept COMSOL long (7 columns) or wide (2+4*N columns) sweep exports."""
    import re
    path=Path(path)
    data=np.loadtxt(path,comments='%',ndmin=2,encoding='utf-8-sig')
    cases={}
    if data.shape[1]==7:
        for lam in np.unique(data[:,2]):
            block=data[np.isclose(data[:,2],lam,rtol=1e-10,atol=1e-12)]
            cases[float(lam)]=np.column_stack([block[:,:2],block[:,3:7]])
    elif data.shape[1]>2 and (data.shape[1]-2)%4==0:
        header=''
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.lstrip().startswith('%'):break
                if 'solid.sx' in line and '@' in line:header=line
        entries=re.findall(r'solid\.(sx|sy|sxy|mises)\s*\([^)]*\)\s*@\s*(?:lamda|lambda|lam)\s*=\s*([+\-\d.eE]+)',header)
        if len(entries)!=data.shape[1]-2:
            raise ValueError('无法从表头识别工况和应力列；需要 @ lamda=... 等标记')
        for offset in range(0,len(entries),4):
            group=entries[offset:offset+4]
            if [item[0] for item in group]!=['sx','sy','sxy','mises'] or len({float(item[1]) for item in group})!=1:
                raise ValueError('每组应按 sx,sy,sxy,mises 排列，且具有相同孔径比')
            lam=float(group[0][1])
            if lam in cases:raise ValueError('导出文件包含重复工况列')
            cases[lam]=np.column_stack([data[:,:2],data[:,2+offset:6+offset]])
    else:
        raise ValueError('不支持的孔端数据列数；需要7列长格式或2+4*N列宽格式')
    for lam,block in cases.items():
        if lam<=0 or not np.all(np.isfinite(block)):
            raise ValueError(f'lambda={lam:g}: 工况或数据无效，请检查导出文件')
        if np.max(np.abs(block[:,1]))>1e-9 or np.any(block[:,0]<=A_MAJOR) or np.any(block[:,0]>=L/2):
            raise ValueError(f'lambda={lam:g}: 需要右孔端前方路径 x>a,y=0，且在板内')
        if np.any(block[:,5]<=0):
            raise ValueError('von Mises应力需为正值，才能使用双对数图')
        ids=np.argsort(block[:,0]);block=block[ids]
        keep=np.r_[True,np.diff(block[:,0])>1e-14]
        cases[lam]=block[keep]
    return cases


def plot_tip_decay_comsol_comparison(data_file,lambdas,cfg,output_dir,show=True,save=True):
    """KM lines + COMSOL markers, evaluated at identical actual FEM coordinates."""
    from matplotlib.lines import Line2D
    cases=read_comsol_tip_data(data_file)
    lambdas=np.asarray(lambdas,dtype=float)
    if not len(lambdas) or np.unique(lambdas).size!=lambdas.size:
        raise ValueError('需要非空且无重复项的孔径比列表')
    output_dir=Path(output_dir)
    colors=['#555555','#F04B4E','#2375D8','#29A36A','#AB6CD6','#10ACB5']
    markers=['s','o','^','v','D','>']
    collected=[];metrics=[]
    for lam in lambdas:
        match=[key for key in cases if np.isclose(key,lam,rtol=1e-10,atol=1e-12)]
        if len(match)!=1:raise ValueError(f'COMSOL文件缺少或重复 lambda={lam:g}')
        block=cases[match[0]]
        distance=block[:,0]-A_MAJOR
        sources,_,mapped,order,coefficients,_=solve_boundary_coefficients(
            float(lam),cfg,output_dir if save else None)
        influence=field_influence(block[:,0]+1j*block[:,1],sources,float(lam),order,mapped,cfg)
        stress=np.column_stack([influence[i]@coefficients*STRESS_TO_PA for i in (2,3,4)])
        km_vm=np.sqrt(stress[:,0]**2-stress[:,0]*stress[:,1]+stress[:,1]**2+3*stress[:,2]**2)
        if not np.all(np.isfinite(km_vm)) or np.any(km_vm<=0):
            raise ValueError('预测应力不能用于双对数坐标，请检查求解结果')
        fem_vm=block[:,5]
        row={'lambda':float(lam),'point_count':len(block),
             'd_min_m':float(distance[0]),'d_max_m':float(distance[-1]),
             'sampled_relative_L2_vm':sampled_relative_l2(km_vm,fem_vm)}
        for j,name in enumerate(('sxx','syy','sxy')):
            row[f'sampled_relative_L2_{name}']=sampled_relative_l2(stress[:,j],block[:,2+j])
        metrics.append(row);collected.append((float(lam),distance,km_vm,fem_vm))
        print(f'lambda={lam:g}: tip-path N={len(block)}, sampled L2(vm)={row["sampled_relative_L2_vm"]:.3e}')
        if save:
            output_dir.mkdir(parents=True,exist_ok=True)
            table=np.column_stack([distance,distance/A_MAJOR,block[:,:2],block[:,2:5],stress,fem_vm,km_vm])
            np.savetxt(output_dir/f'tip_decay_comparison_lambda_{lam:g}.csv',table,delimiter=',',comments='',
                       header='distance_m,d_over_a,x_m,y_m,fem_sxx_pa,fem_syy_pa,fem_sxy_pa,km_sxx_pa,km_syy_pa,km_sxy_pa,fem_vm_pa,km_vm_pa')
    style={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
           'mathtext.fontset':'stix','font.size':13,'pdf.fonttype':42,'svg.fonttype':'none',
           'axes.linewidth':1.1,'xtick.direction':'in','ytick.direction':'in'}
    with plt.rc_context(style):
        fig,ax=plt.subplots(figsize=(7.4,5.6),dpi=170)
        lambda_handles=[];all_vm=[];all_distances=[]
        for i,(lam,distance,km_vm,fem_vm) in enumerate(collected):
            color=colors[i%len(colors)];marker=markers[i%len(markers)]
            ax.loglog(distance/A_MAJOR,km_vm,color=color,lw=1.35,zorder=2)
            # Select actual exported points at approximately equal log-distance intervals.
            log_d=np.log(distance)
            targets=np.linspace(log_d[0],log_d[-1],max(2,TIP_COMPARISON_MARKERS))
            ids=np.unique(np.argmin(abs(log_d[:,None]-targets[None,:]),axis=0))
            ax.loglog(distance[ids]/A_MAJOR,fem_vm[ids],ls='None',marker=marker,
                      ms=5.2,mfc='white',mec=color,mew=1.05,color=color,zorder=3)
            lambda_handles.append(Line2D([],[],color=color,lw=1.1,marker=marker,ms=5,
                                         mfc='white',mec=color,label=rf'$\lambda={lam:g}$'))
            all_vm.extend([km_vm,fem_vm]);all_distances.append(distance/A_MAJOR)
        ax.set_xlabel(r'Distance ahead of the right hole end, $d/a$',fontsize=14)
        ax.set_ylabel(r'von Mises stress, $\sigma_{\mathrm{vM}}$ (Pa)',fontsize=14)
        ax.set_xlim(min(a[0] for a in all_distances),max(a[-1] for a in all_distances))
        ax.set_ylim(min(np.min(a) for a in all_vm)/1.65,max(np.max(a) for a in all_vm)*3.2)
        ax.tick_params(which='major',length=6,width=1.0,labelsize=12,top=True,right=True)
        ax.tick_params(which='minor',length=3,width=.8,top=True,right=True)
        ax.grid(False)
        legend=ax.legend(handles=lambda_handles,loc='upper right',ncol=3,frameon=False,
                         fontsize=TIP_COMPARISON_LEGEND_SIZE,handlelength=1.8,
                         columnspacing=1.1,handletextpad=.45,borderaxespad=.65)
        ax.add_artist(legend)
        method_handles=[Line2D([],[],color='#252525',lw=1.4,label='CV-PIKFNN'),
                        Line2D([],[],color='#252525',ls='None',marker='o',ms=5,
                               mfc='white',mec='#252525',label='COMSOL FEM')]
        ax.legend(handles=method_handles,loc='lower left',frameon=False,fontsize=11,
                  handlelength=2,borderaxespad=.7)
        fig.tight_layout(pad=.8)
        paths=[]
        if save:
            for ext in ('png','pdf','svg'):
                path=output_dir/f'six_tip_decay_comsol_comparison.{ext}'
                fig.savefig(path,dpi=450,bbox_inches='tight',facecolor='white');paths.append(path)
            with (output_dir/'tip_decay_error_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
        if show:plt.show()
        else:plt.close(fig)
    return metrics,paths


def plot_grouped_stress_fields(field_cases, output_dir):
    """Four stress rows, requested case columns, one shared colorbar per row.

    Shared color limits use the pooled 99.5th percentile in each row;
    rectangular colorbar extensions indicate values outside those limits.
    """
    output_dir=Path(output_dir)
    keys=('Sxx','Syy','Sxy','VM')
    labels=(r'$\sigma_{xx}^{\mathrm{KM}}$',r'$\sigma_{yy}^{\mathrm{KM}}$',
            r'$\sigma_{xy}^{\mathrm{KM}}$',r'$\sigma_{\mathrm{vM}}^{\mathrm{KM}}$')
    style={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
           'mathtext.fontset':'stix','font.size':11,'pdf.fonttype':42}
    with plt.rc_context(style):
        nn=len(field_cases)
        fig=plt.figure(figsize=(3*nn+1,11.5),layout='constrained')
        gs=fig.add_gridspec(4,nn+1,width_ratios=[1]*nn+[.055],wspace=.06,hspace=.05)
        for row,key in enumerate(keys):
            values=np.concatenate([result[key][np.isfinite(result[key])] for _,result in field_cases])
            vmax=max(float(np.percentile(values if row==3 else abs(values),99.5)),1e-12)
            vmin=0 if row==3 else -vmax
            for col,(lam,result) in enumerate(field_cases):
                ax=fig.add_subplot(gs[row,col])
                xx,yy=result['XX'],result['YY']
                im=ax.imshow(result[key],origin='lower',extent=[xx.min(),xx.max(),yy.min(),yy.max()],
                             cmap='turbo' if row==3 else 'RdBu_r',vmin=vmin,vmax=vmax,
                             interpolation='bilinear',aspect='equal')
                ax.add_patch(Ellipse((0,0),2*A_MAJOR,2*A_MAJOR*lam,facecolor='white',
                                     edgecolor='#252525',lw=.65,zorder=5))
                ax.set_xlim(-L/2,L/2);ax.set_ylim(-L/2,L/2)
                ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.tick_params(labelsize=9)
                if row==0:ax.set_title(rf'$\lambda={lam:g}$')
                if col==0:ax.set_ylabel(labels[row]+'\n$y$ (m)')
                else:ax.tick_params(labelleft=False)
                if row==3:ax.set_xlabel('$x$ (m)')
                else:ax.tick_params(labelbottom=False)
            cax=fig.add_subplot(gs[row,nn])
            cb=fig.colorbar(im,cax=cax,extend='max' if row==3 else 'both',extendrect=True)
            cb.set_label('Stress (Pa)');cb.formatter.set_powerlimits((0,0));cb.update_ticks()
        for extension in ('png','pdf'):
            fig.savefig(output_dir/f'stress_fields_four_by_six.{extension}',dpi=350)
        plt.close(fig)


def exact_ellipse_uniaxial_fields(z, lam):
    """Infinite-plate ellipse solution, remote sigma_yy=1 in scaled units.

    Used only as an independent verification problem. Its boundary values
    on the finite square differ from the prescribed-displacement example.
    """
    z = np.asarray(z, dtype=complex).reshape(-1, 1)
    w, wz, wzz = joukowski_inverse(z, lam)
    c = A_MAJOR*(1+lam)/2
    mu = (1-lam)/(1+lam)
    aa, bb = c/4, -c*(2+mu)/4
    phi = aa*w+bb/w
    pw, pww = aa-bb/w**2, 2*bb/w**3
    ff, fp = 1/w+mu*w, -1/w**2+mu
    hh, hp = aa-bb/w**2, 2*bb/w**3
    jj, jp = 1-mu/w**2, 2*mu/w**3
    psi = -aa/w-bb*w-ff*hh/jj
    sw = aa/w**2-bb-((fp*hh+ff*hp)*jj-ff*hh*jp)/jj**2
    pz, pzz, sz = pw*wz, pww*wz**2+pw*wzz, sw*wz
    u = (KAPPA*phi-z*np.conj(pz)-np.conj(psi))/(2*G)
    ss, dd = 4*pz.real, 2*(np.conj(z)*pzz+sz)
    return tuple(v.ravel() for v in
                 (u.real,u.imag,(ss-dd.real)/2,(ss+dd.real)/2,dd.imag/2))


def independent_outer_boundary_checks(lam, cfg, sources, mapped, order, coefficients):
    """Validate on shifted points distinct from training points; physical units."""
    edge = -L/2+(np.arange(503)+.317)*L/503
    displacement_error, traction_error = [], []
    for y, target in ((L/2,U0),(-L/2,-U0)):
        f = field_influence(edge+1j*y,sources,lam,order,mapped,cfg)
        displacement_error.extend([abs(f[0]@coefficients),abs(f[1]@coefficients-target)])
    for x in (-L/2,L/2):
        f = field_influence(x+1j*edge,sources,lam,order,mapped,cfg)
        traction_error.append(np.hypot(f[2]@coefficients,f[4]@coefficients))
    return {'holdout_max_loaded_displacement_error_m':float(np.max(displacement_error)/U_SCALE),
            'holdout_max_side_traction_pa':float(np.max(traction_error)*STRESS_TO_PA)}


def validate_hole_refinement(data_path, output_dir, lam=.005):
    """Density check plus a separate exact-solution boundary verification.

    No COMSOL stress is used in fitting. All L2 errors below are discrete
    errors at the original exported coordinates, not arclength integrals.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    fem = read_comsol_tip_data(data_path)[lam]
    z = fem[:,0]+1j*fem[:,1]
    # Diagnose the exported reference's local interpolation, without assuming
    # that the interval equals a particular finite element's size.
    distance = fem[:,0]-A_MAJOR
    if len(fem)>2 and np.all(np.diff(distance)>0):
        slopes = np.diff(fem[:,3])/np.diff(distance)
        tolerance = max(abs(slopes[0])*1e-5,1.0)
        different = np.flatnonzero(abs(slopes-slopes[0])>tolerance)
        last = int(different[0]) if len(different) else len(slopes)
        selected = slice(0,last+1)
        line = np.polyfit(distance[selected]-distance[0],fem[selected,3],1) if last else None
        reference_check = {'lambda':lam,'tip_curvature_radius_m':A_MAJOR*lam**2,
                           'initial_linear_syy_interval_start_m':float(distance[0]),
                           'initial_linear_syy_interval_end_m':float(distance[last]),
                           'n_initial_linear_syy_points':last+1,
                           'relative_slope_tolerance':1e-5}
        if line is not None:
            reference_check['max_linear_fit_residual_pa'] = float(np.max(abs(
                fem[selected,3]-np.polyval(line,distance[selected]-distance[0]))))
        print('Exported reference interpolation check:',reference_check,flush=True)
        with (output_dir/'comsol_near_tip_interpolation_check.json').open('w') as stream:
            json.dump(reference_check,stream,indent=2)
    rows, curves = [], []
    for nh, nt in ((1440,200),(2880,400),(5760,800)):
        cfg = SolverConfig(n_hole=nh,tip_points_per_branch=nt,use_warm_start=False)
        sources, ni, mapped, order, coefficients, diagnostics = solve_boundary_coefficients(lam,cfg)
        influence = field_influence(z,sources,lam,order,mapped,cfg)
        stress = np.column_stack([influence[i]@coefficients*STRESS_TO_PA for i in (2,3,4)])
        vm = np.sqrt(stress[:,0]**2-stress[:,0]*stress[:,1]+stress[:,1]**2+3*stress[:,2]**2)
        matrix,rhs,constraints = assemble_boundary_system(lam,cfg,sources,order,mapped,ni)
        row = {'lambda':lam,'n_hole_uniform':nh,'tip_points_per_branch':nt,
               'weighted_boundary_loss_sum':float(np.sum((matrix@coefficients-rhs)**2)),
               'near_tip_vm_pa':float(vm[0]),
               'relative_boundary_residual':diagnostics['relative_boundary_residual'],
               'holdout_max_hole_traction_pa':diagnostics['holdout_max_hole_traction_pa']}
        for i,label in enumerate(('sxx','syy','sxy')):
            row[f'L2_{label}'] = sampled_relative_l2(stress[:,i],fem[:,i+2])
        row['L2_vm'] = sampled_relative_l2(vm,fem[:,5])
        row.update(independent_outer_boundary_checks(lam,cfg,sources,mapped,order,coefficients))
        rows.append(row)
        curves.append(vm)
        print('Refinement:',row,flush=True)
        if nh==1440:
            # Replace only exterior RHS by exact ellipse-solution boundary data.
            # The homogeneous, traction-free hole constraints remain unchanged.
            exact_rhs = rhs.copy()
            edge = boundary_points(lam,cfg)['edge']
            nn = cfg.n_side
            for k,y in enumerate((L/2,-L/2)):
                fields = exact_ellipse_uniaxial_fields(edge+1j*y,lam)
                for j in (0,1):
                    block = 2*k+j
                    exact_rhs[block*nn:(block+1)*nn] = np.sqrt(cfg.w_displacement/nn)*fields[j]/U0
            for k,x in enumerate((-L/2,L/2)):
                fields = exact_ellipse_uniaxial_fields(x+1j*edge,lam)
                for j,component in enumerate((2,4)):
                    block = 4+2*k+j
                    exact_rhs[block*nn:(block+1)*nn] = np.sqrt(cfg.w_side_traction/nn)*np.sign(x)*fields[component]
            exact_coefficients, exact_diagnostics = solve_scaled_constrained_ridge(
                matrix,exact_rhs,constraints,cfg.ridge_relative)
            expected = exact_ellipse_uniaxial_fields(z,lam)
            verification = {'lambda':lam,'relative_boundary_residual':exact_diagnostics['relative_boundary_residual']}
            for i,label in ((2,'sxx'),(3,'syy')):
                verification[f'L2_{label}'] = sampled_relative_l2(influence[i]@exact_coefficients,expected[i])
            tip = exact_ellipse_uniaxial_fields(np.array([A_MAJOR+0j]),lam)
            verification['exact_tip_syy_scaled'] = float(tip[3][0])
            verification['Inglis_tip_syy_scaled'] = 1+2/lam
            print('Separate analytic verification:',verification,flush=True)
            with (output_dir/'analytic_ellipse_verification.json').open('w') as stream:
                json.dump(verification,stream,indent=2)
    with (output_dir/'hole_refinement_checks.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    distance=fem[:,0]-A_MAJOR
    intervals=[]
    for lo,hi in ((1e-6,1e-5),(1e-5,1e-4),(1e-4,1e-3),(1e-3,1e-2),(1e-2,.251)):
        mask=(distance>=lo*(1-1e-9))&(distance<hi)
        if mask.any():
            intervals.append({'distance_min_m':lo,'distance_max_m':hi,
                              'n_points':int(mask.sum()),
                              'L2_vm':sampled_relative_l2(curves[1][mask],fem[mask,5])})
    with (output_dir/'tip_error_by_distance.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(intervals[0]))
        writer.writeheader();writer.writerows(intervals)
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    ax.loglog(distance,fem[:,5],color='black',lw=1.3,label='COMSOL')
    for row,vm,style in zip(rows,curves,('-','--',':')):
        ax.loglog(distance,vm,style,lw=1.5,label=f"KM: {row['n_hole_uniform']} + 4 x {row['tip_points_per_branch']}")
    ax.set_xlabel('Distance from hole tip (m)');ax.set_ylabel('von Mises stress (Pa)')
    ax.legend(frameon=False,fontsize=9)
    for extension in ('png','pdf'):
        fig.savefig(output_dir/f'hole_refinement_comparison.{extension}',dpi=350)
    plt.close(fig)
    return rows


if __name__ == '__main__':
    args = parse_args()
    if args.plot_only:
        redraw_hole_boundary_from_csv(args.output_dir,args.lambdas,show=SHOW_COMPARISON)
        sys.exit(0)
    if '--validate-hole' in sys.argv:
        validate_hole_refinement(args.tip_data,args.output_dir,
                                 lam=min(args.lambdas))
        sys.exit(0)
    # Check all requested comparison cases before spending time on field maps.
    if not args.tip_data.exists():
        raise FileNotFoundError(f'孔端数据不存在：{args.tip_data}；请放置新导出文件或使用 --tip-data 指定路径。')
    tip_cases=read_comsol_tip_data(args.tip_data)
    missing=[float(lam) for lam in args.lambdas
             if not any(np.isclose(lam,key,rtol=1e-10,atol=1e-12) for key in tip_cases)]
    if missing:
        raise ValueError(f'孔端文件缺少工况 {missing}；需要新六工况数据，已有工况为 {sorted(tip_cases,reverse=True)}')
    if '--tip-only' not in sys.argv:
        if not args.hole_data.exists():
            raise FileNotFoundError(f'孔边数据不存在：{args.hole_data}；请放置新导出文件或使用 --hole-data 指定路径。')
        hole_data=read_comsol_hole_data(args.hole_data)
        missing=[float(lam) for lam in args.lambdas
                 if not np.any(np.isclose(hole_data[:,2],lam,rtol=1e-10,atol=1e-12))]
        if missing:raise ValueError(f'孔边文件缺少工况 {missing}；请使用新导出文件。')
    if '--compare-only' not in sys.argv and '--tip-only' not in sys.argv:
        main()
    comparison_cfg = SolverConfig(
        inner_source_scale=args.inner_source_scale,outer_source_scale=args.outer_source_scale,
        mapped_source_radius=args.mapped_source_radius,n_mapped_sources=args.mapped_sources,
        mapped_lambda_threshold=args.mapped_threshold,ridge_relative=args.ridge,
        optimizer=args.optimizer,use_warm_start=not args.no_warm_start,
        adam_steps=args.adam_steps,adam_lr=args.adam_lr,
        lbfgs_epochs=args.lbfgs_epochs,lbfgs_max_iter=args.lbfgs_max_iter,
        print_every=args.print_every)
    if '--tip-only' not in sys.argv:
        compare_comsol_hole_stress(args.hole_data,comparison_cfg,args.output_dir,
                                  show=SHOW_COMPARISON,save=SAVE_COMPARISON,lambdas=args.lambdas)
    elif '--tip-only' not in sys.argv:
        print(f'孔边对比未运行：请将导出文件放到 {COMSOL_HOLE_FILE}，或修改 COMSOL_HOLE_FILE。')

    if args.tip_data.exists():
        plot_tip_decay_comsol_comparison(args.tip_data,args.lambdas,comparison_cfg,args.output_dir,
                                         show=SHOW_COMPARISON,save=SAVE_COMPARISON)
    else:
        raise FileNotFoundError(f'请将孔端路径数据放到 {COMSOL_TIP_FILE}，或修改 COMSOL_TIP_FILE。')

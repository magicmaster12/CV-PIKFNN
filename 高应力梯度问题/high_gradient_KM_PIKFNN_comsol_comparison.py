"""Corrected KM-PIKFNN solver for a square plate with a central elliptical hole.

The approximation uses KM point-force fundamental-solution kernels.  For
moderate aspect ratios, fictitious sources are placed on an inner curve inside
the excluded hole and on an artificial square outside the physical square.  For
highly flattened holes, the squeezed physical inner curve is replaced by a
fixed source circle in the Joukowski plane, supplemented by inverse-power
analytic enrichment functions.

For fixed source locations, all boundary residuals are linear in the kernel
coefficients.  The coefficients are therefore identified by a column-scaled,
regularized least-squares solve.  This minimizes the same boundary loss without
thousands of Adam/L-BFGS iterations.
"""

import argparse
import csv
import time
from dataclasses import dataclass
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
    n_side: int = 280
    n_hole: int = 720
    n_inner_sources: int = 96
    n_outer_sources_per_side: int = 24
    inner_source_scale: float = 0.70
    outer_source_scale: float = 1.25
    mapped_lambda_threshold: float = 0.20
    n_mapped_sources: int = 64
    mapped_source_radius: float = 0.85
    w_displacement: float = 20.0
    w_side_traction: float = 3.0
    w_hole_traction: float = 15.0
    ridge_relative: float = 1.0e-8


def pole_order_for_lambda(lam: float) -> int:
    """Analytic enrichment used only when the ellipse becomes crack-like."""
    if lam >= 0.20:
        return 0
    if lam >= 0.05:
        return 8
    if lam >= 0.02:
        return 12
    return 16


def build_source_points(lam: float, cfg: SolverConfig) -> Tuple[np.ndarray, int, bool]:
    """Return physical-plane sources and select the mapped-circle representation."""
    b_minor = A_MAJOR * lam
    use_mapped_inner = lam < cfg.mapped_lambda_threshold

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
    eta = (source_radius * np.exp(1j * theta)).reshape(1, -1)

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
    eta = (source_radius * np.exp(1j * theta)).reshape(1, -1)
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


def solve_scaled_constrained_ridge(
    matrix: np.ndarray,
    rhs: np.ndarray,
    constraints: np.ndarray,
    ridge_relative: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Stable constrained ridge solve with unit-norm design-matrix columns."""
    column_scale = np.linalg.norm(matrix, axis=0)
    column_scale = np.maximum(column_scale, 1.0e-14)
    scaled_matrix = matrix / column_scale
    scaled_constraints = constraints / column_scale

    gram = scaled_matrix.T @ scaled_matrix
    ridge = ridge_relative * np.trace(gram) / gram.shape[0]
    regularized_gram = gram + ridge * np.eye(gram.shape[0])
    if constraints.shape[0]:
        kkt = np.block(
            [
                [regularized_gram, scaled_constraints.T],
                [scaled_constraints, np.zeros((constraints.shape[0], constraints.shape[0]))],
            ]
        )
        kkt_rhs = np.concatenate(
            [scaled_matrix.T @ rhs, np.zeros(constraints.shape[0], dtype=float)]
        )
        scaled_solution = np.linalg.solve(kkt, kkt_rhs)[: matrix.shape[1]]
    else:
        scaled_solution = np.linalg.solve(
            regularized_gram, scaled_matrix.T @ rhs
        )
    solution = scaled_solution / column_scale
    residual = matrix @ solution - rhs

    diagnostics = {
        "ridge": float(ridge),
        "relative_boundary_residual": float(np.linalg.norm(residual) / np.linalg.norm(rhs)),
        "coefficient_l2_norm": float(np.linalg.norm(solution)),
        "constraint_residual": float(np.linalg.norm(constraints @ solution)),
        "condition_number_regularized": float(np.linalg.cond(regularized_gram)),
    }
    return solution, diagnostics


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
    chunk_size = 6000
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
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    start_time = time.perf_counter()
    sources, n_inner, use_mapped_inner = build_source_points(lam, cfg)
    pole_order = pole_order_for_lambda(lam)
    matrix, rhs, constraints = assemble_boundary_system(
        lam, cfg, sources, pole_order, use_mapped_inner, n_inner
    )
    coefficients, diagnostics = solve_scaled_constrained_ridge(
        matrix, rhs, constraints, cfg.ridge_relative
    )
    result = evaluate_on_grid(
        lam, cfg, sources, pole_order, use_mapped_inner, coefficients, grid_size
    )
    max_tip_vm = save_tip_profile(
        lam, cfg, sources, pole_order, use_mapped_inner, coefficients, output_dir
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

    plot_source_distribution(
        lam, cfg, sources, n_inner, use_mapped_inner, output_dir
    )
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
                        default=[1.0, 0.5, 0.05, 0.02, 0.01, 0.005])
    parser.add_argument("--grid-size", type=int, default=420)
    parser.add_argument("--inner-source-scale", type=float, default=0.70)
    parser.add_argument("--outer-source-scale", type=float, default=1.25)
    parser.add_argument("--mapped-source-radius", type=float, default=0.85)
    parser.add_argument("--mapped-sources", type=int, default=64)
    parser.add_argument("--mapped-threshold", type=float, default=0.20)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--output-dir", type=Path,
                        default=Path.cwd() / "KM_PIKFNN_Corrected_Results")
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
    )

    diagnostics_rows: List[Dict[str, float]] = []
    for lam in args.lambdas:
        _, diagnostics = solve_case(float(lam), cfg, args.grid_size, output_dir)
        diagnostics_rows.append(diagnostics)

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


def compare_comsol_hole_stress(data_file, cfg, output_dir, show=True, save=True):
    data = read_comsol_hole_data(data_file)
    output_dir = Path(output_dir)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
    comparisons, metrics = [], []
    for lam in np.unique(data[:, 2])[::-1]:
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

        sources, n_inner, mapped = build_source_points(float(lam), cfg)
        order = pole_order_for_lambda(float(lam))
        cache_path = output_dir / f'km_pikfnn_fields_lambda_{lam:g}.npz'
        if cache_path.exists():
            with np.load(cache_path) as cache:
                coefficients = cache['coefficients'].copy()
                cached_sources = cache['sources']
                if not np.array_equal(cached_sources, sources):
                    raise ValueError(f'{cache_path.name}: 缓存源点与当前配置不同，请使用生成缓存时的参数')
                if mapped and not np.isclose(float(cache['mapped_source_radius']), cfg.mapped_source_radius):
                    raise ValueError('缓存映射源半径与当前配置不同')
            expected = 2*len(sources)+4+(4*cfg.n_mapped_sources if mapped else 0)+4*order
            if len(coefficients) != expected:
                raise ValueError('缓存系数数量与当前配置不同')
            origin = 'cached coefficients'
        else:
            matrix, rhs, constraints = assemble_boundary_system(
                float(lam), cfg, sources, order, mapped, n_inner)
            coefficients, _ = solve_scaled_constrained_ridge(
                matrix, rhs, constraints, cfg.ridge_relative)
            origin = 'boundary least-squares solve'

        influence = field_influence(z, sources, float(lam), order, mapped, cfg)
        km = np.column_stack([influence[i] @ coefficients * STRESS_TO_PA for i in (2,3,4)])
        fem = block[:, 3:6]
        km_vm = np.sqrt(km[:,0]**2-km[:,0]*km[:,1]+km[:,1]**2+3*km[:,2]**2)
        # 使用导出的COMSOL von Mises；不替换为另行计算的值。
        fem_vm = block[:, 6]
        tx, ty = -A_MAJOR*np.sin(theta), A_MAJOR*lam*np.cos(theta)
        tangent_length = np.hypot(tx, ty)
        tx, ty = tx/tangent_length, ty/tangent_length
        km_tt = tx**2*km[:,0]+2*tx*ty*km[:,2]+ty**2*km[:,1]
        fem_tt = tx**2*fem[:,0]+2*tx*ty*fem[:,2]+ty**2*fem[:,1]
        degrees = np.degrees(theta)
        maximum_gap = np.degrees(np.max(np.diff(np.r_[theta, theta[0]+2*np.pi])))
        row = {'lambda':float(lam), 'valid_unique_points':len(theta),
               'max_angle_gap_deg':float(maximum_gap),
               'sampled_relative_L2_sigma_tt':sampled_relative_l2(km_tt,fem_tt),
               'sampled_relative_L2_vm':sampled_relative_l2(km_vm,fem_vm),
               'sampled_peak_sigma_tt_comsol_pa':float(np.max(fem_tt)),
               'sampled_peak_sigma_tt_km_pa':float(np.max(km_tt))}
        metrics.append(row)
        comparisons.append((float(lam), degrees, km_tt, fem_tt, km_vm, fem_vm))
        print(f'lambda={lam:g}: N={len(theta)}, sampled L2(sigma_tt)={row["sampled_relative_L2_sigma_tt"]:.3e}, '
              f'max angle gap={maximum_gap:.3f} deg; {origin}')
        if save:
            columns = np.column_stack([degrees, block[:,:2], fem, km, fem_tt, km_tt, fem_vm, km_vm])
            np.savetxt(output_dir/f'hole_boundary_comparison_lambda_{lam:g}.csv', columns,
                       delimiter=',', comments='',
                       header='theta_deg,x_m,y_m,fem_sxx_pa,fem_syy_pa,fem_sxy_pa,km_sxx_pa,km_syy_pa,km_sxy_pa,fem_sigma_tt_pa,km_sigma_tt_pa,fem_vm_pa,km_vm_pa')

    style = {'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
             'mathtext.fontset':'stix', 'font.size':11, 'pdf.fonttype':42}
    with plt.rc_context(style):
        figures = []
        for name, ylabel, ki, fi in [('sigma_tt',r'$\sigma_{tt}$ (Pa)',2,3),
                                     ('von_mises',r'$\sigma_{\mathrm{vM}}$ (Pa)',4,5)]:
            cols = min(3,len(comparisons)); rows = int(np.ceil(len(comparisons)/cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.0*cols,3.2*rows),
                                     squeeze=False, constrained_layout=True)
            for ax, values in zip(axes.flat, comparisons):
                lam, degrees = values[:2]
                fx, fy = split_at_angular_gaps(degrees,values[fi])
                kx, ky = split_at_angular_gaps(degrees,values[ki])
                ax.plot(fx,fy,color='#252525',lw=1.5,label='COMSOL FEM')
                ax.plot(kx,ky,color='#BE4540',lw=1.4,ls='--',label='KM-PIKFNN')
                ax.set(xlim=(0,360),xlabel=r'$\theta$ (deg)',ylabel=ylabel,title=rf'$\lambda={lam:g}$')
                ax.set_xticks([0,90,180,270,360])
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
                ax.grid(alpha=.2,lw=.5)
                ax.legend(frameon=False,fontsize=9)
            for ax in list(axes.flat)[len(comparisons):]:
                ax.set_visible(False)
            if save:
                for ext in ('png','pdf'):
                    fig.savefig(output_dir/f'hole_boundary_{name}_comparison.{ext}',dpi=350,bbox_inches='tight')
            figures.append(fig)
        if show:
            plt.show()
        else:
            for fig in figures:
                plt.close(fig)
    if save:
        with (output_dir/'hole_boundary_error_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metrics[0]))
            writer.writeheader();writer.writerows(metrics)
    return metrics


if __name__ == '__main__':
    args = parse_args()
    if '--compare-only' not in sys.argv:
        main()
    comparison_cfg = SolverConfig(
        inner_source_scale=args.inner_source_scale,outer_source_scale=args.outer_source_scale,
        mapped_source_radius=args.mapped_source_radius,n_mapped_sources=args.mapped_sources,
        mapped_lambda_threshold=args.mapped_threshold,ridge_relative=args.ridge)
    if COMSOL_HOLE_FILE.exists():
        compare_comsol_hole_stress(COMSOL_HOLE_FILE,comparison_cfg,args.output_dir,
                                  show=SHOW_COMPARISON,save=SAVE_COMPARISON)
    else:
        print(f'孔边对比未运行：请将导出文件放到 {COMSOL_HOLE_FILE}，或修改 COMSOL_HOLE_FILE。')

import argparse
import copy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ==========================================================
# 0. Device and precision
# ==========================================================
REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128
torch.set_default_dtype(REAL_DTYPE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ==========================================================
# 1. Physical parameters
# ==========================================================
L = 2.0
E_real = 1.0e6
nu = 0.3
disp_load_real = 1.0e-3
a_major = 0.5

scale_u = 1.0e3
E = 1.0
disp_load = disp_load_real * scale_u
kappa_f = float((3.0 - nu) / (1.0 + nu))
G_f = float(E / (2.0 * (1.0 + nu)))

U_REF = max(abs(disp_load), 1.0)
SIGMA_REF = max(E * abs(disp_load) / (L / 2.0), 1.0)
HOLE_BC_REF = max(SIGMA_REF * L, 1.0)
VM_MAX_PA_DEFAULT = 2.5e3


# ==========================================================
# 2. Training config
# ==========================================================
@dataclass
class TrainConfig:
    hidden: int
    pole_order: int
    n_side: int
    n_hole_uniform: int
    n_hole_tip: int
    n_tip_each: int
    epochs_adam: int
    lbfgs_max_iter: int
    lbfgs_max_eval: int
    adam_lr: float
    resample_every: int
    detach_inverse_map: bool
    w_disp: float
    w_top_shear: float
    w_side: float
    w_hole: float
    w_tip: float
    w_anchor: float


def get_config(lam: float) -> TrainConfig:
    # Keep the model capacity fixed so continuation across lambda stays compatible.
    if lam <= 0.01:
        return TrainConfig(
            hidden=96,
            pole_order=16,
            n_side=420,
            n_hole_uniform=840,
            n_hole_tip=320,
            n_tip_each=220,
            epochs_adam=7000,
            lbfgs_max_iter=2600,
            lbfgs_max_eval=3200,
            adam_lr=3.5e-4,
            resample_every=250,
            detach_inverse_map=True,
            w_disp=20.0,
            w_top_shear=2.0,
            w_side=2.5,
            w_hole=12.0,
            w_tip=24.0,
            w_anchor=0.5,
        )
    if lam <= 0.05:
        return TrainConfig(
            hidden=96,
            pole_order=16,
            n_side=360,
            n_hole_uniform=700,
            n_hole_tip=240,
            n_tip_each=180,
            epochs_adam=6000,
            lbfgs_max_iter=2200,
            lbfgs_max_eval=2800,
            adam_lr=4.0e-4,
            resample_every=300,
            detach_inverse_map=True,
            w_disp=18.0,
            w_top_shear=1.5,
            w_side=1.75,
            w_hole=10.0,
            w_tip=18.0,
            w_anchor=0.5,
        )
    return TrainConfig(
        hidden=96,
        pole_order=16,
        n_side=260,
        n_hole_uniform=480,
        n_hole_tip=120,
        n_tip_each=80,
        epochs_adam=4500,
        lbfgs_max_iter=1800,
        lbfgs_max_eval=2400,
        adam_lr=7.0e-4,
        resample_every=400,
        detach_inverse_map=True,
        w_disp=15.0,
        w_top_shear=1.0,
        w_side=1.0,
        w_hole=5.0,
        w_tip=8.0,
        w_anchor=0.3,
    )


# ==========================================================
# 3. Complex potential model
# ==========================================================
class JoukowskiCVNN(nn.Module):
    def __init__(self, a: float, b: float, hidden: int, pole_order: int, detach_inverse_map: bool):
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.c = float(np.sqrt(max(a**2 - b**2, 0.0)))
        self.pole_order = pole_order
        self.detach_inverse_map = detach_inverse_map

        self.fc1 = nn.Linear(1, hidden, dtype=COMPLEX_DTYPE)
        self.fc2 = nn.Linear(hidden, hidden, dtype=COMPLEX_DTYPE)
        self.fc3 = nn.Linear(hidden, hidden, dtype=COMPLEX_DTYPE)
        self.out_phi = nn.Linear(hidden, 1, dtype=COMPLEX_DTYPE)
        self.out_psi = nn.Linear(hidden, 1, dtype=COMPLEX_DTYPE)

        self.pole_A_R = nn.Parameter(torch.zeros(pole_order, dtype=REAL_DTYPE))
        self.pole_A_I = nn.Parameter(torch.zeros(pole_order, dtype=REAL_DTYPE))
        self.pole_B_R = nn.Parameter(torch.zeros(pole_order, dtype=REAL_DTYPE))
        self.pole_B_I = nn.Parameter(torch.zeros(pole_order, dtype=REAL_DTYPE))

        init_sigma = E * disp_load / (L / 2.0)
        self.phi_z_R = nn.Parameter(torch.tensor(0.25 * init_sigma, dtype=REAL_DTYPE))
        self.phi_z_I = nn.Parameter(torch.tensor(0.0, dtype=REAL_DTYPE))
        self.psi_z_R = nn.Parameter(torch.tensor(0.50 * init_sigma, dtype=REAL_DTYPE))
        self.psi_z_I = nn.Parameter(torch.tensor(0.0, dtype=REAL_DTYPE))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.real)
                nn.init.xavier_uniform_(module.weight.imag)
                module.weight.data *= 1.0 / np.sqrt(2.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias.real, 0.0)
                    nn.init.constant_(module.bias.imag, 0.0)
        self.out_phi.weight.data *= 0.05
        self.out_psi.weight.data *= 0.05

    def inverse_joukowski(self, z: torch.Tensor) -> torch.Tensor:
        # Solve alpha*zeta^2 - z*zeta + beta = 0 and choose the exterior branch.
        root = torch.sqrt(z * z - self.c * self.c)
        denom = self.a + self.b
        zeta_plus = (z + root) / denom
        zeta_minus = (z - root) / denom
        choose_plus = torch.abs(zeta_plus) >= torch.abs(zeta_minus)
        return torch.where(choose_plus, zeta_plus, zeta_minus)

    def forward_from_zeta(self, zeta: torch.Tensor, z_eval: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zeta_in = zeta / 2.5
        h = torch.sin(self.fc1(zeta_in))
        h = torch.sin(self.fc2(h))
        h = torch.sin(self.fc3(h))

        phi = self.out_phi(h)
        psi = self.out_psi(h)

        pole_A = torch.complex(self.pole_A_R, self.pole_A_I)
        pole_B = torch.complex(self.pole_B_R, self.pole_B_I)
        for k in range(self.pole_order):
            p = k + 1
            phi = phi + pole_A[k] / (zeta**p)
            psi = psi + pole_B[k] / (zeta**p)

        phi_z = torch.complex(self.phi_z_R, self.phi_z_I)
        psi_z = torch.complex(self.psi_z_R, self.psi_z_I)
        phi = phi + phi_z * z_eval
        psi = psi + psi_z * z_eval
        return phi, psi


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def get_complex_potential_terms(model: JoukowskiCVNN, x: torch.Tensor, y: torch.Tensor, need_second_derivative: bool = False):
    z_raw = torch.complex(x, y)
    zeta_raw = model.inverse_joukowski(z_raw)

    if model.detach_inverse_map:
        xi = zeta_raw.real.detach().requires_grad_(True)
        eta = zeta_raw.imag.detach().requires_grad_(True)
    else:
        xi = zeta_raw.real
        eta = zeta_raw.imag

    zeta = torch.complex(xi, eta)
    z_eval = (model.a + model.b) / 2.0 * zeta + (model.a - model.b) / (2.0 * zeta)

    phi, psi = model.forward_from_zeta(zeta, z_eval)
    dphi_dxi = torch.complex(grad(phi.real, xi), grad(phi.imag, xi))
    dpsi_dxi = torch.complex(grad(psi.real, xi), grad(psi.imag, xi))

    omega_prime = (model.a + model.b) / 2.0 - (model.a - model.b) / (2.0 * zeta**2)
    dphi_dz = dphi_dxi / omega_prime
    dpsi_dz = dpsi_dxi / omega_prime

    d2phi_dz2 = None
    if need_second_derivative:
        d2phi_dz_dxi = torch.complex(grad(dphi_dz.real, xi), grad(dphi_dz.imag, xi))
        d2phi_dz2 = d2phi_dz_dxi / omega_prime

    return z_eval, phi, psi, dphi_dz, dpsi_dz, d2phi_dz2


def get_phys_fields(model: JoukowskiCVNN, x: torch.Tensor, y: torch.Tensor):
    z_eval, phi, psi, dphi_dz, dpsi_dz, d2phi_dz2 = get_complex_potential_terms(
        model,
        x,
        y,
        need_second_derivative=True,
    )
    assert d2phi_dz2 is not None

    disp = (kappa_f * phi - z_eval * torch.conj(dphi_dz) - torch.conj(psi)) / (2.0 * G_f)
    u_x = disp.real
    u_y = disp.imag

    sum_sig = 4.0 * dphi_dz.real
    diff_sig = 2.0 * (torch.conj(z_eval) * d2phi_dz2 + dpsi_dz)
    sig_xx = (sum_sig - diff_sig.real) / 2.0
    sig_yy = (sum_sig + diff_sig.real) / 2.0
    sig_xy = diff_sig.imag / 2.0

    return u_x, u_y, sig_xx, sig_yy, sig_xy


def complex_abs_sq(values: torch.Tensor) -> torch.Tensor:
    return values.real**2 + values.imag**2


def weighted_complex_mean(values: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    flat_values = values.reshape(-1)
    if weights is None:
        return torch.mean(flat_values)

    flat_weights = weights.reshape(-1)
    return torch.sum(flat_weights * flat_values) / torch.sum(flat_weights)


def hole_boundary_complex_residual(model: JoukowskiCVNN, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z_eval, phi, psi, dphi_dz, _, _ = get_complex_potential_terms(model, x, y, need_second_derivative=False)
    return phi + z_eval * torch.conj(dphi_dz) + torch.conj(psi)


# ==========================================================
# 4. Sampling and loss
# ==========================================================
def to_col(values) -> torch.Tensor:
    if isinstance(values, np.ndarray):
        tensor = torch.from_numpy(values)
    elif isinstance(values, torch.Tensor):
        tensor = values
    else:
        tensor = torch.tensor(values)
    return tensor.reshape(-1, 1).to(device=device, dtype=REAL_DTYPE).requires_grad_(True)


def leafify_coord(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().to(device=device, dtype=REAL_DTYPE).requires_grad_(True)


def freeze_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().to(device=device, dtype=REAL_DTYPE)


def trimmed_edge_points(n: int, side_value: float) -> torch.Tensor:
    dx = L / n
    coords = np.linspace(-L / 2.0 + 0.5 * dx, L / 2.0 - 0.5 * dx, n)
    return to_col(coords)


def build_theta_samples(n_uniform: int, n_tip_cluster: int) -> torch.Tensor:
    theta_uniform = (np.arange(n_uniform, dtype=np.float64) + 0.5) * (2.0 * np.pi / n_uniform)

    if n_tip_cluster > 0:
        eps = np.geomspace(2.0e-4, 0.22 * np.pi, n_tip_cluster)
        theta_tip = np.concatenate(
            [
                eps,
                2.0 * np.pi - eps,
                np.pi - eps,
                np.pi + eps,
            ]
        )
        theta = np.concatenate([theta_uniform, theta_tip])
    else:
        theta = theta_uniform

    theta = np.mod(theta, 2.0 * np.pi)
    theta = np.unique(np.round(theta, 12))
    return to_col(theta)


def ellipse_boundary(theta: torch.Tensor, b_minor: float):
    x = (a_major * torch.cos(theta)).requires_grad_(True)
    y = (b_minor * torch.sin(theta)).requires_grad_(True)
    nx = b_minor * torch.cos(theta)
    ny = a_major * torch.sin(theta)
    norm = torch.sqrt(nx**2 + ny**2)
    ds_dtheta = torch.sqrt((a_major * torch.sin(theta)) ** 2 + (b_minor * torch.cos(theta)) ** 2)
    weights = ds_dtheta / torch.mean(ds_dtheta)
    return x, y, nx / norm, ny / norm, weights


def create_tip_points(lam: float, cfg: TrainConfig):
    if cfg.n_tip_each <= 0:
        return None

    eps = torch.logspace(
        np.log10(2.0e-4),
        np.log10(0.18),
        cfg.n_tip_each,
        device=device,
        dtype=REAL_DTYPE,
    ).reshape(-1, 1)
    theta_tip = torch.cat([eps, 2.0 * np.pi - eps, np.pi - eps, np.pi + eps], dim=0)
    x_tip, y_tip, nx_tip, ny_tip, _ = ellipse_boundary(theta_tip, a_major * lam)
    return x_tip, y_tip, nx_tip, ny_tip


def create_anchor_point():
    x_anchor = to_col(np.array([0.0]))
    y_anchor = to_col(np.array([-L / 2.0]))
    return x_anchor, y_anchor


def create_training_data(lam: float, cfg: TrainConfig):
    b_minor = a_major * lam

    x_t = trimmed_edge_points(cfg.n_side, L / 2.0)
    y_t = torch.full_like(x_t, L / 2.0).requires_grad_(True)
    x_b = trimmed_edge_points(cfg.n_side, -L / 2.0)
    y_b = torch.full_like(x_b, -L / 2.0).requires_grad_(True)

    y_l = trimmed_edge_points(cfg.n_side, -L / 2.0)
    x_l = torch.full_like(y_l, -L / 2.0).requires_grad_(True)
    y_r = trimmed_edge_points(cfg.n_side, L / 2.0)
    x_r = torch.full_like(y_r, L / 2.0).requires_grad_(True)

    theta = build_theta_samples(cfg.n_hole_uniform, cfg.n_hole_tip)
    x_h, y_h, nx_h, ny_h, w_h = ellipse_boundary(theta, b_minor)
    x_h = leafify_coord(x_h)
    y_h = leafify_coord(y_h)
    nx_h = freeze_tensor(nx_h)
    ny_h = freeze_tensor(ny_h)
    w_h = freeze_tensor(w_h)

    data = {
        "top": (x_t, y_t),
        "bot": (x_b, y_b),
        "left": (x_l, y_l),
        "right": (x_r, y_r),
        "hole": (x_h, y_h, nx_h, ny_h, w_h),
        "anchor": create_anchor_point(),
    }

    tip_points = create_tip_points(lam, cfg)
    if tip_points is not None:
        x_tip, y_tip, nx_tip, ny_tip = tip_points
        data["tip"] = (
            leafify_coord(x_tip),
            leafify_coord(y_tip),
            freeze_tensor(nx_tip),
            freeze_tensor(ny_tip),
        )
    return data


def compute_loss(model: JoukowskiCVNN, data: Dict[str, Tuple[torch.Tensor, ...]], cfg: TrainConfig):
    ux_t, uy_t, _, _, sig_xy_t = get_phys_fields(model, *data["top"])
    ux_b, uy_b, _, _, sig_xy_b = get_phys_fields(model, *data["bot"])

    loss_disp = torch.mean(((uy_t - disp_load) / U_REF) ** 2) + torch.mean(((uy_b + disp_load) / U_REF) ** 2)
    loss_shear = torch.mean((sig_xy_t / SIGMA_REF) ** 2) + torch.mean((sig_xy_b / SIGMA_REF) ** 2)

    _, _, sx_l, _, sxy_l = get_phys_fields(model, *data["left"])
    _, _, sx_r, _, sxy_r = get_phys_fields(model, *data["right"])
    loss_side = torch.mean((sx_l / SIGMA_REF) ** 2 + (sxy_l / SIGMA_REF) ** 2)
    loss_side = loss_side + torch.mean((sx_r / SIGMA_REF) ** 2 + (sxy_r / SIGMA_REF) ** 2)

    x_h, y_h, _, _, w_h = data["hole"]
    # Enforce the traction-free hole boundary with the KM boundary identity.
    # This avoids the stress-component cancellation that becomes very stiff near crack-like tips.
    bc_h = hole_boundary_complex_residual(model, x_h, y_h)
    bc_const = weighted_complex_mean(bc_h, w_h)
    loss_hole = torch.sum(w_h * complex_abs_sq(bc_h - bc_const)) / torch.sum(w_h) / (HOLE_BC_REF**2)

    x_a, y_a = data["anchor"]
    ux_a, _, _, _, _ = get_phys_fields(model, x_a, y_a)
    loss_anchor = torch.mean((ux_a / U_REF) ** 2)

    loss_tip = torch.zeros((), device=device, dtype=REAL_DTYPE)
    if "tip" in data:
        xt, yt, _, _ = data["tip"]
        bc_tip = hole_boundary_complex_residual(model, xt, yt)
        loss_tip = torch.mean(complex_abs_sq(bc_tip - bc_const)) / (HOLE_BC_REF**2)

    total = (
        cfg.w_disp * loss_disp
        + cfg.w_top_shear * loss_shear
        + cfg.w_side * loss_side
        + cfg.w_hole * loss_hole
        + cfg.w_tip * loss_tip
        + cfg.w_anchor * loss_anchor
    )

    stats = {
        "total": total.detach(),
        "disp": loss_disp.detach(),
        "top_shear": loss_shear.detach(),
        "side": loss_side.detach(),
        "hole": loss_hole.detach(),
        "tip": loss_tip.detach(),
        "anchor": loss_anchor.detach(),
    }
    return total, stats


# ==========================================================
# 5. Training and evaluation
# ==========================================================
def format_stats(stats: Dict[str, torch.Tensor]) -> str:
    return " | ".join(f"{key}={float(val):.3e}" for key, val in stats.items())


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def geometric_continuation_path(
    target_lams: List[float],
    ratio_limit: float = 2.2,
    small_lambda_ratio_limit: float = 1.55,
    small_lambda_threshold: float = 0.1,
) -> List[float]:
    schedule = [float(target_lams[0])]
    for target in target_lams[1:]:
        current = schedule[-1]
        while current / target > (
            small_lambda_ratio_limit if min(current, target) <= small_lambda_threshold else ratio_limit
        ):
            current = float(np.sqrt(current * target))
            schedule.append(float(f"{current:.6g}"))
        schedule.append(float(target))

    merged = []
    for lam in schedule:
        if not merged or abs(lam - merged[-1]) > 1.0e-12:
            merged.append(lam)
    return merged


def continuation_epochs_scale(lam: float, is_target: bool, epochs_scale: float, intermediate_scale: float) -> float:
    if is_target:
        return epochs_scale

    step_scale = epochs_scale * intermediate_scale
    if lam <= 0.10:
        step_scale = max(step_scale, 0.60 * epochs_scale)
    if lam <= 0.03:
        step_scale = max(step_scale, 0.85 * epochs_scale)
    return step_scale


def evaluate_on_grid(model: JoukowskiCVNN, lam: float, grid_size: int):
    b_minor = a_major * lam
    coords_1d = np.linspace(-0.99, 0.99, grid_size)
    xx, yy = np.meshgrid(coords_1d, coords_1d)
    valid_mask = ((xx**2 / a_major**2) + (yy**2 / b_minor**2)) > 1.02

    x_all = xx[valid_mask].reshape(-1)
    y_all = yy[valid_mask].reshape(-1)

    vm_all, sxx_all, syy_all, sxy_all = [], [], [], []
    stress_scale = E_real / scale_u

    model.eval()
    chunk = 15000
    for start in range(0, x_all.shape[0], chunk):
        xb = torch.tensor(x_all[start:start + chunk], dtype=REAL_DTYPE, device=device).reshape(-1, 1).requires_grad_(True)
        yb = torch.tensor(y_all[start:start + chunk], dtype=REAL_DTYPE, device=device).reshape(-1, 1).requires_grad_(True)
        _, _, sxx, syy, sxy = get_phys_fields(model, xb, yb)

        sxx_r = sxx.detach().cpu().numpy().reshape(-1) * stress_scale
        syy_r = syy.detach().cpu().numpy().reshape(-1) * stress_scale
        sxy_r = sxy.detach().cpu().numpy().reshape(-1) * stress_scale
        vm_r = np.sqrt(sxx_r**2 - sxx_r * syy_r + syy_r**2 + 3.0 * sxy_r**2)

        sxx_all.append(sxx_r)
        syy_all.append(syy_r)
        sxy_all.append(sxy_r)
        vm_all.append(vm_r)

    vm_map = np.full((grid_size, grid_size), np.nan)
    sxx_map = vm_map.copy()
    syy_map = vm_map.copy()
    sxy_map = vm_map.copy()

    vm_map[valid_mask] = np.concatenate(vm_all)
    sxx_map[valid_mask] = np.concatenate(sxx_all)
    syy_map[valid_mask] = np.concatenate(syy_all)
    sxy_map[valid_mask] = np.concatenate(sxy_all)

    mask_outer = (np.abs(xx) > 0.985) | (np.abs(yy) > 0.985)
    for array in [vm_map, sxx_map, syy_map, sxy_map]:
        array[mask_outer] = np.nan

    return xx, yy, sxx_map, syy_map, sxy_map, vm_map


def save_case_plot(lam: float, sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray, vm: np.ndarray, out_dir: Path, vm_max_pa: float):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    data_list = [sxx, syy, sxy, vm]
    titles = [
        "Stress $\\sigma_{xx}$ (Pa)",
        "Stress $\\sigma_{yy}$ (Pa)",
        "Shear $\\sigma_{xy}$ (Pa)",
        "Von Mises Stress (Pa)",
    ]
    extent = [-L / 2.0, L / 2.0, -L / 2.0, L / 2.0]

    for ax, data, title in zip(axes.flatten(), data_list, titles):
        if "Von Mises" in title:
            vmin, vmax = 0.0, vm_max_pa
        else:
            vmax = np.nanpercentile(np.abs(data), 99.5)
            vmin = -vmax

        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="jet" if "Von Mises" in title else "RdBu_r",
            aspect="equal",
            vmax=vmax,
            vmin=vmin,
            interpolation="bilinear",
        )
        ellipse = plt.matplotlib.patches.Ellipse((0.0, 0.0), 2.0 * a_major, 2.0 * (a_major * lam), color="white", zorder=10)
        ax.add_patch(ellipse)
        ax.set_title(title + f" | $\\lambda = {lam}$")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0))

    plt.tight_layout()
    save_path = out_dir / f"Stress_Maps_lambda_{lam}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot -> {save_path}")


def save_history_csv(history: List[Dict[str, float]], out_dir: Path, lam: float):
    if not history:
        return
    keys = list(history[0].keys())
    lines = [",".join(keys)]
    for row in history:
        lines.append(",".join(f"{row[key]:.10e}" for key in keys))
    (out_dir / f"loss_history_lambda_{lam}.csv").write_text("\n".join(lines), encoding="utf-8")


def solve_cvpinn_case(
    lam: float,
    init_state: Optional[Dict[str, torch.Tensor]],
    epochs_scale: float,
    run_lbfgs: bool,
    export_results: bool,
    grid_size: int,
    vm_max_pa: float,
    output_dir: Path,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, torch.Tensor], List[Dict[str, float]]]:
    print(f"\n>> Start solving lambda = {lam}")
    cfg = get_config(lam)
    b_minor = a_major * lam
    model = JoukowskiCVNN(
        a=a_major,
        b=b_minor,
        hidden=cfg.hidden,
        pole_order=cfg.pole_order,
        detach_inverse_map=cfg.detach_inverse_map,
    ).to(device)

    if init_state is not None:
        model.load_state_dict(init_state, strict=False)
        print("Loaded warm-start parameters from previous lambda.")

    data = create_training_data(lam, cfg)
    history: List[Dict[str, float]] = []

    epochs_adam = max(1, int(round(cfg.epochs_adam * epochs_scale)))
    lbfgs_max_iter = max(1, int(round(cfg.lbfgs_max_iter * epochs_scale)))
    lbfgs_max_eval = max(1, int(round(cfg.lbfgs_max_eval * epochs_scale)))
    print_every = max(100, epochs_adam // 8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.adam_lr, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_adam)

    t0 = time.time()
    for epoch in range(1, epochs_adam + 1):
        if cfg.resample_every > 0 and epoch > 1 and epoch % cfg.resample_every == 0:
            data = create_training_data(lam, cfg)

        optimizer.zero_grad(set_to_none=True)
        loss, stats = compute_loss(model, data, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 150.0)
        optimizer.step()
        scheduler.step()

        history.append({"epoch": float(epoch), **{key: float(value) for key, value in stats.items()}})
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs_adam:
            print(f"Adam Epoch {epoch}: {format_stats(stats)}")

    if run_lbfgs:
        print("Starting L-BFGS refinement...")
        data = create_training_data(lam, cfg)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=lbfgs_max_iter,
            max_eval=lbfgs_max_eval,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        lbfgs_calls = {"count": 0}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            current_loss, current_stats = compute_loss(model, data, cfg)
            current_loss.backward()
            lbfgs_calls["count"] += 1
            if lbfgs_calls["count"] % 50 == 0:
                print(f"L-BFGS iter {lbfgs_calls['count']}: {format_stats(current_stats)}")
            return current_loss

        lbfgs.step(closure)

    result = None
    if export_results:
        print(f"Training finished in {time.time() - t0:.1f} s, evaluating...")
        xx, yy, sxx_map, syy_map, sxy_map, vm_map = evaluate_on_grid(model, lam, grid_size)
        result = {
            "XX": xx,
            "YY": yy,
            "Sxx": sxx_map,
            "Syy": syy_map,
            "Sxy": sxy_map,
            "VM": vm_map,
        }
        save_case_plot(lam, sxx_map, syy_map, sxy_map, vm_map, output_dir, vm_max_pa)
        save_history_csv(history, output_dir, lam)
    else:
        print(f"Training finished in {time.time() - t0:.1f} s, continuation step complete.")

    return result, clone_state_dict(model), history


# ==========================================================
# 6. CLI
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Improved KM-PINN solver for an elliptical-hole plate.")
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 0.05, 0.02, 0.01, 0.005],
        help="Target lambda values to solve in descending order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "CV_PINN_Results_Improved",
        help="Directory used to save plots and logs.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=420,
        help="Evaluation grid size per direction.",
    )
    parser.add_argument(
        "--vm-max-pa",
        type=float,
        default=VM_MAX_PA_DEFAULT,
        help="Fixed upper limit for the von Mises colorbar in Pa.",
    )
    parser.add_argument(
        "--epochs-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to Adam/L-BFGS iterations.",
    )
    parser.add_argument(
        "--intermediate-scale",
        type=float,
        default=0.35,
        help="Extra scale for continuation-only intermediate lambda steps.",
    )
    parser.add_argument(
        "--skip-lbfgs",
        action="store_true",
        help="Skip L-BFGS refinement.",
    )
    parser.add_argument(
        "--disable-continuation",
        action="store_true",
        help="Disable lambda continuation and train each target lambda directly.",
    )
    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_lams = [float(lam) for lam in args.lambdas]
    if args.disable_continuation:
        schedule = target_lams
    else:
        schedule = geometric_continuation_path(target_lams)

    print(f"Target lambdas: {target_lams}")
    print(f"Training schedule: {schedule}")

    target_lookup = {round(lam, 12): lam for lam in target_lams}
    prev_state = None

    summary_lines = ["lambda,max_vm_pa,max_vm_kpa"]
    for lam in schedule:
        is_target = round(lam, 12) in target_lookup
        step_scale = continuation_epochs_scale(lam, is_target, args.epochs_scale, args.intermediate_scale)
        run_lbfgs = (not args.skip_lbfgs) and is_target

        result, prev_state, _ = solve_cvpinn_case(
            lam=lam,
            init_state=prev_state,
            epochs_scale=step_scale,
            run_lbfgs=run_lbfgs,
            export_results=is_target,
            grid_size=args.grid_size,
            vm_max_pa=args.vm_max_pa,
            output_dir=output_dir,
        )

        if is_target:
            assert result is not None
            max_vm = float(np.nanmax(result["VM"]))
            summary_lines.append(f"{lam},{max_vm:.10e},{max_vm / 1.0e3:.10e}")
        else:
            print(f"Continuation step lambda = {lam} finished, plot export skipped.")

    (output_dir / "summary.csv").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()

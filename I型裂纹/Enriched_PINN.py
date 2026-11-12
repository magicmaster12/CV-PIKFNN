"""Python translation of the MATLAB folder `二维裂缝问题`.

This script keeps the original MATLAB logic but uses PyTorch autograd to
compute derivatives, which is cleaner than manually rewriting the very long
enrichment derivatives from the supplementary code.

Main contents translated from the MATLAB project:
- Geometry generation for the upper/lower half domains
- Linear elasticity coefficients
- Enriched PINN model with crack-tip asymptotic enrichment
- Boundary / interface loss terms
- Training with Adam + LBFGS
- COD-based SIF extraction
- Post-processing stress cloud from the extracted SIF

The default benchmark is the same as the MATLAB main.m:
center crack, a = 1, b = 2, h = 6.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for this translation.") from exc

torch.set_default_dtype(torch.float64)


# -----------------------------------------------------------------------------
# Global plotting style
# -----------------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class CrackConfig:
    a_crack: float = 1.0
    b_b: float = 2.0
    h_h: float = 6.0

    N_crack: int = 10
    refine_time: int = 1
    refine_ratios: float = 0.7

    N_SIF_Start_1: int = 3
    n_crack_SIF: int = 7

    hidden_width: int = 15
    hidden_layers: int = 5
    activation_function: int = 3
    \

    adam_steps: int = 5000
    lbfgs_steps: int = 5000
    learning_rate: float = 1.0e-3
    print_every: int = 250

    w_neumann_y_load: float = 10.0
    w_neumann_y_zero: float = 1.0

    use_saved_checkpoint: bool = True
    checkpoint_name: str = "XPINN_checkpoint.pt"

    seed: int = 1234
    use_cuda: bool = True

    def xscal_up(self) -> np.ndarray:
        return np.array([0.0, self.b_b], dtype=np.float64)

    def yscal_up(self) -> np.ndarray:
        return np.array([0.0, self.h_h], dtype=np.float64)

    def xscal_down(self) -> np.ndarray:
        return np.array([0.0, self.b_b], dtype=np.float64)

    def yscal_down(self) -> np.ndarray:
        return np.array([-self.h_h, 0.0], dtype=np.float64)

    def refine_length(self) -> float:
        return 0.8 * self.a_crack

    def refine_length_x(self) -> float:
        return self.a_crack + self.refine_length()

    def refine_length_y(self) -> float:
        return self.refine_length()


def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(use_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def initialize_he(shape: Tuple[int, ...], fan_in: int, device: torch.device) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float64, device=device) * math.sqrt(2.0 / fan_in)


def initialize_zeros(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.float64, device=device)


def initialize_ones(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.ones(*shape, dtype=torch.float64, device=device)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float64, device=a.device if a.numel() else b.device)
    return torch.mean((a - b) ** 2)


def indexed_mse(pred: torch.Tensor, target: torch.Tensor, idx: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(idx, np.ndarray):
        idx = torch.as_tensor(idx, dtype=torch.long, device=pred.device)
    if idx.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float64, device=pred.device)
    return torch.mean((pred[idx] - target[idx]) ** 2)


def grad_sum(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs.sum(),
        inputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def safe_acos(x: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(x, -1.0 + 1e-12, 1.0 - 1e-12))


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------
def elasticity_coefficients(ck: int) -> Dict[str, float]:
    """MATLAB: Elasticity_Coefficients.m"""
    if ck not in (1, 2):
        raise ValueError("ck must be 1 (upper) or 2 (lower).")

    q = 1.0
    eg = 1.0
    mu = 0.3
    G_G = eg / 2.0 / (1.0 + mu)
    c_1 = eg * (1.0 - mu) / (1.0 - 2.0 * mu) / (1.0 + mu)
    c_2 = eg / (1.0 + mu) / 2.0
    c_3 = eg / (1.0 + mu) / (1.0 - 2.0 * mu) / 2.0
    a_1 = eg * (1.0 - mu) / ((1.0 + mu) * (1.0 - 2.0 * mu))
    a_2 = a_1 * mu / (1.0 - mu)
    a_3 = eg / 2.0 / (1.0 + mu)

    return {
        "q": q,
        "eg": eg,
        "mu": mu,
        "G_G": G_G,
        "a_1": a_1,
        "a_2": a_2,
        "a_3": a_3,
        "c_1": c_1,
        "c_2": c_2,
        "c_3": c_3,
        "Lamei": 3.0 - 4.0 * mu,
    }


def sif_exact(cfg: CrackConfig, coeffs_up: Dict[str, float]) -> float:
    """MATLAB: SIF_Exact.m"""
    a = cfg.a_crack
    b = cfg.b_b
    q = coeffs_up["q"]
    return q * math.sqrt(math.pi * a) * (
        1.0 - 0.025 * (a / b) ** 2 + 0.06 * (a / b) ** 4
    ) * math.sqrt(1.0 / math.cos(math.pi * a / (2.0 * b)))


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def crack_surface_elements(cfg: CrackConfig, x_scal: np.ndarray, y_scal: np.ndarray) -> Tuple[np.ndarray, float]:
    """MATLAB: Crack_Surface_Elements.m"""
    dx_crack = cfg.a_crack / cfg.N_crack
    L_remaining_half = (x_scal[1] - x_scal[0]) - cfg.a_crack
    N_remaining_half = int(np.fix(L_remaining_half / dx_crack))
    x_crack = np.linspace(0.0, cfg.a_crack, cfg.N_crack + 1)
    x_half_right = np.linspace(cfg.a_crack, x_scal[1], N_remaining_half + 1)
    bp_interface = np.zeros((N_remaining_half + cfg.N_crack + 1, 2), dtype=np.float64)
    bp_interface[:, 0] = np.concatenate([x_crack[:-1], x_half_right])
    bp_interface[:, 1] = y_scal[0]
    return bp_interface, dx_crack


def node_refine(
    cfg: CrackConfig,
    ck: int,
    inp: np.ndarray,
    x_temp: np.ndarray,
    dx: float,
    dy: float,
    refine_length_x: float,
    refine_length_y: float,
    y_temp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """MATLAB: node_refine.m"""
    err = 1e-8
    mask = (np.abs(inp[:, 0]) < refine_length_x - err) & (np.abs(inp[:, 1]) < refine_length_y - err)
    selected = inp[mask]

    if ck == 1:
        dxx, dyy = dx, dy
    elif ck == 2:
        dxx, dyy = dx, -dy
    else:
        raise ValueError("ck must be 1 or 2.")

    if selected.size:
        add = np.vstack([
            np.column_stack([selected[:, 0] + dxx / 2.0, selected[:, 1] + dyy / 2.0]),
            np.column_stack([selected[:, 0], selected[:, 1] + dyy / 2.0]),
            np.column_stack([selected[:, 0] + dxx / 2.0, selected[:, 1]]),
        ])
        inp_new = np.vstack([inp, add])
    else:
        inp_new = inp.copy()

    x_mask = np.abs(x_temp) < refine_length_x - err
    y_mask = np.abs(y_temp) < refine_length_y - err
    x_temp_new = np.sort(np.concatenate([x_temp, x_temp[x_mask] + dxx / 2.0]))
    y_temp_new = np.sort(np.concatenate([y_temp, y_temp[y_mask] + dyy / 2.0]))
    return inp_new, x_temp_new, dx / 2.0, dy / 2.0, y_temp_new


def geodata_rectangle_up_and_down(
    cfg: CrackConfig,
    refine_length_x: float,
    refine_length_y: float,
    x_scal: np.ndarray,
    y_scal: np.ndarray,
    ny: int,
    bp_interface: np.ndarray,
    ck: int,
) -> Dict[str, np.ndarray]:
    """MATLAB: geodata_rectangle_up_and_down.m"""
    x_temp = bp_interface[:, 0].copy()
    y_temp = np.linspace(y_scal[0], y_scal[1], ny)
    nx = len(x_temp)
    allx, ally = np.meshgrid(x_temp, y_temp)
    inp = np.column_stack([allx.reshape(-1), ally.reshape(-1)])

    n2 = np.zeros((ny - 2, 2), dtype=np.float64)
    n3 = np.zeros((nx - 2, 2), dtype=np.float64)
    bp2 = np.zeros((ny - 2, 2), dtype=np.float64)
    bp3 = np.zeros((nx - 2, 2), dtype=np.float64)

    if ck == 1:
        bp2[:, 0] = x_scal[1]
        bp2[:, 1] = y_temp[1:-1]
        bp3[:, 0] = x_temp[1:-1]
        bp3[:, 1] = y_scal[1]
        n2[:, 0] = 1.0
        n3[:, 1] = 1.0
    elif ck == 2:
        bp3[:, 0] = x_temp[1:-1]
        bp3[:, 1] = y_scal[0]
        bp2[:, 0] = x_scal[1]
        bp2[:, 1] = y_temp[1:-1]
        n3[:, 1] = -1.0
        n2[:, 0] = 1.0
    else:
        raise ValueError("ck must be 1 or 2.")

    dx = cfg.a_crack / cfg.N_crack
    dy = (y_scal[1] - y_scal[0]) / (ny - 1)
    for _ in range(cfg.refine_time):
        inp, x_temp, dx, dy, y_temp = node_refine(
            cfg, ck, inp, x_temp, dx, dy, refine_length_x, refine_length_y, y_temp
        )
        refine_length_x = cfg.a_crack + (refine_length_x - cfg.a_crack) * cfg.refine_ratios
        refine_length_y = refine_length_y * cfg.refine_ratios

    bp1 = np.zeros((len(x_temp), 2), dtype=np.float64)
    bp4 = np.zeros((len(y_temp) - 2, 2), dtype=np.float64)
    n1 = np.zeros((len(bp1), 2), dtype=np.float64)
    n4 = np.zeros((len(bp4), 2), dtype=np.float64)

    if ck == 1:
        bp1[:, 0] = x_temp
        bp1[:, 1] = y_scal[0]
        bp4[:, 0] = x_scal[0]
        bp4[:, 1] = y_temp[1:-1]
        n1[:, 1] = -1.0
        n4[:, 0] = -1.0
        y_large = y_scal[1]
        y_small = y_scal[0]
    else:
        bp1[:, 0] = x_temp
        bp1[:, 1] = y_scal[1]
        bp4[:, 0] = x_scal[0]
        bp4[:, 1] = y_temp[1:-1]
        n1[:, 1] = 1.0
        n4[:, 0] = -1.0
        y_large = -y_scal[0]
        y_small = y_scal[1]

    error = 1e-8
    inp_index = np.where(
        (inp[:, 0] < x_scal[1] - error)
        & (inp[:, 0] > x_scal[0] + error)
        & (np.abs(inp[:, 1]) < y_large - error)
        & (np.abs(inp[:, 1]) > y_small + error)
    )[0]
    inp = inp[inp_index]

    crack_tip_index = np.where((np.abs(bp1[:, 0]) >= cfg.a_crack - 1e-8) & (np.abs(bp1[:, 0]) <= cfg.a_crack + 1e-8))[0]
    bp1 = np.delete(bp1, crack_tip_index, axis=0)
    n1 = np.delete(n1, crack_tip_index, axis=0)

    normals = np.vstack([n1, n2, n3, n4])
    boundary = np.vstack([bp1, bp2, bp3, bp4])
    points = np.vstack([boundary, inp])

    return {
        "bp1": bp1,
        "bp2": bp2,
        "bp3": bp3,
        "bp4": bp4,
        "inp": inp,
        "N": normals,
        "bp": boundary,
        "S": points,
    }


def boundary_conditions(
    cfg: CrackConfig,
    ck: int,
    bp: np.ndarray,
    bp1: np.ndarray,
    bp2: np.ndarray,
    bp3: np.ndarray,
    bp4: np.ndarray,
    coeffs: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """MATLAB: Boundary_Conditions.m"""
    N1_crack = cfg.N_crack_new
    N1_1 = cfg.N_interface_half_new
    N2 = len(bp2)
    N3 = len(bp3)
    N4 = len(bp4)

    # Convert MATLAB 1-based indexing to Python 0-based indexing.
    N_crack = np.arange(0, N1_crack, dtype=np.int64)
    start_2 = N1_1 + N1_crack
    start_3 = start_2 + N2
    start_4 = start_3 + N3
    N_2 = np.arange(start_2, start_2 + N2, dtype=np.int64)
    N_3 = np.arange(start_3, start_3 + N3, dtype=np.int64)
    N_4 = np.arange(start_4, start_4 + N4, dtype=np.int64)

    if ck == 1:
        q = coeffs["q"]
        k = 1.0
    elif ck == 2:
        q = coeffs["q"]
        k = -1.0
    else:
        raise ValueError("ck must be 1 or 2.")

    dirichlet_x = N_4
    dirichlet_y = np.array([], dtype=np.int64)
    neumann_x = np.concatenate([N_crack, N_2, N_3])
    neumann_y = np.concatenate([N_crack, N_2, N_3, N_4])
    neumann_y_load = N_3
    neumann_y_zero = np.concatenate([N_crack, N_2, N_4])

    BC_x = np.zeros(len(bp), dtype=np.float64)
    BC_y = np.zeros(len(bp), dtype=np.float64)
    BC_y[N_3] = k * q

    return {
        "BC_x": BC_x,
        "BC_y": BC_y,
        "Dirichlet_BC_x": dirichlet_x,
        "Dirichlet_BC_y": dirichlet_y,
        "Neumann_BC_x": neumann_x,
        "Neumann_BC_y": neumann_y,
        "Neumann_BC_y_load": neumann_y_load,
        "Neumann_BC_y_zero": neumann_y_zero,
    }


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
def activ_function(x: torch.Tensor, activation_function: int = 3) -> torch.Tensor:
    """MATLAB: activ_function.m"""
    if activation_function == 1:
        return torch.sigmoid(x)
    if activation_function == 2:
        return torch.tanh(x)
    if activation_function == 3:
        return x * torch.sigmoid(x)
    if activation_function == 4:
        return torch.log1p(torch.exp(x))
    if activation_function == 5:
        return torch.sin(x)
    if activation_function == 6:
        return torch.atan(x)
    if activation_function == 7:
        return x * torch.tanh(torch.log1p(torch.exp(x)))
    raise ValueError(f"Unsupported activation_function = {activation_function}")


class CrackPINN(nn.Module):
    """MATLAB: Parameters_Initialize + model.m"""

    def __init__(
        self,
        hidden_width: int = 15,
        hidden_layers: int = 5,
        activation_function: int = 3,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.activation_function = activation_function
        self.layers = nn.ModuleList()
        in_features = 2
        for layer_idx in range(hidden_layers):
            out_features = hidden_width if layer_idx < hidden_layers - 1 else hidden_width
            self.layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.out = nn.Linear(hidden_width, 2)
        self.enrich = nn.Parameter(torch.ones(1, dtype=torch.float64))

        self.apply(self._init_weights)
        if device is not None:
            self.to(device=device, dtype=torch.float64)
        else:
            self.double()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / module.in_features))
            nn.init.zeros_(module.bias)
            module.weight.data = module.weight.data.double()
            module.bias.data = module.bias.data.double()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, y], dim=1)
        for layer in self.layers:
            z = layer(z)
            z = activ_function(z, self.activation_function)
        z = self.out(z)
        return z


def extend_functions(
    ck: int,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: CrackConfig,
    coeffs: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MATLAB: extend_functions.m.

    This keeps the original enrichment field and, unlike the first draft,
    returns explicit analytic derivatives so the training logic matches the
    MATLAB code much more closely.
    """

    x_new = x - cfg.a_crack
    y_new = y.clone()
    eps = 1e-23

    if ck == 1:
        y_new = y_new + eps
        rho = torch.sqrt(x_new**2 + y_new**2)
        theta = safe_acos(x_new / rho)
    elif ck == 2:
        y_new = y_new - eps
        rho = torch.sqrt(x_new**2 + y_new**2)
        theta = -safe_acos(x_new / rho)
    else:
        raise ValueError("ck must be 1 or 2.")

    G_G = coeffs["G_G"]
    mu = coeffs["mu"]
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    prefactor = torch.sqrt(rho) / (G_G * sqrt_2pi)
    pref_grad = 1.0 / (G_G * sqrt_2pi * torch.sqrt(rho))

    sin_half = torch.sin(theta / 2.0)
    cos_half = torch.cos(theta / 2.0)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    h = 1.5 - 2.0 * mu - 0.5 * cos_theta

    u1 = prefactor * cos_half * h
    u2 = prefactor * sin_half * h

    # Explicit derivatives, consistent with the MATLAB supplementary code.
    u1_x = pref_grad * (
        0.5 * cos_half * h * cos_theta
        + 0.5 * sin_half * h * sin_theta
        - 0.5 * cos_half * sin_theta**2
    )
    u1_y = pref_grad * (
        0.5 * cos_half * h * sin_theta
        - 0.5 * sin_half * h * cos_theta
        + 0.5 * cos_half * sin_theta * cos_theta
    )
    u2_x = pref_grad * (
        0.5 * sin_half * h * cos_theta
        - 0.5 * cos_half * h * sin_theta
        - 0.5 * sin_half * sin_theta**2
    )
    u2_y = pref_grad * (
        0.5 * sin_half * h * sin_theta
        + 0.5 * cos_half * h * cos_theta
        + 0.5 * sin_half * sin_theta * cos_theta
    )

    return u1, u2, u1_x, u1_y, u2_x, u2_y


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def function_lossF(
    coeffs: Dict[str, float],
    u1_xx: torch.Tensor,
    u1_yy: torch.Tensor,
    u1_xy: torch.Tensor,
    u2_xx: torch.Tensor,
    u2_yy: torch.Tensor,
    u2_xy: torch.Tensor,
    idx_interior: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    f_1 = coeffs["c_1"] * u1_xx[idx_interior] + coeffs["c_2"] * u1_yy[idx_interior] + coeffs["c_3"] * u2_xy[idx_interior]
    f_2 = coeffs["c_1"] * u2_yy[idx_interior] + coeffs["c_2"] * u2_xx[idx_interior] + coeffs["c_3"] * u1_xy[idx_interior]
    zeros_1 = torch.zeros_like(f_1)
    zeros_2 = torch.zeros_like(f_2)
    return mse(f_1, zeros_1), mse(f_2, zeros_2)


def function_lossN(
    f_neumann_x: torch.Tensor,
    f_neumann_y: torch.Tensor,
    neumann_x: np.ndarray,
    neumann_y: np.ndarray,
    bc_x: torch.Tensor,
    bc_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return indexed_mse(f_neumann_x, bc_x, neumann_x), indexed_mse(f_neumann_y, bc_y, neumann_y)


def function_lossU(
    u1: torch.Tensor,
    u2: torch.Tensor,
    dirichlet_x: np.ndarray,
    dirichlet_y: np.ndarray,
    bc_x: torch.Tensor,
    bc_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return indexed_mse(u1, bc_x, dirichlet_x), indexed_mse(u2, bc_y, dirichlet_y)


def loss_function_half(
    model: CrackPINN,
    data: Dict[str, torch.Tensor | np.ndarray],
    coeffs: Dict[str, float],
    cfg: CrackConfig,
    ck: int,
) -> Dict[str, torch.Tensor]:
    """MATLAB: Loss_Function.m"""

    x = data["x"].clone().detach().requires_grad_(True)
    y = data["y"].clone().detach().requires_grad_(True)
    normal = data["normal"]
    boundary_count = int(data["boundary_count"])
    interior_idx = torch.arange(boundary_count, x.shape[0], device=x.device, dtype=torch.long)

    base = model(x, y)
    u1_base = base[:, 0:1]
    u2_base = base[:, 1:2]

    # MATLAB logic:
    #   - second derivatives in PDE residuals come only from the base network
    #   - enrichment is added only to displacement and first derivatives
    u1_x_base = grad_sum(u1_base, x)
    u1_y_base = grad_sum(u1_base, y)
    u2_x_base = grad_sum(u2_base, x)
    u2_y_base = grad_sum(u2_base, y)
    u1_xx_base = grad_sum(u1_x_base, x)
    u1_yy_base = grad_sum(u1_y_base, y)
    u1_xy_base = grad_sum(u1_x_base, y)
    u2_xx_base = grad_sum(u2_x_base, x)
    u2_yy_base = grad_sum(u2_y_base, y)
    u2_xy_base = grad_sum(u2_x_base, y)

    u1_add, u2_add, u1_x_add, u1_y_add, u2_x_add, u2_y_add = extend_functions(ck, x, y, cfg, coeffs)
    enrich = model.enrich.view(1, 1)

    u1 = u1_base + enrich * u1_add
    u2 = u2_base + enrich * u2_add

    u1_x = u1_x_base + enrich * u1_x_add
    u1_y = u1_y_base + enrich * u1_y_add
    u2_x = u2_x_base + enrich * u2_x_add
    u2_y = u2_y_base + enrich * u2_y_add

    u1_xx = u1_xx_base
    u1_yy = u1_yy_base
    u1_xy = u1_xy_base
    u2_xx = u2_xx_base
    u2_yy = u2_yy_base
    u2_xy = u2_xy_base

    n = normal
    f_neumann_x = (
        coeffs["a_1"] * n[:, 0:1] * u1_x[:boundary_count]
        + coeffs["a_3"] * n[:, 1:2] * u1_y[:boundary_count]
        + coeffs["a_2"] * n[:, 0:1] * u2_y[:boundary_count]
        + coeffs["a_3"] * n[:, 1:2] * u2_x[:boundary_count]
    )
    f_neumann_y = (
        coeffs["a_2"] * n[:, 1:2] * u1_x[:boundary_count]
        + coeffs["a_3"] * n[:, 0:1] * u1_y[:boundary_count]
        + coeffs["a_3"] * n[:, 0:1] * u2_x[:boundary_count]
        + coeffs["a_1"] * n[:, 1:2] * u2_y[:boundary_count]
    )

    lossF1, lossF2 = function_lossF(coeffs, u1_xx, u1_yy, u1_xy, u2_xx, u2_yy, u2_xy, interior_idx)
    lossU1, lossU2 = function_lossU(u1, u2, data["Dirichlet_BC_x"], data["Dirichlet_BC_y"], data["BC_x"], data["BC_y"])
    lossN1 = indexed_mse(f_neumann_x, data["BC_x"], data["Neumann_BC_x"])
    lossN2_load = indexed_mse(f_neumann_y, data["BC_y"], data["Neumann_BC_y_load"])
    lossN2_zero = indexed_mse(f_neumann_y, data["BC_y"], data["Neumann_BC_y_zero"])
    lossN2 = cfg.w_neumann_y_load * lossN2_load + cfg.w_neumann_y_zero * lossN2_zero

    if torch.isnan(lossU1):
        lossU1 = torch.tensor(0.0, dtype=torch.float64, device=x.device)
    if torch.isnan(lossU2):
        lossU2 = torch.tensor(0.0, dtype=torch.float64, device=x.device)

    return {
        "lossF1": lossF1,
        "lossF2": lossF2,
        "lossU1": lossU1,
        "lossU2": lossU2,
        "lossN1": lossN1,
        "lossN2": lossN2,
        "lossN2_load": lossN2_load,
        "lossN2_zero": lossN2_zero,
        "U1_interface": u1[data["interface"]],
        "U2_interface": u2[data["interface"]],
        "N1_interface": f_neumann_x[data["interface"]],
        "N2_interface": f_neumann_y[data["interface"]],
        "u1": u1,
        "u2": u2,
    }


def modelGradients(
    model_up: CrackPINN,
    data_up: Dict[str, torch.Tensor | np.ndarray],
    coeffs_up: Dict[str, float],
    model_down: CrackPINN,
    data_down: Dict[str, torch.Tensor | np.ndarray],
    coeffs_down: Dict[str, float],
    cfg: CrackConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MATLAB: modelGradients.m"""
    out_up = loss_function_half(model_up, data_up, coeffs_up, cfg, ck=1)
    out_down = loss_function_half(model_down, data_down, coeffs_down, cfg, ck=2)

    lossInterface_Dirichlet_x = 0.5*mse(out_up["U1_interface"], out_down["U1_interface"])
    lossInterface_Dirichlet_y = 0.5*mse(out_up["U2_interface"], out_down["U2_interface"])
    lossInterface_Neumann_x = 0.5*mse(out_up["N1_interface"], -out_down["N1_interface"])
    lossInterface_Neumann_y = 0.5*mse(out_up["N2_interface"], -out_down["N2_interface"])
    wN2 = 5.0
    loss = (
        out_up["lossF1"] + out_up["lossF2"] + out_up["lossU1"] + out_up["lossU2"] + out_up["lossN1"] + wN2*out_up["lossN2"]
        + out_down["lossF1"] + out_down["lossF2"] + out_down["lossU1"] + out_down["lossU2"] + out_down["lossN1"] + wN2*out_down["lossN2"]
        + lossInterface_Dirichlet_x + lossInterface_Dirichlet_y
        + lossInterface_Neumann_x + lossInterface_Neumann_y
    )

    parts = {
        "loss": float(loss.detach().cpu().item()),
        "lossF1_up": float(out_up["lossF1"].detach().cpu().item()),
        "lossF2_up": float(out_up["lossF2"].detach().cpu().item()),
        "lossU1_up": float(out_up["lossU1"].detach().cpu().item()),
        "lossU2_up": float(out_up["lossU2"].detach().cpu().item()),
        "lossN1_up": float(out_up["lossN1"].detach().cpu().item()),
        "lossN2_up": float(out_up["lossN2"].detach().cpu().item()),
        "lossN2_load_up": float(out_up["lossN2_load"].detach().cpu().item()),
        "lossN2_zero_up": float(out_up["lossN2_zero"].detach().cpu().item()),
        "lossF1_down": float(out_down["lossF1"].detach().cpu().item()),
        "lossF2_down": float(out_down["lossF2"].detach().cpu().item()),
        "lossU1_down": float(out_down["lossU1"].detach().cpu().item()),
        "lossU2_down": float(out_down["lossU2"].detach().cpu().item()),
        "lossN1_down": float(out_down["lossN1"].detach().cpu().item()),
        "lossN2_down": float(out_down["lossN2"].detach().cpu().item()),
        "lossN2_load_down": float(out_down["lossN2_load"].detach().cpu().item()),
        "lossN2_zero_down": float(out_down["lossN2_zero"].detach().cpu().item()),
        "lossInterface_Dirichlet_x": float(lossInterface_Dirichlet_x.detach().cpu().item()),
        "lossInterface_Dirichlet_y": float(lossInterface_Dirichlet_y.detach().cpu().item()),
        "lossInterface_Neumann_x": float(lossInterface_Neumann_x.detach().cpu().item()),
        "lossInterface_Neumann_y": float(lossInterface_Neumann_y.detach().cpu().item()),
    }
    return loss, parts


def format_loss_report(phase: str, step: int, parts: Dict[str, float]) -> str:
    return (
        f"{phase} {step:5d} | loss={parts['loss']:.3e} | "
        f"up[F1={parts['lossF1_up']:.2e}, F2={parts['lossF2_up']:.2e}, U1={parts['lossU1_up']:.2e}, U2={parts['lossU2_up']:.2e}, "
        f"N1={parts['lossN1_up']:.2e}, N2={parts['lossN2_up']:.2e} "
        f"(load={parts['lossN2_load_up']:.2e}, zero={parts['lossN2_zero_up']:.2e})] | "
        f"down[F1={parts['lossF1_down']:.2e}, F2={parts['lossF2_down']:.2e}, U1={parts['lossU1_down']:.2e}, U2={parts['lossU2_down']:.2e}, "
        f"N1={parts['lossN1_down']:.2e}, N2={parts['lossN2_down']:.2e} "
        f"(load={parts['lossN2_load_down']:.2e}, zero={parts['lossN2_zero_down']:.2e})] | "
        f"interface[Dx={parts['lossInterface_Dirichlet_x']:.2e}, Dy={parts['lossInterface_Dirichlet_y']:.2e}, "
        f"Nx={parts['lossInterface_Neumann_x']:.2e}, Ny={parts['lossInterface_Neumann_y']:.2e}]"
    )

# -----------------------------------------------------------------------------
# Model initialization / MATLAB compatibility wrappers
# -----------------------------------------------------------------------------
def Parameters_Initialize(
    numLayers: int,
    numNeurons: List[int],
    num_physics: int,
    num_dimensions: int,
    activation_function: int,
    device: torch.device,
) -> CrackPINN:
    """MATLAB: Parameters_Initialize.m"""
    hidden_layers = numLayers - 1
    hidden_width = int(numNeurons[0])
    model = CrackPINN(hidden_width=hidden_width, hidden_layers=hidden_layers, activation_function=activation_function, device=device)
    return model.double()


# -----------------------------------------------------------------------------
# Problem assembly
# -----------------------------------------------------------------------------
def build_problem(cfg: CrackConfig, device: torch.device) -> Dict[str, Dict]:
    global coeffs_up, coeffs_down

    coeffs_up = elasticity_coefficients(1)
    coeffs_down = elasticity_coefficients(2)

    bp_interface, dx_crack = crack_surface_elements(cfg, cfg.xscal_up(), cfg.yscal_up())
    ny_up = int(np.fix((cfg.yscal_up()[1] - cfg.yscal_up()[0]) / dx_crack) + 1)
    ny_down = int(np.fix((cfg.yscal_down()[1] - cfg.yscal_down()[0]) / dx_crack) + 1)

    geom_up = geodata_rectangle_up_and_down(
        cfg,
        cfg.refine_length_x(),
        cfg.refine_length_y(),
        cfg.xscal_up(),
        cfg.yscal_up(),
        ny_up,
        bp_interface,
        1,
    )
    geom_down = geodata_rectangle_up_and_down(
        cfg,
        cfg.refine_length_x(),
        cfg.refine_length_y(),
        cfg.xscal_down(),
        cfg.yscal_down(),
        ny_down,
        bp_interface,
        2,
    )

    cfg.N_interface_half_new = int(np.sum(geom_up["bp1"][:, 0] > cfg.a_crack))
    cfg.N_crack_new = int(np.sum(geom_up["bp1"][:, 0] < cfg.a_crack))

    bc_up = boundary_conditions(cfg, 1, geom_up["bp"], geom_up["bp1"], geom_up["bp2"], geom_up["bp3"], geom_up["bp4"], coeffs_up)
    bc_down = boundary_conditions(cfg, 2, geom_down["bp"], geom_down["bp1"], geom_down["bp2"], geom_down["bp3"], geom_down["bp4"], coeffs_down)

    interface = np.arange(cfg.N_crack_new, cfg.N_crack_new + cfg.N_interface_half_new, dtype=np.int64)

    problem = {
        "cfg": cfg,
        "coeffs_up": coeffs_up,
        "coeffs_down": coeffs_down,
        "bp_interface": bp_interface,
        "dx_crack": dx_crack,
        "up": {
            **geom_up,
            **bc_up,
            "boundary_count": len(geom_up["bp"]),
            "interface": interface,
        },
        "down": {
            **geom_down,
            **bc_down,
            "boundary_count": len(geom_down["bp"]),
            "interface": interface,
        },
    }

    for side in ("up", "down"):
        problem[side]["x"] = torch.as_tensor(problem[side]["S"][:, 0:1], dtype=torch.float64, device=device)
        problem[side]["y"] = torch.as_tensor(problem[side]["S"][:, 1:2], dtype=torch.float64, device=device)
        problem[side]["normal"] = torch.as_tensor(problem[side]["N"], dtype=torch.float64, device=device)
        problem[side]["BC_x"] = torch.as_tensor(problem[side]["BC_x"], dtype=torch.float64, device=device)
        problem[side]["BC_y"] = torch.as_tensor(problem[side]["BC_y"], dtype=torch.float64, device=device)
        problem[side]["Dirichlet_BC_x"] = torch.as_tensor(problem[side]["Dirichlet_BC_x"], dtype=torch.long, device=device)
        problem[side]["Dirichlet_BC_y"] = torch.as_tensor(problem[side]["Dirichlet_BC_y"], dtype=torch.long, device=device)
        problem[side]["Neumann_BC_x"] = torch.as_tensor(problem[side]["Neumann_BC_x"], dtype=torch.long, device=device)
        problem[side]["Neumann_BC_y"] = torch.as_tensor(problem[side]["Neumann_BC_y"], dtype=torch.long, device=device)
        problem[side]["interface"] = torch.as_tensor(problem[side]["interface"], dtype=torch.long, device=device)

    return problem


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_models(
    model_up: CrackPINN,
    model_down: CrackPINN,
    problem: Dict[str, Dict],
    cfg: CrackConfig,
) -> Tuple[Dict, List[Dict[str, float]]]:
    params = list(model_up.parameters()) + list(model_down.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)
    history: List[Dict[str, float]] = []

    def closure_loss() -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, parts = modelGradients(
            model_up, problem["up"], problem["coeffs_up"],
            model_down, problem["down"], problem["coeffs_down"],
            cfg,
        )
        return loss, parts

    t0 = time.time()
    for step in range(1, cfg.adam_steps + 1):
        optimizer.zero_grad()
        loss, parts = closure_loss()
        loss.backward()
        optimizer.step()
        if step == 1 or step % cfg.print_every == 0 or step == cfg.adam_steps:
            item = {"step": step, **parts}
            history.append(item)
            print(format_loss_report("Adam", step, parts))

    lbfgs = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=cfg.lbfgs_steps,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    lbfgs_iter = {"count": 0}

    def lbfgs_closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss, parts = closure_loss()
        loss.backward()
        lbfgs_iter["count"] += 1
        step = lbfgs_iter["count"]
        if step == 1 or step % 100 == 0 or step == cfg.lbfgs_steps:
            item = {"step": cfg.adam_steps + step, "phase": "LBFGS", **parts}
            history.append(item)
            print(format_loss_report("LBFGS", step, parts))
        return loss

    if cfg.lbfgs_steps > 0:
        print("Starting LBFGS refinement...")
        lbfgs.step(lbfgs_closure)

    elapsed = time.time() - t0
    result = {
        "elapsed": elapsed,
        "model_up": model_up,
        "model_down": model_down,
        "problem": problem,
    }
    return result, history


# -----------------------------------------------------------------------------
# Prediction / SIF
# -----------------------------------------------------------------------------
def predict_half_displacements(
    model: CrackPINN,
    data: Dict[str, torch.Tensor | np.ndarray],
    coeffs: Dict[str, float],
    cfg: CrackConfig,
    ck: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = data["x"]
    y = data["y"]
    with torch.no_grad():
        base = model(x, y)
        u1_add, u2_add, _, _, _, _ = extend_functions(ck, x, y, cfg, coeffs)
        u1 = base[:, 0:1] + model.enrich.view(1, 1) * u1_add
        u2 = base[:, 1:2] + model.enrich.view(1, 1) * u2_add
    return u1.detach().cpu().numpy().reshape(-1), u2.detach().cpu().numpy().reshape(-1)


def sif_displacement(
    u1_up: np.ndarray,
    u1_down: np.ndarray,
    u2_up: np.ndarray,
    u2_down: np.ndarray,
    bp1_up: np.ndarray,
    coeffs_up: Dict[str, float],
    coeffs_down: Dict[str, float],
    cfg: CrackConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB: SIF_Displacement.m"""
    sif_1 = np.zeros(cfg.n_crack_SIF, dtype=np.float64)
    r_kk = np.zeros(cfg.n_crack_SIF, dtype=np.float64)
    cp = bp1_up
    cracp_tip = cfg.N_crack_new
    debug_rows = []
    for kk in range(1, cfg.n_crack_SIF + 1):
        kk_new = kk - 1
        idx = cfg.N_crack_new - 1 - kk_new
        du_y = u2_up[idx] - u2_down[idx]
        rh = cfg.a_crack - cp[idx, 0]
        if rh <= 0:
            raise ValueError(f"Non-positive rh={rh} at idx={idx}.")
        sif_1[kk_new] = math.sqrt(2.0 * math.pi / rh) * coeffs_up["G_G"] * du_y / 4.0 / (1.0 - coeffs_up["mu"])
        r_kk[kk_new] = rh
        debug_rows.append((kk, idx, float(cp[idx, 0]), float(rh), float(du_y), float(sif_1[kk_new])))

    print("SIF DEM points:")
    for kk, idx, xk, rhk, duy, ki in debug_rows:
        print(f"  k={kk:02d} idx={idx:03d} x={xk:.8f} r={rhk:.8f} du_y={duy:.8e} K_I={ki:.8f}")
    return sif_1, r_kk


# -----------------------------------------------------------------------------
# Post-processing
# -----------------------------------------------------------------------------
def plot_sif_curve(rh: np.ndarray, sif_old: np.ndarray, cfg: CrackConfig, sif_exact_value: float, out_dir: Path) -> Tuple[float, float]:
    sl = slice(cfg.N_SIF_Start_1 - 1, cfg.n_crack_SIF)
    rh_fit = np.asarray(rh[sl], dtype=np.float64)
    sif_fit = np.asarray(sif_old[sl], dtype=np.float64)
    order = np.argsort(rh_fit)
    rh_fit = rh_fit[order]
    sif_fit = sif_fit[order]

    p = np.polyfit(rh_fit, sif_fit, 1)
    x_i = np.linspace(0.0, float(np.max(rh_fit)), 200)
    sif_line = x_i * p[0] + p[1]
    sif_num = float(p[1])
    sif_error = abs(sif_num - sif_exact_value) / abs(sif_exact_value)

    print("SIF extrapolation points:")
    for rr, kk in zip(rh_fit, sif_fit):
        print(f"  r={rr:.8f}  K_I={kk:.8f}")
    print(f"SIF linear fit: slope={p[0]:.8e}, intercept={p[1]:.8e}, exact={sif_exact_value:.8e}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=180)
    ax.plot(rh, sif_old, "b-o", lw=1.5, ms=4, label="DEM extraction")
    ax.plot(x_i, sif_line, "r--", lw=1.5, label="Linear extrapolation")
    ax.set_xlabel(r"$r$", fontsize=12)
    ax.set_ylabel(r"$K_I$", fontsize=12)
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "SIF.png", dpi=600, bbox_inches="tight")
    plt.show()

    return sif_num, sif_error


def plot_cod(bp1_up: np.ndarray, u2_up: np.ndarray, u2_down: np.ndarray, cfg: CrackConfig, out_dir: Path) -> None:
    idx = np.arange(max(cfg.N_crack_new, 1))
    rrr = bp1_up[idx]
    du_num = np.real(u2_up[idx] - u2_down[idx])

    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=180)
    ax.plot(rrr[:, 0], du_num, "ro", ms=4)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$u_y^{up} - u_y^{down}$", fontsize=12)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, lw=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "COD.png", dpi=600, bbox_inches="tight")
    plt.show()


def plot_williams_stress_cloud(sif_value: float, cfg: CrackConfig, coeffs_up: Dict[str, float], out_dir: Path) -> None:
    """MATLAB stress post-processing block appended to main.m."""
    print("Generating stress cloud from the extracted SIF...")
    res = 300
    x_plot = np.linspace(0.0, cfg.b_b, res)
    y_plot = np.linspace(-cfg.b_b, cfg.b_b, res)
    X_grid, Y_grid = np.meshgrid(x_plot, y_plot)

    r_grid = np.sqrt((X_grid - cfg.a_crack) ** 2 + Y_grid**2)
    theta_grid = np.arctan2(Y_grid, X_grid - cfg.a_crack)
    r_grid = np.maximum(r_grid, 1e-6)

    coef = sif_value / np.sqrt(2.0 * np.pi * r_grid)
    S_xx = coef * np.cos(theta_grid / 2.0) * (1.0 - np.sin(theta_grid / 2.0) * np.sin(3.0 * theta_grid / 2.0))
    S_yy = coef * np.cos(theta_grid / 2.0) * (1.0 + np.sin(theta_grid / 2.0) * np.sin(3.0 * theta_grid / 2.0))
    S_xy = coef * np.sin(theta_grid / 2.0) * np.cos(theta_grid / 2.0) * np.cos(3.0 * theta_grid / 2.0)

    S_zz = coeffs_up["mu"] * (S_xx + S_yy)
    Von_Mises = np.sqrt(
        0.5 * (
            (S_xx - S_yy) ** 2
            + (S_yy - S_zz) ** 2
            + (S_zz - S_xx) ** 2
            + 6.0 * S_xy**2
        )
    )

    # Fixed display ranges to better match the MATLAB reference figures.
    fields = [
        (S_xx, r"$\sigma_{xx}$", (-0.20, 1.60), [0.0, 0.5, 1.0, 1.5]),
        (S_yy, r"$\sigma_{yy}$", (-0.20, 1.60), [0.0, 0.5, 1.0, 1.5]),
        (S_xy, r"$\tau_{xy}$", (-1.00, 1.00), [-1.0, -0.5, 0.0, 0.5, 1.0]),
        (Von_Mises, r"Von Mises Equivalent Stress", (0.00, 1.20), [0.0, 0.4, 0.8, 1.2]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), dpi=180)
    for ax, (field, title, clim, ticks) in zip(axes, fields):
        vmin, vmax = clim
        levels = np.linspace(vmin, vmax, 80)
        cf = ax.contourf(X_grid, Y_grid, field, levels=levels, cmap="jet", extend="both")
        cf.set_clim(vmin, vmax)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_ticks(ticks)
        ax.plot([0.0, cfg.a_crack], [0.0, 0.0], "w-", lw=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.tick_params(direction="in", top=True, right=True)
    fig.suptitle(f"XPINN Computed Stress Fields (Extrapolated SIF = {sif_value:.4f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "stress.png", dpi=600, bbox_inches="tight")
    plt.show()
    print("Stress plot saved.")


def save_checkpoint(path: Path, model_up: CrackPINN, model_down: CrackPINN, history: List[Dict[str, float]], cfg: CrackConfig, result: Dict[str, object]) -> None:
    payload = {
        "model_up": model_up.state_dict(),
        "model_down": model_down.state_dict(),
        "history": history,
        "result": {
            "elapsed": result.get("elapsed", 0.0),
        },
        "config": cfg.__dict__,
    }
    torch.save(payload, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(path: Path, model_up: CrackPINN, model_down: CrackPINN, device: torch.device) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
    payload = torch.load(path, map_location=device)
    model_up.load_state_dict(payload["model_up"])
    model_down.load_state_dict(payload["model_down"])
    history = payload.get("history", [])
    result = payload.get("result", {})
    print(f"Checkpoint loaded from: {path}")
    return result, history


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main() -> Dict[str, object]:
    cfg = CrackConfig()
    set_seed(cfg.seed)
    device = make_device(cfg.use_cuda)
    out_dir = Path(__file__).resolve().parent
    checkpoint_path = out_dir / cfg.checkpoint_name

    problem = build_problem(cfg, device)

    numNeurons = [cfg.hidden_width] * cfg.hidden_layers
    numLayers = len(numNeurons) + 1
    num_physics = 2
    num_dimensions = 2

    model_up = Parameters_Initialize(numLayers, numNeurons, num_physics, num_dimensions, cfg.activation_function, device)
    model_down = Parameters_Initialize(numLayers, numNeurons, num_physics, num_dimensions, cfg.activation_function, device)

    result: Dict[str, object]
    history: List[Dict[str, float]] = []

    if cfg.use_saved_checkpoint and checkpoint_path.exists():
        try:
            result, history = load_checkpoint(checkpoint_path, model_up, model_down, device)
        except Exception as exc:
            print(f"Checkpoint load failed, retraining from scratch: {exc}")
            result, history = train_models(model_up, model_down, problem, cfg)
            save_checkpoint(checkpoint_path, model_up, model_down, history, cfg, result)
    else:
        print("Training upper and lower crack subdomains...")
        result, history = train_models(model_up, model_down, problem, cfg)
        save_checkpoint(checkpoint_path, model_up, model_down, history, cfg, result)

    coeffs_up = problem["coeffs_up"]
    coeffs_down = problem["coeffs_down"]
    up_data = problem["up"]
    down_data = problem["down"]

    u1_up, u2_up = predict_half_displacements(model_up, up_data, coeffs_up, cfg, ck=1)
    u1_down, u2_down = predict_half_displacements(model_down, down_data, coeffs_down, cfg, ck=2)

    sif_exact_value = sif_exact(cfg, coeffs_up)
    sif_old, rh = sif_displacement(u1_up, u1_down, u2_up, u2_down, up_data["bp1"], coeffs_up, coeffs_down, cfg)
    sif_num, sif_error = plot_sif_curve(rh, sif_old, cfg, sif_exact_value, out_dir)
    plot_cod(up_data["bp1"], u2_up, u2_down, cfg, out_dir)
    plot_williams_stress_cloud(sif_num, cfg, coeffs_up, out_dir)

    print(f"Elapsed total_time is: {result.get('elapsed', 0.0):.2f} seconds.")
    print(f"SIF_exact = {sif_exact_value:.4f}    SIF_num = {sif_num:.4f}    SIF_error = {sif_error:.4E}")

    return {
        "cfg": cfg,
        "problem": problem,
        "history": history,
        "model_up": model_up,
        "model_down": model_down,
        "u1_up": u1_up,
        "u2_up": u2_up,
        "u1_down": u1_down,
        "u2_down": u2_down,
        "sif_exact": sif_exact_value,
        "sif_num": sif_num,
        "sif_error": sif_error,
        "elapsed": result.get("elapsed", 0.0),
        "checkpoint_path": str(checkpoint_path),
    }


if __name__ == "__main__":
    main()

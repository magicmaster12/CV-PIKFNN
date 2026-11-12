# %% [markdown]
# # 自适应源点 KM-PIKFNN：含光滑非规则孔的有限方板
#
# 本算例直接在物理平面定义非规则孔，不使用保角映射。KM 点力基本解解析满足
# 无体力二维各向同性弹性控制方程，线性核系数通过最小二乘求解；小型神经网络
# 只学习孔内虚拟源点的空间变化位置 alpha(theta)。
#
# 运行前请把 COMSOL 导出的 `irregular_hole_M1/M2/M3_grid_801.txt`
# 放在本 notebook 同一目录。程序优先读取 M3，其次 M2、M1。

# %%
from __future__ import annotations

import copy
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# ------------------------- physical parameters -------------------------
DTYPE = torch.float64
# Float64 is normally more stable on CPU. Set KM_DEVICE=cuda manually if desired.
DEVICE = torch.device(os.environ.get("KM_DEVICE", "cpu"))

L_REAL = 1.0                 # half width of square plate, m
R0_REAL = 0.42               # mean hole radius, m
EPS3 = 0.22
EPS5 = 0.08
E_REAL = 1000.0              # Pa
NU = 0.30
U0_REAL = 1.0e-3             # m

# x*=x/L, u*=u/U0, sigma*=sigma/(E U0/L)
R0 = R0_REAL / L_REAL
E_STAR = 1.0
G_STAR = E_STAR / (2.0 * (1.0 + NU))
KAPPA = (3.0 - NU) / (1.0 + NU)       # plane stress
SIGMA_SCALE = E_REAL * U0_REAL / L_REAL

# ------------------------- numerical configuration ---------------------
N_HOLE_SOURCES = 32
N_OUTER_SOURCES_PER_SIDE = 18
N_HOLE_BND = 480
N_SIDE_BND = 140
OUTER_SOURCE_OFFSET = 0.25

ALPHA_MIN = 0.20
ALPHA_MAX = 0.92
FIXED_DEFAULT_ALPHA = 0.65
FIXED_ALPHA_GRID = np.arange(0.30, 0.901, 0.05)

RCOND = 1.0e-11
W_HOLE = 3.0
W_DISP = 10.0
W_TRACTION = 1.0
W_ANCHOR = 1.0

FAST_DEMO = os.environ.get("KM_FAST_DEMO", "0") == "1"
ADAPT_CYCLES = 3 if FAST_DEMO else 14
SOURCE_STEPS_PER_CYCLE = 10 if FAST_DEMO else 80
SOURCE_LR = 3.0e-3
REG_SMOOTH = 2.0e-4
REG_MOVE = 2.0e-6

# Set an explicit Path here if automatic discovery does not find the COMSOL file.
COMSOL_FILE: Path | None = None
OUTPUT_DIR = Path("irregular_hole_km_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Adaptive-source KM-PIKFNN for an irregular cavity ===")
print(f"device={DEVICE}, dtype={DTYPE}, plane-stress kappa={KAPPA:.8f}")
print(f"E*={E_STAR:.1f}, G*={G_STAR:.8f}, stress scale={SIGMA_SCALE:g} Pa")
print(f"fast demo={FAST_DEMO}")

# %% [markdown]
# ## 1. 几何、边界采样与 COMSOL 数据读取

# %%
def hole_radius(theta):
    return R0 * (1.0 + EPS3 * np.cos(3.0 * theta) + EPS5 * np.sin(5.0 * theta))


def hole_radius_derivatives(theta):
    r = hole_radius(theta)
    rp = R0 * (-3.0 * EPS3 * np.sin(3.0 * theta) + 5.0 * EPS5 * np.cos(5.0 * theta))
    rpp = R0 * (-9.0 * EPS3 * np.cos(3.0 * theta) - 25.0 * EPS5 * np.sin(5.0 * theta))
    return r, rp, rpp


def hole_curve(theta):
    r = hole_radius(theta)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def hole_geometry(theta):
    """Boundary points, physical-domain outward normals, arc weights, curvature."""
    r, rp, rpp = hole_radius_derivatives(theta)
    c, s = np.cos(theta), np.sin(theta)
    x = r * c
    y = r * s
    xp = rp * c - r * s
    yp = rp * s + r * c
    xpp = rpp * c - 2.0 * rp * s - r * c
    ypp = rpp * s + 2.0 * rp * c - r * s
    ds = np.sqrt(xp**2 + yp**2)
    # CCW hole curve: outward normal of the material domain points into the hole.
    nx = -yp / ds
    ny = xp / ds
    curvature = np.abs(xp * ypp - yp * xpp) / np.maximum(ds**3, 1.0e-15)
    return np.column_stack((x, y)), np.column_stack((nx, ny)), ds, curvature


def midpoint_parameter(n: int, phase: float) -> np.ndarray:
    return (np.arange(int(n), dtype=float) + phase) / float(n)


def make_boundary_data(n_hole=N_HOLE_BND, n_side=N_SIDE_BND, phase=0.5):
    theta = 2.0 * np.pi * midpoint_parameter(n_hole, phase)
    hole_xy, hole_n, hole_ds, _ = hole_geometry(theta)

    t = -1.0 + 2.0 * midpoint_parameter(n_side, phase)
    top = np.column_stack((t, np.ones_like(t)))
    bottom = np.column_stack((t, -np.ones_like(t)))
    right = np.column_stack((np.ones_like(t), t))
    left = np.column_stack((-np.ones_like(t), t))
    return {
        "theta": theta,
        "hole": hole_xy,
        "hole_n": hole_n,
        "hole_ds": hole_ds,
        "top": top,
        "bottom": bottom,
        "right": right,
        "left": left,
    }


def combine_boundary_data(first, second):
    combined = {}
    for key in ("theta", "hole", "hole_n", "hole_ds", "top", "bottom", "right", "left"):
        combined[key] = np.concatenate((first[key], second[key]), axis=0)
    return combined


TRAIN_DATA = make_boundary_data(phase=0.25)
VALID_DATA = make_boundary_data(phase=0.75)
COMBINED_DATA = combine_boundary_data(TRAIN_DATA, VALID_DATA)
TEST_DATA = make_boundary_data(n_hole=2 * N_HOLE_BND, n_side=2 * N_SIDE_BND, phase=0.50)


def discover_comsol_file() -> Path | None:
    if COMSOL_FILE is not None:
        path = Path(COMSOL_FILE)
        return path if path.exists() else None
    patterns = [
        "*M3*grid*801*.txt", "*M3*.txt",
        "*M2*grid*801*.txt", "*M2*.txt",
        "*M1*grid*801*.txt", "*M1*.txt",
    ]
    search_dirs = [Path.cwd(), Path.cwd() / "data"]
    for pattern in patterns:
        for directory in search_dirs:
            candidates = sorted(directory.glob(pattern)) if directory.exists() else []
            if candidates:
                return candidates[0]
    return None


def load_comsol_grid(path: Path):
    """Read x,y,u,v,sxx,syy,sxy,vm from a COMSOL spreadsheet text/CSV export."""
    first_data_line = ""
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("%"):
                first_data_line = stripped
                break
    delimiter = "," if "," in first_data_line else None
    data = np.genfromtxt(path, comments="%", delimiter=delimiter, dtype=float)
    data = np.atleast_2d(data)
    data = data[~np.all(np.isnan(data), axis=1)]
    if data.shape[1] < 8:
        raise ValueError(
            f"COMSOL file has {data.shape[1]} columns; expected at least 8 in the order "
            "x,y,u,v,solid.sx,solid.sy,solid.sxy,solid.mises."
        )
    return {
        "x": data[:, 0], "y": data[:, 1],
        "ux": data[:, 2], "uy": data[:, 3],
        "sxx": data[:, 4], "syy": data[:, 5], "sxy": data[:, 6],
        "vm": data[:, 7], "raw": data,
    }


theta_plot = np.linspace(0.0, 2.0 * np.pi, 1000, endpoint=True)
hole_plot = hole_curve(theta_plot)
fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k-", lw=1.3)
ax.fill(hole_plot[:, 0], hole_plot[:, 1], color="white", ec="k", lw=1.3)
ax.set_aspect("equal")
ax.set_xlim(-1.08, 1.08)
ax.set_ylim(-1.08, 1.08)
ax.set_xlabel(r"$x/L$")
ax.set_ylabel(r"$y/L$")
ax.set_title("Square plate with a smooth irregular cavity")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. KM 点力基本解和线性边界系统

# %%
def outer_source_points(n_per_side=N_OUTER_SOURCES_PER_SIDE, offset=OUTER_SOURCE_OFFSET):
    t = -1.0 + 2.0 * midpoint_parameter(n_per_side, 0.5)
    bottom = np.column_stack((t, -(1.0 + offset) * np.ones_like(t)))
    right = np.column_stack(((1.0 + offset) * np.ones_like(t), t))
    top = np.column_stack((t[::-1], (1.0 + offset) * np.ones_like(t)))
    left = np.column_stack((-(1.0 + offset) * np.ones_like(t), t[::-1]))
    return np.vstack((bottom, right, top, left))


SOURCE_THETA = 2.0 * np.pi * midpoint_parameter(N_HOLE_SOURCES, 0.5)
SOURCE_RADIUS = hole_radius(SOURCE_THETA)
OUTER_SOURCES = outer_source_points()


def fixed_source_points(alpha: float) -> np.ndarray:
    hole_sources = np.column_stack((
        alpha * SOURCE_RADIUS * np.cos(SOURCE_THETA),
        alpha * SOURCE_RADIUS * np.sin(SOURCE_THETA),
    ))
    return np.vstack((hole_sources, OUTER_SOURCES))


def coefficient_count(n_sources: int) -> int:
    return 2 * int(n_sources) + 8


def unpack_coefficients(coeffs, n_sources):
    coeffs = np.asarray(coeffs, dtype=float)
    force = coeffs[:n_sources] + 1j * coeffs[n_sources:2 * n_sources]
    i = 2 * n_sources
    phi0 = coeffs[i] + 1j * coeffs[i + 1]
    phi1 = coeffs[i + 2] + 1j * coeffs[i + 3]
    psi0 = coeffs[i + 4] + 1j * coeffs[i + 5]
    psi1 = coeffs[i + 6] + 1j * coeffs[i + 7]
    return force, phi0, phi1, psi0, psi1


def fields_from_coefficients_np(points, sources, coeffs):
    points = np.asarray(points, dtype=float)
    sources = np.asarray(sources, dtype=float)
    z = points[:, 0] + 1j * points[:, 1]
    zeta = sources[:, 0] + 1j * sources[:, 1]
    q = z[:, None] - zeta[None, :]
    logq = np.log(np.maximum(np.abs(q), 1.0e-300)) + 1j * np.angle(q)
    d1 = 1.0 / q
    d2 = -1.0 / q**2
    force, phi0, phi1, psi0, psi1 = unpack_coefficients(coeffs, len(sources))
    cf = 1.0 / (2.0 * np.pi * (KAPPA + 1.0))
    phi = -cf * (logq @ force) + phi0 + phi1 * z
    dphi = -cf * (d1 @ force) + phi1
    d2phi = -cf * (d2 @ force)
    psi = cf * (KAPPA * (logq @ np.conj(force)) + (d1 * np.conj(zeta)[None, :]) @ force) + psi0 + psi1 * z
    dpsi = cf * (KAPPA * (d1 @ np.conj(force)) + (d2 * np.conj(zeta)[None, :]) @ force) + psi1
    disp = (KAPPA * phi - z * np.conj(dphi) - np.conj(psi)) / (2.0 * G_STAR)
    stress_sum = 4.0 * dphi.real
    stress_diff = 2.0 * (np.conj(z) * d2phi + dpsi)
    sxx = 0.5 * (stress_sum - stress_diff.real)
    syy = 0.5 * (stress_sum + stress_diff.real)
    sxy = 0.5 * stress_diff.imag
    return disp.real, disp.imag, sxx, syy, sxy


def _components_from_response(z, phi, dphi, d2phi, psi, dpsi):
    disp = (KAPPA * phi - z[:, None] * np.conj(dphi) - np.conj(psi)) / (2.0 * G_STAR)
    stress_sum = 4.0 * dphi.real
    stress_diff = 2.0 * (np.conj(z)[:, None] * d2phi + dpsi)
    return {
        "ux": disp.real,
        "uy": disp.imag,
        "sxx": 0.5 * (stress_sum - stress_diff.real),
        "syy": 0.5 * (stress_sum + stress_diff.real),
        "sxy": 0.5 * stress_diff.imag,
    }


def field_design_matrices_np(points, sources):
    """Matrices mapping real KM coefficients to the five physical fields."""
    points = np.asarray(points, dtype=float)
    sources = np.asarray(sources, dtype=float)
    z = points[:, 0] + 1j * points[:, 1]
    zeta = sources[:, 0] + 1j * sources[:, 1]
    q = z[:, None] - zeta[None, :]
    logq = np.log(np.maximum(np.abs(q), 1.0e-300)) + 1j * np.angle(q)
    d1 = 1.0 / q
    d2 = -1.0 / q**2
    zb = np.conj(zeta)[None, :]
    cf = 1.0 / (2.0 * np.pi * (KAPPA + 1.0))

    # Unit real point-force coefficients.
    phi_r = -cf * logq
    dphi_r = -cf * d1
    d2phi_r = -cf * d2
    psi_r = cf * (KAPPA * logq + d1 * zb)
    dpsi_r = cf * (KAPPA * d1 + d2 * zb)
    real_resp = _components_from_response(z, phi_r, dphi_r, d2phi_r, psi_r, dpsi_r)

    # Unit imaginary point-force coefficients. The conjugated-force term changes sign.
    phi_i = 1j * phi_r
    dphi_i = 1j * dphi_r
    d2phi_i = 1j * d2phi_r
    psi_i = 1j * cf * (-KAPPA * logq + d1 * zb)
    dpsi_i = 1j * cf * (-KAPPA * d1 + d2 * zb)
    imag_resp = _components_from_response(z, phi_i, dphi_i, d2phi_i, psi_i, dpsi_i)

    # Eight affine columns: Re/Im(phi0), Re/Im(phi1), Re/Im(psi0), Re/Im(psi1).
    p = len(z)
    phi = np.zeros((p, 8), dtype=complex)
    dphi = np.zeros_like(phi)
    d2phi = np.zeros_like(phi)
    psi = np.zeros_like(phi)
    dpsi = np.zeros_like(phi)
    phi[:, 0], phi[:, 1] = 1.0, 1j
    phi[:, 2], phi[:, 3] = z, 1j * z
    dphi[:, 2], dphi[:, 3] = 1.0, 1j
    psi[:, 4], psi[:, 5] = 1.0, 1j
    psi[:, 6], psi[:, 7] = z, 1j * z
    dpsi[:, 6], dpsi[:, 7] = 1.0, 1j
    affine_resp = _components_from_response(z, phi, dphi, d2phi, psi, dpsi)

    matrices = {}
    for key in real_resp:
        matrices[key] = np.concatenate((real_resp[key], imag_resp[key], affine_resp[key]), axis=1)
    return matrices


def assemble_boundary_system(sources, data):
    rows, targets = [], []

    def add_group(matrix, target, weight, sample_weight=None):
        target = np.broadcast_to(np.asarray(target, dtype=float), (matrix.shape[0],))
        if sample_weight is None:
            normalized_weight = np.ones(matrix.shape[0], dtype=float)
        else:
            normalized_weight = np.asarray(sample_weight, dtype=float)
            normalized_weight = normalized_weight / np.mean(normalized_weight)
        scale = np.sqrt(weight * normalized_weight / max(matrix.shape[0], 1))
        rows.append(scale[:, None] * matrix)
        targets.append(scale * target)

    hole = field_design_matrices_np(data["hole"], sources)
    nx, ny = data["hole_n"][:, 0:1], data["hole_n"][:, 1:2]
    tx = nx * hole["sxx"] + ny * hole["sxy"]
    ty = nx * hole["sxy"] + ny * hole["syy"]
    add_group(tx, 0.0, W_HOLE, data["hole_ds"])
    add_group(ty, 0.0, W_HOLE, data["hole_ds"])

    outer_mats = {}
    for name in ("top", "bottom", "right", "left"):
        outer_mats[name] = field_design_matrices_np(data[name], sources)

    add_group(outer_mats["top"]["uy"], +1.0, W_DISP)
    add_group(outer_mats["top"]["sxy"], 0.0, W_TRACTION)
    add_group(outer_mats["bottom"]["uy"], -1.0, W_DISP)
    add_group(outer_mats["bottom"]["sxy"], 0.0, W_TRACTION)
    add_group(outer_mats["right"]["sxx"], 0.0, W_TRACTION)
    add_group(outer_mats["right"]["sxy"], 0.0, W_TRACTION)
    add_group(outer_mats["left"]["sxx"], 0.0, W_TRACTION)
    add_group(outer_mats["left"]["sxy"], 0.0, W_TRACTION)

    # Gauge condition equivalent to mean(ux)=0 on the outer boundary.
    outer_ux = np.vstack([outer_mats[name]["ux"] for name in ("top", "bottom", "right", "left")])
    add_group(np.mean(outer_ux, axis=0, keepdims=True), 0.0, W_ANCHOR)
    return np.vstack(rows), np.concatenate(targets)


def solve_linear_coefficients(sources, data, progress=False):
    tic = time.time()
    A, b = assemble_boundary_system(sources, data)
    coeffs, _, rank, singular_values = np.linalg.lstsq(A, b, rcond=RCOND)
    residual = A @ coeffs - b
    cond_raw = np.inf
    cond = np.inf
    if singular_values.size and singular_values[-1] > 0.0:
        cond_raw = singular_values[0] / singular_values[-1]
    if singular_values.size:
        retained = singular_values[singular_values > RCOND * singular_values[0]]
        if retained.size:
            cond = singular_values[0] / retained[-1]
    info = {
        "rows": int(A.shape[0]),
        "unknowns": int(A.shape[1]),
        "rank": int(rank),
        "loss": float(np.dot(residual, residual)),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "max": float(np.max(np.abs(residual))),
        "condition": float(cond),
        "condition_raw": float(cond_raw),
        "coefficient_norm": float(np.linalg.norm(coeffs)),
        "time_s": float(time.time() - tic),
    }
    if progress:
        print(info)
    return coeffs, info


def boundary_loss_np(sources, coeffs, data):
    A, b = assemble_boundary_system(sources, data)
    residual = A @ coeffs - b
    return float(np.dot(residual, residual))

# %% [markdown]
# ## 3. 固定源点基线：扫描并选择最佳常数缩放系数
#
# 固定基线不使用任意选定的单一源点距离。每个候选 alpha 仅在训练边界点上
# 拟合线性系数，再用错位的验证边界点选择最佳 alpha。

# %%
fixed_scan = []
print("\n--- Fixed-source alpha sweep ---")
for alpha in FIXED_ALPHA_GRID:
    sources = fixed_source_points(float(alpha))
    coeffs, train_info = solve_linear_coefficients(sources, TRAIN_DATA)
    val_loss = boundary_loss_np(sources, coeffs, VALID_DATA)
    fixed_scan.append({
        "alpha": float(alpha),
        "train_loss": train_info["loss"],
        "validation_loss": val_loss,
        "condition": train_info["condition"],
        "rank": train_info["rank"],
    })
    print(
        f"alpha={alpha:.2f} | train={train_info['loss']:.3e} | "
        f"validation={val_loss:.3e} | cond={train_info['condition']:.3e}"
    )

best_scan = min(fixed_scan, key=lambda item: item["validation_loss"])
BEST_FIXED_ALPHA = best_scan["alpha"]
FIXED_BEST_SOURCES = fixed_source_points(BEST_FIXED_ALPHA)
FIXED_BEST_COEFFS, FIXED_BEST_INFO = solve_linear_coefficients(FIXED_BEST_SOURCES, COMBINED_DATA)

FIXED_DEFAULT_SOURCES = fixed_source_points(FIXED_DEFAULT_ALPHA)
FIXED_DEFAULT_COEFFS, FIXED_DEFAULT_INFO = solve_linear_coefficients(FIXED_DEFAULT_SOURCES, COMBINED_DATA)

print(f"\nBest fixed alpha selected without COMSOL data: {BEST_FIXED_ALPHA:.3f}")
print("Best fixed refit:", FIXED_BEST_INFO)

scan_alpha = np.array([item["alpha"] for item in fixed_scan])
scan_val = np.array([item["validation_loss"] for item in fixed_scan])
scan_cond = np.array([item["condition"] for item in fixed_scan])
fig, ax1 = plt.subplots(figsize=(6.2, 4.2), dpi=150)
ax1.semilogy(scan_alpha, scan_val, "o-", color="#1f77b4", label="Validation loss")
ax1.axvline(BEST_FIXED_ALPHA, color="0.25", ls="--", lw=1.0)
ax1.set_xlabel(r"Constant source scale $\alpha$")
ax1.set_ylabel("Boundary validation loss", color="#1f77b4")
ax2 = ax1.twinx()
ax2.semilogy(scan_alpha, scan_cond, "s--", color="#d62728", label="Condition estimate")
ax2.set_ylabel("Condition estimate", color="#d62728")
ax1.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. 神经网络自适应孔内源点
#
# 网络输入为角度傅里叶特征和归一化曲率，输出有界的 alpha(theta)。
# 每轮先精确拟合线性核系数，再固定核系数更新源点网络，形成块坐标/变投影训练。

# %%
def to_tensor(array):
    return torch.as_tensor(array, dtype=DTYPE, device=DEVICE)


def numpy_boundary_to_torch(data):
    return {key: to_tensor(value) for key, value in data.items()}


TRAIN_TORCH = numpy_boundary_to_torch(TRAIN_DATA)


class SourcePlacementNet(nn.Module):
    def __init__(self, initial_alpha: float, hidden: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.to(device=DEVICE, dtype=DTYPE)
        p = (initial_alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
        p = float(np.clip(p, 1.0e-5, 1.0 - 1.0e-5))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(p / (1.0 - p)))

    def forward(self, features):
        return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * torch.sigmoid(self.net(features).squeeze(-1))


class AdaptiveSourceKM(nn.Module):
    def __init__(self, initial_alpha=BEST_FIXED_ALPHA):
        super().__init__()
        _, _, _, curvature = hole_geometry(SOURCE_THETA)
        curvature_feature = (curvature - curvature.mean()) / max(curvature.std(), 1.0e-12)
        features = np.column_stack((
            np.cos(SOURCE_THETA), np.sin(SOURCE_THETA),
            np.cos(2.0 * SOURCE_THETA), np.sin(2.0 * SOURCE_THETA),
            curvature_feature,
        ))
        self.register_buffer("theta", to_tensor(SOURCE_THETA))
        self.register_buffer("radius", to_tensor(SOURCE_RADIUS))
        self.register_buffer("features", to_tensor(features))
        self.register_buffer("outer_sources", to_tensor(OUTER_SOURCES))
        self.source_net = SourcePlacementNet(initial_alpha)
        n_total = N_HOLE_SOURCES + len(OUTER_SOURCES)
        self.coeffs = nn.Parameter(torch.zeros(coefficient_count(n_total), dtype=DTYPE, device=DEVICE))
        self.initial_alpha = float(initial_alpha)

    def alpha(self):
        return self.source_net(self.features)

    def source_points(self):
        alpha = self.alpha()
        hole = torch.stack((
            alpha * self.radius * torch.cos(self.theta),
            alpha * self.radius * torch.sin(self.theta),
        ), dim=1)
        return torch.cat((hole, self.outer_sources), dim=0)

    def source_points_numpy(self):
        with torch.no_grad():
            return self.source_points().detach().cpu().numpy()

    def set_coefficients(self, coeffs):
        with torch.no_grad():
            self.coeffs.copy_(to_tensor(coeffs))

    def source_regularization(self):
        alpha = self.alpha()
        cyclic_diff = alpha - torch.roll(alpha, shifts=1)
        smooth = torch.mean(cyclic_diff.square()) / (ALPHA_MAX - ALPHA_MIN)**2
        move = torch.mean((alpha - self.initial_alpha)**2) / (ALPHA_MAX - ALPHA_MIN)**2
        return REG_SMOOTH * smooth + REG_MOVE * move

    def fields(self, points):
        points = points.to(device=DEVICE, dtype=DTYPE)
        sources = self.source_points()
        z = torch.complex(points[:, 0], points[:, 1])
        zeta = torch.complex(sources[:, 0], sources[:, 1])
        q = z[:, None] - zeta[None, :]
        logq = torch.log(torch.clamp(torch.abs(q), min=1.0e-30)) + 1j * torch.angle(q)
        d1 = 1.0 / q
        d2 = -1.0 / q.square()
        n_sources = len(zeta)
        c = self.coeffs
        force = torch.complex(c[:n_sources], c[n_sources:2 * n_sources])
        i = 2 * n_sources
        phi0 = torch.complex(c[i], c[i + 1])
        phi1 = torch.complex(c[i + 2], c[i + 3])
        psi0 = torch.complex(c[i + 4], c[i + 5])
        psi1 = torch.complex(c[i + 6], c[i + 7])
        cf = 1.0 / (2.0 * np.pi * (KAPPA + 1.0))
        phi = -cf * (logq @ force) + phi0 + phi1 * z
        dphi = -cf * (d1 @ force) + phi1
        d2phi = -cf * (d2 @ force)
        psi = cf * (KAPPA * (logq @ torch.conj(force)) + (d1 * torch.conj(zeta)[None, :]) @ force) + psi0 + psi1 * z
        dpsi = cf * (KAPPA * (d1 @ torch.conj(force)) + (d2 * torch.conj(zeta)[None, :]) @ force) + psi1
        disp = (KAPPA * phi - z * torch.conj(dphi) - torch.conj(psi)) / (2.0 * G_STAR)
        stress_sum = 4.0 * dphi.real
        stress_diff = 2.0 * (torch.conj(z) * d2phi + dpsi)
        return {
            "ux": disp.real,
            "uy": disp.imag,
            "sxx": 0.5 * (stress_sum - stress_diff.real),
            "syy": 0.5 * (stress_sum + stress_diff.real),
            "sxy": 0.5 * stress_diff.imag,
        }


def adaptive_boundary_loss(model, data):
    hole = model.fields(data["hole"])
    nx, ny = data["hole_n"][:, 0], data["hole_n"][:, 1]
    tx = nx * hole["sxx"] + ny * hole["sxy"]
    ty = nx * hole["sxy"] + ny * hole["syy"]
    hole_weight = data["hole_ds"] / torch.mean(data["hole_ds"])
    loss_hole = W_HOLE * (
        torch.mean(hole_weight * tx.square()) + torch.mean(hole_weight * ty.square())
    )

    top = model.fields(data["top"])
    bottom = model.fields(data["bottom"])
    right = model.fields(data["right"])
    left = model.fields(data["left"])
    loss_disp = W_DISP * (
        torch.mean((top["uy"] - 1.0).square()) +
        torch.mean((bottom["uy"] + 1.0).square())
    )
    loss_traction = W_TRACTION * (
        torch.mean(top["sxy"].square()) + torch.mean(bottom["sxy"].square()) +
        torch.mean(right["sxx"].square()) + torch.mean(right["sxy"].square()) +
        torch.mean(left["sxx"].square()) + torch.mean(left["sxy"].square())
    )
    mean_ux = torch.mean(torch.cat((top["ux"], bottom["ux"], right["ux"], left["ux"])))
    loss_anchor = W_ANCHOR * mean_ux.square()
    physics = loss_hole + loss_disp + loss_traction + loss_anchor
    regularization = model.source_regularization()
    return physics + regularization, {
        "physics": physics,
        "hole": loss_hole,
        "disp": loss_disp,
        "traction": loss_traction,
        "anchor": loss_anchor,
        "regularization": regularization,
    }


model = AdaptiveSourceKM(BEST_FIXED_ALPHA)
initial_sources = model.source_points_numpy().copy()
initial_coeffs, _ = solve_linear_coefficients(initial_sources, TRAIN_DATA)
model.set_coefficients(initial_coeffs)
model.coeffs.requires_grad_(False)

optimizer = torch.optim.Adam(model.source_net.parameters(), lr=SOURCE_LR)
training_history = []
best_state = copy.deepcopy(model.state_dict())
best_validation_loss = boundary_loss_np(initial_sources, initial_coeffs, VALID_DATA)

print("\n--- Adaptive source optimization ---")
for cycle in range(ADAPT_CYCLES):
    # Exact linear refit at the current source positions.
    current_sources = model.source_points_numpy()
    current_coeffs, _ = solve_linear_coefficients(current_sources, TRAIN_DATA)
    model.set_coefficients(current_coeffs)

    for _ in range(SOURCE_STEPS_PER_CYCLE):
        optimizer.zero_grad()
        total_loss, terms = adaptive_boundary_loss(model, TRAIN_TORCH)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.source_net.parameters(), max_norm=10.0)
        optimizer.step()

    # Validation is evaluated after an exact coefficient refit.
    current_sources = model.source_points_numpy()
    current_coeffs, train_info = solve_linear_coefficients(current_sources, TRAIN_DATA)
    model.set_coefficients(current_coeffs)
    val_loss = boundary_loss_np(current_sources, current_coeffs, VALID_DATA)
    alpha_now = model.alpha().detach().cpu().numpy()
    training_history.append((cycle, train_info["loss"], val_loss, alpha_now.min(), alpha_now.max(), alpha_now.std()))
    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
    print(
        f"cycle={cycle + 1:02d}/{ADAPT_CYCLES} | train={train_info['loss']:.3e} | "
        f"validation={val_loss:.3e} | alpha=[{alpha_now.min():.3f},{alpha_now.max():.3f}] | "
        f"std={alpha_now.std():.3e}"
    )

model.load_state_dict(best_state)
ADAPTIVE_SOURCES = model.source_points_numpy()
ADAPTIVE_COEFFS, ADAPTIVE_INFO = solve_linear_coefficients(ADAPTIVE_SOURCES, COMBINED_DATA)
model.set_coefficients(ADAPTIVE_COEFFS)
ADAPTIVE_ALPHA = model.alpha().detach().cpu().numpy()

fixed_test_loss = boundary_loss_np(FIXED_BEST_SOURCES, FIXED_BEST_COEFFS, TEST_DATA)
adaptive_test_loss = boundary_loss_np(ADAPTIVE_SOURCES, ADAPTIVE_COEFFS, TEST_DATA)

print("\n--- Boundary-only comparison ---")
print(f"Fixed default alpha           : {FIXED_DEFAULT_ALPHA:.3f}")
print(f"Best fixed alpha              : {BEST_FIXED_ALPHA:.3f}")
print(f"Best fixed dense-test loss    : {fixed_test_loss:.6e}")
print(f"Adaptive dense-test loss      : {adaptive_test_loss:.6e}")
print(f"Best fixed condition estimate : {FIXED_BEST_INFO['condition']:.6e}")
print(f"Adaptive condition estimate   : {ADAPTIVE_INFO['condition']:.6e}")
print(f"Adaptive alpha range          : [{ADAPTIVE_ALPHA.min():.4f}, {ADAPTIVE_ALPHA.max():.4f}]")

# %% [markdown]
# ## 5. 源点位置和自适应规律

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7), dpi=160)
for ax in axes:
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k-", lw=1.0)
    ax.plot(hole_plot[:, 0], hole_plot[:, 1], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$y/L$")

axes[0].scatter(FIXED_BEST_SOURCES[:N_HOLE_SOURCES, 0], FIXED_BEST_SOURCES[:N_HOLE_SOURCES, 1], s=22, c="#1f77b4", label="Hole sources")
axes[0].scatter(FIXED_BEST_SOURCES[N_HOLE_SOURCES:, 0], FIXED_BEST_SOURCES[N_HOLE_SOURCES:, 1], s=10, c="0.55", label="Outer sources")
axes[0].set_title(f"Best fixed sources: alpha={BEST_FIXED_ALPHA:.2f}")
axes[0].legend(fontsize=8)

axes[1].scatter(ADAPTIVE_SOURCES[:N_HOLE_SOURCES, 0], ADAPTIVE_SOURCES[:N_HOLE_SOURCES, 1], s=25, c=ADAPTIVE_ALPHA, cmap="viridis", label="Adaptive hole sources")
axes[1].scatter(ADAPTIVE_SOURCES[N_HOLE_SOURCES:, 0], ADAPTIVE_SOURCES[N_HOLE_SOURCES:, 1], s=10, c="0.55", label="Fixed outer sources")
axes[1].set_title("Neural-network adaptive sources")
axes[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "source_locations.png", dpi=300, bbox_inches="tight")
plt.show()

_, _, _, source_curvature = hole_geometry(SOURCE_THETA)
curvature_normalized = source_curvature / source_curvature.max()
order = np.argsort(SOURCE_THETA)
fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=150)
ax1.plot(SOURCE_THETA[order], ADAPTIVE_ALPHA[order], "o-", color="#1f77b4", label=r"Learned $\alpha(\theta)$")
ax1.axhline(BEST_FIXED_ALPHA, color="0.25", ls="--", label="Best constant alpha")
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"Source scale $\alpha$")
ax2 = ax1.twinx()
ax2.plot(SOURCE_THETA[order], curvature_normalized[order], "s-", ms=3, color="#d62728", alpha=0.65, label="Normalized curvature")
ax2.set_ylabel("Normalized curvature")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
ax1.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "adaptive_alpha_vs_curvature.png", dpi=300, bbox_inches="tight")
plt.show()



# %% [markdown]
# ## 6. 与 COMSOL 规则栅格参考解比较
#
# COMSOL 仅用于最终评价，不用于固定 alpha 选择或自适应源点训练。

# %%
def evaluate_in_chunks(points, sources, coeffs, chunk_size=12000):
    fields = {key: [] for key in ("ux", "uy", "sxx", "syy", "sxy")}
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size]
        values = fields_from_coefficients_np(chunk, sources, coeffs)
        for key, value in zip(fields, values):
            fields[key].append(value)
    return {key: np.concatenate(parts) for key, parts in fields.items()}


def von_mises_plane_stress(sxx, syy, sxy):
    return np.sqrt(np.maximum(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2, 0.0))


def relative_l2(prediction, reference):
    denominator = max(np.linalg.norm(reference), 1.0e-15)
    return float(np.linalg.norm(prediction - reference) / denominator)


def error_metrics(pred, ref, mask):
    metrics = {}
    for key in ("ux", "uy", "sxx", "syy", "sxy", "vm"):
        metrics[key] = relative_l2(pred[key][mask], ref[key][mask])
    return metrics


comsol_path = discover_comsol_file()
COMSOL_RESULTS = None
if comsol_path is None:
    print(
        "\nCOMSOL grid file was not found. Put irregular_hole_M1/M2/M3_grid_801.txt "
        "next to this notebook or set COMSOL_FILE explicitly, then rerun this cell."
    )
else:
    print(f"\nLoading COMSOL reference: {comsol_path}")
    if "M1" in comsol_path.name.upper():
        print("WARNING: M1 is a preliminary mesh. Replace it with converged M3 data for the paper.")
    comsol = load_comsol_grid(comsol_path)
    finite = np.isfinite(np.column_stack((
        comsol["ux"], comsol["uy"], comsol["sxx"], comsol["syy"], comsol["sxy"], comsol["vm"]
    ))).all(axis=1)
    points_star = np.column_stack((comsol["x"] / L_REAL, comsol["y"] / L_REAL))
    valid_points = points_star[finite]

    reference = {
        "ux": comsol["ux"][finite] / U0_REAL,
        "uy": comsol["uy"][finite] / U0_REAL,
        "sxx": comsol["sxx"][finite] / SIGMA_SCALE,
        "syy": comsol["syy"][finite] / SIGMA_SCALE,
        "sxy": comsol["sxy"][finite] / SIGMA_SCALE,
        "vm": comsol["vm"][finite] / SIGMA_SCALE,
    }
    fixed_prediction = evaluate_in_chunks(valid_points, FIXED_BEST_SOURCES, FIXED_BEST_COEFFS)
    adaptive_prediction = evaluate_in_chunks(valid_points, ADAPTIVE_SOURCES, ADAPTIVE_COEFFS)
    fixed_prediction["vm"] = von_mises_plane_stress(fixed_prediction["sxx"], fixed_prediction["syy"], fixed_prediction["sxy"])
    adaptive_prediction["vm"] = von_mises_plane_stress(adaptive_prediction["sxx"], adaptive_prediction["syy"], adaptive_prediction["sxy"])

    # Align the arbitrary horizontal rigid translation before comparing ux.
    fixed_shift = np.mean(fixed_prediction["ux"] - reference["ux"])
    adaptive_shift = np.mean(adaptive_prediction["ux"] - reference["ux"])
    fixed_prediction["ux"] -= fixed_shift
    adaptive_prediction["ux"] -= adaptive_shift

    all_mask = np.ones(len(valid_points), dtype=bool)
    radial_distance = np.linalg.norm(valid_points, axis=1)
    near_hole_mask = radial_distance <= 0.80
    fixed_global = error_metrics(fixed_prediction, reference, all_mask)
    adaptive_global = error_metrics(adaptive_prediction, reference, all_mask)
    fixed_near = error_metrics(fixed_prediction, reference, near_hole_mask)
    adaptive_near = error_metrics(adaptive_prediction, reference, near_hole_mask)

    print("\n=== Relative L2 errors against COMSOL ===")
    print(f"{'Field':<12}{'Fixed-global':>16}{'Adaptive-global':>18}{'Fixed-near':>16}{'Adaptive-near':>18}")
    for key in ("ux", "uy", "sxx", "syy", "sxy", "vm"):
        print(
            f"{key:<12}{fixed_global[key]:>16.4e}{adaptive_global[key]:>18.4e}"
            f"{fixed_near[key]:>16.4e}{adaptive_near[key]:>18.4e}"
        )

    summary_path = OUTPUT_DIR / "comsol_error_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["field", "fixed_global", "adaptive_global", "fixed_near_hole", "adaptive_near_hole"])
        for key in ("ux", "uy", "sxx", "syy", "sxy", "vm"):
            writer.writerow([key, fixed_global[key], adaptive_global[key], fixed_near[key], adaptive_near[key]])

    COMSOL_RESULTS = {
        "path": str(comsol_path),
        "finite_mask": finite,
        "points_star": points_star,
        "reference": reference,
        "fixed": fixed_prediction,
        "adaptive": adaptive_prediction,
        "fixed_global": fixed_global,
        "adaptive_global": adaptive_global,
        "fixed_near": fixed_near,
        "adaptive_near": adaptive_near,
    }

    # Common-scale von Mises comparison on the exported regular grid.
    x_all, y_all = comsol["x"], comsol["y"]
    x_unique, y_unique = np.unique(x_all), np.unique(y_all)
    shape = (len(y_unique), len(x_unique))
    def grid_map(valid_values):
        full = np.full(len(x_all), np.nan)
        full[finite] = valid_values
        ix = np.searchsorted(x_unique, x_all)
        iy = np.searchsorted(y_unique, y_all)
        image = np.full(shape, np.nan)
        image[iy, ix] = full
        return image

    vm_ref_map = grid_map(reference["vm"] * SIGMA_SCALE)
    vm_fixed_map = grid_map(fixed_prediction["vm"] * SIGMA_SCALE)
    vm_adapt_map = grid_map(adaptive_prediction["vm"] * SIGMA_SCALE)
    vmax = np.nanpercentile(vm_ref_map, 99.8)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=150, constrained_layout=True)
    for ax, image_data, title in zip(
        axes,
        (vm_ref_map, vm_fixed_map, vm_adapt_map),
        ("COMSOL", "Best fixed KM", "Adaptive-source KM-NN"),
    ):
        im = ax.imshow(image_data, origin="lower", extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()], cmap="turbo", vmin=0.0, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=axes, label="von Mises stress (Pa)", shrink=0.88)
    fig.savefig(OUTPUT_DIR / "comsol_fixed_adaptive_vm.png", dpi=300, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 7. 保存核系数、源点和训练记录

# %%
history_array = np.asarray(training_history, dtype=float)
scan_array = np.column_stack((scan_alpha, scan_val, scan_cond))
save_items = {
    "source_theta": SOURCE_THETA,
    "fixed_default_sources": FIXED_DEFAULT_SOURCES,
    "fixed_default_coeffs": FIXED_DEFAULT_COEFFS,
    "fixed_best_alpha": np.asarray(BEST_FIXED_ALPHA),
    "fixed_best_sources": FIXED_BEST_SOURCES,
    "fixed_best_coeffs": FIXED_BEST_COEFFS,
    "adaptive_alpha": ADAPTIVE_ALPHA,
    "adaptive_sources": ADAPTIVE_SOURCES,
    "adaptive_coeffs": ADAPTIVE_COEFFS,
    "fixed_alpha_scan": scan_array,
    "training_history": history_array,
    "fixed_dense_test_loss": np.asarray(fixed_test_loss),
    "adaptive_dense_test_loss": np.asarray(adaptive_test_loss),
}
np.savez_compressed(OUTPUT_DIR / "irregular_hole_adaptive_km_results.npz", **save_items)
print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")

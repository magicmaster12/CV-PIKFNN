# -*- coding: utf-8 -*-
"""独立绘制源点布置示意图：仅依赖 numpy 和 matplotlib，无需求解程序。"""
from types import SimpleNamespace

# ---------------- 可调整的示意图参数 ----------------
L = 2.0                       # 方板边长
A_MAJOR = 0.5                 # 椭圆孔长半轴
LAMBDA_VALUES = (0.5, 0.05)    # 两张图的长短轴比
MAPPED_THRESHOLD = 0.2
INNER_SOURCE_SCALE = 0.70
OUTER_SOURCE_SCALE = 1.25
MAPPED_SOURCE_RADIUS = 0.85
SHOW_INNER = 20               # 普通分支内部展示点数
SHOW_OUTER_PER_SIDE = 6        # 外部每条边展示点数
SHOW_MAPPED = 20              # 辅助平面展示极点数
# --------------------------------------------------

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def plot_source_distribution(
    lam: float,
    cfg,
    sources: np.ndarray,
    n_inner: int,
    use_mapped_inner: bool,
    output_dir: Path,
) -> None:
    """Publication schematic, without any elasticity solve.

    Uses the geometry constants defined in this standalone script. The auxiliary panel
    shows the neighborhood of the mapped hole, not the entire mapped domain.
    """
    # Adjust these three values to change display density only.
    show_inner = SHOW_INNER
    show_outer_per_side = SHOW_OUTER_PER_SIDE
    show_mapped = SHOW_MAPPED

    def subset(points, count):
        points = np.asarray(points).reshape(-1)
        if len(points) <= count:
            return points
        indices = np.floor((np.arange(count) + 0.5) * len(points) / count).astype(int)
        return points[indices]

    b_minor = A_MAJOR * lam
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    inner = np.asarray(sources[:n_inner])
    outer = np.asarray(sources[n_inner:])
    source_extent = cfg.outer_source_scale * L / 2.0
    # The solver concatenates the four square sides in boundary order.
    side_size = cfg.n_outer_sources_per_side
    if len(outer) == 4 * side_size:
        outer_show = np.concatenate([
            subset(outer[i * side_size:(i + 1) * side_size], show_outer_per_side)
            for i in range(4)
        ])
    else:
        outer_show = subset(outer, 4 * show_outer_per_side)

    black, gray, red, blue = "#252525", "#8C8C8C", "#BE4540", "#326A93"
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        if use_mapped_inner:
            fig, axes = plt.subplots(1, 2, figsize=(9.0, 6.0))
            ax, ax_zeta = axes
            fig.subplots_adjust(left=0.025, right=0.975, top=0.90,
                                bottom=0.24, wspace=0.05)
        else:
            fig, ax = plt.subplots(figsize=(5.4, 6.0))
            ax_zeta = None
            fig.subplots_adjust(left=0.055, right=0.945, top=0.90, bottom=0.24)

        ax.add_patch(Rectangle(
            (-L / 2, -L / 2), L, L, fill=False, color=black, lw=1.5))
        ax.add_patch(Rectangle(
            (-source_extent, -source_extent), 2 * source_extent, 2 * source_extent,
            fill=False, color=gray, lw=1.0, ls=(0, (5, 4))))
        ax.plot(A_MAJOR * np.cos(theta), b_minor * np.sin(theta),
                color=black, lw=1.5)
        ax.scatter(outer_show.real, outer_show.imag, s=25, color=red,
                   marker="s", edgecolors="white", linewidths=0.4, zorder=4)
        if n_inner:
            inner_show = subset(inner, show_inner)
            # 按极角排序 + 闭合，保证沿椭圆一圈连续
            order = np.argsort(np.angle(inner_show - inner_show.mean()))
            inner_sorted = inner_show[order]
            inner_line = np.append(inner_sorted, inner_sorted[0])
            ax.plot(inner_line.real, inner_line.imag,
                    color=gray, lw=1.0, ls=(0, (4, 2)), zorder=5)
            ax.scatter(inner_show.real, inner_show.imag, s=24, color=blue,
                    edgecolors="white", linewidths=0.4, zorder=4)
        extent = max(source_extent, L / 2, A_MAJOR) * 1.07
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
        prefix = "(a) " if use_mapped_inner else ""
        ax.set_title(prefix + rf"Physical plane, $\lambda={lam:g}$", pad=14)

        handles = [
            Line2D([], [], color=black, lw=1.5, label="Physical boundary"),
            Line2D([], [], color=gray, lw=1, ls="--", label="Artificial source boundary"),
            Line2D([], [], color=red, marker="s", ls="none", markersize=5,
                   label="Outer sources"),
        ]
        if n_inner:
            handles.append(Line2D([], [], color=blue, marker="o", ls="none",
                                  markersize=5, label="Inner sources"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.4),
                  frameon=False, ncol=1, fontsize=13, handlelength=2.3,
                  borderaxespad=0, labelspacing=0.45)

        if ax_zeta is not None:
            radius = cfg.mapped_source_radius
            unit_circle = np.exp(1j * theta)
            source_circle = radius * unit_circle
            angles = (np.arange(cfg.n_mapped_sources) + 0.5) * (
                2 * np.pi / cfg.n_mapped_sources)
            mapped_show = subset(radius * np.exp(1j * angles), show_mapped)
            ax_zeta.plot(unit_circle.real, unit_circle.imag, color=black, lw=1.5,
                         label=r"Mapped hole boundary")
            ax_zeta.plot(source_circle.real, source_circle.imag, color=gray,
                         lw=1.0, ls=(0, (5, 4)),
                         label=rf"Artificial source boundary")
            ax_zeta.scatter(mapped_show.real, mapped_show.imag, s=27, color=blue,
                            edgecolors="white", linewidths=0.4, zorder=4,
                            label="Inner sources")
            mapped_extent = 1.08 * max(1.0, radius)
            ax_zeta.set(xlim=(-mapped_extent, mapped_extent),
                        ylim=(-mapped_extent, mapped_extent), aspect="equal")
            ax_zeta.set_axis_off()
            ax_zeta.set_title(r"(b) Auxiliary $\zeta$-plane (local view)", pad=14)
            ax_zeta.legend(loc="center", bbox_to_anchor=(0.5, 0.45),
                           frameon=False, fontsize=14, handlelength=2.3,
                           borderaxespad=0, labelspacing=0.45)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / f"source_distribution_lambda_{lam:g}"
        try:
            for suffix in ("png", "pdf", "svg"):
                fig.savefig(Path(str(stem) + "." + suffix), dpi=400,
                            bbox_inches="tight", pad_inches=0.12, facecolor="white")
        finally:
            plt.close(fig)


def main():
    # 仅生成绘图坐标，不读取求解结果，不进行网络训练或方程求解。
    cfg = SimpleNamespace(
        outer_source_scale=OUTER_SOURCE_SCALE,
        n_outer_sources_per_side=SHOW_OUTER_PER_SIDE,
        mapped_source_radius=MAPPED_SOURCE_RADIUS,
        n_mapped_sources=SHOW_MAPPED,
    )
    extent = OUTER_SOURCE_SCALE * L / 2.0
    # 对称分布，避免在四个角点重复绘制。
    values = np.linspace(-extent, extent, SHOW_OUTER_PER_SIDE + 2)[1:-1]
    outer = np.concatenate([
        values - 1j * extent,
        extent + 1j * values,
        values[::-1] + 1j * extent,
        -extent + 1j * values[::-1],
    ])
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    output_dir = base_dir / "source_schematic_figures"
    for lam in LAMBDA_VALUES:
        if lam <= 0:
            raise ValueError("lambda 必须大于零。")
        use_mapping = lam < MAPPED_THRESHOLD
        if use_mapping:
            inner = np.empty(0, dtype=complex)
        else:
            angles = (np.arange(SHOW_INNER) + 0.5) * 2 * np.pi / SHOW_INNER
            inner = INNER_SOURCE_SCALE * A_MAJOR * (
                np.cos(angles) + 1j * lam * np.sin(angles))
        plot_source_distribution(
            lam, cfg, np.concatenate([inner, outer]),
            len(inner), use_mapping, output_dir,
        )
    print(f"绘图完成，PNG / PDF / SVG 已保存到：{output_dir}")


if __name__ == "__main__":
    main()

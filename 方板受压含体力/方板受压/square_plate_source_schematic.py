# -*- coding: utf-8 -*-
"""方板受压源点示意图：独立运行，仅依赖 numpy 和 matplotlib。

保留 KM-PIKFNN-point-force(3).ipynb 的方板、源点与配点坐标生成规则。
采用 source_schematic_standalone(1).py 的颜色、字体和线条样式。
运行：python square_plate_source_schematic.py
也可直接把完整脚本粘贴至独立 Jupyter 单元运行，无需训练或参考解。
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------------- 用户可调整参数 ----------------
L = 1.0                          # 物理方板边长，区域 [0,L] × [0,L]
SOURCE_SCALE = 1.20               # alpha，人工源点边界的边长为 alpha * L
N_SOURCES_PER_SIDE = 24           # 实际源点数：每边24，总计96
N_BOUNDARY_PER_SIDE = 200         # 实际配点数：每边200，总计800
SHOW_SOURCES_PER_SIDE = 6         # 示意图抽样展示的每边源点数
SHOW_BOUNDARY_PER_SIDE = 14       # 示意图抽样展示的每边配点数
SHOW_ALL_POINTS = False           # True：显示全部96个源点和800个配点
SHOW_BOUNDARY_POINTS = True       # False：仅显示边界轮廓与源点
SHOW_TITLE = True
SHOW_FIGURE = True                # 在窗口或Jupyter中展示图片
OUTPUT_DIR = None                # None：保存到脚本目录/source_schematic_figures
# ------------------------------------------------

# 与示例代码完全一致的四种颜色。
BLACK = "#252525"
GRAY = "#8C8C8C"
RED = "#BE4540"
BLUE = "#326A93"

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


def square_side_points(lo, hi, n_per_side):
    """按下、右、上、左四条边生成内部点，排除角点，避免重复。"""
    if n_per_side < 1 or int(n_per_side) != n_per_side:
        raise ValueError("每边点数必须为正整数。")
    t = np.linspace(lo, hi, int(n_per_side) + 2)[1:-1]
    return [
        np.column_stack((t, np.full_like(t, lo))),
        np.column_stack((np.full_like(t, hi), t)),
        np.column_stack((t[::-1], np.full_like(t, hi))),
        np.column_stack((np.full_like(t, lo), t[::-1])),
    ]


def display_subset(points, count):
    """从实际坐标抽取展示点，不改变源点边界或用于训练的数量。"""
    if count < 1 or int(count) != count:
        raise ValueError("展示点数必须为正整数。")
    if count >= len(points):
        return points
    if count == 1:
        return points[len(points) // 2:len(points) // 2 + 1]
    # 抽取内部位置，避免示意图中相邻边的点在角点附近视觉重叠。
    indices = np.rint(np.linspace(0, len(points) - 1, count + 2)[1:-1]).astype(int)
    return points[indices]


def plot_square_plate_sources(output_dir=None, show=True):
    """仅生成绘图坐标，绘制并导出源点示意图；不执行有限元或网络训练。"""
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L必须大于零且为有限数。")
    if not np.isfinite(SOURCE_SCALE) or SOURCE_SCALE <= 1:
        raise ValueError("SOURCE_SCALE必须大于1，以保证源点位于物理域之外。")

    offset = 0.5 * (SOURCE_SCALE - 1.0) * L
    lo, hi = -offset, L + offset
    source_sides = square_side_points(lo, hi, N_SOURCES_PER_SIDE)
    boundary_sides = square_side_points(0.0, L, N_BOUNDARY_PER_SIDE)
    sources = np.vstack([
        side if SHOW_ALL_POINTS else display_subset(side, SHOW_SOURCES_PER_SIDE)
        for side in source_sides
    ])
    boundary = np.vstack([
        side if SHOW_ALL_POINTS else display_subset(side, SHOW_BOUNDARY_PER_SIDE)
        for side in boundary_sides
    ])

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.4, 6.0))
        fig.subplots_adjust(left=0.055, right=0.945, top=0.90, bottom=0.16)
        ax.set_facecolor("white")
        ax.add_patch(Rectangle(
            (0.0, 0.0), L, L, fill=False, color=BLACK, lw=1.5, zorder=1))
        ax.add_patch(Rectangle(
            (lo, lo), hi - lo, hi - lo,
            fill=False, color=GRAY, lw=1.0, ls=(0, (5, 4)), zorder=1))
        ax.scatter(
            sources[:, 0], sources[:, 1], s=25, color=RED,
            marker="s", edgecolors="white", linewidths=0.4, zorder=4)
        if SHOW_BOUNDARY_POINTS:
            ax.scatter(
                boundary[:, 0], boundary[:, 1], s=18, color=BLUE,
                marker="o", edgecolors="white", linewidths=0.4, zorder=4)

        extent = 0.5 * SOURCE_SCALE * L * 1.07
        center = 0.5 * L
        ax.set(xlim=(center - extent, center + extent),
               ylim=(center - extent, center + extent), aspect="equal")
        ax.set_axis_off()
        if SHOW_TITLE:
            ax.set_title(
                rf"Square plate, $\alpha={SOURCE_SCALE:g}$", pad=14)

        handles = [
            Line2D([], [], color=BLACK, lw=1.5, label="Physical boundary"),
            Line2D([], [], color=GRAY, lw=1.0, ls=(0, (5, 4)),
                   label="Artificial source boundary"),
            Line2D([], [], color=RED, marker="s", ls="none", markersize=5,
                   label="Outer sources"),
        ]
        if SHOW_BOUNDARY_POINTS:
            handles.append(Line2D(
                [], [], color=BLUE, marker="o", ls="none", markersize=5,
                label="Boundary collocation points"))
        ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.5),
                  frameon=False, ncol=1, fontsize=13, handlelength=2.3,
                  borderaxespad=0, labelspacing=0.45)

        if output_dir is None:
            base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
            output_dir = base / "source_schematic_figures"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / f"square_plate_sources_alpha_{SOURCE_SCALE:g}"
        paths = []
        for suffix in ("png", "pdf", "svg"):
            path = Path(str(stem) + "." + suffix)
            fig.savefig(path, dpi=400, bbox_inches="tight",
                        pad_inches=0.12, facecolor="white")
            paths.append(path)
        if show:
            plt.show()
        plt.close(fig)

    print(f"外扩比例 alpha={SOURCE_SCALE:g}，边界外偏移距离={offset:g} m")
    print(f"实际设置：{4 * N_SOURCES_PER_SIDE}个源点，"
          f"{4 * N_BOUNDARY_PER_SIDE}个边界配点（均不包含角点）。")
    print(f"本图显示：{len(sources)}个源点，"
          f"{len(boundary) if SHOW_BOUNDARY_POINTS else 0}个边界配点。")
    for path in paths:
        print(f"已保存：{path.resolve()}")
    return paths


if __name__ == "__main__":
    plot_square_plate_sources(output_dir=OUTPUT_DIR, show=SHOW_FIGURE)

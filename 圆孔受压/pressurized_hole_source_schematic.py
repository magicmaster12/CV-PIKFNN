"""圆孔受压算例源点示意图；独立运行，仅需 NumPy 和 Matplotlib。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

# ===== 可修改参数：与圆孔受压 KM-PIKFNN 算例一致 =====
A = 1.0                         # 圆孔半径
L_OUT = 80.0 * A                 # 外部正方形边长
SOURCE_SCALE = 1.20              # 外部源边界边长 / 物理外边界边长
HOLE_SOURCE_SCALE = 0.70         # 内部源圆半径 / 圆孔半径
N_SOURCES_PER_SIDE = 24          # 外部源点：4 × 24 = 96
N_HOLE_SOURCES = 64              # 内部源点：64；总源点数为 160
N_BOUNDARY_PER_SIDE = 160
N_HOLE_BOUNDARY = 360

# 示意图默认只显示真实点集的子集；不改变源点位置或实际数量。
SHOW_ALL_POINTS = False
SHOW_BOUNDARY_POINTS = True
SHOW_SOURCES_PER_SIDE = 6
SHOW_BOUNDARY_PER_SIDE = 14
SHOW_HOLE_SOURCES = 16
SHOW_HOLE_BOUNDARY = 28
LEGEND_FONTSIZE = 11
SHOW_TITLES = True
SAVE_FIGURES = True
SHOW_FIGURE = True
OUTPUT_DIR = None                # None：保存至脚本旁的 pressurized_hole_figures

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


def square_points(half, n):
    """每边 n 点，排除角点；顺序与算例一致。"""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("每边点数必须为正整数")
    t = np.linspace(-half, half, n + 2, dtype=np.float32)[1:-1]
    return [np.column_stack((t, np.full_like(t, -half))),
            np.column_stack((np.full_like(t, half), t)),
            np.column_stack((t[::-1], np.full_like(t, half))),
            np.column_stack((np.full_like(t, -half), t[::-1]))]


def circle_points(radius, n):
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("圆周点数必须为正整数")
    t = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t)))


def subset(points, count, circular=False):
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("显示点数必须为正整数")
    if SHOW_ALL_POINTS or count >= len(points):
        return points
    if circular:
        ids = np.floor(np.arange(count) * len(points) / count).astype(int)
    else:
        ids = np.rint(np.linspace(0, len(points) - 1, count + 2)[1:-1]).astype(int)
    return points[ids]


def draw_points(ax, points, source=False):
    ax.scatter(points[:, 0], points[:, 1], s=25 if source else 18,
               color=RED if source else BLUE, marker="s" if source else "o",
               edgecolors="white", linewidths=0.4, zorder=5)


def draw_circle(ax, radius, source=False):
    ax.add_patch(Circle((0, 0), radius, fill=False,
                        edgecolor=GRAY if source else BLACK,
                        linewidth=1.0 if source else 1.5,
                        linestyle=(0, (5, 4)) if source else "-", zorder=2))


def plot_schematic():
    if not (A > 0 and L_OUT > 2 * A and SOURCE_SCALE > 1
            and 0 < HOLE_SOURCE_SCALE < 1):
        raise ValueError("要求 A>0、L_OUT>2A、SOURCE_SCALE>1、0<HOLE_SOURCE_SCALE<1")
    half = L_OUT / 2
    source_half = half * SOURCE_SCALE
    inner = circle_points(HOLE_SOURCE_SCALE * A, N_HOLE_SOURCES)
    hole_boundary = circle_points(A, N_HOLE_BOUNDARY)
    outer = square_points(source_half, N_SOURCES_PER_SIDE)
    boundary = square_points(half, N_BOUNDARY_PER_SIDE)

    with plt.rc_context(STYLE):
        fig, (ax, zoom) = plt.subplots(1, 2, figsize=(10, 5), dpi=150,
                                      gridspec_kw={"width_ratios": [1.15, 1]})
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.04, top=0.90, wspace=0.16)
        ax.add_patch(Rectangle((-half, -half), L_OUT, L_OUT,
                               fill=False, edgecolor=BLACK, linewidth=1.5))
        ax.add_patch(Rectangle((-source_half, -source_half), 2 * source_half, 2 * source_half,
                               fill=False, edgecolor=GRAY, linewidth=1,
                               linestyle=(0, (5, 4))))
        for points in outer:
            draw_points(ax, subset(points, SHOW_SOURCES_PER_SIDE), source=True)
        if SHOW_BOUNDARY_POINTS:
            for points in boundary:
                draw_points(ax, subset(points, SHOW_BOUNDARY_PER_SIDE))

        for panel in (ax, zoom):
            draw_circle(panel, A)
            draw_circle(panel, HOLE_SOURCE_SCALE * A, source=True)
            # 全局图的孔径很小，孔边点仅在放大图显示，避免符号堆叠。
            if panel is zoom:
                draw_points(panel, subset(inner, SHOW_HOLE_SOURCES, circular=True), source=True)
                if SHOW_BOUNDARY_POINTS:
                    draw_points(panel, subset(hole_boundary, SHOW_HOLE_BOUNDARY, circular=True))
            panel.set_aspect("equal", adjustable="box")
            panel.set_axis_off()

        # 整体图保留 80a 的真实尺寸，孔边点集在右图放大显示。
        lim = source_half * 1.08
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        zoom.set_xlim(-1.24 * A, 1.24 * A)
        zoom.set_ylim(-1.24 * A, 1.24 * A)
        ax.annotate("Hole detail", xy=(0, 1.3 * A), xytext=(0, 0.26 * half),
                    ha="center", color=BLACK, fontsize=11,
                    arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8})

        handles = [Line2D([], [], color=BLACK, lw=1.5, label="Physical boundaries"),
                   Line2D([], [], color=GRAY, lw=1, linestyle=(0, (5, 4)),
                          label="Artificial source boundaries"),
                   Line2D([], [], color=RED, marker="s", linestyle="None", markersize=5,
                          markeredgecolor="white", markeredgewidth=0.4, label="Source points")]
        if SHOW_BOUNDARY_POINTS:
            handles.append(Line2D([], [], color=BLUE, marker="o", linestyle="None", markersize=4.5,
                                  markeredgecolor="white", markeredgewidth=0.4,
                                  label="Boundary collocation points"))
        ax.legend(handles=handles, loc="center", bbox_to_anchor=(0, -0.58 * half),
                  bbox_transform=ax.transData, frameon=False, fontsize=LEGEND_FONTSIZE,
                  handlelength=2.1, labelspacing=0.6)
        if SHOW_TITLES:
            ax.set_title("(a) Overall distribution", pad=12)
            zoom.set_title("(b) Hole detail", pad=12)

        paths = []
        if SAVE_FIGURES:
            root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
            folder = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else root / "pressurized_hole_figures"
            folder.mkdir(parents=True, exist_ok=True)
            for extension in ("png", "pdf", "svg"):
                path = folder / f"pressurized_hole_sources.{extension}"
                fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
                paths.append(path)
                print(f"Saved: {path}")
        if SHOW_FIGURE:
            plt.show()
        return fig, (ax, zoom), paths


if __name__ == "__main__":
    plot_schematic()

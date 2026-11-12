# -*- coding: utf-8 -*-
"""I 型中心裂纹四分之一域源点示意图，仅依赖 numpy、matplotlib。

在原 notebook 中：保留原配点/源点生成代码，用本文件替换绘图部分，调用
    fig, ax = plot_mode_i_center_crack_sources(cfg, bnd_pts, src_pts)
独立运行：python mode_i_center_crack_source_schematic.py
独立示例采用 a=1、b=2、h=6；请按真实工况修改底部 DEMO_CFG。
默认为均匀布局示意图：仅重排显示点，训练坐标不变。
show_all_points=True 显示原始坐标；schematic_points=False 抽取实际点。
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 与方板受压模板完全一致。
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


def _as_numpy_points(points):
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _display_subset(points, count):
    """只从现有坐标抽取点；不插值，不移动或修改实际配点。"""
    if len(points) <= count:
        return points
    indices = np.rint(np.linspace(0, len(points) - 1, count + 2)[1:-1]).astype(int)
    return points[np.unique(indices)]


def plot_mode_i_center_crack_sources(
    cfg, bnd_pts, src_pts,
    save_name="mode_i_center_crack_source_distribution.png",
    dpi=600, *, show_all_points=False, show_boundary_points=True,
    show_title=False, show=True,
    boundary_points_short_side=14, source_points_short_side=6,
    legend_fontsize=16, schematic_points=True,
):
    """保留用户压缩布局、模板配色；导出 PNG、PDF、SVG。

    boundary_points_short_side/source_points_short_side 是短边展示点数。
    默认按画面长度生成均匀示意点，避免压缩后左右边过密。
    图例位于域内；裂面及 ligament 属于四分之一域的边界段。
    """
    a, b, h = (float(cfg[key]) for key in ("a", "b", "h"))
    mx = float(cfg["src_margin_x_factor"]) * b
    my = float(cfg["src_margin_y_factor"]) * h
    if not all(np.isfinite(v) for v in (a, b, h, mx, my)):
        raise ValueError("几何参数必须为有限数。")
    if not (0 < a < b and h > 0 and mx > 0 and my > 0):
        raise ValueError("需要 0<a<b、h>0，并设置正的源点边界外偏移。")
    if boundary_points_short_side < 1 or source_points_short_side < 1:
        raise ValueError("展示点数必须为正数。")
    sources = _as_numpy_points(src_pts)
    boundary = {name: _as_numpy_points(p) for name, p in bnd_pts.items()}
    if not np.isfinite(sources).all() or any(not np.isfinite(p).all() for p in boundary.values()):
        raise ValueError("坐标中存在 NaN 或无穷值。")
    short = min(b, h)

    # 原生成函数四边源点交错排列，先按边分类、排序，再分别抽样。
    # 角点最终去重；只影响图中显示，不修改传入的 src_pts。
    displayed_sources = sources
    if not show_all_points:
        groups = []
        covered = np.zeros(len(sources), dtype=bool)
        for dim, value, sort_dim, length in (
            (1, -my, 0, b + 2 * mx), (1, h + my, 0, b + 2 * mx),
            (0, -mx, 1, h + 2 * my), (0, b + mx, 1, h + 2 * my),
        ):
            mask = np.isclose(sources[:, dim], value, rtol=1e-6, atol=1e-8 * max(b, h))
            covered |= mask
            side = sources[mask]
            side = side[np.argsort(side[:, sort_dim])]
            n_show = max(2, round(source_points_short_side * length / min(b + 2 * mx, h + 2 * my)))
            groups.append(_display_subset(side, n_show))
        # 若用户传入不在矩形四边上的源点，也保留它们。
        groups.append(sources[~covered])
        displayed_sources = np.unique(np.vstack(groups), axis=0)

    # 保持等比例；狭长模型增加画布高度，避免横向挤压图例。
    figsize = (5.4, 6.0) 
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.035,
                            top=0.95 if show_title else 0.975)
        ax.set_facecolor("white")
        ax.plot([-mx, b + mx, b + mx, -mx, -mx],
                [-my, -my, h + my, h + my, -my],
                color=GRAY, lw=1.0, ls=(0, (5, 4)), zorder=1)
        # 三条非底边；底部裂面与对称韧带分别绘制。
        ax.plot([0, 0, b, b], [0, h, h, 0], color=BLACK, lw=1.5, zorder=2)
        ax.plot([0, a], [0, 0], color=RED, lw=2.0,
                solid_capstyle="butt", zorder=3)
        ax.plot([a, b], [0, 0], color=BLACK, lw=1.25,
                ls=(0, (5, 4)), zorder=3)

        # 先确定画面坐标，再按像素长度控制展示密度。
        xpad = .04 * (b + 2 * mx)
        ypad = .035 * (h + 2 * my)  # 为顶部/底部标记留空间，避免方块被裁剪
        ax.set_xlim(-mx - xpad, b + mx + xpad)
        ax.set_ylim(-my - ypad, h + my + ypad)
        ax.set_aspect("auto")
        ax.set_axis_off()
        fig.canvas.draw()
        tr = ax.transData.transform
        px_width = abs(tr((b, 0))[0] - tr((0, 0))[0])
        px_height = abs(tr((0, h))[1] - tr((0, 0))[1])
        boundary_spacing = min(px_width, px_height) / boundary_points_short_side
        outer_width = abs(tr((b + mx, 0))[0] - tr((-mx, 0))[0])
        outer_height = abs(tr((0, h + my))[1] - tr((0, -my))[1])
        source_spacing = min(outer_width, outer_height) / source_points_short_side

        def uniform_rectangle(x0, x1, y0, y1, spacing):
            width = abs(tr((x1, y0))[0] - tr((x0, y0))[0])
            height = abs(tr((x0, y1))[1] - tr((x0, y0))[1])
            nx, ny = max(2, round(width / spacing)), max(2, round(height / spacing))
            # 单边等距，半个间距错开角点，无重复点。
            xs = x0 + (np.arange(nx) + .5) * (x1 - x0) / nx
            ys = y0 + (np.arange(ny) + .5) * (y1 - y0) / ny
            return {
                "top": np.column_stack((xs, np.full(nx, y1))),
                "bottom": np.column_stack((xs, np.full(nx, y0))),
                "left": np.column_stack((np.full(ny, x0), ys)),
                "right": np.column_stack((np.full(ny, x1), ys)),
            }

        uniform_boundary = None
        if schematic_points and not show_all_points:
            rectangle = uniform_rectangle(0, b, 0, h, boundary_spacing)
            bottom = rectangle.pop("bottom")
            # 裂面/韧带共享同一均匀底边点列，避免分段抽样造成密度差。
            distance = np.abs(tr(bottom)[:, 0] - tr((a, 0))[0])
            bottom = bottom[distance > 7.]  # 星形裂尖周围不叠加圆点
            rectangle["crack"] = bottom[bottom[:, 0] < a].copy()
            rectangle["crack"][:, 1] = float(cfg.get("eps_crack", 0.))
            rectangle["ligament"] = bottom[bottom[:, 0] > a]
            uniform_boundary = rectangle
            displayed_sources = np.vstack(list(uniform_rectangle(
                -mx, b + mx, -my, h + my, source_spacing).values()))

        shown_boundary = 0
        if show_boundary_points:
            for key, length, sort_dim in (
                ("top", b, 0), ("right", h, 1), ("left", h, 1),
                ("crack", a, 0), ("ligament", b - a, 0),
            ):
                pts = boundary.get(key, np.empty((0, 2)))
                pts = pts[np.argsort(pts[:, sort_dim])]
                if uniform_boundary is not None:
                    pts = uniform_boundary[key]
                elif not show_all_points:
                    pixel_length = px_height if key in ("left", "right") else px_width * length / b
                    pts = _display_subset(pts, max(2, round(pixel_length / boundary_spacing)))
                # 所有边界配点用同一蓝色；红色仅用于源点及裂面线。
                ax.scatter(pts[:, 0], pts[:, 1], s=18, color=BLUE,
                           edgecolors="white", linewidths=0.4, zorder=4)
                shown_boundary += len(pts)
        ax.scatter(displayed_sources[:, 0], displayed_sources[:, 1],
                   s=25, color=RED, marker="s", edgecolors="white",
                   linewidths=0.4, zorder=5)
        ax.scatter([a], [0], s=150, marker="*", color=BLACK,
                   edgecolors="white", linewidths=0.5, zorder=6)

        handles = [
            Line2D([], [], color=BLACK, lw=1.5, label="Physical boundary"),
            Line2D([], [], color=GRAY, lw=1.0, ls=(0, (5, 4)), label="Artificial source boundary"),
            Line2D([], [], color=RED, marker="s", ls="none", markersize=5, label="Outer sources"),
        ]
        if show_boundary_points:
            handles.append(Line2D([], [], color=BLUE, marker="o", ls="none",
                                  markersize=5, label="Boundary collocation points"))
        handles.extend([
            Line2D([], [], color=RED, lw=2.0, label="Crack face"),
            Line2D([], [], color=BLACK, lw=1.25, ls=(0, (5, 4)), label="Antisymmetry ligament"),
            Line2D([], [], color=BLACK, marker="*", ls="none", markersize=9, label="Crack tip"),
        ])
        legend = ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.53),
                          frameon=False, fontsize=legend_fontsize, handlelength=2.3,
                          labelspacing=1.15, borderaxespad=0, borderpad=0, handletextpad=.65)
        if show_title:
            ax.set_title("Mode-I center crack (quarter domain)", pad=14)
        # 最大化图例字号，约占内部宽度94%，同时保证上下留白。
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = legend.get_window_extent(renderer)
        factor = min(.94 * px_width / box.width, .74 * px_height / box.height)
        fitted_size = legend_fontsize * factor
        legend.remove()
        legend = ax.legend(handles=handles, loc="center",
                           bbox_to_anchor=(b / 2, h * .53), bbox_transform=ax.transData,
                           frameon=False, fontsize=fitted_size, handlelength=2.0,
                           labelspacing=1.15, borderaxespad=0, borderpad=0,
                           handletextpad=.65)

        output = Path(save_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        stem = output.with_suffix("") if output.suffix.lower() in (".png", ".pdf", ".svg") else output
        for suffix in ("png", "pdf", "svg"):
            path = Path(str(stem) + "." + suffix)
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        print(f"实际源点 {len(sources)} 个；图中展示 {len(displayed_sources)} 个。")
        print(f"图中边界配点 {shown_boundary} 个；训练坐标未改变。")
        if schematic_points and not show_all_points:
            print("显示点按画面均匀排布，仅用于布局示意，并非原始坐标逐点展示。")
        print(f"已保存 PNG / PDF / SVG：{stem.resolve()}")
        if show:
            plt.show()
    return fig, ax


def generate_demo_points(cfg):
    """独立示例：严格复现用户提供的均匀配点及矩形源点生成规则。"""
    a, b, h = (float(cfg[key]) for key in ("a", "b", "h"))
    def horizontal(x0, x1, y, n):
        x = np.linspace(x0, x1, n)
        return np.column_stack((x, np.full_like(x, y)))
    def vertical(x, n):
        y = np.linspace(0, h, n)
        return np.column_stack((np.full_like(y, x), y))
    boundary = {
        "top": horizontal(0, b, h, cfg["n_top"]),
        "right": vertical(b, cfg["n_side"]),
        "left": vertical(0, cfg["n_side"]),
        "crack": horizontal(.02 * a, .98 * a, cfg["eps_crack"], cfg["n_crack"]),
        "ligament": horizontal(1.02 * a, b, 0, cfg["n_ligament"]),
    }
    mx, my = cfg["src_margin_x_factor"] * b, cfg["src_margin_y_factor"] * h
    xs = np.linspace(-mx, b + mx, cfg["n_src_side"])
    ys = np.linspace(-my, h + my, cfg["n_src_side"])
    sources = []
    for x in xs:
        sources.extend([(x, -my), (x, h + my)])
    for y in ys[1:-1]:
        sources.extend([(-mx, y), (b + mx, y)])
    return boundary, np.asarray(sources)


if __name__ == "__main__":
    DEMO_CFG = dict(a=1., b=2., h=6., eps_crack=1e-6,
                    src_margin_x_factor=.15, src_margin_y_factor=.05,
                    n_top=200, n_side=200, n_crack=100, n_ligament=100,
                    n_src_side=80)
    demo_boundary, demo_sources = generate_demo_points(DEMO_CFG)
    output_dir = Path(__file__).resolve().parent
    plot_mode_i_center_crack_sources(
    DEMO_CFG, demo_boundary, demo_sources,
    save_name=output_dir / "mode_i_center_crack_source_distribution_uniform.png", show=False)


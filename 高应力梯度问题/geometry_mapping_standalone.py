# -*- coding: utf-8 -*-
"""Geometry and conformal-mapping schematics; no source or collocation points.

Standalone: numpy and matplotlib only. No solver results or training are needed.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# ---------------- Adjustable figure settings ----------------
L = 2.0
A_MAJOR = 0.5
LAMBDA_VALUES = (0.5, 0.05)
MAPPED_THRESHOLD = 0.2
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
LINEWIDTH = 1.6
DPI = 400
SAVE_FORMATS = ("png",)     # Add "pdf" or "svg" if desired.
SHOW = False               # True: also display the figures.
OUTPUT_FOLDER = "geometry_mapping_figures"
# ------------------------------------------------------------


def plot_geometry_mapping(lam, output_dir):
    if L <= 0 or A_MAJOR <= 0 or not 0 < lam <= 1:
        raise ValueError("Require L>0, A_MAJOR>0, and 0<lambda<=1.")
    if A_MAJOR >= L/2:
        raise ValueError("The hole must lie strictly inside the square plate.")
    b_minor = A_MAJOR*lam
    theta = np.linspace(0., 2.*np.pi, 1000)
    use_mapping = lam <= MAPPED_THRESHOLD
    black, blue = "#252525", "#326A93"
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": LABEL_FONTSIZE,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        if use_mapping:
            fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
            ax, mapped_ax = axes
            fig.subplots_adjust(left=.035, right=.965, bottom=.09,
                                top=.88, wspace=.37)
        else:
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            mapped_ax = None
            fig.subplots_adjust(left=.08, right=.92, bottom=.09, top=.88)

        # Physical boundaries only: no artificial boundaries or point markers.
        ax.add_patch(Rectangle((-L/2, -L/2), L, L, fill=False,
                               edgecolor=black, linewidth=LINEWIDTH))
        ax.plot(A_MAJOR*np.cos(theta), b_minor*np.sin(theta),
                color=black, linewidth=LINEWIDTH)
        extent = 1.07*L/2
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        prefix = "(a) " if use_mapping else ""
        ax.set_title(prefix + rf"Physical $z$-plane, $\lambda={lam:g}$",
                     fontsize=TITLE_FONTSIZE, pad=12)

        if mapped_ax is not None:
            # Local mapped-hole view; the outer square is not mapped to a circle.
            mapped_ax.plot(np.cos(theta), np.sin(theta), color=black,
                           linewidth=LINEWIDTH)
            mapped_ax.text(0., 0., r"$|w|=1$", ha="center", va="center",
                           fontsize=TITLE_FONTSIZE)
            mapped_ax.set_xlim(-1.08, 1.08)
            mapped_ax.set_ylim(-1.08, 1.08)
            mapped_ax.set_aspect("equal", adjustable="box")
            mapped_ax.set_axis_off()
            mapped_ax.set_title(r"(b) Mapped $w$-plane (local view)",
                                fontsize=TITLE_FONTSIZE, pad=12)

            fig.canvas.draw()
            left, right = ax.get_position(), mapped_ax.get_position()
            y_center = .5*(left.y0+left.y1)
            start = (left.x1+.006, y_center)
            end = (right.x0-.006, y_center)
            arrow = FancyArrowPatch(start, end, transform=fig.transFigure,
                                    arrowstyle="-|>", mutation_scale=22,
                                    linewidth=2., color=blue)
            fig.add_artist(arrow)
            fig.text(.5*(start[0]+end[0]), y_center+.07,
                     "Conformal\nmapping", ha="center", va="bottom",
                     fontsize=LABEL_FONTSIZE, color=blue)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir/f"geometry_mapping_lambda_{lam:g}"
        for suffix in SAVE_FORMATS:
            path = Path(str(stem)+"."+suffix)
            fig.savefig(path, dpi=DPI, bbox_inches="tight",
                        pad_inches=.12, facecolor="white")
            print("Saved:", path.resolve())
        if SHOW:
            plt.show()
        plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for lam in LAMBDA_VALUES:
        plot_geometry_mapping(lam, base_dir/OUTPUT_FOLDER)


if __name__ == "__main__":
    main()

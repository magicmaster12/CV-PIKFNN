"""
Plot the six KM-PIKFNN NPZ results as one 4 x 6 stress-map figure.

Rows:
    sigma_xx, sigma_yy, sigma_xy, von Mises stress
Columns:
    lambda = 1, 0.5, 0.05, 0.02, 0.01, 0.005

The six NPZ files are written by:
    高应力梯度问题_KM-PIKFNN.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, Rectangle


# Physical geometry used by the notebook.
L = 2.0
A_MAJOR = 0.5

LAMBDA_ORDER = (1.0, 0.5, 0.05, 0.02, 0.01, 0.005)
FIELD_ORDER = ("Sxx", "Syy", "Sxy", "VM")
FIELD_TITLES = (
    r"$\sigma_{xx}$",
    r"$\sigma_{yy}$",
    r"$\sigma_{xy}$",
    r"$\sigma_{\mathrm{vM}}$",
)


def lambda_text(value: float) -> str:
    """Format lambda values without unnecessary trailing zeros."""
    return f"{value:g}"


def npz_path(input_dir: Path, lam: float) -> Path:
    return input_dir / f"km_pikfnn_fields_lambda_{lambda_text(lam)}.npz"


def load_results(input_dir: Path) -> list[tuple[float, dict[str, np.ndarray]]]:
    """Load results in the requested physical-case order."""
    results = []
    missing = []

    for lam in LAMBDA_ORDER:
        path = npz_path(input_dir, lam)
        if not path.exists():
            missing.append(str(path))
            continue

        with np.load(path) as data:
            required = set(FIELD_ORDER) | {"XX", "YY"}
            absent = required.difference(data.files)
            if absent:
                raise KeyError(f"{path} is missing fields: {sorted(absent)}")
            result = {key: np.asarray(data[key]) for key in required}
        results.append((lam, result))

    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Could not find these NPZ files:\n{missing_text}")
    return results


def row_limits(
    results: list[tuple[float, dict[str, np.ndarray]]],
    percentile: float,
) -> dict[str, tuple[float, float]]:
    """Compute one symmetric color range for the six panels in each row."""
    limits = {}
    for field in FIELD_ORDER:
        values = np.concatenate(
            [
                np.abs(result[field][np.isfinite(result[field])]).ravel()
                for _, result in results
            ]
        )
        vmax = float(np.nanpercentile(values, percentile))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.nanmax(values))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        limits[field] = (-vmax, vmax)
    return limits


def add_geometry(ax: plt.Axes, lam: float) -> None:
    """Draw the square boundary and the excluded elliptical hole."""
    ax.add_patch(
        Rectangle(
            (-L / 2.0, -L / 2.0),
            L,
            L,
            fill=False,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )
    )
    ax.add_patch(
        Ellipse(
            (0.0, 0.0),
            2.0 * A_MAJOR,
            2.0 * A_MAJOR * lam,
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
        )
    )


def plot_group(
    results: list[tuple[float, dict[str, np.ndarray]]],
    output_path: Path,
    percentile: float = 99.5,
    dpi: int = 400,
) -> None:
    """Create and save a 4 x 6 plot with one colorbar per row."""
    limits = row_limits(results, percentile)
    norms = {
        field: Normalize(vmin=vmin, vmax=vmax)
        for field, (vmin, vmax) in limits.items()
    }

    # The first column is reserved for row labels; the last column contains
    # one colorbar for each row. Each row's six panels share that row's norm.
    fig = plt.figure(figsize=(18.0, 10.5), dpi=dpi)
    grid = fig.add_gridspec(
        nrows=4,
        ncols=8,
        width_ratios=[0.42, 1, 1, 1, 1, 1, 1, 0.075],
        wspace=0.06,
        hspace=0.13,
        left=0.055,
        right=0.965,
        bottom=0.075,
        top=0.925,
    )

    axes = np.empty((4, 6), dtype=object)
    row_images = {}
    ROW_CMAPS = {
    "Sxx": "RdBu_r",
    "Syy": "RdBu_r",
    "Sxy": "RdBu_r",
    "VM": "turbo",   
}

    for col, (lam, result) in enumerate(results, start=1):
        xx = result["XX"]
        yy = result["YY"]
        extent = [
            float(np.nanmin(xx)),
            float(np.nanmax(xx)),
            float(np.nanmin(yy)),
            float(np.nanmax(yy)),
        ]

        for row, (field, title) in enumerate(zip(FIELD_ORDER, FIELD_TITLES)):
            ax = fig.add_subplot(grid[row, col])
            axes[row, col - 1] = ax
            ax.set_facecolor("white")
            image = ax.imshow(
                result[field],
                origin="lower",
                extent=extent,
                cmap=ROW_CMAPS[field],
                norm=norms[field],
                interpolation="bilinear",
                aspect="equal",
            )
            row_images[field] = image
            add_geometry(ax, lam)

            ax.set_xlim(-L / 2.0, L / 2.0)
            ax.set_ylim(-L / 2.0, L / 2.0)
            ax.set_xticks([-1.0, 0.0, 1.0])
            ax.set_yticks([-1.0, 0.0, 1.0])
            ax.tick_params(labelsize=8, length=2.5, pad=2)
            ax.grid(False)

            if row == 0:
                ax.set_title(rf"$\lambda={lambda_text(lam)}$", fontsize=12, pad=6)
            if row < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("$x$ (m)", fontsize=9, labelpad=2)
            if col > 1:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("$y$ (m)", fontsize=9, labelpad=2)

    # Row labels sit outside the panels and do not consume a panel's axes.
    for row, title in enumerate(FIELD_TITLES):
        row_ax = fig.add_subplot(grid[row, 0])
        row_ax.axis("off")
        row_ax.text(
            0.92,
            0.5,
            title,
            ha="right",
            va="center",
            rotation=90,
            fontsize=14,
        )

    if not row_images:
        raise RuntimeError("No images were created.")

    for row, field in enumerate(FIELD_ORDER):
        colorbar_ax = fig.add_subplot(grid[row, 7])
        colorbar = fig.colorbar(row_images[field], cax=colorbar_ax)
        colorbar.set_label("Stress (Pa)", fontsize=10, labelpad=7)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.formatter.set_powerlimits((0, 0))
        colorbar.update_ticks()

        colorbar.ax.yaxis.set_label_coords(3.2, 0.5)

    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
    for field in FIELD_ORDER:
        vmin, vmax = limits[field]
        print(f"{field} row color range: [{vmin:.6g}, {vmax:.6g}] Pa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 4 x 6 KM-PIKFNN stress-map group plot."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            r"C:\Users\magic\Desktop\KM-PIKFNN\高应力梯度问题"
            r"\KM_PIKFNN_Corrected_Results"
        ),
        help="Directory containing km_pikfnn_fields_lambda_*.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to input-dir/stress_group_4x6.png",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Percentile used for the shared symmetric color limit.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.percentile <= 100.0:
        raise ValueError("--percentile must be in (0, 100].")

    input_dir = args.input_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_dir / "stress_group_4x6.png"
    )
    results = load_results(input_dir)
    plot_group(
        results,
        output_path=output_path,
        percentile=args.percentile,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

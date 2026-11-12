"""Redraw saved Mode-II geometry results; no solver or training is imported.

Place this script beside modeii_width_height_results, then run:
    python replot_modeii_geometry.py
Or specify the CSV:
    python replot_modeii_geometry.py --csv /path/to/kii_geometry_comparison.csv
Output: two separate figures, each saved as 600-dpi PNG and vector PDF.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

# Edit these settings to adjust the appearance without changing the results.
CFG = {
    "csv_path": "modeii_width_height_results/kii_geometry_comparison.csv",
    "figsize": (6.6, 4.8),
    "font_size": 12,
    "label_size": 15,
    "legend_size": 14,
    "line_width": 1.6,
    "marker_size": 6.0,
    "legend_columns": 2,
    "parameter_legend_anchor": (0.985, 0.985),
    "method_legend_anchor": (0.015, 0.015),
    "dpi": 600,
}

COLORS = ["#606060", "#FF4545", "#267DFF", "#20A976", "#B568ED", "#00B8C4"]
MARKERS = ["s", "o", "^", "v", "D", ">", "<", "p", "h", "*"]


def resolve_csv(value):
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "kii_geometry_comparison.csv"
    if path.is_file():
        return path
    beside_script = Path(__file__).resolve().parent / path
    if not path.is_absolute() and beside_script.is_file():
        return beside_script
    raise FileNotFoundError(
        f"Saved results not found: {path}\n"
        "Set CFG['csv_path'] or run with --csv /path/to/kii_geometry_comparison.csv.\n"
        "This script only reads saved results; it never runs the solvers."
    )


def read_results(path, raw=False):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"b_over_a", "h_over_a", "KII_KM", "KII_BEM"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        rows = []
        seen = set()
        for line, original in enumerate(reader, start=2):
            row = {key: float(original[key]) for key in required}
            pair = row["b_over_a"], row["h_over_a"]
            if pair in seen:
                raise ValueError(f"Duplicate geometry at CSV line {line}: {pair}")
            seen.add(pair)
            if "bem_elements" in original and int(float(original["bem_elements"])) != 2304:
                raise ValueError(f"CSV line {line} does not use 2304 BEM elements.")
            if raw:
                row["plot_KM"], row["plot_BEM"] = row["KII_KM"], row["KII_BEM"]
            elif original.get("FII_KM") and original.get("FII_BEM"):
                row["plot_KM"] = float(original["FII_KM"])
                row["plot_BEM"] = float(original["FII_BEM"])
            else:
                if not original.get("a") or not original.get("tau0"):
                    raise ValueError("Normalized columns FII_KM/FII_BEM are missing. "
                                     "Use --raw for unnormalized KII.")
                scale = float(original["tau0"]) * np.sqrt(np.pi * float(original["a"]))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"Invalid normalization scale at CSV line {line}.")
                row["plot_KM"], row["plot_BEM"] = row["KII_KM"] / scale, row["KII_BEM"] / scale
            if not all(np.isfinite(v) for v in row.values()):
                raise ValueError(f"Nonfinite result at CSV line {line}.")
            rows.append(row)
    if not rows:
        raise ValueError("The saved results CSV is empty.")
    return rows


def draw_panel(rows, fixed_key, varying_key, fixed_label, xlabel, stem, folder, raw):
    fig, ax = plt.subplots(figsize=CFG["figsize"], dpi=150)
    fig.subplots_adjust(left=.16, right=.975, bottom=.15, top=.975)
    values = sorted({r[fixed_key] for r in rows})
    handles = []
    for j, value in enumerate(values):
        group = sorted((r for r in rows if r[fixed_key] == value), key=lambda r: r[varying_key])
        x = np.array([r[varying_key] for r in group])
        km = np.array([r["plot_KM"] for r in group])
        bem = np.array([r["plot_BEM"] for r in group])
        color, marker = COLORS[j % len(COLORS)], MARKERS[j % len(MARKERS)]
        # Connect only the actual computed points; no smoothing or fitted values.
        ax.plot(x, km, color=color, lw=CFG["line_width"], zorder=2)
        ax.plot(x, bem, linestyle="none", marker=marker, ms=CFG["marker_size"],
                mfc="white", mec=color, mew=1.1, zorder=3)
        handles.append(Line2D([], [], color=color, lw=CFG["line_width"],
                              marker=marker, mfc="white", mew=1.1,
                              ms=CFG["marker_size"] - 1,
                              label=rf"${fixed_label}={value:g}$"))

    ax.set_xlabel(xlabel, fontsize=CFG["label_size"])
    ax.set_ylabel(r"$K_{II}$" if raw else r"$K_{II}/(\tau_0\sqrt{\pi a})$",
                  fontsize=CFG["label_size"])
    x_values = sorted({r[varying_key] for r in rows})
    xspan = max(x_values[-1] - x_values[0], .1 * max(abs(x_values[0]), 1.))
    ax.set_xlim(x_values[0] - .04 * xspan, x_values[-1] + .04 * xspan)
    # Create clear space inside the frame above and below the curves for legends.
    y = np.array([r[k] for r in rows for k in ("plot_KM", "plot_BEM")])
    yspan = max(float(np.ptp(y)), .05 * max(float(np.max(np.abs(y))), 1.))
    legend_rows = int(np.ceil(len(values) / CFG["legend_columns"]))
    ax.set_ylim(float(y.min()) - .20 * yspan,
                float(y.max()) + (.12 + .065 * legend_rows) * yspan)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=5.5, width=.9, pad=6)
    ax.tick_params(which="minor", length=2.8, width=.65)
    ax.grid(False)
    parameter_legend = ax.legend(
        handles=handles, loc="upper right", bbox_to_anchor=CFG["parameter_legend_anchor"],
        ncol=CFG["legend_columns"], frameon=False, fontsize=CFG["legend_size"],
        handlelength=1.75, handletextpad=.55, columnspacing=1.0, labelspacing=.4,
        borderaxespad=.1,
    )
    ax.add_artist(parameter_legend)
    ax.legend(handles=[
        Line2D([], [], color="#303030", lw=CFG["line_width"], label="CV-PIKFNN"),
        Line2D([], [], color="#303030", ls="none", marker="o", mfc="white",
               ms=5.5, mew=1.1, label="BEM (2304 elements)"),
    ], loc="lower left", bbox_to_anchor=CFG["method_legend_anchor"], frameon=False,
        fontsize=CFG["legend_size"], handlelength=2.0, handletextpad=.7,
        labelspacing=.4, borderaxespad=.1)
    for suffix in ("png", "pdf"):
        target = folder / f"{stem}.{suffix}"
        fig.savefig(target, dpi=CFG["dpi"], facecolor="white")
        print(f"Saved: {target.resolve()}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=CFG["csv_path"], help="Saved CSV or its containing folder")
    parser.add_argument("--output", help="Output folder; default: CSV folder / replot")
    parser.add_argument("--raw", action="store_true", help="Plot unnormalized KII")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    path = resolve_csv(args.csv)
    rows = read_results(path, raw=args.raw)
    folder = Path(args.output).expanduser() if args.output else path.parent / "replot"
    folder.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": CFG["font_size"],
        "axes.linewidth": .9, "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        draw_panel(rows, "h_over_a", "b_over_a", "h/a", r"$b/a$",
                   "kii_vs_plate_width", folder, args.raw)
        draw_panel(rows, "b_over_a", "h_over_a", "b/a", r"$h/a$",
                   "kii_vs_plate_height", folder, args.raw)
        if args.no_show:
            plt.close("all")
        else:
            plt.show()


if __name__ == "__main__":
    main()

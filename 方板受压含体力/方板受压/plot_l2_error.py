from pathlib import Path
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter


# ==========================================
# 0. Global style
# ==========================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


METHODS = ["D-PINN", "KM-Net", "MFS", "FSNet", "KM-FSNet"]
FIELD_KEYS = ["ux", "uy", "sxx", "syy", "sxy", "von_mises"]
FIELD_LABELS = [
    r"$u_x$",
    r"$u_y$",
    r"$\sigma_{xx}$",
    r"$\sigma_{yy}$",
    r"$\sigma_{xy}$",
    r"$\sigma_{\mathrm{vm}}$",
]

# If fig5_l2_errors.csv does not exist, the script creates a template.
DEFAULT_CSV = Path(__file__).with_name("fig5_l2_errors.csv")
DEFAULT_OUT_DIR = Path(__file__).with_name("l2_error条形图")

# Optional inline data. Keep np.nan until you want to hard-code results here.
INLINE_ERRORS = {
    "D-PINN": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    "KM-Net": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    "MFS": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    "FSNet": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    "KM-FSNet": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
}

METHOD_STYLES = {
    "D-PINN": {"color": "#5D8EBF", "hatch": None},
    "KM-Net": {"color": "#FDB96A", "hatch": "///"},
    "MFS": {"color": "#82C46C", "hatch": "\\\\"},
    "FSNet": {"color": "#B48AD6", "hatch": ".."},
    "KM-FSNet": {"color": "#E85D5D", "hatch": "xx"},
}


def create_template_csv(csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *FIELD_KEYS])
        for method in METHODS:
            writer.writerow([method, "", "", "", "", "", ""])


def _to_float(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    return float(text)


def load_errors_from_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        create_template_csv(csv_path)
        raise FileNotFoundError(
            f"已创建模板: {csv_path}\n"
            "请填入 5 种方法的六个物理量相对 L2 误差后重新运行。"
        )

    values = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            if method:
                values[method] = [_to_float(row.get(key)) for key in FIELD_KEYS]

    missing = [m for m in METHODS if m not in values]
    if missing:
        raise ValueError(f"CSV 缺少这些方法: {missing}")

    errors = np.array([values[m] for m in METHODS], dtype=float)
    if np.isnan(errors).any():
        nan_where = np.argwhere(np.isnan(errors))
        details = [
            f"{METHODS[i]}:{FIELD_KEYS[j]}" for i, j in nan_where[:12]
        ]
        raise ValueError(
            "CSV 中仍有空值，请补齐后再画图。示例空值位置: "
            + ", ".join(details)
        )
    if np.any(errors <= 0):
        raise ValueError("对数坐标要求所有 L2 误差都必须大于 0。")
    return errors


def load_errors(csv_path, use_inline=False):
    if use_inline:
        errors = np.array([INLINE_ERRORS[m] for m in METHODS], dtype=float)
        if np.isnan(errors).any():
            raise ValueError("INLINE_ERRORS 仍有 np.nan，请先填入真实 L2 误差。")
        if np.any(errors <= 0):
            raise ValueError("对数坐标要求所有 L2 误差都必须大于 0。")
        return errors
    return load_errors_from_csv(csv_path)


def log_formatter(val, pos):
    if val <= 0:
        return ""
    exp = int(np.floor(np.log10(val)))
    coeff = val / (10 ** exp)
    if np.isclose(coeff, 1.0):
        if exp == -1:
            return "0.1"
        return f"1E{exp}"
    return ""


def auto_ylim(errors):
    finite = np.asarray(errors)[np.isfinite(errors)]
    finite = finite[finite > 0]
    ymin = 10 ** np.floor(np.log10(finite.min()) - 0.25)
    ymax = 10 ** np.ceil(np.log10(finite.max()) + 0.25)
    return ymin, ymax


def plot_grouped_bars(
    errors,
    field_indices,
    title,
    save_path,
    figsize=(8.2, 5.2),
    ylim=None,
):
    labels = [FIELD_LABELS[i] for i in field_indices]
    data = errors[:, field_indices]

    x = np.arange(len(field_indices), dtype=float)
    n_methods = len(METHODS)
    width = 0.13 if len(field_indices) >= 5 else 0.16
    gap = 0.015
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * (width + gap)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for i, method in enumerate(METHODS):
        style = METHOD_STYLES[method]
        ax.bar(
            x + offsets[i],
            data[i],
            width,
            label=method,
            color=style["color"],
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )

    ax.set_yscale("log")
    if ylim is None:
        ylim = auto_ylim(data)
    ax.set_ylim(*ylim)

    ax.set_ylabel(r"Relative $L_2$-norm Error", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_title(title, fontsize=14)

    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=6,
        width=1,
        labelsize=12,
        pad=8,
    )
    ax.tick_params(axis="y", which="minor", direction="in", length=3, width=1)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.35)

    ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot Fig. 5 grouped L2 error bars for five methods and six fields."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Use INLINE_ERRORS in this script instead of reading the CSV.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures after saving.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    errors = load_errors(args.csv, use_inline=args.inline)

    common_ylim = auto_ylim(errors)
    plot_grouped_bars(
        errors,
        field_indices=list(range(6)),
        title="Fig. 5. Relative L2 errors of different methods",
        save_path=args.out / "fig5_l2_error_bars_all_fields.png",
        figsize=(9.2, 5.4),
        ylim=common_ylim,
    )
    plot_grouped_bars(
        errors,
        field_indices=[0, 1],
        title="Displacement errors",
        save_path=args.out / "fig5_l2_error_bars_displacement.png",
        figsize=(6.2, 5.2),
        ylim=common_ylim,
    )
    plot_grouped_bars(
        errors,
        field_indices=[2, 3, 4, 5],
        title="Stress errors",
        save_path=args.out / "fig5_l2_error_bars_stress.png",
        figsize=(7.6, 5.2),
        ylim=common_ylim,
    )

    print(f"读取误差表: {args.csv}")
    print(f"图片已保存到: {args.out}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

"""Read saved NPZ data; plot 5 rows x 3 columns without rerunning either solver.

Run from the directory containing the two result folders, or edit CFG paths.
Dependencies: numpy, matplotlib. Can also be pasted into a notebook cell.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter, MaxNLocator
BASE_DIR = Path(__file__).resolve().parent
CFG = dict(
    km_file=BASE_DIR / "literature23_body_force_results/fields_and_coefficients.npz",
    pinn_file=BASE_DIR / "literature23_pinn_results/fields_and_coefficients.npz",
    output_dir=BASE_DIR /"literature23_comparison_figures",
    figsize=(10.4, 15.6),
    title_fontsize=17, row_fontsize=19, label_fontsize=12, tick_fontsize=11,
    colorbar_fontsize=11, cmap="coolwarm", levels=101, save_dpi=400,
    save_outputs=True, show=True,
)

QUANTITIES = ("ux", "uy", "sxx", "syy", "sxy")
ROW_LABELS = (r"$u_x$", r"$u_y$", r"$\sigma_{xx}$",
              r"$\sigma_{yy}$", r"$\sigma_{xy}$")


def read_fields(filename):
    path = Path(filename).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}\nEdit CFG['km_file'] or CFG['pinn_file'].")
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "Y", "prediction", "reference", "quantities"}
        if not required.issubset(data.files):
            raise ValueError(f"{path}: missing keys {sorted(required-set(data.files))}")
        X, Y = data["X"].copy(), data["Y"].copy()
        names = [str(q) for q in data["quantities"]]
        if not set(QUANTITIES).issubset(names):
            raise ValueError(f"{path}: quantities must include {QUANTITIES}")
        columns = [names.index(q) for q in QUANTITIES]
        n = X.size
        fields = []
        for key in ("reference", "prediction"):
            arr = data[key]
            if arr.shape == (n, len(names)):
                arr = arr[:, columns].reshape(*X.shape, len(columns))
            elif arr.shape == (*X.shape, len(names)):
                arr = arr[..., columns]
            else:
                raise ValueError(f"{path}: unsupported {key} shape {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{path}: {key} contains non-finite values")
            fields.append(arr.copy())
    if X.ndim != 2 or X.shape != Y.shape or not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("X and Y must be finite 2D grids with the same shape")
    return X, Y, fields[0], fields[1]


def plot_comparison(cfg=CFG):
    X, Y, exact, km = read_fields(cfg["km_file"])
    XP, YP, exact_p, pinn = read_fields(cfg["pinn_file"])
    if XP.shape != X.shape or not np.allclose(XP, X) or not np.allclose(YP, Y):
        raise ValueError("The two saved grids differ. Export both on the same grid first.")
    if not np.allclose(exact_p, exact, rtol=1e-8, atol=1e-12):
        raise ValueError("The saved reference fields differ: check material, loading, units and scales.")

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 12})
    fig = plt.figure(figsize=cfg["figsize"], dpi=150)
    # Data panels all have equal square boxes. Colorbar axes are explicitly
    # placed after aspect adjustment to match each row's actual panel height.
    grid = fig.add_gridspec(5, 3, left=.12, right=.865, bottom=.055, top=.965,
                           wspace=.14, hspace=.16)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(5)])
    maps = []
    for r in range(5):
        arrays = [a[..., r] for a in (exact, km, pinn)]
        all_min = min(float(a.min()) for a in arrays)
        all_max = max(float(a.max()) for a in arrays)
        # Signed fields: symmetric limits about zero. The exact uy is
        # nonnegative: retain its actual full range including any PINN undershoot.
        if r != 1:
            vmax = max(abs(all_min), abs(all_max), 1e-30)
            vmin = -vmax
        else:
            vmin, vmax = min(0., all_min), max(all_max, 1e-30)
        levels = np.linspace(vmin, vmax, cfg["levels"])
        norm = Normalize(vmin=vmin, vmax=vmax)
        for c, (ax, values) in enumerate(zip(axes[r], arrays)):
            im = ax.contourf(X, Y, values, levels=levels,
                             norm=norm, cmap=cfg["cmap"], extend="neither")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(float(X.min()), float(X.max()))
            ax.set_ylim(float(Y.min()), float(Y.max()))
            ax.set_xticks(np.linspace(float(X.min()), float(X.max()), 3))
            ax.set_yticks(np.linspace(float(Y.min()), float(Y.max()), 3))
            ax.tick_params(direction="in", top=True, right=True,
                           labelsize=cfg["tick_fontsize"],
                           labelbottom=(r == 4), labelleft=(c == 0))
            for spine in ax.spines.values():
                spine.set_color("#666666"); spine.set_linewidth(.8)
            if c == 0:
                ax.set_ylabel(r"$y$ (m)", fontsize=cfg["label_fontsize"])
            if r == 4:
                ax.set_xlabel(r"$x$ (m)", fontsize=cfg["label_fontsize"])
            if r == 0:
                ax.set_title(("Analytical", "CV-PIKFNN", "PINN")[c],
                             fontsize=cfg["title_fontsize"], pad=11)
        maps.append(im)

    fig.canvas.draw()
    for r, im in enumerate(maps):
        box = axes[r, 2].get_position()
        cax = fig.add_axes([box.x1+.018, box.y0, .017, box.height])
        cb = fig.colorbar(im, cax=cax, ticks=np.linspace(im.norm.vmin, im.norm.vmax, 5))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        cb.formatter = formatter; cb.update_ticks()
        cb.ax.tick_params(labelsize=cfg["colorbar_fontsize"], length=3)
        cb.ax.yaxis.get_offset_text().set_fontsize(cfg["colorbar_fontsize"])
        cb.set_label("Displacement (m)" if r < 2 else "Stress (Pa)",
                     fontsize=cfg["label_fontsize"], labelpad=6)
        left = axes[r, 0].get_position()
        fig.text(.025, (left.y0+left.y1)/2, ROW_LABELS[r],
                 rotation=90, va="center", ha="center", fontsize=cfg["row_fontsize"])

    if cfg["save_outputs"]:
        out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            file = out/f"full_field_comparison_5x3.{extension}"
            fig.savefig(file, dpi=cfg["save_dpi"], bbox_inches="tight", facecolor="white")
            print("Saved:", file.resolve())
    if cfg["show"]:
        plt.show()
    return fig, axes


if __name__ == "__main__":
    plot_comparison()

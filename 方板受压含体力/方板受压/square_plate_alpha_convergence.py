# -*- coding: utf-8 -*-
"""KM点力核方板：双精度、正则化变量投影与自动源点比例优化。

独立运行只需 numpy/scipy/matplotlib，不依赖 PyTorch 或参考解。
源点优化不接收 BEM/COMSOL 场标签，参考解只用于最后的误差评价。
算法是核网络线性权重的最小二乘训练 + 一个非线性源点比例的有界优化，
不是 Adam 联合训练。全区间自动括定会进行多次目标函数计算，计入耗时。
"""
from pathlib import Path
import csv
import json
import time
import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize_scalar

L = 1.0
E_REAL = 1.0e6
NU = 0.3
DISP_TOP_REAL = -1.0e-4
SCALE_U = 1.0e4
E = 1.0
G = E / (2.0 * (1.0 + NU))
KAPPA = 3.0 - 4.0 * NU
DISP_TOP = DISP_TOP_REAL * SCALE_U
N_SOURCES_PER_SIDE = 24
N_BOUNDARY_PER_EDGE = 200
N_VALIDATION_PER_EDGE = 257
ALPHA_BOUNDS = (1.02, 1.50)
FIXED_ALPHA_VALUES = (1.05, 1.10, 1.20, 1.30, 1.40)
ADAPTIVE_ALPHA_INITIALS = (1.05, 1.20, 1.40)
# 相对最大奇异值的平滑Tikhonov滤波尺度。
# 固定与自适应组共用；不能利用参考场误差调到某个期望alpha。
REGULARIZATION_REL = 1.0e-8
BRACKET_POINTS = 13
ALPHA_XTOL = 2.0e-5
SHOW_FIGURES = True


def source_points(alpha, n_per_side=N_SOURCES_PER_SIDE):
    if not (np.isfinite(alpha) and alpha > 1):
        raise ValueError("alpha必须为大于1的有限数。")
    half = 0.5 * L * alpha
    lo, hi = 0.5 * L - half, 0.5 * L + half
    t = np.linspace(lo, hi, n_per_side + 2, dtype=np.float64)[1:-1]
    return np.vstack((np.column_stack((t, np.full_like(t, lo))),
                      np.column_stack((np.full_like(t, hi), t)),
                      np.column_stack((t[::-1], np.full_like(t, hi))),
                      np.column_stack((np.full_like(t, lo), t[::-1]))))


def boundary_points(n, midpoint=False):
    # validation采用单元中点网格；与原等分内部训练点采用不同采样规则。
    t = L * ((np.arange(n) + 0.5) / n if midpoint
             else np.arange(1, n + 1) / (n + 1))
    return {"bottom": np.column_stack((t, np.zeros_like(t))),
            "top": np.column_stack((t, np.full_like(t, L))),
            "left": np.column_stack((np.zeros_like(t), t)),
            "right": np.column_stack((np.full_like(t, L), t))}


def potential_cache(points, sources):
    points = np.asarray(points, dtype=np.float64)
    z = points[:, 0] + 1j * points[:, 1]
    s = sources[:, 0] + 1j * sources[:, 1]
    dz = z[:, None] - s[None, :]
    angle = np.angle(0.5 * L * (1 + 1j) - s)
    rotated = dz * np.exp(-1j * angle[None, :])
    scale = np.sqrt(len(s))
    return {"z": z,
            "basis": (np.log(np.abs(rotated)) + 1j * (
                np.angle(rotated) + angle[None, :])) / scale,
            "d1": 1.0 / dz / scale, "d2": -1.0 / dz**2 / scale,
            "sbar": np.conj(s), "n": len(s)}


def fields_from_coefficients(coeff, cache):
    """接受一个系数向量或多个列向量，均使用原KM点力核公式。"""
    coeff = np.asarray(coeff, dtype=np.float64)
    n = cache["n"]
    z = cache["z"][:, None] if coeff.ndim == 2 else cache["z"]
    p = coeff[:n] + 1j * coeff[n:2*n]
    phi0 = coeff[2*n] + 1j * coeff[2*n+1]
    phi1 = coeff[2*n+2] + 1j * coeff[2*n+3]
    psi0 = coeff[2*n+4] + 1j * coeff[2*n+5]
    psi1 = coeff[2*n+6] + 1j * coeff[2*n+7]
    b, d1, d2 = cache["basis"], cache["d1"], cache["d2"]
    cf = 1.0 / (2.0 * np.pi * (KAPPA + 1.0))
    phi = -cf * (b @ p) + phi0 + phi1 * z
    dphi = -cf * (d1 @ p) + phi1
    d2phi = -cf * (d2 @ p)
    psi = cf * (KAPPA * (b @ np.conj(p)) +
                (d1 * cache["sbar"][None, :]) @ p) + psi0 + psi1*z
    dpsi = cf * (KAPPA * (d1 @ np.conj(p)) +
                 (d2 * cache["sbar"][None, :]) @ p) + psi1
    u = (KAPPA*phi - z*np.conj(dphi) - np.conj(psi))/(2*G)
    total = 4*dphi.real
    difference = 2*(np.conj(z)*d2phi + dpsi)
    return u.real, u.imag, 0.5*(total-difference.real), \
        0.5*(total+difference.real), 0.5*difference.imag


def boundary_matrix(alpha, n_points, n_sources=N_SOURCES_PER_SIDE, midpoint=False):
    sources = source_points(alpha, n_sources)
    n = len(sources)
    # 原表示中的phi0/psi0包含2个等价常数自由度。
    # 固定psi0=0消除规范冗余，并不改变可表示的位移和应力场。
    active = np.delete(np.arange(2*n+8), [2*n+4, 2*n+5])
    identity = np.eye(2*n+8, dtype=np.float64)[:, active]
    sides = boundary_points(n_points, midpoint)
    f = {k: fields_from_coefficients(identity, potential_cache(v, sources))
         for k, v in sides.items()}
    matrix = np.vstack((f["bottom"][0], f["bottom"][1],
                        f["top"][0], f["top"][1],
                        f["left"][2], f["left"][4],
                        f["right"][2], f["right"][4]))
    rhs = np.zeros(8*n_points, dtype=np.float64)
    rhs[3*n_points:4*n_points] = DISP_TOP
    return matrix, rhs, active


def fit_at_alpha(alpha, regularization_rel=REGULARIZATION_REL,
                 n_train=N_BOUNDARY_PER_EDGE, n_validation=N_VALIDATION_PER_EDGE,
                 n_sources=N_SOURCES_PER_SIDE):
    """每个alpha重新拟合线性系数；不使用任何参考解。"""
    a, b, active = boundary_matrix(alpha, n_train, n_sources)
    u, s, vt = svd(a, full_matrices=False, lapack_driver="gesdd")
    tau = regularization_rel * s[0]
    if regularization_rel > 0:
        weights = s / (s*s + tau*tau)
    else:
        cutoff = np.finfo(float).eps * max(a.shape) * s[0]
        weights = np.divide(1.0, s, out=np.zeros_like(s), where=s > cutoff)
    reduced = vt.T @ (weights * (u.T @ b))
    coeff = np.zeros(2*4*n_sources+8)
    coeff[active] = reduced
    av, bv, _ = boundary_matrix(alpha, n_validation, n_sources, midpoint=True)
    train_loss = float(np.sum((a @ reduced-b)**2)/n_train)
    validation_loss = float(np.sum((av @ reduced-bv)**2)/n_validation)
    kept = s > max(tau, np.finfo(float).eps*max(a.shape)*s[0])
    return {"alpha": float(alpha), "coefficients": coeff,
            "n_sources_per_side": int(n_sources),
            "train_loss": train_loss, "validation_loss": validation_loss,
            "coefficient_norm": float(np.linalg.norm(reduced)),
            "effective_rank": int(np.count_nonzero(kept)),
            "effective_condition": float(s[0]/s[kept][-1]),
            "regularization_rel": float(regularization_rel)}


def optimize_source_scale(alpha_initial, bounds=ALPHA_BOUNDS,
                          bracket_points=BRACKET_POINTS,
                          regularization_rel=REGULARIZATION_REL,
                          n_sources=N_SOURCES_PER_SIDE,
                          n_train=N_BOUNDARY_PER_EDGE,
                          n_validation=N_VALIDATION_PER_EDGE):
    """全区间自动括定 + 每个可见低谷内连续优化，返回真正计算到的最优。

    全区间探索为降低初值依赖提供保障，不宣称数学上的全局最优。
    不同运行不共享缓存；每行耗时都包含完整搜索成本。
    """
    lo, hi = map(float, bounds)
    if not (1 < lo < hi and lo <= alpha_initial <= hi):
        raise ValueError("需要1<lower<upper且初值位于允许范围内。")
    cache, history = {}, []
    start = time.perf_counter()

    def evaluate(alpha):
        alpha = float(alpha)
        if alpha not in cache:
            fit = fit_at_alpha(alpha, regularization_rel, n_train,
                               n_validation, n_sources)
            cache[alpha] = fit
            history.append({k: fit[k] for k in ["alpha", "train_loss",
                "validation_loss", "coefficient_norm", "effective_rank"]})
        return cache[alpha]["validation_loss"]

    evaluate(alpha_initial)
    grid = np.linspace(lo, hi, bracket_points)
    values = np.array([evaluate(a) for a in grid])
    intervals = []
    for i in range(1, len(grid)-1):
        if values[i] <= values[i-1] and values[i] <= values[i+1]:
            intervals.append((grid[i-1], grid[i+1]))
    if values[0] <= values[1]:
        intervals.append((grid[0], grid[1]))
    if values[-1] <= values[-2]:
        intervals.append((grid[-2], grid[-1]))
    for interval in intervals:
        result = minimize_scalar(lambda a: np.log(max(evaluate(a), 1e-300)),
            bounds=interval, method="bounded",
            options={"xatol": ALPHA_XTOL, "maxiter": 80})
        evaluate(result.x)
    best = min(cache.values(), key=lambda f: f["validation_loss"])
    best = dict(best)
    best.update(alpha_initial=float(alpha_initial),
                search_time_s=time.perf_counter()-start,
                objective_evaluations=len(cache), history=history,
                selection_objective="boundary_validation_loss",
                algorithm="regularized_variable_projection_auto_bracketing")
    return best


def predict_fields(result, points, batch_size=2000):
    sources = source_points(result["alpha"], result.get("n_sources_per_side", N_SOURCES_PER_SIDE))
    chunks = [[] for _ in range(5)]
    for i in range(0, len(points), batch_size):
        f = fields_from_coefficients(result["coefficients"],
                                    potential_cache(points[i:i+batch_size], sources))
        for bucket, values in zip(chunks, f):
            bucket.append(values)
    ux, uy, sxx, syy, sxy = [np.concatenate(v) for v in chunks]
    ux, uy = ux/SCALE_U, uy/SCALE_U
    scale = E_REAL/SCALE_U
    sxx, syy, sxy = sxx*scale, syy*scale, sxy*scale
    szz = NU*(sxx+syy)
    vm = np.sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2)+3*sxy**2)
    return dict(ux=ux, uy=uy, sxx=sxx, syy=syy, sxy=sxy, von_mises=vm)


def relative_l2(pred, reference):
    return float(np.linalg.norm(pred-reference)/max(np.linalg.norm(reference), 1e-30))


def evaluate_reference(result, points, reference):
    """必须在源点优化完成后调用；这些误差不参与alpha选择。"""
    predicted = predict_fields(result, points)
    return {"L2_"+key: relative_l2(predicted[key], np.asarray(value))
            for key, value in reference.items() if key in predicted}


def load_comsol_reference(path):
    """原COMSOL导出8列：x,y,ux,uy,sxx,syy,sxy,mises。"""
    a = np.loadtxt(path, comments="%")
    if a.ndim != 2 or a.shape[1] < 8:
        raise ValueError("COMSOL文件至少需要8列。")
    a = a[np.all(np.isfinite(a[:, :8]), axis=1)]
    if not len(a):
        raise ValueError("COMSOL文件没有有效数据行。")
    return a[:, :2], {k: a[:, i+2] for i, k in enumerate(
        ["ux", "uy", "sxx", "syy", "sxy", "von_mises"])}


def run_experiment(reference_points=None, reference=None, output_dir=None):
    """五组固定比例+三个初值自动优化，保存论文表1及完整多初值结果。"""
    if output_dir is None:
        base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        output_dir = base/"adaptive_source_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = None
    rows, fixed, adaptive = [], [], []
    for alpha in FIXED_ALPHA_VALUES:
        start = time.perf_counter()
        f = fit_at_alpha(alpha)
        f.update(alpha_initial=alpha, search_time_s=time.perf_counter()-start,
                 objective_evaluations=1)
        fixed.append(f)
    for alpha in ADAPTIVE_ALPHA_INITIALS:
        f = optimize_source_scale(alpha)
        adaptive.append(f)
        print(f"init={alpha:.4f} -> alpha={f['alpha']:.6f}, "
              f"validation={f['validation_loss']:.3e}, "
              f"evaluations={f['objective_evaluations']}, "
              f"time={f['search_time_s']:.2f}s", flush=True)
    for mode, results in [("fixed", fixed), ("adaptive", adaptive)]:
        for f in results:
            row = dict(mode=mode, alpha_initial=f["alpha_initial"],
                       alpha_final=f["alpha"], train_loss=f["train_loss"],
                       validation_loss=f["validation_loss"],
                       coefficient_norm=f["coefficient_norm"],
                       effective_rank=f["effective_rank"],
                       effective_condition=f["effective_condition"],
                       regularization_rel=f["regularization_rel"],
                       objective_evaluations=f["objective_evaluations"],
                       time_s=f["search_time_s"])
            if reference_points is not None and reference is not None:
                row.update(evaluate_reference(f, reference_points, reference))
            rows.append(row)
    keys = list(rows[0])
    with (output_dir/"all_initializations.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    # 论文最后一列预先指定采用首个初值，避免参考误差挑选最佳。
    selected = adaptive[0]
    table_rows = rows[:len(fixed)] + [rows[len(fixed)]]
    labels = [f"Fixed alpha={f['alpha']:.2f}" for f in fixed] + [
        f"Adaptive (init={ADAPTIVE_ALPHA_INITIALS[0]:.2f})"]
    metrics = ["alpha_initial", "alpha_final", "L2_ux", "L2_uy", "L2_von_mises", "time_s"]
    with (output_dir/"table1.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file); writer.writerow(["Quantity"] + labels)
        for key in metrics:
            writer.writerow([key]+[r.get(key, "Reference not supplied") for r in table_rows])
    serial = [{k: v for k, v in f.items() if k != "coefficients"} for f in adaptive]
    (output_dir/"adaptive_history.json").write_text(
        json.dumps(serial, indent=2), encoding="utf-8")
    np.savez(output_dir/"selected_model.npz", alpha=selected["alpha"],
             coefficients=selected["coefficients"], regularization_rel=REGULARIZATION_REL,
             n_sources_per_side=selected["n_sources_per_side"])
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "stix"}):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for f, color in zip(adaptive, ["#326A93", "#BE4540", "#8C8C8C"]):
            h = sorted(f["history"], key=lambda x: x["alpha"])
            ax.semilogy([v["alpha"] for v in h], [v["validation_loss"] for v in h],
                        ".-", lw=1, color=color, label=f"init={f['alpha_initial']:.2f}")
            ax.scatter([f["alpha"]], [f["validation_loss"]], marker="*", s=90, color=color)
        ax.set_xlabel(r"Source-boundary scaling factor $\alpha$")
        ax.set_ylabel("Boundary validation loss")
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(output_dir/"adaptive_objective.png", dpi=250)
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)
    if limiter is not None:
        limiter.restore_original_limits()
    print("已保存结果：", output_dir.resolve())
    if reference is None:
        print("未提供参考解，因此未生成L2误差；不会填入虚构误差。")
    return rows, selected, adaptive



# ---------------- 收敛历程图的参数 ----------------
CONVERGENCE_INITIALS = (1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.48)
CONVERGENCE_COLORS = ("#565656", "#FF3B3B", "#176CFF", "#2CB570", "#B274DB", "#D59A00", "#00BFCB")
CONVERGENCE_MARKERS = ("s", "o", "^", "v", "D", "<", ">")
CONVERGENCE_LINESTYLES = ("-", "--", "-.", ":", "-", "--", "-.")
# ------------------------------------------------


def incumbent_trace(history):
    """按真实评价顺序，记录截至当前评价的最佳比例，不排序、不平滑、不补点。"""
    best_loss = np.inf
    best_alpha = None
    rows = []
    for i, h in enumerate(history, start=1):
        if h["validation_loss"] < best_loss:
            best_loss = h["validation_loss"]
            best_alpha = h["alpha"]
        rows.append({"evaluation": i, "candidate_alpha": h["alpha"],
                     "candidate_loss": h["validation_loss"],
                     "best_alpha": best_alpha, "best_loss": best_loss})
    return rows


def draw_convergence(runs, output_dir, show=True):
    """图中实线/标记连接的是历史最佳alpha，候选alpha另存CSV供核查。"""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "serif",
             "font.serif": ["Times New Roman", "DejaVu Serif"],
             "mathtext.fontset": "stix", "font.size": 14,
             "pdf.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.2, 5.5))
        fig.subplots_adjust(left=0.15, right=0.97, bottom=0.17, top=0.96)
        final_alpha = runs[0]["alpha"]
        final_alphas = np.array([r["alpha"] for r in runs])
        same_final = np.ptp(final_alphas) < 1e-4
        for j, (r, color, marker, ls) in enumerate(zip(
                runs, CONVERGENCE_COLORS, CONVERGENCE_MARKERS, CONVERGENCE_LINESTYLES)):
            trace = incumbent_trace(r["history"])
            x = np.array([t["evaluation"] for t in trace])
            y = np.array([t["best_alpha"] for t in trace])
            # 各曲线不同位置放置标记，重合轨迹仍可识别；数据本身不位移。
            positions = sorted(set([0, len(x)-1] + list(range(j+1, len(x), 7))))
            ax.plot(x, y, color=color, marker=marker, ls=ls, lw=1.15,
                    markersize=5.0, markeredgewidth=0.7,
                    markerfacecolor=color, markevery=positions,
                    label=rf"$\alpha_0={r['alpha_initial']:.2f}$", zorder=3+j*.1)
        if same_final:
            ax.axhline(final_alpha, color="#555555", lw=0.9,
                       ls=(0, (5, 4)), zorder=1)
            ax.text(0.96, 0.29, rf"$\alpha^*={final_alpha:.6f}$",
                    transform=ax.transAxes, ha="right", va="center", fontsize=16,
                    bbox=dict(facecolor="white", edgecolor="none", pad=2))
        ax.set_xlabel("Objective evaluations", fontsize=20, labelpad=10)
        ax.set_ylabel(r"Best-so-far $\alpha$", fontsize=20, labelpad=9)
        ax.set_xlim(0, max(len(r["history"]) for r in runs)+1)
        ax.set_ylim(1.025, 1.55)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", direction="in", top=True, right=True,
                       length=6, width=1, labelsize=15, pad=7)
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       length=3, width=.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
        ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=13,
                  handlelength=1.8, handletextpad=.3, columnspacing=.65,
                  borderaxespad=.45, labelspacing=.4)
        # 对真实末段进行放大；不是人为生成平滑轨迹。
        inset = ax.inset_axes([.48, .39, .47, .26])
        for j, (r, color, marker, ls) in enumerate(zip(
                runs, CONVERGENCE_COLORS, CONVERGENCE_MARKERS, CONVERGENCE_LINESTYLES)):
            trace = incumbent_trace(r["history"])
            x = np.array([t["evaluation"] for t in trace])
            y = np.array([t["best_alpha"] for t in trace])
            inset.plot(x, y, color=color, ls=ls, lw=1.05,
                       marker=marker, markersize=3.7,
                       markevery=list(range(j+1, len(x), 7)))
        inset.set_xlim(14, max(len(r["history"]) for r in runs)+.5)
        if same_final:
            inset.set_ylim(final_alpha-.003, final_alpha+.003)
        inset.tick_params(direction="in", top=True, right=True, labelsize=9,
                          length=3, pad=3)
        inset.set_title("Final-stage detail", fontsize=11, pad=4)
        paths=[]
        for suffix in ["png", "pdf", "svg"]:
            p=output_dir/f"alpha_convergence.{suffix}"
            fig.savefig(p, dpi=400, bbox_inches="tight", pad_inches=.08, facecolor="white")
            paths.append(p)
        if show:
            plt.show()
        plt.close(fig)
    return paths


def run_convergence_experiment(output_dir=None, show=True):
    """7个初值完整独立运行，没有参考误差参与alpha选择。"""
    if output_dir is None:
        base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        output_dir = base / "alpha_convergence_results"
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    try:
        from threadpoolctl import threadpool_limits
        limiter=threadpool_limits(limits=1)
    except ImportError:
        limiter=None
    runs=[]
    for alpha0 in CONVERGENCE_INITIALS:
        r=optimize_source_scale(alpha0)
        runs.append(r)
        print(f"alpha0={alpha0:.2f} -> alpha={r['alpha']:.9f}; "
              f"evaluations={r['objective_evaluations']}; time={r['search_time_s']:.2f}s",flush=True)
    if limiter is not None:
        limiter.restore_original_limits()
    flat=[]
    for r in runs:
        for row in incumbent_trace(r["history"]):
            flat.append({"alpha_initial":r["alpha_initial"],**row})
    with (output_dir/"alpha_convergence_history.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    summary=[{k:r[k] for k in ["alpha_initial","alpha","validation_loss",
                               "objective_evaluations","search_time_s"]} for r in runs]
    with (output_dir/"alpha_convergence_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    serial=[{k:v for k,v in r.items() if k!="coefficients"} for r in runs]
    (output_dir/"alpha_convergence_raw.json").write_text(json.dumps(serial,indent=2),encoding="utf-8")
    draw_convergence(runs,output_dir,show)
    print("已保存：",output_dir.resolve())
    return runs


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="真实的源点比例选择历程，不是Adam epoch曲线")
    parser.add_argument("--no-show",action="store_true")
    parser.add_argument("--output-dir",default=None)
    parser.add_argument("--history",help="可选：读取已保存raw.json，只重绘，不重新优化")
    args=parser.parse_args()
    if args.history:
        runs=json.loads(Path(args.history).read_text(encoding="utf-8"))
        out=Path(args.output_dir) if args.output_dir else Path(args.history).resolve().parent
        draw_convergence(runs,out,not args.no_show)
    else:
        run_convergence_experiment(args.output_dir,not args.no_show)

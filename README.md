# CV-PIKFNN：复变物理信息核函数神经网络求解弹性力学和断裂力学 —— 论文配套代码

**中文** · [English](#english)

本仓库是论文配套的可复现代码集，收录 **CV-PIKFNN** 求解器及其全部算例（孔洞、椭圆开孔、I/II/混合型裂纹、边缘裂纹、高应力梯度问题）与基线方法（PINN、CV-PINN、BEM、FEM）的 notebook、独立脚本、结果图表和误差表。

---

## 目录

- [环境准备](#环境准备)
- [仓库结构](#仓库结构)
- [未随仓库上传的数据文件（重要）](#未随仓库上传的数据文件重要)
- [各算例入口与运行方式](#各算例入口与运行方式)
- [方法缩写对照](#方法缩写对照)
- [论文图表 ↔ 文件对应关系](#论文图表--文件对应关系)
- [已知问题与阅读提示](#已知问题与阅读提示)
- [引用](#引用)
- [English](#english)

---

## 环境准备

```bash
conda create -n pikfnn python=3.10 -y
conda activate pikfnn
pip install -r requirements.txt
```

- 多数 notebook 记录的解释器为 **Python 3.10.20**（少数 3.9 / 3.12）。
- PyTorch 全程使用 **float64**，**CPU 即可跑通全部算例**；有 GPU 会自动利用。
- 若 `numba` 与 `torch` 同时加载报 OpenMP 重复初始化错误，设置环境变量
  `KMP_DUPLICATE_LIB_OK=TRUE`（部分脚本内部已这样处理）。
- 独立 `.py` 脚本大多使用 `Path(__file__).parent` 相对定位，clone 后可直接运行；
  **notebook 中存在硬编码的绝对路径**，运行前需修改，见[已知问题](#已知问题与阅读提示)。

## 仓库结构

| 目录 | 研究内容 | 参考解 |
|---|---|---|
| `圆孔受压/` | 无限大平面内受均匀内压圆孔（a=1 m, P₀=1 MPa, E=1 GPa, ν=0.3，外边界 L=80a 施加解析位移模拟无限远） | 自写多核 BEM |
| `椭圆开孔受拉/` | 有限域板含中心椭圆孔、单向拉伸 | 多核 BEM |
| `方板受压含体力/` | 两块内容：**`方板受压/`** 子目录＝顶部位移压缩方板（无体力，COMSOL 参考）；**顶层**＝文献[23]基准的单位方板复体力问题 | COMSOL 6.3 / 解析解 |
| `不规则圆孔/` | 含光滑非规则孔（r(θ) 带 3、5 阶谐波 ε₃=0.22, ε₅=0.08）受拉方板，**不使用保角映射**，直接在物理平面定义 | COMSOL 6.3（M1/M2/M3 三套网格） |
| `I型裂纹/` | 中心穿透裂纹 I 型（a=1, b=2, h=6），Williams n=1,3,5 富集，扫 b/a=1.1–3.0 求 K_I | Tada 有限宽公式 / MATLAB 版 Enriched PINN |
| `II型裂纹/` | II 型（面内反平面剪）中心裂纹四分之一模型；另含 b/a×h/a 几何扫描 | 独立多核 BEM（不参与训练） |
| `混合型裂纹/` | I–II 混合型中心**斜**裂纹，倾角 15°/30°/45°/60°（H=4, 2H=8, a=1, σ₀=1，平面应力 ν=0.3），利用中心反演对称只解上半域 | Williams/Tada 组合参考 (K_I,K_II) |
| `边缘裂纹/` | 左边界单边缘裂纹 I 型，上半域 0≤x≤b, 0≤y≤h/2，远场 σ₂₂=σ₀ | notebook 内诊断输出 |
| `高应力梯度问题/` | 方板（L=2）中心椭圆孔，长短轴比 **λ 从 1 压到 0.005** 制造高应力梯度；含 KM-PINN / CV-PINN 基线 | COMSOL 6.3 + BEM |
| `KM_PIKFNN_Corrected_Results/`<br>`KM_PIKFNN_Refined_Results/`<br>`KM_PIKFNN_Six_Cases_Results/` | 高应力梯度 λ 扫描的**三次输出快照**（Corrected＝最小二乘直解；Refined/Six_Cases＝SVD 热启动 + Adam/L-BFGS） | 同上 |

## 未随仓库上传的数据文件（重要）

GitHub 对**单个文件设 100 MB 硬上限**，因此以下大体积数据不入库，已在 `.gitignore` 中显式排除（共 106 个文件 / 476 MB）。仓库本体 818 个文件、232 MB，**未使用 Git LFS**，`git clone` 即得全部代码与结果图。

| 被排除的文件 | 体积 | 影响 | 如何补齐 |
|---|---|---|---|
| `不规则圆孔/irregular_hole_M1/M2/M3_grid_801.txt` | 3 × 118.6 MB | **不规则圆孔算例无法运行**（其余算例不受影响） | 用 COMSOL 6.3 对 `irregular_hole_M1/M2/M3.mph` 导出 801×801 规则网格，文件名保持原样放回 `不规则圆孔/`。程序读取优先级为 **M3 → M2 → M1**；M1 只是初步网格，论文结果用 M3 |
| `**/*.npz` | 116 MB / 102 个 | 仅缺少**原始数值场**；由这些 npz 生成的 csv/图表已全部保留 | 重跑对应 notebook 或脚本即可重新生成，输出目录名在代码中固定 |
| `高应力梯度问题/ellipse_lambda_0p005.vtu` | 4.5 MB | λ=0.005 的 VTK 网格导出缺失；该算例的 COMSOL 对照数据 `hole stress.txt`、`tip_stress.txt` **仍在仓库内** | 需要时从 COMSOL 重新导出 |

另外，`II型裂纹/modeii_width_height_results/bem_cache/*.json`（40 个哈希命名的 BEM 缓存）**已上传**，可让几何扫描跳过重算。

## 各算例入口与运行方式

推荐顺序：先看结果图，再跑 notebook 复现。

| 算例 | 主入口 | 基线 / 对照 | 输出位置 |
|---|---|---|---|
| 受压圆孔 | `圆孔受压-自适应源点KM-PIKFNN.ipynb`、`pressurized_hole_KM_PIKFNN_adaptive.ipynb` | `圆孔受压-BEM.ipynb`（参考解）、`-PINN.ipynb`、`-PIKFNN.ipynb`、`KMINN.ipynb`（CV-PINN+Laurent 富集） | `hole_adaptive_results/`、`figures/` |
| 椭圆开孔受拉 | `椭圆开孔受拉-KM-PIKFNN.ipynb`（另两个 `-BEM-analytic-stress` / `-BEM-lstsq-stress` 差别在应力由复势解析式还是最小二乘求出） | `椭圆开孔受拉.ipynb`（BEM+CV-PINN） | 结果保存在 notebook 内 |
| 方板受压（无体力） | `方板受压/KM-PIKFNN_square_adaptive_varpro.ipynb` 或独立脚本 `方板受压/square_plate_alpha_convergence.py`（**算法相同，后者可直接命令行运行**） | `方板受压comsol参考解.ipynb`、`PINN.ipynb`、`MFS.ipynb`、`PIKFNN.ipynb`(FSNet)、`KM-PIKFNN-point-force*.ipynb` | `alpha_convergence_results/`、`*_prediction_clouds/`、`物理量误差云图/`、`l2_error条形图/` |
| 方板复体力（文献[23]基准） | `KM_PIKFNN_body_force_cold_conditioned (2).ipynb`、`KM_PIKFNN_body_force_joint_LM_4starts (2).ipynb`、`KM_complex_body_force_boundary_only.ipynb` | `PINN_literature23_body_force.ipynb`、`PINN_bodyforce构造ex1.ipynb` | `literature23_*_results/`（各含 `method_comparison_table.md/.csv/.pdf`） |
| 不规则圆孔 | `不规则圆孔/irregular_hole_adaptive_km_pikfnn.ipynb`（或同名的 `.py` 导出版，已改为同目录自动发现网格文件） | COMSOL M1/M2/M3 网格 | `irregular_hole_km_outputs/` |
| I 型裂纹 | `KM_PIKFNN_Williams_ba_sweep_adaptive.ipynb`、`KM_PIKFNN_Williams_ba_sweep.ipynb`、`KM_PIKFNN_ModeI_cold_start_source_study.ipynb` | `Enriched_PINN.py`（MATLAB 版富集 PINN 移植，含 XPINN 分区） | `ba_sif_results/`、`ba_sif_adaptive_results/`、`mode_i_cold_start_source_study/` |
| II 型裂纹 | `KM-PIKFNN_ModeII_adaptive_BEM_reference.ipynb`、`modeii_width_height_bem_sweep.py`（几何扫描，含 SVD 热启动 + 系数级 L-BFGS，**末位物理系数即 K_II**） | 同文件内的独立 BEM；`二型(1).ipynb` | `modeii_width_height_results/`（含 `kii_geometry_comparison.md`：KM 与 BEM 相对差 ≤0.54%） |
| 混合型裂纹 | `KM_PIKFNN_mixed_mode_adaptive_cold_start.ipynb`、`KM_PIKFNN_mixed_mode_slant_crack.ipynb` | `Mixed.ipynb`、`Mixed_mode_source_study_case41_style.ipynb` | `mixed_mode_adaptive_results/`、`mixed_mode_adaptive_cold_results/` |
| 边缘裂纹 | `KM-PIKFNN_edgecrackf.ipynb` | notebook 内核矩阵可辨识性诊断 | notebook 内 |
| 高应力梯度 | `高应力梯度问题_KM-PIKFNN.ipynb`（最小二乘直解）、`高应力梯度问题_KM-PIKFNN_Adam_LBFGS.ipynb`、根目录 `high_gradient_KM_PIKFNN_comsol_comparison (5).py` | `KM-PINN.py` / `KM-PINN/`（Joukowski 保角映射复变量 PINN + λ 连续延拓）、`CV_PINN_Results_Improved/` | 三个 `KM_PIKFNN_*_Results/` 目录 |
| 源点/几何示意图（无需训练） | `*/source_schematic*.py`、`pressurized_hole_source_schematic.py`、`square_plate_source_schematic.py`、`mixed_mode_crack_source_schematic.py`、`geometry_mapping_standalone.py` | — | `*_source_figures/`、`source_schematic_figures/` |
| 纯重绘脚本（读取已有 csv，不训练） | `replot_pressurized_hole_adaptive.py`、`replot_modeI_adaptive.py`、`replot_modeII_adaptive.py`、`replot_modeii_geometry.py`、`replot_mixed_mode_adaptive.py`、`plot_l2_error.py`、`plot_km_pikfnn_4x6.py`、`plot_literature23_full_field_comparison.py` | — | 对应结果目录 |

## 方法缩写对照

以下释义均来自代码与 notebook 标题原文；仓库中**没有出现 PIKFNN 的逐字母英文全称**。

| 缩写 | 含义（依据） |
|---|---|
| **KM-PIKFNN** | 本文主角。K-M（Kolosov–Muskhelishvili）**复势点力基本解**为核的物理信息基本解网络；结果文件前缀普遍为 `km_pikfnn_*` |
| **PIKFNN** | 同框架的**实值 Kelvin 核**版本（`圆孔受压-PIKFNN.ipynb` 标题写作"实值 Kelvin 核 PIKFNN"；`方板受压/PIKFNN.ipynb` 标题写作 "FSNet: Fundamental Solution Network"） |
| **PINN / D-PINN** | 传统物理信息神经网络（`方板受压/PINN.ipynb` 标 "Vanilla PINN"，其输出在对比表中记作 D-PINN） |
| **CV-PINN** | 复变量 PINN（含 Laurent 负幂次项富集） |
| **KM-PINN** | Joukowski 保角映射复变量 PINN + λ 连续延拓（`高应力梯度问题/KM-PINN.py`，类名 `JoukowskiCVNN`） |
| **XPINN** | 分区（上/下半域）域分解 PINN，`I型裂纹/XPINN_checkpoint.pt` 为其权重 |
| **Enriched PINN** | 裂尖渐近富集 PINN + COD 提取 SIF（`I型裂纹/Enriched_PINN.py`，MATLAB 项目移植） |
| **MFS** | 基本解法（Method of Fundamental Solutions），`方板受压/MFS.ipynb` |
| **FSNet / KM-Net / KM-FSNet** | 单层基本解网络族，在 `fig5_l2_errors.csv` 与云图目录中与 KM-PIKFNN 并列对比 |
| **BEM** | 自写多核边界元（numba 加速，`njit/prange`），在裂纹算例中作**独立参考解**：明确声明不参与训练损失、初始化或最优状态选择 |
| **COMSOL** | COMSOL 6.3.0.290 有限元参考解，以 txt/csv 导出后被脚本读取 |

## 论文图表 ↔ 文件对应关系

| 论文图 / 表 | 文件 |
|---|---|
| 全场 3×3 对比（受压圆孔） | `圆孔受压/figures/full_field_comparison_3x3.{png,pdf,tiff}`（由 `plot_3x3_full_field_comparison.ipynb` 生成） |
| 各场相对 L2 误差（对 BEM） | `圆孔受压/figures/relative_l2_errors_vs_bem.{csv,pdf,png}` |
| 图 4：KM-FSNet vs COMSOL 误差云图 | `方板受压含体力/方板受压/kmfsnet_comsol_error_clouds/fig4_kmfsnet_comsol_error_*_absolute.png` |
| 图 5：五方法 × 六物理量 L2 柱状图 | `方板受压含体力/方板受压/l2_error条形图/fig5_l2_error_bars_*.png`，数据源 `fig5_l2_errors.csv`，绘图脚本 `plot_l2_error.py` |
| α 收敛性 | `方板受压含体力/方板受压/alpha_convergence_results/alpha_convergence.{pdf,png,svg}` |
| 6 个 λ 的 4×6 应力场 | `KM_PIKFNN_Six_Cases_Results/stress_fields_four_by_six.{png,pdf}`（`plot_km_pikfnn_4x6.py` 生成） |
| 裂尖衰减 vs COMSOL | `KM_PIKFNN_Refined_Results/six_tip_decay_comsol_comparison.*`、`six_tip_decay_curves.*` |
| K_I 随 b/a | `I型裂纹/ba_sif(_adaptive_)results/km_pikfnn_sif_vs_ba.{csv,png,svg}`、`sif_vs_ba_reference3.*` |
| K_II 几何扫描 | `II型裂纹/modeii_width_height_results/kii_*.{csv,md,pdf,png}` + `replot/` |
| 混合模式收敛（30°） | `混合型裂纹/mixed_mode_adaptive(_cold)_results/adaptive_source_convergence_30deg.{png,pdf}` |
| 文献[23] 方法对比表 | `方板受压含体力/literature23_*_results/method_comparison_table.{md,csv,pdf,png}`（Fixed vs Adaptive KM-PIKFNN，α₀=1.10–1.50，与 PINN 对照） |

## 已知问题与阅读提示

诚实列出，便于读者复现时避坑（均为仓库现状，不影响主结果）：

1. **notebook 里的绝对路径**：多个 notebook 保存了开发机路径 `C:\Users\magic\Desktop\KM-PIKFNN\...`，他人 clone 后直接运行会 `FileNotFoundError`，需改为本地路径。已知涉及：
   - `混合型裂纹/replot_mixed_mode_adaptive.py`（`DATA_DIR`）
   - `高应力梯度问题/plot_km_pikfnn_4x6.py`
   - `方板受压含体力/方板受压/KM-PIKFNN-point-force.ipynb`、`PINN.ipynb`（`COMSOL_REF_PATH`）
   - `方板受压含体力/方板受压/MFS.ipynb`、`PIKFNN.ipynb`、`方板受压comsol参考解.ipynb`（指向另一并行项目 `Desktop\KM-MFSNet\...`）
   - `不规则圆孔/irregular_hole_adaptive_km_pikfnn.ipynb`（M1/M2/M3 路径写死；同目录 `.py` 导出版已改为自动发现）
   另外，部分 notebook 里"方板受压"写作顶层目录，而本仓库实际路径是 `方板受压含体力/方板受压`，层级不一致。
2. **重复文件**：根目录 `high_gradient_KM_PIKFNN_comsol_comparison (5).py` 与 `高应力梯度问题/high_gradient_KM_PIKFNN_comsol_comparison (5) .py`（文件名多一个空格）**内容字节完全一致**（各 1581 行）。
3. **`KM_PIKFNN_Refined_Results/`** 是历史输出快照：当前没有任何脚本以该目录名作为默认输出，推断为 `(5)` 脚本改名前的产物，请勿据此判断版本。
4. **目录归位瑕疵**：`圆孔受压/square_complex_body_force_results/` 实为方板体力算例的输出；`II型裂纹/source_schematic.py` 的标题写的是"I 型中心裂纹四分之一域源点示意图"，画的也是 `mode_i_*` 图。
5. `椭圆开孔受拉/` 与 `边缘裂纹/` 的结果只保存在 notebook 输出单元中，没有独立结果目录。

## 引用

若本仓库的代码或结果对您的研究有帮助，请引用：

```
（论文标题 / 作者 / 期刊 / 年份 / DOI —— 待补充）
```

- 仓库名 `CV-PIKFNN` 为 GitHub 项目标识，代码中的方法名为 **KM-PIKFNN**。
- 本仓库暂未添加 `LICENSE` 文件；如需明确授权方式，建议补充（例如 MIT 或 CC-BY-4.0）。

---

## English

**Reproduction code accompanying the paper on KM-PIKFNN**, a physics-informed *fundamental-solution* network for two-dimensional linear elasticity.

The solver uses **Kolosov–Muskhelishvili (K-M) complex-potential point-force fundamental solutions as an analytic kernel**. The linear kernel coefficients are obtained by a **column-scaled, regularized least-squares solve with an SVD warm start**, optionally refined with Adam / L-BFGS, while a *tiny* neural network learns **only the positions (or the ratio α) of the virtual source points** — the "adaptive source" scheme. Because the governing equations are satisfied exactly, pointwise, by the kernel, training only has to fit boundary data. This yields BEM/COMSOL-level accuracy on stress-concentration and high-gradient problems at a fraction of the training cost.

**Layout.** One directory per benchmark: `圆孔受压` (pressurized circular hole in an infinite plane), `椭圆开孔受拉` (elliptical hole under tension), `方板受压含体力` (square plate: a displacement-driven case with COMSOL reference in `方板受压/`, plus the complex-body-force benchmark of reference [23] at the top level), `不规则圆孔` (irregular hole, *no* conformal mapping), `I型裂纹` / `II型裂纹` / `混合型裂纹` / `边缘裂纹` (Mode I, Mode II, mixed-mode and edge cracks with Williams enrichment), and `高应力梯度问题` (elliptical hole with axis ratio λ driven from 1 down to 0.005 to create a steep stress gradient). Baselines shipped alongside: PINN / D-PINN, CV-PINN, KM-PINN (Joukowski-mapped complex-variable PINN with λ-continuation), XPINN, Enriched PINN, MFS, FSNet / KM-Net / KM-FSNet, an in-house multi-kernel **BEM** used as an *independent* reference that explicitly does not enter the loss, the initialization or the model selection, and COMSOL 6.3 exports.

**Environment.** Python 3.10 (conda), `pip install -r requirements.txt`. PyTorch runs in float64; **every benchmark runs on CPU**. Set `KMP_DUPLICATE_LIB_OK=TRUE` if `numba` and `torch` clash over OpenMP. Stand-alone `.py` scripts resolve paths relative to `__file__` and run as-is; **several notebooks contain hard-coded absolute paths from the authors' machine and must be edited before running** (see *已知问题*, item 1).

**Data files not in the repository.** GitHub rejects any single file above 100 MB, so 106 files (476 MB) are excluded via `.gitignore`: three 118.6 MB COMSOL mesh exports `不规则圆孔/irregular_hole_M{1,2,3}_grid_801.txt`, all `.npz` raw field arrays (116 MB) and one `.vtu` mesh export. The repository itself is **818 files / 232 MB and does not use Git LFS** — `git clone` gives you all code, figures, error tables and CSVs. To reproduce the irregular-hole case, export the three 801×801 grids from COMSOL 6.3 and drop them into `不规则圆孔/`; the solver reads **M3 first, then M2, then M1** (M1 is a preliminary mesh — paper numbers use M3). All derived `.csv` summaries and figures survive exclusion of the `.npz` files, and the BEM cache in `II型裂纹/.../bem_cache/` is included so the Mode-II geometry sweep needs no recomputation.

**Reading the results.** Every figure in the paper maps to a file listed in [论文图表 ↔ 文件对应关系](#论文图表--文件对应关系): the 4×6 stress-field panel for six λ values, the fig.4 KM-FSNet-vs-COMSOL error clouds, the fig.5 six-field L2 bar charts (from `fig5_l2_errors.csv`, plot-only script `plot_l2_error.py`), the α-convergence study (`square_plate_alpha_convergence.py`, identical to the varpro notebook), K_I vs b/a and the K_II geometry sweep (KM vs BEM relative difference ≤ 0.54%), the mixed-mode 30° convergence, and the `method_comparison_table.*` files contrasting Fixed vs Adaptive KM-PIKFNN (α₀ = 1.10–1.50) against PINN.

**Known quirks** are listed honestly at [已知问题与阅读提示](#已知问题与阅读提示): hard-coded notebook paths, one duplicated 1581-line script, a legacy results snapshot (`KM_PIKFNN_Refined_Results/`) that no current script writes to, and two result/plot directories that sit under a misleading parent folder.

**Citation / license.** The BibTeX entry is a placeholder to be filled in when the paper is published; no `LICENSE` file has been added yet.

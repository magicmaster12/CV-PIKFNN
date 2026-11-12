# -*- coding: utf-8 -*-
"""Redraw saved mixed-mode source histories. No solver or training is run."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR = DATA_DIR = Path(
    r"C:\Users\magic\Desktop\KM-PIKFNN\混合型裂纹\mixed_mode_adaptive_cold_results"
)
print("结果目录存在：", DATA_DIR.is_dir())
print("汇总表存在：",
      (DATA_DIR / "adaptive_source_initial_final_summary.csv").is_file())

if DATA_DIR.is_dir():
    print("目录中的实际文件：")
    for p in DATA_DIR.iterdir():
        print(repr(p.name))
PLOT_ANGLE = 30
PLOT_CFG = dict(figsize=(20, 9.4), label_fontsize=17, tick_fontsize=13,
                legend_fontsize=15, line_width=1.8, save_dpi=400,
                save_outputs=True, show=True, show_best_markers=False)

def read_mixed_histories(folder, angle=30):
    import re

    folder = Path(folder)
    runs = []

    # 与原求解代码一致：H=4，sigma0=1。
    H = 4.0
    references = {
        15: (1.9705, 0.4641),
        30: (1.6020, 0.8180),
        45: (1.0838, 0.9666),
        60: (0.5494, 0.8551),
    }

    angle = float(angle)
    if angle not in references:
        raise ValueError(f"没有 {angle:g}° 工况的参考值。")

    # 直接读取原代码保存的历史文件。
    pattern = re.compile(
        r"angle_([0-9.]+)deg_mx_([0-9.]+)_my_([0-9.]+)_history\.csv"
    )

    for path in folder.glob("angle_*deg_mx_*_my_*_history.csv"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue

        saved_angle, mx, my = map(float, match.groups())
        if not np.isclose(saved_angle, angle):
            continue

        with path.open(newline="", encoding="utf-8-sig") as f:
            history = list(csv.DictReader(f))

        if not history:
            raise ValueError(f"历史文件为空：{path}")

        best_path = path.with_name(
            path.name.replace("_history.csv", "_best.npz")
        )
        if best_path.is_file():
            with np.load(best_path, allow_pickle=False) as data:
                best_record = int(data["best_record"].item())
        else:
            # 与训练时一致，按最低总损失确定采用状态。
            best_row = min(
                history, key=lambda row: float(row["total_loss"])
            )
            best_record = int(float(best_row["record"]))

        runs.append({
            "adaptive_history": history,
            "initial_offsets": (mx, my),
            "best_record": best_record,
        })

    if not runs:
        raise FileNotFoundError(
            f"{folder.resolve()} 中没有找到 {angle:g}° 的训练历史。"
        )

    runs.sort(key=lambda run: run["initial_offsets"])
    print(f"已读取 {len(runs)} 组 {angle:g}° 训练历史。")

    return runs, references[angle], H

def plot_mixed_histories(runs, refs, H, output_dir, angle=30, settings=None):
    pc=dict(PLOT_CFG)
    if settings: pc.update(settings)
    if not runs: raise ValueError('No adaptive source histories available.')
    colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    panels=[('offset_x_over_H',r'$m_x/H$',False),('offset_y_over_H',r'$m_y/H$',False),
            ('KI',r'Stress intensity factor $K_I$',False),('KII',r'Stress intensity factor $K_{II}$',False),
            ('boundary_residual',r'Boundary residual $\sqrt{L_b}$',True),
            ('err_KI',r'Difference from BEM in $K_I$ (%)',True),
            ('err_KII',r'Difference from BEM in $K_{II}$ (%)',True),('total_loss','Total training loss',True)]
    style={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
           'mathtext.fontset':'stix','font.size':13,'axes.linewidth':.85,'pdf.fonttype':42}
    with plt.rc_context(style):
        fig,axes=plt.subplots(2,4,figsize=pc['figsize'],dpi=150,constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=.07,h_pad=.07,wspace=.07,hspace=.07)
        for panel,(ax,(key,label,logscale)) in enumerate(zip(axes.ravel(),panels)):
            xmax=0
            for j,case in enumerate(runs):
                history=case['adaptive_history']
                x=np.asarray([float(r['record']) for r in history]);y=np.asarray([float(r[key]) for r in history])
                if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError(f'Non-finite values in {key}.')
                if key.startswith('err_'): y=y*100 # Stored errors are fractions.
                y=np.where(y>0,y,np.nan) if logscale else y
                mx,my=case['initial_offsets'];name=rf'$({mx/H:g},\,{my/H:g})$'
                ax.plot(x,y,color=colors[j%4],lw=pc['line_width'],label=name)
                xmax=max(xmax,float(x.max()))
                if pc['show_best_markers']:
                    k=next((k for k,r in enumerate(history) if int(float(r['record']))==int(case['best_record'])),None)
                    if k is not None: ax.plot(x[k],y[k],'o',ms=4.5,mfc='white',color=colors[j%4])
            if key in ('KI','KII'):
                ref=refs[0 if key=='KI' else 1]
                ax.axhline(ref,color='#444444',ls='--',lw=1.2,zorder=1)
                ax.annotate('BEM reference',xy=(.98,ref),xycoords=('axes fraction','data'),
                            xytext=(0,-8),textcoords='offset points',ha='right',va='top',fontsize=12,color='#444444')
            if logscale: ax.set_yscale('log')
            ax.set_xlim(0,xmax)
            ax.set_xlabel('Iteration',fontsize=pc['label_fontsize'])
            ax.set_ylabel(label,fontsize=pc['label_fontsize'])
            ax.tick_params(direction='in',top=True,right=True,labelsize=pc['tick_fontsize'])
            ax.grid(False)
            ax.margins(y=.12)
            if panel==0:
                ax.legend(loc='lower right',ncol=2,fontsize=pc['legend_fontsize'],frameon=False,
                          handlelength=1.5,columnspacing=.7,handletextpad=.4)
            elif panel!=1:
                ax.legend(loc='lower right' if key in ('KI','KII') else 'upper right',ncol=1,
                          fontsize=pc['legend_fontsize'],frameon=False,handlelength=1.5,handletextpad=.4)
        if pc['save_outputs']:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            for ext in ('png','pdf'):
                file=folder/f'adaptive_source_convergence_{angle:g}deg.{ext}'
                fig.savefig(file,dpi=pc['save_dpi'],bbox_inches='tight',facecolor='white')
                print('Saved:',file.resolve())
        if pc['show']: plt.show()
    return fig,axes

if __name__=='__main__':
    runs,refs,H=read_mixed_histories(DATA_DIR,PLOT_ANGLE)
    plot_mixed_histories(runs,refs,H,DATA_DIR,PLOT_ANGLE)

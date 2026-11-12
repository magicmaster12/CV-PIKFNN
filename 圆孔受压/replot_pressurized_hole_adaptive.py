# -*- coding: utf-8 -*-
"""Read history_run_*.csv and redraw circular-hole adaptive curves; no training."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
BASE_DIR=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR=BASE_DIR/'hole_adaptive_results'
PLOT_CFG=dict(label_fontsize=17,tick_fontsize=13,legend_fontsize=15,line_width=1.8,
              figsize=(12.8,9.6),single_figsize=(6.8,5.2),save_dpi=400,
              save_outputs=True,show=True,save_single=True)
SPECS=[('outer_loss','Outer-boundary loss',True),('hole_loss','Hole-boundary loss',True),
       ('rho',r'$r_s/a$',False),('stress_relative_l2',r'Relative stress $L_2$ error',True)]
COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728']

def read_saved_hole_runs(folder):
    folder=Path(folder)
    files=sorted(folder.glob('history_run_*.csv'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
    if not files:
        raise FileNotFoundError(f'No history_run_*.csv in {folder.resolve()}. Set DATA_DIR to the existing result folder.')
    runs=[]
    for file in files:
        with file.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        if not rows: raise ValueError(f'Empty history: {file}')
        history=[{k:float(v) for k,v in row.items()} for row in rows]
        runs.append(dict(history=history,initial_scales=(history[0]['rho'],history[0]['alpha'])))
    return runs

def draw_hole_panel(ax,runs,spec,panel,pc):
    key,label,logscale=spec
    for j,run in enumerate(runs):
        history=run['history'];rho,alpha=run['initial_scales']
        x=np.asarray([float(r['iteration']) for r in history])
        y=np.asarray([float(r[key]) for r in history])
        if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError(f'Non-finite values in {key}')
        if logscale: y=np.where(y>0,y,np.nan)
        ax.plot(x,y,color=COLORS[j%len(COLORS)],lw=pc['line_width'],label=rf'$({rho:g},\,{alpha:g})$')
    if logscale: ax.set_yscale('log')
    ax.set_xlabel('Iteration',fontsize=pc['label_fontsize'])
    ax.set_ylabel(label,fontsize=pc['label_fontsize'])
    ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=pc['tick_fontsize'])
    ax.grid(False);ax.margins(x=.02,y=.12)
    if panel==0:
        ax.legend(loc='upper right' if key!='rho' else 'best',ncol=1,frameon=False,
                  fontsize=pc['legend_fontsize'],handlelength=1.5,handletextpad=.4)

def plot_hole_histories(runs,output_dir,settings=None):
    pc=dict(PLOT_CFG)
    if settings: pc.update(settings)
    folder=Path(output_dir)
    if pc['save_outputs']:folder.mkdir(parents=True,exist_ok=True)
    style={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
           'mathtext.fontset':'stix','font.size':13,'axes.linewidth':.85,'pdf.fonttype':42}
    with plt.rc_context(style):
        fig,axes=plt.subplots(2,2,figsize=pc['figsize'],dpi=150,constrained_layout=True)
        for i,(ax,spec) in enumerate(zip(axes.flat,SPECS)): draw_hole_panel(ax,runs,spec,i,pc)
        if pc['save_outputs']:
            for ext in ('png','pdf'):fig.savefig(folder/f'adaptive_training_4panels.{ext}',dpi=pc['save_dpi'],bbox_inches='tight',facecolor='white')
        if pc['save_single']:
            for i,spec in enumerate(SPECS):
                single,ax=plt.subplots(figsize=pc['single_figsize'],dpi=150,constrained_layout=True)
                draw_hole_panel(ax,runs,spec,i,pc)
                if pc['save_outputs']:
                    for ext in ('png','pdf'):single.savefig(folder/f'single_{spec[0]}.{ext}',dpi=pc['save_dpi'],bbox_inches='tight',facecolor='white')
                if not pc['show']:plt.close(single)
        if pc['show']:plt.show()
    return fig,axes

if __name__=='__main__':
    if not list(DATA_DIR.glob('history_run_*.csv')):
        candidate=BASE_DIR.parent/'hole_adaptive_results'
        if list(candidate.glob('history_run_*.csv')):DATA_DIR=candidate
    plot_hole_histories(read_saved_hole_runs(DATA_DIR),DATA_DIR)

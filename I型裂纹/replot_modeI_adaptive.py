# -*- coding: utf-8 -*-
"""Replot Mode-I adaptive-source histories in the style of the body-force case.

No training. Requires numpy and matplotlib only. Reads run_*_history.csv and
initial_final_summary.csv from the original solver output folder.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR = BASE_DIR / 'mode_i_cold_start_source_study'

PLOT_CFG = dict(
    figsize=(15, 8.6), label_fontsize=15, tick_fontsize=12,
    legend_fontsize=13, line_width=1.8,
    save_dpi=400, show_grid=False, show_best_markers=False,
    save_outputs=True, show=True,
)


def read_saved_source_study(folder):
    folder = Path(folder).expanduser()
    summary_path = folder / 'initial_final_summary.csv'
    if not summary_path.is_file():
        raise FileNotFoundError(f'Missing {summary_path.resolve()}. Set DATA_DIR to the original solver output folder.')
    with summary_path.open(newline='', encoding='utf-8-sig') as stream:
        summaries = list(csv.DictReader(stream))
    results = []
    for source in summaries:
        summary = dict(source)
        for key in ('run', 'best_record'):
            summary[key] = int(float(summary[key]))
        for key in ('initial_margin_x', 'initial_margin_y', 'KI_reference'):
            summary[key] = float(summary[key])
        path = folder / f"run_{summary['run']}_history.csv"
        with path.open(newline='', encoding='utf-8-sig') as stream:
            history = list(csv.DictReader(stream))
        if not history:
            raise ValueError(f'Empty history: {path}')
        results.append(dict(summary=summary, history=history))
    if not results:
        raise ValueError('The summary file contains no completed runs.')
    return results


def plot_source_initialization_study(results, base_cfg, plot_cfg=None):
    settings = dict(PLOT_CFG)
    if plot_cfg is not None:
        settings.update(plot_cfg)
    if not results:
        raise ValueError('No source-study results are available.')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    panels = [
        ('margin_x', r'$\alpha_x=m_x/b$', False),
        ('margin_y', r'$\alpha_y=m_y/h$', False),
        ('KI', r'Stress intensity factor $K_I$', False),
        ('boundary_residual', r'Boundary residual $\sqrt{L_{\mathrm{boundary}}}$', True),
        ('KI_relative_error', r'Relative difference in $K_I$ (%)', True),
        ('total_loss', 'Total training loss', True),
    ]
    style = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.linewidth': .85,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=settings['figsize'],
                                 dpi=150, constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=.07, h_pad=.07, wspace=.06, hspace=.06)
        for ax, (key, ylabel, logscale) in zip(axes.ravel(), panels):
            all_values, all_iterations = [], []
            for index, result in enumerate(results):
                history, summary = result['history'], result['summary']
                x = np.asarray([float(row['record']) for row in history])
                values = np.asarray([float(row[key]) for row in history])
                if key == 'KI_relative_error':
                    values = 100.*values
                if not np.isfinite(x).all() or not np.isfinite(values).all():
                    raise ValueError(f'Non-finite history values in {key}.')
                all_values.extend(values)
                all_iterations.extend(x)
                # A zero cannot be represented on a logarithmic axis.
                plotted = np.where(values > 0., values, np.nan) if logscale else values
                label = rf"$({float(summary['initial_margin_x']):g},\,{float(summary['initial_margin_y']):g})$"
                color = colors[index % len(colors)]
                ax.plot(x, plotted, color=color, lw=settings['line_width'], label=label)
                if settings['show_best_markers']:
                    best_index = next((j for j, row in enumerate(history)
                                       if int(float(row['record'])) == int(summary['best_record'])), None)
                    if best_index is not None and np.isfinite(plotted[best_index]):
                        ax.plot(x[best_index], plotted[best_index], 'o', color=color,
                                ms=4.5, mfc='white', mew=1.1)
            if logscale:
                ax.set_yscale('log')
            if key == 'KI':
                refs = np.asarray([float(r['summary']['KI_reference']) for r in results])
                if not np.allclose(refs, refs[0], rtol=1e-10, atol=1e-12):
                    raise ValueError('The runs use different KI reference values.')
                ax.axhline(refs[0], color='#444444', ls='--', lw=1.2,
                           label='_nolegend_', zorder=1)
                ax.annotate(r'$K_I^{\mathrm{ref}}$',
                            xy=(.98, refs[0]), xycoords=('axes fraction', 'data'),
                            xytext=(-3, -12), textcoords='offset points',
                            ha='right', va='top', color='#444444',
                            fontsize=settings['legend_fontsize'])
            if key in ('margin_x', 'margin_y'):
                low, high = min(all_values), max(all_values)
                span = max(high-low, .02)
                # Only the first source-parameter panel needs legend space.
                headroom = .35 if key == 'margin_x' else .06
                ax.set_ylim(low-.06*span, high+headroom*span)
            ax.set_xlim(min(all_iterations), max(all_iterations))
            ax.set_xlabel('Iteration', fontsize=settings['label_fontsize'])
            ax.set_ylabel(ylabel, fontsize=settings['label_fontsize'])
            ax.tick_params(direction='in', top=True, right=True,
                           labelsize=settings['tick_fontsize'], length=4, width=.8)
            if settings['show_grid']:
                ax.grid(which='major', color='#dddddd', lw=.5, alpha=.55)
            else:
                ax.grid(False)
            if key != 'margin_y':
                legend = ax.legend(
                    loc='lower right' if key == 'KI' else 'upper right',
                    ncol=2 if key == 'margin_x' else 1, frameon=False,
                    fontsize=settings['legend_fontsize'],
                    handlelength=1.8, handletextpad=.45,
                    columnspacing=.85, labelspacing=.35, borderaxespad=.55)
                for line in legend.get_lines():
                    line.set_linewidth(settings['line_width'])
        if settings['save_outputs']:
            folder = Path(base_cfg['output_dir']).expanduser()
            folder.mkdir(parents=True, exist_ok=True)
            for extension in ('png', 'pdf'):
                path = folder / f'cold_start_source_convergence_case41_style.{extension}'
                fig.savefig(path, dpi=settings['save_dpi'], bbox_inches='tight', facecolor='white')
                print('Saved:', path.resolve())
        if settings['show']:
            plt.show()
        plt.close(fig)
    print('Iteration follows the recorded updates: one Adam step or one L-BFGS outer call; closure evaluations are not counted separately.')
    return fig, axes


if __name__ == '__main__':
    saved_results = read_saved_source_study(DATA_DIR)
    plot_source_initialization_study(saved_results, {'output_dir': str(DATA_DIR)})

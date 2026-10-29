import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import os

from . import common

# IEEEtran single column. Figures are emitted at exactly this width so LaTeX
# never rescales them -- rescaling is what shrank the tick labels to nothing.
COLUMN_IN = 3.45
ROW_IN    = 1.1

# metric, axis label, scale. AoII and reward span four orders of magnitude
# once reactive is in the picture, so both need a log axis; AoII is
# non-negative and reward is not, hence log against symlog.
PANELS = [('avg_aoii', 'AoII', 'log'),
          ('utilization', 'Offered load $G$', 'linear'),
          ('reward', 'Reward', 'symlog')]

FLOOR = 1e-3

STYLE = {'font.size': 6, 'axes.labelsize': 6, 'axes.titlesize': 6.5,
         'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'legend.fontsize': 6,
         'lines.linewidth': 0.9, 'axes.linewidth': 0.5,
         'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
         'xtick.major.size': 2, 'ytick.major.size': 2,
         'legend.frameon': False, 'figure.dpi': 200}

# the figures of the paper: LaTeX text in Times, so that the policy names are
# set in small caps exactly as in the text
PAPER_STYLE = {**STYLE, 'text.usetex': True, 'font.family': 'serif',
               'text.latex.preamble': r'\usepackage{mathptmx}'}

def thousands(v, _):
    return f'{v/1000:g}k' if v else '0'

def plot_panel(args, dataset, solvers=None, out_dir=None):
    """One row of panels for a dataset, at exact column width."""
    solvers = solvers or ['reactive', 'random', 'hybrid', 'threshold', 'ucb', 'vhct']
    out_dir = out_dir or args.figure_dir
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(PANELS), figsize=(COLUMN_IN, ROW_IN))
        for ax, (metric, label, scale) in zip(axes, PANELS):
            args.metric = metric
            for solver in solvers:
                x, y, sd = common.load(args, dataset, solver)
                lo, hi = y - sd, y + sd
                if scale != 'symlog':
                    # the metric cannot go negative, so neither should its band
                    lo = np.maximum(lo, FLOOR if scale == 'log' else 0.0)
                    y = np.maximum(y, FLOOR if scale == 'log' else 0.0)
                line, = ax.plot(x, y, label=common.SOLVER_LABEL[solver])
                ax.fill_between(x, lo, hi, color=line.get_color(),
                                alpha=0.15, lw=0)
            if scale == 'log':
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(mticker.LogLocator(numticks=5))
                ax.yaxis.set_minor_locator(mticker.NullLocator())
            elif scale == 'symlog':
                ax.set_yscale('symlog', linthresh=1e-1)
                # one tick every two decades keeps the axis legible at this size
                lo, hi = ax.get_ylim()
                ticks = [0.0]
                k = -1
                while -10.0 ** k > lo:
                    ticks.append(-10.0 ** k); k += 2
                ax.set_yticks(sorted(ticks))
                ax.yaxis.set_minor_locator(mticker.NullLocator())
            else:
                ax.locator_params(axis='y', nbins=4)
            ax.set_title(label, pad=2)
            ax.set_xlabel('Slot', labelpad=1)
            ax.margins(x=0.02)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands))
            ax.locator_params(axis='x', nbins=3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=len(solvers), loc='upper center',
                   bbox_to_anchor=(0.5, 1.03), columnspacing=0.8,
                   handlelength=1.2, handletextpad=0.4)
        fig.tight_layout(rect=(0, 0, 1, 0.87), pad=0.2, w_pad=0.5)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'plot_panel_{dataset}.pdf')
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_panel_all(args, datasets=None, solvers=None, out_dir=None):
    """All traces in one figure: one row per dataset, the same three panels,
    one legend. Replaces the per-dataset figures in the paper."""
    datasets = datasets or ['berkeley', 'bts', 'sindit', 'synthetic']
    solvers  = solvers  or ['reactive', 'random', 'hybrid', 'threshold', 'ucb', 'vhct']
    out_dir  = out_dir  or args.figure_dir
    names = {'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT', 'synthetic': 'Synthetic'}
    with plt.rc_context(STYLE):
        fig, rows = plt.subplots(len(datasets), len(PANELS),
                                 figsize=(COLUMN_IN, 0.4 * len(datasets) + 0.25))
        for r, (dataset, axes) in enumerate(zip(datasets, rows)):
            for ax, (metric, label, scale) in zip(axes, PANELS):
                args.metric = metric
                for solver in solvers:
                    x, y, sd = common.load(args, dataset, solver)
                    lo, hi = y - sd, y + sd
                    if scale != 'symlog':
                        lo = np.maximum(lo, FLOOR if scale == 'log' else 0.0)
                        y  = np.maximum(y, FLOOR if scale == 'log' else 0.0)
                    line, = ax.plot(x, y, label=common.SOLVER_LABEL[solver])
                    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.15, lw=0)
                if scale == 'log':
                    ax.set_yscale('log')
                    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=4))
                    ax.yaxis.set_minor_locator(mticker.NullLocator())
                elif scale == 'symlog':
                    ax.set_yscale('symlog', linthresh=1e-1)
                    lo_, hi_ = ax.get_ylim(); ticks = [0.0]; k = -1
                    while -10.0 ** k > lo_:
                        ticks.append(-10.0 ** k); k += 2
                    ax.set_yticks(sorted(ticks))
                    ax.yaxis.set_minor_locator(mticker.NullLocator())
                else:
                    ax.locator_params(axis='y', nbins=3)
                if r == 0:
                    ax.set_title(label, pad=2)
                if r == len(datasets) - 1:
                    ax.set_xlabel('Slot', labelpad=1)
                ax.margins(x=0.02)
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands))
                ax.locator_params(axis='x', nbins=3)
            axes[0].set_ylabel(names[dataset], labelpad=2)
        handles, labels = rows[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=len(solvers), loc='upper center',
                   bbox_to_anchor=(0.5, 1.015), columnspacing=0.8,
                   handlelength=1.2, handletextpad=0.4)
        fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.2, w_pad=0.5, h_pad=0.4)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'plot_panel_all.pdf')
        fig.savefig(path)
        plt.close(fig)
    return path

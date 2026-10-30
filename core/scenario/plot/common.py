import matplotlib.ticker as mticker
import numpy as np

COLUMN_IN = 3.45

STYLE = {'font.size': 6, 'axes.labelsize': 6, 'axes.titlesize': 6.5,
         'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'legend.fontsize': 6,
         'lines.linewidth': 0.9, 'axes.linewidth': 0.5,
         'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
         'xtick.major.size': 2, 'ytick.major.size': 2,
         'legend.frameon': False, 'figure.dpi': 200}

PAPER_STYLE = {**STYLE, 'text.usetex': True, 'font.family': 'serif',
               'text.latex.preamble': r'\usepackage{mathptmx}'}

SOLVER_TEX = {'reactive': r'\textsc{Reactive}', 'random': r'\textsc{Throughput}',
              'hybrid': r'\textsc{Hybrid}', 'threshold': r'\textsc{Threshold}',
              'ucb': r'\textsc{UCB}', 'vhct': r'\textsc{VHCT}', 'mpgts': r'\textsc{Mpgts}'}

SOLVER_STYLE = {
    'reactive' : dict(color='#000000', marker='s', ls='-'),
    'threshold': dict(color='#555555', marker='^', ls='--'),
    'hybrid'   : dict(color='#808080', marker='D', ls='-.'),
    'random'   : dict(color='#9a9a9a', marker='v', ls=':'),
    'ucb'      : dict(color='#1f4e9c', marker='o', ls='--'),
    'vhct'     : dict(color='#5b9bd5', marker='P', ls='-.'),
    'mpgts'    : dict(color='#d62728', marker='*', ls='-'),
}

def reward_axis(ax, which='y', floor=None):
    axis = ax.yaxis if which == 'y' else ax.xaxis
    (ax.set_yscale if which == 'y' else ax.set_xscale)('log')
    axis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    axis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: r'$-10^{%d}$' % round(np.log10(v)) if v > 0 and abs(np.log10(v) - round(np.log10(v))) < 1e-6 else ''))
    axis.set_minor_formatter(mticker.NullFormatter())
    lim = ax.get_ylim if which == 'y' else ax.get_xlim
    lo, hi = sorted(lim())
    (ax.set_ylim if which == 'y' else ax.set_xlim)(floor or hi, lo)

def reward_err(v, sd):
    v, sd = np.asarray(v, float), np.nan_to_num(np.asarray(sd, float))
    return np.array([np.minimum(sd, 0.9 * v), sd])


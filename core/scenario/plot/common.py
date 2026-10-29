import matplotlib.ticker as mticker
import numpy as np

# IEEEtran single column. Figures are emitted at exactly this width so LaTeX
# never rescales them, which would shrink the tick labels.
COLUMN_IN = 3.45

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

# policy names as the paper sets them (small caps, LaTeX-rendered figures)
SOLVER_TEX = {'reactive': r'\textsc{Reactive}', 'random': r'\textsc{Throughput}',
              'hybrid': r'\textsc{Hybrid}', 'threshold': r'\textsc{Threshold}',
              'ucb': r'\textsc{UCB}', 'vhct': r'\textsc{VHCT}', 'mpgts': r'\textsc{Mpgts}'}

# one line style per policy: the proposed policy in red with a star; the
# non-learning baselines in a black-gray group and the learned supplements in a
# blue group, each member with its own marker and dash pattern
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
    """Axis for a reward that is always negative and spans decades: positions
    are |reward| on a log scale, flipped so that the best reward (closest to
    zero) is at the top or right, and the ticks read as the reward, -10^k.
    `floor`, the worst end of the axis, is where bars start, so a longer bar is
    a higher reward."""
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
    """Error bar of |reward| v for a log axis: the side toward zero is cut at
    90% of v, since a standard deviation larger than the mean would reach past
    zero, which a log axis cannot show."""
    v, sd = np.asarray(v, float), np.nan_to_num(np.asarray(sd, float))
    return np.array([np.minimum(sd, 0.9 * v), sd])


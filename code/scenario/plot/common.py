import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import glob
import os

LABEL = {'avg_aoii'   : 'Average AoII',
         'max_aoii'   : 'Peak AoII',
         'utilization': 'Offered load $G$',
         'p_success'  : 'Success probability',
         'reward'     : 'Reward'}

SOLVER_LABEL = {'reactive': 'reactive', 'random': 'throughput',
                'hybrid'  : 'hybrid',   'threshold': 'threshold',
                'ucb'    : 'UCB', 'vhct'  : 'VHCT', 'mpgts': 'Mpgts (ours)'}

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

def smooth(x, y, n_smooth):
    """Block-average into windows of n_smooth slots."""
    T = (len(x) // n_smooth) * n_smooth
    return x[:T].reshape(-1, n_smooth)[:, 0], y[:T].reshape(-1, n_smooth).mean(axis=1)

def metric(df, args):
    if args.metric == 'p_success':
        n_success, n_send = df['n_success'].to_numpy(), df['n_send'].to_numpy()
        return np.divide(n_success, n_send, out=np.ones_like(n_success, dtype=float),
                         where=n_send != 0)
    if args.metric == 'utilization':
        return df['n_send'].to_numpy() / args.n_irsa_slot
    return df[args.metric].to_numpy()

def load(args, dataset, solver):
    """Mean and standard deviation of one metric across every seed on disk."""
    paths = sorted(glob.glob(os.path.join(args.csv_dir, args.mode,
                                          f'{dataset}_{solver}_s*.csv')))
    if not paths:
        raise FileNotFoundError(f'no runs for {dataset}/{solver} under {args.csv_dir}')
    curves = []
    for path in paths:
        df   = pd.read_csv(path)
        x, y = smooth(df['step'].to_numpy(), metric(df, args), args.n_smooth)
        curves.append(y)
    # a shorter run would misalign the block average, so keep the common prefix
    n = min(len(y) for y in curves)
    Y = np.stack([y[:n] for y in curves])
    return x[:n], Y.mean(axis=0), Y.std(axis=0)

def save(args, path):
    plt.legend()
    plt.xlabel('Timeslot')
    plt.ylabel(LABEL.get(args.metric, args.metric))
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.cla()
    plt.clf()

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


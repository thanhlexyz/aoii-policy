import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from .plot_panel import COLUMN_IN, PAPER_STYLE
from . import common

NAMES    = {'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT', 'synthetic': 'Synthetic'}
# the channel is the shade of the bar: solid, light, hatched
CHANNELS = [('ideal', 'slot occupancy', dict(alpha=1.0)),
            ('software', 'software waveform', dict(alpha=0.45)),
            ('ota', 'radios', dict(alpha=1.0, fill=False, hatch='////'))]
ORDER    = ['reactive', 'threshold', 'hybrid', 'random', 'ucb', 'vhct', 'mpgts']

def plot_ota(args, out_dir=None):
    """RQ4 figure: reward per policy under slot-occupancy decoding, the
    software waveform and the radios, one row per trace with radio runs. The
    policies are sorted from worst to best under slot occupancy and coloured
    as in the other figures; the channel is the shade of the bar, so a policy
    whose three bars agree is one the simulator predicts."""
    out_dir = out_dir or os.path.join(args.paper_figure_dir, 'simulation')
    ch = pd.read_csv(os.path.join(args.csv_dir, 'channel.csv'))
    datasets = [d for d in NAMES if 'ota' in set(ch[ch.dataset == d].channel)]
    width = 0.27
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(len(datasets), 1, figsize=(COLUMN_IN, 1.05 * len(datasets) + 0.25), squeeze=False)
        for ax, dataset in zip(axes[:, 0], datasets):
            d = ch[(ch.dataset == dataset) & ch.solver.isin(ORDER)]
            ideal = d[d.channel == 'ideal'].set_index('solver').reward
            solvers = list(ideal.sort_values().index)
            floor = 10 ** np.ceil(np.log10((-d.reward + d.reward_std.fillna(0)).max()))
            for j, (channel, _, shade) in enumerate(CHANNELS):
                c = d[d.channel == channel].set_index('solver')
                for k, solver in enumerate(solvers):
                    if solver not in c.index: continue
                    color = common.SOLVER_STYLE[solver]['color']
                    x = k + (j - 1) * width
                    v = -c.loc[solver, 'reward']
                    ax.bar(x, v - floor, bottom=floor, width=width * 0.92, color=color, edgecolor=color,
                           lw=0.5, **shade)
                    if c.loc[solver, 'n_seed'] > 1:
                        ax.errorbar(x, v, yerr=common.reward_err([v], [c.loc[solver, 'reward_std']]), color='k', lw=0.4, capsize=0.8)
            ax.set_ylim(floor, 0.5 * (-d.reward).min())
            common.reward_axis(ax, 'y', floor)
            ax.set_ylabel(rf'\texttt{{{NAMES[dataset]}}} reward', labelpad=2)
            ax.set_xticks(range(len(solvers)))
            ax.set_xticklabels([common.SOLVER_TEX[s] for s in solvers], rotation=15, ha='right')
            for tick, solver in zip(ax.get_xticklabels(), solvers):
                tick.set_color(common.SOLVER_STYLE[solver]['color'])
            ax.tick_params(axis='x', length=0)
            ax.grid(axis='y', lw=0.3, alpha=0.4); ax.set_axisbelow(True)
        handles = [plt.Rectangle((0, 0), 1, 1, color='0.35', ec='0.35', lw=0.5, **shade) for _, _, shade in CHANNELS]
        fig.legend(handles, [l for _, l, _ in CHANNELS], ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.4)
        fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.2, h_pad=0.4)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'plot_ota.pdf')
        fig.savefig(path); plt.close(fig)
    print(f'[+] wrote {path}')
    return path

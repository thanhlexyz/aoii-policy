import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from .plot_panel import COLUMN_IN, PAPER_STYLE
from . import common

NAMES    = {'synthetic': 'Synthetic', 'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT'}
# the channel is the marker: hollow for the two simulated channels, filled
# over the radios; the policy is the colour, as in the other figures
CHANNELS = [('ideal', 'slot occupancy', 'o', False, -0.25), ('software', 'software waveform', 's', False, 0.0),
            ('ota', 'radios', '^', True, 0.25)]
# the local-record rules collapse to rewards below -10 on every channel and
# are left off the linear axis (the caption says so); the rest are sorted by
# their slot-occupancy reward
SHOWN = ['hybrid', 'random', 'ucb', 'vhct', 'mpgts']

def plot_ota(args, out_dir=None, datasets=None, name='plot_ota.pdf'):
    """RQ4 figure: reward per policy under slot-occupancy decoding, the
    software waveform and the radios, on a linear axis so that the gaps among
    the competitive policies are visible; one row per trace with radio runs,
    the policies sorted from worst to best under slot occupancy, the proposed
    policy highlighted."""
    out_dir = out_dir or os.path.join(args.paper_figure_dir, 'simulation')
    ch = pd.read_csv(os.path.join(args.csv_dir, 'channel.csv'))
    datasets = datasets or [d for d in NAMES if 'ota' in set(ch[ch.dataset == d].channel)]
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(len(datasets), 1, figsize=(COLUMN_IN, 1.0 * len(datasets) + 0.3), squeeze=False)
        for ax, dataset in zip(axes[:, 0], datasets):
            d = ch[(ch.dataset == dataset) & ch.solver.isin(SHOWN)]
            solvers = list(d[d.channel == 'ideal'].set_index('solver').reward.sort_values().index)
            # the axis spans the competitive rewards; a collapsed run (below
            # -1) is drawn at the bottom edge with its value
            lo = 1.25 * d[d.reward > -1].reward.min()
            ax.set_ylim(lo, 0.08 * abs(lo))
            for k, solver in enumerate(solvers):
                color = common.SOLVER_STYLE[solver]['color']
                ours = solver == 'mpgts'
                for j, (channel, _, marker, filled, dx) in enumerate(CHANNELS):
                    c = d[(d.channel == channel) & (d.solver == solver)]
                    if c.empty: continue
                    r, sd = c.reward.iloc[0], c.reward_std.iloc[0]
                    if r < lo:
                        # off the axis: a marker at the edge, the value stacked above by channel
                        ax.plot(k + dx, lo, marker='v', ms=3.5, color=color, ls='none', clip_on=False, zorder=3)
                        ax.annotate(f'${r:.0f}\\pm{sd:.0f}$', (k + dx, lo), xytext=(0, 4 + 7 * j), textcoords='offset points',
                                    ha='center', va='bottom', fontsize=4.5, color=color)
                        continue
                    ax.errorbar(k + dx, r, yerr=sd, color=color, lw=0.6, capsize=1.2,
                                marker=marker, ms=4.5 if ours else 3.5, mfc=color if filled else 'white', mew=0.9,
                                ls='none', zorder=3 if ours else 2)
                if ours:
                    ax.axvspan(k - 0.45, k + 0.45, color=color, alpha=0.08, lw=0)
            ax.set_ylabel(rf'\texttt{{{NAMES[dataset]}}} reward', labelpad=2)
            ax.set_xticks(range(len(solvers)))
            ax.set_xticklabels([common.SOLVER_TEX[s] for s in solvers])
            for tick, solver in zip(ax.get_xticklabels(), solvers):
                tick.set_color(common.SOLVER_STYLE[solver]['color'])
            ax.tick_params(axis='x', length=0); ax.set_xlim(-0.5, len(solvers) - 0.5)
            ax.grid(axis='y', lw=0.3, alpha=0.4); ax.set_axisbelow(True)
        handles = [plt.Line2D([], [], marker=m, color='0.35', mfc='0.35' if f else 'white', mew=0.9, ls='none', ms=3.5)
                   for _, _, m, f, _ in CHANNELS]
        fig.legend(handles, [l for _, l, _, _, _ in CHANNELS], ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.0)
        fig.tight_layout(rect=(0.03, 0, 1, 0.93), pad=0.2, h_pad=0.4)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        fig.savefig(path); plt.close(fig)
    print(f'[+] wrote {path}')
    return path

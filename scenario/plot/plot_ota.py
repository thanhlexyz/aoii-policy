import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os

from .plot_panel import COLUMN_IN, PAPER_STYLE
from . import common

NAMES    = {'synthetic': 'Synthetic', 'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT'}
# the channel is the fill: hollow under idealized slot-occupancy decoding,
# filled over the radios; the policy is the colour, as in the other figures
CHANNELS = [('ideal', 'slot occupancy', False), ('ota', 'radios', True)]
# the local-record rules reach AoII orders of magnitude above the rest on every
# channel and are left out (the caption says so)
SHOWN    = ['hybrid', 'random', 'vhct', 'ucb', 'mpgts']
PANEL_H  = 1.2    # height of every panel, inch
LABEL_IN = 0.5    # extra width of the first panel, which carries the policy names

def plot_ota(args, out_dir=None):
    """RQ4 figure, as one PDF per trace with radio runs, placed side by side
    with subfigures (plot_ota_<dataset>.pdf), and a shared legend
    (plot_ota_legend.pdf): steady-state AoII per policy as horizontal bars on
    a log axis, hollow under slot occupancy and filled over the radios, mean
    and standard deviation over the seeds. One policy order for every panel,
    worst (top) to best by the mean rank of the slot-occupancy AoII, so that
    the policy names are printed once, on the first panel."""
    out_dir = out_dir or os.path.join(args.paper_figure_dir, 'simulation')
    os.makedirs(out_dir, exist_ok=True)
    ch = pd.read_csv(os.path.join(args.csv_dir, 'channel.csv'))
    ch = ch[ch.solver.isin(SHOWN)]
    datasets = [d for d in NAMES if 'ota' in set(ch[ch.dataset == d].channel)]
    ideal = ch[(ch.channel == 'ideal') & ch.dataset.isin(datasets)]
    rank  = ideal.pivot(index='solver', columns='dataset', values='avg_aoii').rank().mean(axis=1)
    order = list(rank.sort_values(ascending=False).index)
    y     = np.arange(len(order))[::-1]
    width = (COLUMN_IN - LABEL_IN) / len(datasets) - 0.02
    paths = []
    with plt.rc_context(PAPER_STYLE):
        for i, dataset in enumerate(datasets):
            fig, ax = plt.subplots(figsize=(width + (LABEL_IN if i == 0 else 0), PANEL_H))
            d  = ch[ch.dataset == dataset]
            lo = 10 ** np.floor(np.log10(0.5 * d.avg_aoii.min()))
            hi = 10 ** np.ceil(np.log10(1.2 * (d.avg_aoii + d.avg_aoii_std.fillna(0)).max()))
            for yy, solver in zip(y, order):
                color = common.SOLVER_STYLE[solver]['color']
                for j, (channel, _, filled) in enumerate(CHANNELS):
                    c = d[(d.channel == channel) & (d.solver == solver)]
                    if c.empty: continue
                    m, sd = c.avg_aoii.iloc[0], np.nan_to_num(c.avg_aoii_std.iloc[0])
                    yj = yy + (0.19 if j == 0 else -0.19)
                    ax.barh(yj, m - lo, left=lo, height=0.36, color=color if filled else 'white',
                            edgecolor=color, lw=0.7, zorder=2)
                    # a standard deviation above the mean cannot be drawn below
                    # it on a log axis: that side is cut at 80% of the mean
                    ax.errorbar(m, yj, xerr=[[min(sd, 0.8 * m)], [sd]], color='0.25', lw=0.5,
                                capsize=1.0, ls='none', zorder=3)
                if solver == 'mpgts':
                    ax.axhspan(yy - 0.48, yy + 0.48, color=color, alpha=0.08, lw=0, zorder=0)
            ax.set_xscale('log'); ax.set_xlim(lo, hi)
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.set_xlabel('AoII (lower is better)', labelpad=1)
            ax.set_yticks(y); ax.set_ylim(-0.6, len(order) - 0.4)
            if i == 0:
                ax.set_yticklabels([common.SOLVER_TEX[s] for s in order])
                for tick, solver in zip(ax.get_yticklabels(), order):
                    tick.set_color(common.SOLVER_STYLE[solver]['color'])
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', lw=0.3, alpha=0.4); ax.set_axisbelow(True)
            fig.tight_layout(pad=0.15)
            path = os.path.join(out_dir, f'plot_ota_{dataset}.pdf')
            fig.savefig(path); plt.close(fig); paths.append(path)
            print(f'[+] wrote {path}')
        fig = plt.figure(figsize=(COLUMN_IN, 0.2))
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor='0.45' if f else 'white', edgecolor='0.45', lw=0.7)
                   for _, _, f in CHANNELS]
        fig.legend(handles, [l for _, l, _ in CHANNELS], ncol=2, loc='center', handlelength=1.4,
                   columnspacing=1.5, fontsize=6)
        path = os.path.join(out_dir, 'plot_ota_legend.pdf')
        fig.savefig(path); plt.close(fig); paths.append(path)
        print(f'[+] wrote {path}')
    return paths

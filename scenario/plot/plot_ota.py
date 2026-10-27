import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from .plot_panel import COLUMN_IN, STYLE
from . import common

NAMES    = {'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT', 'synthetic': 'Synthetic'}
CHANNELS = [('ideal', 'slot occupancy', 'o'), ('software', 'software waveform', 's'), ('ota', 'radios', '^')]
ORDER    = ['reactive', 'threshold', 'hybrid', 'random', 'grid', 'vhct', 'repair']

def plot_ota(args, out_dir=None):
    """RQ4 figure: per policy, delivery and reward under the three channels,
    one row per trace with radio runs. The policies are ordered by offered
    load, so the slope of each delivery line is the price of airtime over
    the air."""
    out_dir = out_dir or os.path.join(args.paper_figure_dir, 'simulation')
    ch = pd.read_csv(os.path.join(args.csv_dir, 'channel.csv'))
    datasets = [d for d in NAMES if 'ota' in set(ch[ch.dataset == d].channel)]
    with plt.rc_context(STYLE):
        fig, rows = plt.subplots(len(datasets), 2, figsize=(COLUMN_IN, 0.95 * len(datasets) + 0.35), squeeze=False)
        for (ax1, ax2), dataset in zip(rows, datasets):
            d = ch[ch.dataset == dataset]
            solvers = [s for s in ORDER if s in set(d.solver)]
            x = np.arange(len(solvers))
            for channel, label, marker in CHANNELS:
                c = d[d.channel == channel].set_index('solver')
                p = [c.loc[s, 'p_success'] if s in c.index else np.nan for s in solvers]
                r = [-c.loc[s, 'reward'] if s in c.index else np.nan for s in solvers]
                ax1.plot(x, p, marker=marker, ms=2.5, lw=0.8, label=label)
                ax2.plot(x, r, marker=marker, ms=2.5, lw=0.8, label=label)
            ax1.set_ylabel(f'{NAMES[dataset]}\ndelivery', labelpad=2)
            ax2.set_ylabel('$|$reward$|$', labelpad=2); ax2.set_yscale('log')
            for ax in (ax1, ax2):
                ax.set_xticks(x); ax.set_xticklabels([common.SOLVER_LABEL[s].replace(' (ours)', '') for s in solvers], rotation=35, ha='right')
                ax.grid(lw=0.3, alpha=0.4)
            ax1.set_ylim(0, 1.05)
        handles, labels = rows[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02), handlelength=1.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.2, w_pad=0.6, h_pad=0.3)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'plot_ota.pdf')
        fig.savefig(path); plt.close(fig)
    print(f'[+] wrote {path}')
    return path

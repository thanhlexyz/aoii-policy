import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os

from .plot_panel import COLUMN_IN, STYLE

NAMES = {'synthetic': 'Synthetic', 'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT'}
CLIP  = 50   # percent; gains beyond this saturate the colour scale

def plot_robustness(args, out_dir=None):
    """RQ3 figure: two heatmaps over the protocol variations (rows) and traces
    (columns). Left, the proposed policy's reward gain over the best baseline
    with its rank printed in the cell; right, the AoII collapse factor of the
    local-record rules relative to the proposed policy."""
    out_dir = out_dir or os.path.join(args.paper_figure_dir, 'simulation')
    rob = pd.read_csv(os.path.join(args.csv_dir, 'robustness.csv'), keep_default_na=False)
    for c in ['gain', 'rank', 'collapse']:
        rob[c] = pd.to_numeric(rob[c], errors='coerce')
    rob['row'] = rob['config'].str.replace('\\\\%', '%', regex=True)
    # the later block and the subsets are trace-specific rows; keep one label
    rob.loc[rob.tag.str.startswith('_o'), 'row'] = 'other streams ' + rob[rob.tag.str.startswith('_o')].groupby('dataset').cumcount().add(1).astype(str)
    rows = list(dict.fromkeys(rob['row']))
    datasets = [d for d in NAMES if d in set(rob.dataset)]
    gain = np.full((len(rows), len(datasets)), np.nan)
    rank = np.full_like(gain, np.nan)
    collapse = np.full_like(gain, np.nan)
    for i, r in enumerate(rows):
        for j, d in enumerate(datasets):
            c = rob[(rob.row == r) & (rob.dataset == d)]
            if len(c):
                gain[i, j] = c.gain.iloc[0]; rank[i, j] = c['rank'].iloc[0]; collapse[i, j] = c.collapse.iloc[0]
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN_IN, 0.13 * len(rows) + 0.6),
                                       gridspec_kw={'width_ratios': [1.25, 1]}, sharey=True)
        im1 = ax1.imshow(np.clip(gain, -CLIP, CLIP), cmap='RdBu', vmin=-CLIP, vmax=CLIP, aspect='auto')
        for i in range(len(rows)):
            for j in range(len(datasets)):
                if np.isnan(gain[i, j]): continue
                ax1.text(j, i, f'{gain[i, j]:+.0f}%\n#{int(rank[i, j])}', ha='center', va='center', fontsize=4.8,
                         color='white' if abs(gain[i, j]) > 0.6 * CLIP else 'black', linespacing=0.9)
        im2 = ax2.imshow(np.log10(collapse), cmap='Oranges', vmin=0, vmax=5, aspect='auto')
        for i in range(len(rows)):
            for j in range(len(datasets)):
                if np.isnan(collapse[i, j]): continue
                v = collapse[i, j]
                ax2.text(j, i, f'{v:.1f}' if v < 10 else f'{v:.0f}' if v < 1000 else f'$10^{{{np.log10(v):.1f}}}$', ha='center', va='center',
                         fontsize=4.8, color='white' if v > 10 ** 3.5 else 'black')
        for ax, title in [(ax1, 'reward gain over best baseline'), (ax2, 'AoII, local rules / ours')]:
            ax.set_xticks(range(len(datasets))); ax.set_xticklabels([NAMES[d] for d in datasets])
            ax.set_title(title, pad=2); ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, len(datasets)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
            ax.grid(which='minor', color='white', lw=0.6); ax.tick_params(which='minor', length=0)
        ax1.set_yticks(range(len(rows))); ax1.set_yticklabels(rows)
        fig.tight_layout(pad=0.2, w_pad=0.4)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'plot_robustness.pdf')
        fig.savefig(path); plt.close(fig)
    print(f'[+] wrote {path}')
    return path

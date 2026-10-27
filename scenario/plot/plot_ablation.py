import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os

from .plot_panel import COLUMN_IN, STYLE

NAMES = {'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT', 'synthetic': 'Synthetic'}

def plot_ablation(args, out_dir=None):
    """RQ2 figure: the reward each ablated variant reaches, one panel per
    trace, bars at |reward| on a log axis (lower is better), the full policy
    highlighted and the best baseline of tab:main as a dashed line."""
    out_dir = out_dir or os.path.join(args.paper_figure_dir, 'simulation')
    abl = pd.read_csv(os.path.join(args.csv_dir, 'ablation.csv'), keep_default_na=False)
    summ = pd.read_csv(os.path.join(args.csv_dir, 'summary.csv'))
    datasets = [d for d in NAMES if d in set(abl.dataset)]
    variants = abl.drop_duplicates('tag')[['tag', 'variant']].values.tolist()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(datasets), figsize=(COLUMN_IN, 0.16 * len(variants) + 0.55), sharey=True)
        y = np.arange(len(variants))[::-1]
        for ax, dataset in zip(np.atleast_1d(axes), datasets):
            d = abl[abl.dataset == dataset].set_index('tag')
            vals = np.array([-d.loc[t, 'reward'] if t in d.index else np.nan for t, _ in variants])
            errs = np.array([d.loc[t, 'reward_std'] if t in d.index else 0 for t, _ in variants])
            colors = ['C3' if t == '' else 'C0' for t, _ in variants]
            ax.barh(y, vals, xerr=errs, color=colors, height=0.65, error_kw={'lw': 0.5, 'capsize': 1.2})
            base = summ[(summ.dataset == dataset) & (summ.solver != 'repair')]['reward'].max()
            ax.axvline(-base, color='k', ls='--', lw=0.6)
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.set_title(NAMES[dataset], pad=2)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', lw=0.3, alpha=0.4)
        np.atleast_1d(axes)[0].set_yticks(y)
        np.atleast_1d(axes)[0].set_yticklabels([v for _, v in variants])
        fig.supxlabel('$|$reward$|$ over the final 1000 slots (lower is better; dashed: best baseline)', fontsize=6, y=0.01)
        fig.tight_layout(rect=(0, 0.04, 1, 1), pad=0.2, w_pad=0.6)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'plot_ablation.pdf')
        fig.savefig(path); plt.close(fig)
    print(f'[+] wrote {path}')
    return path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os

from .plot_panel import COLUMN_IN, PAPER_STYLE
from . import common

# the variants the text of RQ2 discusses; ablation.csv keeps every variant
SHOWN = {'': r'\textsc{Mpgts} (full policy)', '_norepeat': 'no repeats', '_nobg': 'no background',
         '_std': 'std.\\ dev.\\ instead of robust scale', '_ucb': 'UCB1, one arm per round'}

NAMES = {'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT', 'synthetic': 'Synthetic'}

def plot_ablation(args, out_dir=None):
    """RQ2 figure: the reward each ablated variant reaches, one panel per
    trace, bars of reward on a log axis that grow toward zero (higher is
    better), the full policy highlighted and the best baseline of tab:main as
    a dashed line; variants sorted from worst (top) to best."""
    out_dir = out_dir or os.path.join(args.paper_figure_dir, 'simulation')
    abl = pd.read_csv(os.path.join(args.csv_dir, 'ablation.csv'), keep_default_na=False)
    summ = pd.read_csv(os.path.join(args.csv_dir, 'summary.csv'))
    datasets = [d for d in NAMES if d in set(abl.dataset)]
    variants = [v for v in abl.drop_duplicates('tag')[['tag', 'variant']].values.tolist() if v[0] in SHOWN]
    # worst to best from the top, by the mean rank of |reward| over the traces
    rank = abl.pivot(index='tag', columns='dataset', values='reward').rank().mean(axis=1)
    variants.sort(key=lambda v: rank[v[0]])
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, len(datasets), figsize=(COLUMN_IN, 0.16 * len(variants) + 0.55), sharey=True)
        y = np.arange(len(variants))[::-1]
        for ax, dataset in zip(np.atleast_1d(axes), datasets):
            d = abl[abl.dataset == dataset].set_index('tag')
            vals = np.array([-d.loc[t, 'reward'] if t in d.index else np.nan for t, _ in variants])
            errs = np.array([d.loc[t, 'reward_std'] if t in d.index else 0 for t, _ in variants])
            colors = [common.SOLVER_STYLE['mpgts']['color'] if t == '' else '#5b9bd5' for t, _ in variants]
            base = summ[(summ.dataset == dataset) & (summ.solver != 'mpgts')]['reward'].max()
            floor = 10 ** np.ceil(np.log10(np.nanmax(vals + errs)))
            ax.barh(y, vals - floor, left=floor, color=colors, height=0.65)
            ax.errorbar(vals, y, xerr=common.reward_err(vals, errs), ls='none', color='k', lw=0.5, capsize=1.2)
            ax.axvline(-base, color='k', ls='--', lw=0.6)
            ax.set_xlim(floor, 0.5 * np.nanmin(vals - common.reward_err(vals, errs)[0]))
            common.reward_axis(ax, 'x', floor)
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
            ax.set_title(rf'\texttt{{{NAMES[dataset]}}}', pad=2)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', lw=0.3, alpha=0.4)
        np.atleast_1d(axes)[0].set_yticks(y)
        np.atleast_1d(axes)[0].set_yticklabels([SHOWN[t] for t, _ in variants])
        fig.supxlabel('reward over the final 1000 slots (higher is better; dashed: best baseline)', fontsize=6, y=0.01)
        fig.tight_layout(rect=(0, 0.04, 1, 1), pad=0.2, w_pad=0.6)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'plot_ablation.pdf')
        fig.savefig(path); plt.close(fig)
    print(f'[+] wrote {path}')
    return path

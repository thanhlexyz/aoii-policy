import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os

from .common import COLUMN_IN, PAPER_STYLE
from . import common

# the variants the text of RQ2 discusses; ablation.csv keeps every variant
SHOWN = {'': r'\textsc{Mpgts} (full policy)', '_q': 'learn $q$ as well', '_norepeat': 'no repeats', '_nobg': 'no background',
         '_ucb': 'UCB1, one arm per round'}

NAMES = {'synthetic': 'Synthetic', 'berkeley': 'Berkeley', 'bts': 'BTS', 'sindit': 'SINDIT'}

def ablation_order(reward):
    """Rows of `reward` (variant x trace, higher is better) from worst to
    best. Variants are layered by Pareto dominance over the traces: the
    non-dominated ones form the last layer, those dominated only by it the one
    before, and so on. Within a layer they are ordered by their rank on each
    trace normalized to [0, 1] (0 worst, 1 best) and averaged over the traces,
    ties broken by the mean ratio of the best reward of each trace to theirs."""
    tags = list(reward.index)
    def dominated(a, b):   # b dominates a
        return (reward.loc[b] >= reward.loc[a]).all() and (reward.loc[b] > reward.loc[a]).any()
    layer, rest, depth = {}, set(tags), 0
    while rest:
        front = {a for a in rest if not any(dominated(a, b) for b in rest if b != a)}
        for a in front: layer[a] = depth
        rest -= front; depth += 1
    norm  = ((reward.rank() - 1) / max(len(tags) - 1, 1)).mean(axis=1)
    ratio = (reward.max() / reward).mean(axis=1)
    # the last front found is the worst: sort fronts in reverse discovery order
    return sorted(tags, key=lambda t: (-layer[t], norm[t], ratio[t]))

def plot_ablation(args, out_dir=None):
    """RQ2 figure: the reward each ablated variant reaches, one panel per
    trace, bars of reward on a log axis that grow toward zero (higher is
    better), the full policy highlighted and the best baseline of tab:main as
    a dashed line; variants sorted from worst (top) to best."""
    out_dir = out_dir or os.path.join(args.figure_dir, 'simulation')
    abl = pd.read_csv(os.path.join(args.csv_dir, 'ablation.csv'), keep_default_na=False)
    summ = pd.read_csv(os.path.join(args.csv_dir, 'summary.csv'))
    datasets = [d for d in NAMES if d in set(abl.dataset)]
    variants = [v for v in abl.drop_duplicates('tag')[['tag', 'variant']].values.tolist() if v[0] in SHOWN]
    order = ablation_order(abl[abl.tag.isin(SHOWN)].pivot(index='tag', columns='dataset', values='reward')[datasets])
    variants.sort(key=lambda v: order.index(v[0]))
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

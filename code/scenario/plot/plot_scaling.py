import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from .plot_panel import COLUMN_IN, PAPER_STYLE
from . import common

ORDER = ['reactive', 'threshold', 'hybrid', 'random', 'ucb', 'vhct', 'mpgts']

PANEL_IN = (COLUMN_IN / 2, 1.3)   # one subfigure of the paper

def _sensors_axis(ax, Ks):
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks]); ax.minorticks_off()
    ax.set_xlabel('sensors $K$ ($M = K/6$)', labelpad=1)
    ax.grid(lw=0.3, alpha=0.4)

def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path); plt.close(fig)
    print(f'[+] wrote {path}')
    return path

def plot_scaling(args, out_dir=None):
    """Scaling figure, as three PDFs the paper places with subfigures:
    plot_scaling_reward.pdf, reward per policy against the number of sensors K
    (M = K / 6); plot_scaling_consensus.pdf, the rounds the proposed learner
    needs before 80% of the devices share one schedule, one box per K over the
    seeds; and plot_scaling_legend.pdf, the policy legend spanning both."""
    out_dir = out_dir or os.path.join(args.paper_figure_dir, 'simulation')
    sc = pd.read_csv(os.path.join(args.csv_dir, 'scaling.csv'))
    Ks = sorted(sc.K.unique())
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_IN)
        for solver in ORDER:
            d = sc[sc.solver == solver].sort_values('K')
            if d.empty: continue
            ours = solver == 'mpgts'
            st = common.SOLVER_STYLE[solver]
            ax.errorbar(d.K, -d.reward, yerr=common.reward_err(-d.reward, d.reward_std), color=st['color'],
                        marker=st['marker'], ls=st['ls'], ms=5 if ours else 2.8, lw=1.4 if ours else 0.8,
                        capsize=1.2, zorder=3 if ours else 2, label=common.SOLVER_TEX[solver])
        common.reward_axis(ax, 'y')
        ax.set_ylabel('reward (higher is better)', labelpad=2)
        _sensors_axis(ax, Ks)
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout(pad=0.2)
        _save(fig, out_dir, 'plot_scaling_reward.pdf')

        # a standard box per K over the seeds: quartiles, median, whiskers at
        # 1.5 IQR and the seeds beyond them as points
        fig, ax = plt.subplots(figsize=PANEL_IN)
        ps = pd.read_csv(os.path.join(args.csv_dir, 'scaling_per_seed.csv'))
        ps = ps[ps.solver == 'mpgts']
        color = common.SOLVER_STYLE['mpgts']['color']
        data = [ps[ps.K == K].rounds_to_agree.dropna().to_numpy() for K in Ks]
        ax.boxplot(data, positions=Ks, widths=[0.35 * K for K in Ks], patch_artist=True,
                   boxprops=dict(facecolor='#f6cfcf', edgecolor=color, lw=0.8),
                   medianprops=dict(color=color, lw=1.4), whiskerprops=dict(color=color, lw=0.8),
                   capprops=dict(color=color, lw=0.8),
                   flierprops=dict(marker='o', ms=2.5, mfc='white', mec=color, mew=0.7))
        ax.set_ylim(0, None)
        ax.set_ylabel('rounds to consensus', labelpad=2)
        _sensors_axis(ax, Ks)
        fig.tight_layout(pad=0.2)
        _save(fig, out_dir, 'plot_scaling_consensus.pdf')

        fig = plt.figure(figsize=(COLUMN_IN, 0.3))
        fig.legend(handles, labels, ncol=4, loc='center', handlelength=2.6, columnspacing=0.8,
                   handletextpad=0.4, fontsize=5.5)
        _save(fig, out_dir, 'plot_scaling_legend.pdf')
    return out_dir

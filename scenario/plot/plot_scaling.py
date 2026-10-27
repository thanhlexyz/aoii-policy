import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

from .plot_panel import COLUMN_IN, STYLE
from . import common

ORDER = ['reactive', 'threshold', 'hybrid', 'random', 'grid', 'vhct', 'repair']

def plot_scaling(args, out_dir=None):
    """Scaling figure: left, reward per policy against the number of sensors
    K (M = K / 6); right, the share of devices the proposed learner has on one
    schedule against the round, one line per K."""
    out_dir = out_dir or os.path.join(args.paper_figure_dir, 'simulation')
    sc = pd.read_csv(os.path.join(args.csv_dir, 'scaling.csv'))
    Ks = sorted(sc.K.unique())
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.55))
        for solver in ORDER:
            d = sc[sc.solver == solver].sort_values('K')
            if d.empty: continue
            ax1.errorbar(d.K, -d.reward, yerr=d.reward_std, marker='o', ms=2.2, lw=0.8, capsize=1.2,
                         label=common.SOLVER_LABEL[solver].replace(' (ours)', ''))
        ax1.set_xscale('log', base=2); ax1.set_yscale('log')
        ax1.set_xticks(Ks); ax1.set_xticklabels([str(k) for k in Ks])
        ax1.set_xlabel('sensors $K$ ($M = K/6$)', labelpad=1); ax1.set_ylabel('$|$reward$|$', labelpad=2)
        for K in Ks:
            curves = []
            for path in sorted(glob.glob(os.path.join(args.csv_dir, args.mode, f'synthetic_repair_scaleK{K}_s[0-9]*.csv'))):
                df = pd.read_csv(path)
                curves.append(df['agree'].groupby(df.index // 25).mean().to_numpy())
            if not curves: continue
            n = min(len(c) for c in curves)
            a = np.mean([c[:n] for c in curves], axis=0)
            # ten-round moving average, for legibility
            a = np.convolve(a, np.ones(10) / 10, mode='valid')
            ax2.plot(np.arange(len(a)) + 5, a, lw=0.8, label=f'$K = {K}$')
        ax2.set_xlabel('round', labelpad=1); ax2.set_ylabel('devices on modal schedule', labelpad=2)
        ax2.set_ylim(0, 1.02); ax2.legend(handlelength=1.2, loc='lower right', fontsize=5)
        for ax in (ax1, ax2): ax.grid(lw=0.3, alpha=0.4)
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.03), handlelength=1.2,
                   columnspacing=0.8, handletextpad=0.4, fontsize=5.5)
        fig.tight_layout(rect=(0, 0, 1, 0.82), pad=0.2, w_pad=0.8)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'plot_scaling.pdf')
        fig.savefig(path); plt.close(fig)
    print(f'[+] wrote {path}')
    return path

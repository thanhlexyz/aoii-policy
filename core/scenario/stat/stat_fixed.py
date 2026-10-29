import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

def stat_fixed(args):
    """RQ2: how much a schedule fixed in advance costs on the other traces.
    `make fixed` runs every candidate schedule of the proposed policy fixed for
    the whole run (the learner off), on every trace and seed. Per trace, the
    hindsight-best schedule is the one with the best mean reward; `ratio` is
    the reward of a schedule over that best reward (both negative, so 1 is the
    best and larger is worse). The proposed policy, which is told neither the
    trace nor the schedule, is placed on the same scale from summary.csv."""
    rows = []
    for dataset in util.DATASETS:
        pattern = os.path.join(args.csv_dir, args.mode, f'{dataset}_mpgts_fix_n*_bg*_s[0-9]*.csv')
        for path in sorted(glob.glob(pattern)):
            stem = os.path.basename(path)[:-len('.csv')]
            n_r, bg = stem.split('_fix_n')[1].split('_s')[0].split('_bg')
            seed = int(stem.rsplit('_s', 1)[1])
            rows.append({'dataset': dataset, 'n_r': int(n_r), 'bg': float(bg), 'seed': seed,
                         **tail_metrics(path, args.n_irsa_slot)})
    if not rows:
        print('[!] no fixed-schedule runs on disk (make fixed)'); return
    df  = pd.DataFrame(rows)
    agg = (df.groupby(['dataset', 'n_r', 'bg'], sort=False)
             .agg(n_seed=('seed', 'count'), avg_aoii=('avg_aoii', 'mean'),
                  load=('utilization', 'mean'), reward=('reward', 'mean'))
             .reset_index())
    best = agg.groupby('dataset').reward.transform('max')
    agg['ratio'] = agg.reward / best
    path = os.path.join(args.csv_dir, 'fixed_schedules.csv')
    agg.to_csv(path, index=None, float_format='%.6f')

    summary = pd.read_csv(os.path.join(args.csv_dir, 'summary.csv')).set_index(['dataset', 'solver'])
    out = []
    for dataset in util.DATASETS:
        a = agg[agg.dataset == dataset]
        if a.empty: continue
        b = a.loc[a.reward.idxmax()]
        row = {'dataset': dataset, 'best_n_r': int(b.n_r), 'best_bg': b.bg, 'best_reward': b.reward,
               'best_aoii': b.avg_aoii, 'best_load': b.load}
        # cost of carrying the schedule that is best here to every other trace
        for other in util.DATASETS:
            o = agg[(agg.dataset == other) & (agg.n_r == b.n_r) & (agg.bg == b.bg)]
            row[f'ratio_on_{other}'] = o.ratio.iloc[0] if len(o) else float('nan')
        if (dataset, 'mpgts') in summary.index:
            row['mpgts_reward'] = summary.loc[(dataset, 'mpgts'), 'reward']
            row['mpgts_ratio']  = row['mpgts_reward'] / b.reward
        out.append(row)
    path_best = os.path.join(args.csv_dir, 'fixed_best.csv')
    pd.DataFrame(out).to_csv(path_best, index=None, float_format='%.6f')

    # the single schedule with the smallest worst-case ratio over the traces
    worst = agg.groupby(['n_r', 'bg']).ratio.agg(['max', 'idxmax']).reset_index()
    worst['worst_dataset'] = agg.loc[worst['idxmax'], 'dataset'].values
    worst = worst.drop(columns='idxmax').rename(columns={'max': 'worst_ratio'}).sort_values('worst_ratio')
    path_worst = os.path.join(args.csv_dir, 'fixed_worst.csv')
    worst.to_csv(path_worst, index=None, float_format='%.6f')

    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False)); print(pd.DataFrame(out).to_string(index=False)); print(worst.to_string(index=False))
    print(f'[+] wrote {path}, {path_best} and {path_worst}')

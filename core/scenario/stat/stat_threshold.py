import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

# (tau, rho) is chosen on the first CALIB fraction of the run, while every
# policy is scored on the last 1000 slots: the choice never sees the window it
# is reported on
CALIB = 0.4

def calib_reward(path):
    df = pd.read_csv(path)
    return df['reward'].iloc[:int(CALIB * len(df))].mean()

def stat_threshold(args):
    """One (tau, rho) for the Threshold baseline on every trace. `make
    threshold` runs each pair of the grid on every trace and seed (tag
    _tune_t<tau>_r<rho>); per trace, the pairs are ranked by their mean
    calibration reward (1 = best), and the pair with the best mean rank over
    the traces is chosen, ties broken by the mean ratio of its calibration
    reward to the best one of each trace. Ranks are used because the rewards
    of the traces differ by orders of magnitude. The chosen pair is the default
    of util.py; nothing is copied."""
    rows = []
    for dataset in util.DATASETS:
        pattern = os.path.join(args.csv_dir, args.mode, f'{dataset}_threshold_tune_t*_r*_s[0-9]*.csv')
        for path in sorted(glob.glob(pattern)):
            stem = os.path.basename(path)[:-len('.csv')]
            tau, rho = stem.split('_tune_t')[1].rsplit('_s', 1)[0].split('_r')
            rows.append({'dataset': dataset, 'tau': float(tau), 'rho': float(rho),
                         'seed': int(stem.rsplit('_s', 1)[1]), 'calib_reward': calib_reward(path),
                         **tail_metrics(path, args.n_irsa_slot)})
    if not rows:
        print('[!] no threshold tuning runs on disk (make threshold)'); return
    df  = pd.DataFrame(rows)
    agg = (df.groupby(['dataset', 'tau', 'rho'], sort=False)
             .agg(n_seed=('seed', 'count'), calib_reward=('calib_reward', 'mean'),
                  avg_aoii=('avg_aoii', 'mean'), load=('utilization', 'mean'),
                  reward=('reward', 'mean')).reset_index())
    agg['rank']  = agg.groupby('dataset').calib_reward.rank(ascending=False)
    agg['ratio'] = agg.calib_reward / agg.groupby('dataset').calib_reward.transform('max')
    path = os.path.join(args.csv_dir, 'threshold_tuning.csv')
    agg.to_csv(path, index=None, float_format='%.6f')
    pairs = (agg.groupby(['tau', 'rho'])
                .agg(mean_rank=('rank', 'mean'), mean_ratio=('ratio', 'mean'))
                .reset_index().sort_values(['mean_rank', 'mean_ratio']))
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.pivot_table(index=['tau', 'rho'], columns='dataset', values='rank').to_string())
    print(pairs.to_string(index=False))
    best = pairs.iloc[0]
    print(f'[+] chosen for every trace: tau={best.tau:g} rho={best.rho:g}; wrote {path}')

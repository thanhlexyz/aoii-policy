import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

def stat_round_interval(args):
    """RQ4: how much does VHCT depend on how often the base station closes a
    feedback round? A short trace buys few rounds, which caps how far the tree
    can refine before the run ends."""
    rows = []
    for dataset in util.DATASETS:
        pattern = os.path.join(args.csv_dir, args.mode, f'{dataset}_vhct_N*_s[0-9]*.csv')
        for path in sorted(glob.glob(pattern)):
            stem = os.path.basename(path).replace('.csv', '')
            n    = int(stem.split('_N')[1].split('_s')[0])
            seed = int(stem.rsplit('_s', 1)[1])
            rows.append({'dataset': dataset, 'round_interval': n, 'seed': seed,
                         **tail_metrics(path, args.n_irsa_slot)})
    if not rows:
        print('[!] no round-interval runs on disk')
        return
    df  = pd.DataFrame(rows)
    df['n_round'] = df['n_step'] // df['round_interval']
    agg = (df.groupby(['dataset', 'round_interval'], sort=False)
             .agg(n_seed=('seed', 'count'), n_round=('n_round', 'max'),
                  avg_aoii=('avg_aoii', 'mean'),
                  utilization=('utilization', 'mean'),
                  reward=('reward', 'mean'), reward_std=('reward', 'std'))
             .reset_index())
    path = os.path.join(args.csv_dir, 'round_interval.csv')
    agg.to_csv(path, index=None, float_format='%.6f')
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False))
    print(f'[+] wrote {path}')

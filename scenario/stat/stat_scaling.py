import pandas as pd
import numpy as np
import glob
import os

import util

from .stat_final import tail_metrics

AGREE  = 0.8   # the share of devices on one schedule that counts as converged
WINDOW = 5     # rounds

def rounds_to_agree(path, round_interval):
    """First round from which the share of devices on the modal schedule stays
    at or above AGREE for WINDOW consecutive rounds (nan if never)."""
    df = pd.read_csv(path)
    if 'agree' not in df:
        return float('nan')
    a = df['agree'].groupby(df.index // round_interval).mean().to_numpy()
    for r in range(len(a) - WINDOW + 1):
        if a[r:r + WINDOW].min() >= AGREE:
            return float(r)
    return float('nan')

def stat_scaling(args):
    """Scaling in the number of sensors K on the synthetic trace, with the IRSA
    budget M = K / 6 held proportional: reward per policy, and how many rounds
    the proposed learner needs before the devices agree on one schedule."""
    rows = []
    for solver in util.SOLVERS:
        for path in sorted(glob.glob(os.path.join(args.csv_dir, args.mode, f'synthetic_{solver}_scaleK*_s[0-9]*.csv'))):
            stem = os.path.basename(path)[:-4]
            K = int(stem.split('_scaleK')[1].split('_s')[0]); seed = int(stem.rsplit('_s', 1)[1])
            m = tail_metrics(path, K // 6)
            m['rounds_to_agree'] = rounds_to_agree(path, args.round_interval)
            rows.append({'solver': solver, 'K': K, 'seed': seed, **m})
    if not rows:
        print('[!] no scaling runs on disk'); return
    df  = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.csv_dir, 'scaling_per_seed.csv'), index=None, float_format='%.6f')
    agg = (df.groupby(['solver', 'K'], sort=False)
             .agg(n_seed=('seed', 'count'), avg_aoii=('avg_aoii', 'mean'), load=('utilization', 'mean'),
                  p_success=('p_success', 'mean'), reward=('reward', 'mean'), reward_std=('reward', 'std'),
                  rounds_to_agree=('rounds_to_agree', 'mean'),
                  rounds_to_agree_std=('rounds_to_agree', 'std')).reset_index())
    path = os.path.join(args.csv_dir, 'scaling.csv')
    agg.to_csv(path, index=None, float_format='%.6f')
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False)); print(f'[+] wrote {path}')

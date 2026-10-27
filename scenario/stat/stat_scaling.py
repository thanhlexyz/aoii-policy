import pandas as pd
import numpy as np
import glob
import os

import util

from .stat_final import tail_metrics

AGREE  = 0.8   # the share of devices on one schedule that counts as converged
WINDOW = 20    # rounds

def rounds_to_agree(path, round_interval):
    """First round at which the share of devices on the modal schedule reaches
    AGREE (nan if never); the regime change later in the run re-opens the
    exploration, which the figure shows."""
    df = pd.read_csv(path)
    if 'agree' not in df:
        return float('nan')
    a = df['agree'].groupby(df.index // round_interval).mean().to_numpy()
    # the devices start together on the unplayed arms, so the share is 1
    # until the posteriors diversify them; count from the first window of
    # WINDOW rounds whose mean share is back above AGREE after that
    start = (a < AGREE).nonzero()[0]
    if not len(start):
        return 0.0
    w = np.convolve(a[start[0]:], np.ones(WINDOW) / WINDOW, mode='valid')
    above = (w >= AGREE).nonzero()[0]
    return float(start[0] + above[0]) if len(above) else float('nan')

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
            m['rounds_to_agree'] = rounds_to_agree(path, 25)
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

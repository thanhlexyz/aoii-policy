import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

VARIANTS = [('',          'full policy: TS over $(n_r, x_{\\mathrm{bg}})$'),
            ('_bgonly',   'learn $x_{\\mathrm{bg}}$ only'),
            ('_q',        'learn $q$ as well (36 schedules)'),
            ('_fixed',    'fixed schedule $(5, 0.01)$'),
            ('_norepeat', 'no repeats'),
            ('_nobg',     'no background'),
            ('_ucb',      'UCB, one arm per round'),
            ('_device',   'UCB per device'),
            ('_N25',      'round $N = 25$')]

def stat_ablation(args):
    rows = []
    for dataset in util.DATASETS:
        for tag, label in VARIANTS:
            pattern = os.path.join(args.csv_dir, args.mode, f'{dataset}_mpgts{tag}_s[0-9]*.csv')
            for path in sorted(glob.glob(pattern)):
                seed = int(path.rsplit('_s', 1)[1].split('.')[0])
                df   = pd.read_csv(path)
                m    = tail_metrics(path, args.n_irsa_slot)
                m['bg'] = df['bg'].tail(1000).mean() if 'bg' in df else float('nan')
                rows.append({'dataset': dataset, 'tag': tag, 'variant': label, 'seed': seed, **m})
    if not rows:
        print('[!] no ablation runs on disk'); return
    df  = pd.DataFrame(rows)
    agg = (df.groupby(['dataset', 'tag', 'variant'], sort=False)
             .agg(n_seed=('seed', 'count'), avg_aoii=('avg_aoii', 'mean'),
                  load=('utilization', 'mean'), bg=('bg', 'mean'),
                  reward=('reward', 'mean'), reward_std=('reward', 'std'),
                  reward_min=('reward', 'min'))
             .reset_index())
    path = os.path.join(args.csv_dir, 'ablation.csv')
    agg.to_csv(path, index=None, float_format='%.6f')
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False)); print(f'[+] wrote {path}')

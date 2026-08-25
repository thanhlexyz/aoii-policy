import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

# tag -> what the run is
VARIANTS = {'': 'VHCT (model descriptor + event trigger)',
            '_generic': 'generic descriptor (nu = 1)',
            '_norx': 'no event trigger',
            '_generic_norx': 'generic, no event trigger'}

def stat_ablation(args):
    """What the two ingredients of the proposed policy buy, each on its own."""
    rows = []
    for dataset in util.DATASETS:
        for tag, label in list(VARIANTS.items()) + [('_grid', 'grid-UCB + event trigger (adaptive hybrid)'),
                                                    ('_grid_norx', 'grid-UCB, no trigger (adaptive random)')]:
            solver = 'grid' if tag.startswith('_grid') else 'vhct'
            tag    = tag[5:] if tag.startswith('_grid') else tag
            pattern = os.path.join(args.csv_dir, args.mode, f'{dataset}_{solver}{tag}_s[0-9]*.csv')
            for path in sorted(glob.glob(pattern)):
                seed = int(path.rsplit('_s', 1)[1].split('.')[0])
                df   = pd.read_csv(path)
                m    = tail_metrics(path, args.n_irsa_slot)
                # learning curve summary: where the supplement settled
                m['p_learned'] = df['p_learned'].tail(1000).mean() if 'p_learned' in df else float('nan')
                rows.append({'dataset': dataset, 'variant': label, 'seed': seed, **m})
    if not rows:
        print('[!] no ablation runs on disk'); return
    df  = pd.DataFrame(rows)
    agg = (df.groupby(['dataset', 'variant'], sort=False)
             .agg(n_seed=('seed', 'count'), avg_aoii=('avg_aoii', 'mean'),
                  load=('utilization', 'mean'), p_learned=('p_learned', 'mean'),
                  reward=('reward', 'mean'), reward_std=('reward', 'std'))
             .reset_index())
    path = os.path.join(args.csv_dir, 'ablation.csv')
    agg.to_csv(path, index=None, float_format='%.6f')
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False)); print(f'[+] wrote {path}')

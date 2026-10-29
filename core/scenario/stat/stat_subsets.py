import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

def stat_subsets(args):
    """Robustness across sensor populations: the same policies replayed on
    three disjoint 30-stream subsets of each trace (offset in the stream
    index). Reports the per-subset ranking, which is what the robustness
    claims rest on."""
    rows = []
    for dataset in util.DATASETS:
        for path in sorted(glob.glob(os.path.join(args.csv_dir, args.mode, f'{dataset}_*_o[0-9]*_s[0-9]*.csv'))):
            stem = os.path.basename(path)[:-4]
            solver = stem.split('_')[1]
            offset = int(stem.split('_o')[1].split('_s')[0]); seed = int(stem.rsplit('_s', 1)[1])
            rows.append({'dataset': dataset, 'solver': solver, 'offset': offset, 'seed': seed,
                         **tail_metrics(path, args.n_irsa_slot)})
    if not rows:
        print('[!] no subset runs'); return
    df  = pd.DataFrame(rows)
    agg = (df.groupby(['dataset', 'offset', 'solver'], sort=False)
             .agg(n=('seed', 'count'), avg_aoii=('avg_aoii', 'mean'), load=('utilization', 'mean'),
                  reward=('reward', 'mean')).reset_index())
    agg['rank'] = agg.groupby(['dataset', 'offset'])['reward'].rank(ascending=False).astype(int)
    path = os.path.join(args.csv_dir, 'subsets.csv')
    agg.to_csv(path, index=None, float_format='%.6f')
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False))
    print('\nrank by subset (rows: solver; cols: dataset/offset):')
    print(agg.pivot_table(index='solver', columns=['dataset', 'offset'], values='rank').to_string())
    print(f'[+] wrote {path}')

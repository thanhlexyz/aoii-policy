import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

CHANNEL_TAG = {'ideal': '', 'software': '_sw', 'ota': '_ota'}

def stat_channel(args):
    """RQ5: does the policy ranking survive a real waveform? The ideal channel
    counts slot occupancy; the software and radio channels have to actually
    decode a CRC and cancel what they decoded."""
    rows = []
    for channel, tag in CHANNEL_TAG.items():
        for dataset in util.DATASETS:
            for solver in util.SOLVERS:
                pattern = os.path.join(args.csv_dir, args.mode,
                                       f'{dataset}_{solver}{tag}_s[0-9]*.csv')
                for path in sorted(glob.glob(pattern)):
                    seed = int(path.rsplit('_s', 1)[1].split('.')[0])
                    rows.append({'channel': channel, 'dataset': dataset,
                                 'solver': solver, 'seed': seed,
                                 **tail_metrics(path, args.n_irsa_slot)})
    if not rows:
        print('[!] no runs on disk')
        return
    df  = pd.DataFrame(rows)
    agg = (df.groupby(['dataset', 'solver', 'channel'], sort=False)
             .agg(n_seed=('seed', 'count'), n_step=('n_step', 'max'),
                  avg_aoii=('avg_aoii', 'mean'), avg_aoii_std=('avg_aoii', 'std'),
                  utilization=('utilization', 'mean'), utilization_std=('utilization', 'std'),
                  p_success=('p_success', 'mean'),
                  reward=('reward', 'mean'), reward_std=('reward', 'std'))
             .reset_index())
    path = os.path.join(args.csv_dir, 'channel.csv')
    agg.to_csv(path, index=None, float_format='%.6f')
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False))
    print(f'[+] wrote {path}')

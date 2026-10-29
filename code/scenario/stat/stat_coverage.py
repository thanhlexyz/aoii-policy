import pandas as pd
import os

import util

def stat_coverage(args):
    """Whether every replayed stream is still delivering readings at the end
    of the run: a stream that has ended freezes under the zero-order hold and
    would score a spurious zero in the reporting window."""
    for dataset in util.DATASETS:
        d  = os.path.join(args.data_dir, dataset)
        mn = float(open(os.path.join(d, 'min_timestamp.txt')).read())
        N  = len(pd.read_csv(os.path.join(d, 'index.csv')))
        slot, T = util.SLOT_INTERVAL[dataset], util.N_STEP[dataset]
        for offset in [0, 30, 60]:
            ends, n_events_window = [], []
            for k in range(args.n_device):
                df = pd.read_csv(os.path.join(d, f'{(k + offset) % N}.csv'))
                ts = df['timestamp'].to_numpy() - mn
                ends.append(ts[-1] / slot)
                n_events_window.append(((ts >= (T - 1000) * slot) & (ts < T * slot)).sum())
            live = sum(e >= T for e in ends)
            print(f'{dataset:9s} offset={offset:2d}: {live}/{args.n_device} streams live through slot {T}; '
                  f'shortest ends at slot {min(ends):.0f}; events per stream in the last 1000 slots: '
                  f'min {min(n_events_window)} median {sorted(n_events_window)[len(n_events_window)//2]}')

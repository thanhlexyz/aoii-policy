import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

# tag -> label, in the order the table lists them
CONFIGS = [('', 'reference'), ('_causal', 'causal scaling'), ('_eps0.005', r'$\epsilon = 0.005$'),
           ('_eps0.02', r'$\epsilon = 0.02$'), ('_alpha0.5', r'$\alpha = 0.5$'), ('_alpha2', r'$\alpha = 2$'),
           ('_K20', '$K = 20$'), ('_K40', '$K = 40$'), ('_L3', '$L = 3$'),
           ('_t1500', 'later block'), ('_t100000', 'later block')]
LOCAL    = ['reactive', 'threshold']
LEARNERS = ['grid', 'vhct']

def load_config(args, dataset, tag):
    rows = []
    for solver in util.SOLVERS:
        for path in sorted(glob.glob(os.path.join(args.csv_dir, args.mode, f'{dataset}_{solver}{tag}_s[0-9]*.csv'))):
            # K changes the slot count of the reference table: read M from args, K is in the run
            m = tail_metrics(path, args.n_irsa_slot)
            rows.append({'solver': solver, **m})
    return pd.DataFrame(rows)

def summarize(df):
    """The two numbers each robustness claim rests on: how far the local-record
    policies sit above the best learner's AoII (collapse factor), and the best
    learner's reward relative to the throughput rule's (ratio > 1 = better)."""
    g = df.groupby('solver').mean(numeric_only=True)
    if not set(LOCAL + LEARNERS + ['random']) <= set(g.index):
        return None
    best_learner_aoii = g.loc[LEARNERS, 'avg_aoii'].min()
    collapse = g.loc[LOCAL, 'avg_aoii'].min() / max(best_learner_aoii, 1e-6)
    ratio = g.loc[LEARNERS, 'reward'].max() / g.loc['random', 'reward']
    ranking = g['reward'].sort_values(ascending=False).index.tolist()
    return collapse, 1 / ratio, ranking   # reward negative: learner/throughput < 1 means learner better

def stat_robustness(args):
    lines, table = [], []
    for tag, label in CONFIGS:
        cells = []
        for dataset in util.DATASETS:
            df = load_config(args, dataset, tag)
            s = summarize(df) if len(df) else None
            if s is None:
                cells.append('--'); continue
            collapse, ratio, ranking = s
            cells.append(f'{collapse:.0f}x / {ratio:.2f}')
            lines.append(f'{label:16s} {dataset:9s} collapse={collapse:9.0f}x  learner/throughput reward ratio={ratio:.2f}  rank: ' + '>'.join(ranking))
        if any(c != '--' for c in cells):
            table.append((label, cells))
    print('\n'.join(lines))
    out = os.path.join(args.csv_dir, 'robustness.txt')
    with open(out, 'w') as fp:
        fp.write('\n'.join(lines))
    print('\nTABLE (collapse factor / learner-to-throughput reward ratio):')
    for label, cells in table:
        print(f'  {label:18s} & ' + ' & '.join(cells) + r' \\')
    print(f'[+] wrote {out}')

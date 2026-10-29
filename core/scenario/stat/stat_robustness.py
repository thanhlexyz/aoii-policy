import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

# tag -> label, in the order the figure lists them. Every policy is replayed
# under each variation (five seeds); the sensor subsets
# and the protocol checks (causal scaling, frozen learners) join the list.
CONFIGS = [('', 'reference'), ('_causal', 'causal scaling'), ('_frozen', 'frozen at 40\\%'),
           ('_eps0.005', r'$\epsilon = 0.005$'), ('_eps0.02', r'$\epsilon = 0.02$'),
           ('_alpha0.5', r'$\alpha = 0.5$'), ('_alpha2', r'$\alpha = 2$'),
           ('_K20', '$K = 20$'), ('_K40', '$K = 40$'), ('_L3', '$L = 3$'),
           ('_t1500', 'later block'), ('_t100000', 'later block'), ('_t20000', 'later block')]
SUBSETS  = {'berkeley': [30, 60], 'bts': [150, 180], 'sindit': [10, 20], 'synthetic': [30, 60]}
LOCAL    = ['reactive', 'threshold']
PROPOSED = 'mpgts'

def load_config(args, dataset, tag):
    rows = []
    for solver in util.SOLVERS:
        for path in sorted(glob.glob(os.path.join(args.csv_dir, args.mode, f'{dataset}_{solver}{tag}_s[0-9]*.csv'))):
            rows.append({'solver': solver, **tail_metrics(path, args.n_irsa_slot)})
    return pd.DataFrame(rows)

def summarize(df):
    """What each robustness claim rests on: the proposed policy's reward
    against the best baseline's (gain > 0 means it is best), its rank, and
    how far the local-record policies exceed its AoII (collapse factor)."""
    g = df.groupby('solver').mean(numeric_only=True)
    if PROPOSED not in g.index or len(g) < 2:
        return None
    baselines = g.drop(PROPOSED)
    best = baselines['reward'].idxmax()
    gain = (g.loc[PROPOSED, 'reward'] - baselines.loc[best, 'reward']) / abs(baselines.loc[best, 'reward']) * 100
    rank = int((g['reward'] > g.loc[PROPOSED, 'reward']).sum() + 1)
    local = [s for s in LOCAL if s in g.index]
    collapse = g.loc[local, 'avg_aoii'].min() / max(g.loc[PROPOSED, 'avg_aoii'], 1e-6) if local else float('nan')
    return {'mpgts_reward': g.loc[PROPOSED, 'reward'], 'best_baseline': best,
            'best_baseline_reward': baselines.loc[best, 'reward'], 'gain': gain,
            'rank': rank, 'n_policy': len(g), 'collapse': collapse,
            'mpgts_aoii': g.loc[PROPOSED, 'avg_aoii'], 'mpgts_load': g.loc[PROPOSED, 'utilization']}

def stat_robustness(args):
    """RQ3: the proposed policy under every variation of the protocol."""
    rows = []
    for dataset in util.DATASETS:
        for tag, label in CONFIGS:
            df = load_config(args, dataset, tag)
            s = summarize(df) if len(df) else None
            if s: rows.append({'dataset': dataset, 'tag': tag, 'config': label, **s})
        for offset in SUBSETS[dataset]:
            df = load_config(args, dataset, f'_o{offset}')
            s = summarize(df) if len(df) else None
            if s: rows.append({'dataset': dataset, 'tag': f'_o{offset}',
                               'config': f'streams {offset}--{offset + args.n_device - 1}', **s})
    if not rows:
        print('[!] no robustness runs on disk'); return
    out = pd.DataFrame(rows)
    path = os.path.join(args.csv_dir, 'robustness.csv')
    out.to_csv(path, index=None, float_format='%.6f')
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(out.to_string(index=False)); print(f'[+] wrote {path}')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

LABEL = {'avg_aoii'   : 'Average AoII',
         'max_aoii'   : 'Peak AoII',
         'utilization': 'Offered load $G$',
         'p_success'  : 'Success probability',
         'reward'     : 'Reward'}

SOLVER_LABEL = {'reactive': 'reactive', 'random': 'throughput',
                'hybrid'  : 'hybrid',   'threshold': 'threshold',
                'grid'    : 'grid-UCB', 'vhct'  : 'VHCT'}

def smooth(x, y, n_smooth):
    """Block-average into windows of n_smooth slots."""
    T = (len(x) // n_smooth) * n_smooth
    return x[:T].reshape(-1, n_smooth)[:, 0], y[:T].reshape(-1, n_smooth).mean(axis=1)

def metric(df, args):
    if args.metric == 'p_success':
        n_success, n_send = df['n_success'].to_numpy(), df['n_send'].to_numpy()
        return np.divide(n_success, n_send, out=np.ones_like(n_success, dtype=float),
                         where=n_send != 0)
    if args.metric == 'utilization':
        return df['n_send'].to_numpy() / args.n_irsa_slot
    return df[args.metric].to_numpy()

def load(args, dataset, solver):
    """Mean and standard deviation of one metric across every seed on disk."""
    paths = sorted(glob.glob(os.path.join(args.csv_dir, args.mode,
                                          f'{dataset}_{solver}_s*.csv')))
    if not paths:
        raise FileNotFoundError(f'no runs for {dataset}/{solver} under {args.csv_dir}')
    curves = []
    for path in paths:
        df   = pd.read_csv(path)
        x, y = smooth(df['step'].to_numpy(), metric(df, args), args.n_smooth)
        curves.append(y)
    # a shorter run would misalign the block average, so keep the common prefix
    n = min(len(y) for y in curves)
    Y = np.stack([y[:n] for y in curves])
    return x[:n], Y.mean(axis=0), Y.std(axis=0)

def save(args, path):
    plt.legend()
    plt.xlabel('Timeslot')
    plt.ylabel(LABEL.get(args.metric, args.metric))
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.cla()
    plt.clf()

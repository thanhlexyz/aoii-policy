import matplotlib.pyplot as plt
import numpy as np
import os

from simulator import TimeSeries
from .plot_panel import COLUMN_IN, STYLE
import util

N_LAG = 10

def autocorr(x, lags):
    return np.array([1.0 if l == 0 else np.corrcoef(x[l:], x[:-l])[0][1] for l in lags])

def plot_autocorr(args):
    """Autocorrelation of the replayed signal, averaged over sensors, one line
    per dataset. Replay -- not raw events -- is what the policies see."""
    plt.rcParams.update(STYLE)
    plt.figure(figsize=(COLUMN_IN, 1.45))
    for dataset in util.DATASETS:
        args.dataset       = dataset
        args.slot_interval = util.SLOT_INTERVAL[dataset]
        # replay every device for the whole run
        series = [TimeSeries(i, args) for i in range(args.n_device)]
        Y = np.array([[ts.sample(t) for ts in series] for t in range(args.n_step)])
        # average over the sensors whose signal is not constant
        Z = [z for z in (autocorr(Y[:, i], np.arange(N_LAG)) for i in range(args.n_device))
             if not np.isnan(z).any()]
        plt.plot(np.mean(Z, axis=0), '-o', markersize=2, label=dataset)
    plt.legend()
    plt.xlabel('Lag (slot)', labelpad=1)
    plt.ylabel('Autocorrelation', labelpad=1)
    plt.tight_layout(pad=0.25)
    # straight into the paper, like the simulation figures
    path = os.path.join(args.paper_figure_dir, 'system', f'{args.scenario}.pdf')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    print(f'[+] wrote {path}')

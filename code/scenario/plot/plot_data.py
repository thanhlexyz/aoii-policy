import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from simulator import TimeSeries

def plot_data(args):
    time_series = []
    max_timestamp = 0
    for i in range(args.n_device):
        ts = TimeSeries(i, args)
        time_series.append(ts)
        if ts.max_timestamp > max_timestamp:
            max_timestamp = ts.max_timestamp
    x = []
    Y = []
    for t in range(args.n_step):
        x.append(t)
        y = []
        for i in range(args.n_device):
            y.append(time_series[i].sample(t))
        Y.append(y)
    x = np.array(x)
    Y = np.array(Y)
    for i in range(args.n_device):
        plt.plot(x, Y[:, i], label=f'sensor {i}')
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.dataset}.pdf')
    plt.savefig(path)

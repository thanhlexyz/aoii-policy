import matplotlib.pyplot as plt
import os

from . import common

def plot_line_v2(args, datasets=None, solvers=None, out_dir=None):
    """One figure per solver, one line per dataset."""
    datasets = datasets or ['berkeley', 'bts', 'sindit']
    solvers  = solvers  or ['reactive', 'random', 'hybrid', 'vhct']
    out_dir  = out_dir  or args.figure_dir
    for solver in solvers:
        for dataset in datasets:
            x, y, sd = common.load(args, dataset, solver)
            line, = plt.plot(x, y, '-', label=dataset)
            plt.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=0.15, lw=0)
        name = f'plot_line_v2_{solver}_{args.metric}.pdf'
        common.save(args, os.path.join(out_dir, name))

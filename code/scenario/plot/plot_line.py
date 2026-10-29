import matplotlib.pyplot as plt
import os

from . import common

def plot_line(args, datasets=None, solvers=None, out_dir=None, suffix=''):
    """One figure per dataset, one line per solver, averaged over seeds with a
    one-standard-deviation band."""
    datasets = datasets or ['berkeley', 'bts', 'sindit']
    solvers  = solvers  or ['reactive', 'random', 'hybrid', 'vhct']
    out_dir  = out_dir  or args.figure_dir
    for dataset in datasets:
        for solver in solvers:
            x, y, sd = common.load(args, dataset, solver)
            line, = plt.plot(x, y, '-', label=common.SOLVER_LABEL[solver])
            plt.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=0.15, lw=0)
        name = f'plot_line_{dataset}_{args.metric}{suffix}.pdf'
        common.save(args, os.path.join(out_dir, name))

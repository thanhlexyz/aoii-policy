import os

from .plot_line_v2 import plot_line_v2
from .plot_panel import plot_panel
from .plot_line import plot_line

DATASETS = ['berkeley', 'bts', 'sindit']
METRICS  = ['avg_aoii', 'utilization', 'reward']
# with and without the reactive baseline, whose scale hides the other three
SOLVER_SETS = [['reactive', 'random', 'hybrid', 'threshold', 'grid', 'vhct'],
               ['random', 'hybrid', 'threshold', 'grid', 'vhct']]

def make_figures(args):
    """Regenerate every simulation figure the paper includes, straight into
    paper/figure/simulation/ under the filenames paper.tex cites."""
    out_dir = os.path.join(args.paper_figure_dir, 'simulation')
    n = 0
    # the composite rows the paper actually includes
    for dataset in DATASETS:
        print('[+] ' + plot_panel(args, dataset, out_dir=out_dir))
        n += 1
    for metric in METRICS:
        args.metric = metric
        for solvers in SOLVER_SETS:
            plot_line(args, DATASETS, solvers, out_dir, '_' + '_'.join(solvers))
            n += len(DATASETS)
        # one policy across the three traces, which is how the deck frames it
        plot_line_v2(args, DATASETS, ['vhct'], out_dir)
        n += 1
    print(f'[+] wrote {n} figures to {out_dir}')

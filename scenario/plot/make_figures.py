import os

from .plot_line_v2 import plot_line_v2
from .plot_panel import plot_panel, plot_panel_all
from .plot_line import plot_line
from .plot_ablation import plot_ablation
from .plot_robustness import plot_robustness
from .plot_ota import plot_ota
from .plot_scaling import plot_scaling

DATASETS = ['berkeley', 'bts', 'sindit', 'synthetic']
METRICS  = ['avg_aoii', 'utilization', 'reward']
# with and without the reactive baseline, whose scale hides the other three
SOLVER_SETS = [['reactive', 'random', 'hybrid', 'threshold', 'ucb', 'vhct', 'mpgts'],
               ['random', 'hybrid', 'threshold', 'ucb', 'vhct', 'mpgts']]

def make_figures(args):
    """Regenerate every simulation figure the paper includes, straight into
    paper/figure/simulation/ under the filenames paper.tex cites."""
    out_dir = os.path.join(args.paper_figure_dir, 'simulation')
    n = 0
    # RQ2-RQ4, from the csv summaries `make results` writes
    for plot in [plot_ablation, plot_robustness, plot_ota, plot_scaling]:
        try:
            plot(args, out_dir=out_dir); n += 1
        except FileNotFoundError as e:
            print(f'[!] {e}')
    # the composite rows the paper actually includes
    for dataset in DATASETS:
        print('[+] ' + plot_panel(args, dataset, solvers=SOLVER_SETS[0], out_dir=out_dir))
        n += 1
    print('[+] ' + plot_panel_all(args, solvers=SOLVER_SETS[0], out_dir=out_dir)); n += 1
    for metric in METRICS:
        args.metric = metric
        for solvers in SOLVER_SETS:
            plot_line(args, DATASETS, solvers, out_dir, '_' + '_'.join(solvers))
            n += len(DATASETS)
        # one policy across the three traces, which is how the deck frames it
        plot_line_v2(args, DATASETS, ['vhct'], out_dir)
        n += 1
    print(f'[+] wrote {n} figures to {out_dir}')

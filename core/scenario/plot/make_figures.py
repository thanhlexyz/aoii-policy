import os

from .plot_ablation import plot_ablation
from .plot_ota import plot_ota
from .plot_scaling import plot_scaling

def make_figures(args):
    """Regenerate the simulation figures of the paper (Fig. 4-6) into
    <figure_dir>/simulation/ under the filenames paper.tex includes; Table I
    is written by stat_final. `make paper` copies both into the paper."""
    out_dir = os.path.join(args.figure_dir, 'simulation')
    for plot in [plot_ablation, plot_ota, plot_scaling]:
        try:
            plot(args, out_dir=out_dir)
        except FileNotFoundError as e:
            print(f'[!] {e}')

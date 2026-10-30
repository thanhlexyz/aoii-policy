import os

from .plot_ablation import plot_ablation
from .plot_scaling import plot_scaling

def make_figure(args):
    out_dir = os.path.join(args.figure_dir, 'simulation')
    for plot in [plot_ablation, plot_scaling]:
        try:
            plot(args, out_dir=out_dir)
        except FileNotFoundError as e:
            print(f'[!] {e}')

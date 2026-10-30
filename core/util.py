import argparse
import torch
import os

DATASETS = ['synthetic', 'berkeley', 'bts', 'sindit']
SOLVERS  = ['random', 'reactive', 'hybrid', 'threshold', 'ucb', 'vhct', 'mpgts']

SLOT_INTERVAL = {'berkeley': 35, 'bts': 18, 'sindit': 3, 'synthetic': 1}

HORIZON = {'berkeley': 10000, 'bts': 3000, 'sindit': 10000, 'synthetic': 20000}

def create_folders(args):
    for folder in [args.figure_dir, args.data_dir, args.csv_dir, args.raw_dir]:
        os.makedirs(folder, exist_ok=True)

def set_default_device(args):
    torch.set_default_device(args.device)

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--scenario', type=str, default='solve')
    parser.add_argument('--mode', type=str, default='test', choices=['test'])
    parser.add_argument('--no_progress_bar', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_test_episode', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--n_sensor', type=int, default=30)
    parser.add_argument('--n_irsa_slot', type=int, default=5)
    parser.add_argument('--avg_n_replica', type=float, default=2)
    parser.add_argument('--nominal_load', type=float, default=0.55)
    parser.add_argument('--solver', type=str, default='random', choices=SOLVERS)
    parser.add_argument('--error_tolerance', type=float, default=0.01)
    parser.add_argument('--cost_weight', type=float, default=1.0)
    parser.add_argument('--supplement_scale', type=float, default=0.05)
    parser.add_argument('--round_length', type=int, default=100)
    parser.add_argument('--threshold_tau', type=float, default=2.0)
    parser.add_argument('--threshold_rho', type=float, default=1.0)
    parser.add_argument('--n_ucb_arm', type=int, default=8)
    parser.add_argument('--repeat_probability_grid', type=str, default='0.1')
    parser.add_argument('--repeat_budget_grid', type=str, default='3,5')
    parser.add_argument('--background_rate_grid', type=str, default='0,0.002,0.01,0.03')
    parser.add_argument('--learner', type=str, default='thompson', choices=['fixed', 'thompson', 'ucb_server', 'ucb_per_sensor'])
    parser.add_argument('--warm_up', type=int, default=16)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataset', type=str, default='berkeley', choices=DATASETS)
    parser.add_argument('--slot_interval', type=float, default=None)
    parser.add_argument('--stream_offset', type=int, default=0)
    parser.add_argument('--causal_norm', type=float, default=0.0)
    parser.add_argument('--start_slot', type=int, default=0)
    parser.add_argument('--freeze_after', type=float, default=0.0)
    parser.add_argument('--data_dir', type=str, default='../data/data')
    parser.add_argument('--raw_dir', type=str, default='../data/data/raw')
    parser.add_argument('--csv_dir', type=str, default='../data/csv')
    parser.add_argument('--figure_dir', type=str, default='../data/figure')
    default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device)
    args = parser.parse_args()
    set_default_device(args)
    create_folders(args)
    if args.slot_interval is None:
        args.slot_interval = SLOT_INTERVAL[args.dataset]
    if args.horizon is None:
        args.horizon = HORIZON[args.dataset]
    args.throughput_rate = args.nominal_load * args.n_irsa_slot / args.n_sensor
    return args

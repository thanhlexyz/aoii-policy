import argparse
import torch
import os

DATASETS = ['berkeley', 'bts', 'sindit']
SOLVERS  = ['random', 'reactive', 'hybrid', 'threshold', 'grid', 'vhct']
CHANNELS = ['ideal', 'software', 'ota']

# Slot duration in second, set to the measured median inter-event interval of
# each trace (scenario/stat/stat_sampling_interval_per_dataset).
SLOT_INTERVAL = {'berkeley': 35, 'bts': 18, 'sindit': 3}

# Run length in slot. Capped so the whole run replays live readings: past the
# end of a trace the zero-order hold freezes and every policy scores a
# degenerate zero. BTS covers a single day, hence the shorter run.
N_STEP = {'berkeley': 10000, 'bts': 3000, 'sindit': 10000}

def create_folders(args):
    for folder in [args.figure_dir, args.data_dir, args.csv_dir, args.raw_dir]:
        os.makedirs(folder, exist_ok=True)

def set_default_device(args):
    torch.set_default_device(args.device)

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='solve')
    parser.add_argument('--mode', type=str, default='test', choices=['test'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_progress_bar', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # simulator
    parser.add_argument('--n_test_episode', type=int, default=1)
    parser.add_argument('--n_step', type=int, default=None)
    parser.add_argument('--n_device', type=int, default=30)
    # irsa channel
    parser.add_argument('--channel', type=str, default='ideal', choices=CHANNELS)
    parser.add_argument('--n_irsa_slot', type=int, default=5)
    parser.add_argument('--avg_n_irsa_replica', type=float, default=2)
    parser.add_argument('--target_load', type=float, default=0.55)
    # solver
    parser.add_argument('--solver', type=str, default='random', choices=SOLVERS)
    parser.add_argument('--error_threshold', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--supplement_scale', type=float, default=0.05)
    parser.add_argument('--round_interval', type=int, default=100)
    # threshold policy: transmit w.p. rho once the local aoii proxy reaches tau
    parser.add_argument('--threshold_tau', type=float, default=0.1)
    parser.add_argument('--threshold_rho', type=float, default=1.0)
    # grid learner: ucb1 over this many equispaced arms on [0, p_send]
    parser.add_argument('--n_grid_arm', type=int, default=8)
    # vhct ablations: which resolution descriptor, and whether the learned
    # probability rides on top of event-triggered transmissions
    parser.add_argument('--vhct_descriptor', type=str, default='model', choices=['generic', 'model'])
    parser.add_argument('--vhct_reactive', type=int, default=1, choices=[0, 1])
    parser.add_argument('--tag', type=str, default='')
    # dataset
    parser.add_argument('--dataset', type=str, default='berkeley', choices=DATASETS)
    parser.add_argument('--slot_interval', type=float, default=None) # second
    # over-the-air link
    parser.add_argument('--ota_host', type=str, default=None)
    parser.add_argument('--ota_dir', type=str, default='~/aoii-dataset')
    parser.add_argument('--ota_python', type=str, default='~/venv/bin/python')
    # dB SNR on the software channel, pluto transmit gain on the radios
    parser.add_argument('--ota_tx_power', type=float, default=20.0)
    parser.add_argument('--ota_crowd_power', type=float, default=20.0)
    # which device holds the clean path; -1 rotates it every frame
    parser.add_argument('--ota_focus', type=int, default=-1)
    # radios. Pointing the crowd at the same uri as the device puts every
    # device on one radio, which is what a two-radio bench supports.
    parser.add_argument('--ota_tx_uri', type=str, default='ip:192.168.3.1')
    parser.add_argument('--ota_crowd_uri', type=str, default='ip:192.168.3.1')
    parser.add_argument('--ota_rx_uri', type=str, default='ip:192.168.2.1')
    # directory
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--raw_dir', type=str, default='data/raw')
    parser.add_argument('--csv_dir', type=str, default='result/csv')
    parser.add_argument('--figure_dir', type=str, default='result/figure')
    parser.add_argument('--paper_figure_dir', type=str, default='../paper/figure')
    # plot
    parser.add_argument('--metric', type=str, default='reward')
    parser.add_argument('--n_smooth', type=int, default=500)
    # other
    default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device)
    # parse args
    args = parser.parse_args()
    set_default_device(args)
    create_folders(args)
    # derived args
    if args.slot_interval is None:
        args.slot_interval = SLOT_INTERVAL[args.dataset]
    if args.n_step is None:
        args.n_step = N_STEP[args.dataset]
    args.p_send = args.target_load * args.n_irsa_slot / args.n_device
    return args

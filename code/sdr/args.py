"""OFDM/radio parameters shared by the Pi server and the software loopback."""
import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="aoii-dataset over-the-air irsa link")
    # the device under test, the aggregate of the other N-1 devices,
    # and the base station
    p.add_argument("--tx_uri", default="ip:192.168.3.1")
    p.add_argument("--crowd_uri", default="ip:192.168.4.1")
    p.add_argument("--rx_uri", default="ip:192.168.2.1")
    p.add_argument("--sample_rate", type=int, default=int(2e6))
    p.add_argument("--carrier_freq", type=int, default=int(915e6))
    p.add_argument("--tx_power", type=float, default=-25.0)
    p.add_argument("--crowd_power", type=float, default=-25.0)
    p.add_argument("--rx_gain", type=float, default=30.0)
    p.add_argument("--gain_control_mode", default="manual")
    p.add_argument("--n_clear_buffer", type=int, default=4)
    p.add_argument("--n_preamble", type=int, default=11)
    p.add_argument("--n_symbol", type=int, default=256)
    p.add_argument("--n_sps", type=int, default=1)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--n_cp", type=int, default=64)
    p.add_argument("--pilot_spacing", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_irsa_slot", type=int, default=5)
    args = p.parse_args(argv)
    args.n_data = args.n_symbol
    args.n_air_symbol = args.n_fft + args.n_cp
    return args

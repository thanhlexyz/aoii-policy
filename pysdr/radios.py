"""ADALM-Pluto radios for the 3-node testbed: TX, RX, and the adversary.

All three are driven from one Raspberry Pi over USB; their SMA ports are wired
into a combiner so the adversary superimposes on the transmitter at the
receiver. TX and the adversary run cyclic TX buffers, so once the adversary is
started it broadcasts its fixed (image/label-agnostic) perturbation waveform
continuously while the receiver captures superposed frames.

Only imported on the Pi (needs pyadi-iio); the software experiments never touch
this module.
"""
import numpy as np

try:
    import adi
except ImportError:
    adi = None

from pysdr import util


class _PlutoTx:
    """Common cyclic-buffer transmitter (used by TX and by the adversary)."""

    def __init__(self, uri, args):
        assert adi is not None, 'pyadi-iio not available (run on the Pi)'
        self.args = args
        sdr = adi.Pluto(uri)
        sdr.sample_rate = args.sample_rate
        sdr.tx_rf_bandwidth = args.sample_rate
        sdr.tx_lo = args.carrier_freq
        sdr.tx_hardwaregain_chan0 = args.tx_power
        sdr.tx_cyclic_buffer = True
        self.sdr = sdr
        self.samples = None

    def prepare(self, z, preamble=True):
        """z: complex OFDM air block. Frame it, peak-scale into the DAC."""
        args = self.args
        z = np.asarray(z, dtype=np.complex64).reshape(-1)
        if preamble:
            symbols = np.concatenate([util.get_preamble(), z]).astype(np.complex64)
        else:
            symbols = z
        if getattr(args, 'n_air_symbol', None) is not None and args.n_sps == 1:
            samples = symbols
        else:
            samples = util.apply_pulse_shaping(symbols, n_sps=args.n_sps)
        peak = float(np.max(np.abs(samples))) + 1e-12
        self.samples = (samples / peak * (2 ** 14)).astype(np.complex64)

    # the Pluto transmit gain is an attenuation
    TX_GAIN_RANGE = (-89.75, 0.0)

    def set_power(self, tx_power):
        lo, hi = self.TX_GAIN_RANGE
        if not lo <= tx_power <= hi:
            raise ValueError(f'tx gain {tx_power} dB outside the pluto range {lo}..{hi}')
        self.sdr.tx_hardwaregain_chan0 = float(tx_power)

    def transmit(self):
        self.sdr.tx(self.samples)

    def stop(self):
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass


class Transmitter(_PlutoTx):
    """The legitimate transmitter: sends the victim's channel symbols."""


class Adversary(_PlutoTx):
    """The third radio: broadcasts a perturbation waveform (no victim pilots).

    A universal, image/label-agnostic perturbation is a *fixed* waveform, so the
    adversary is prepared once and left transmitting in cyclic mode; it needs no
    per-frame synchronization to the victim.
    """

    def prepare_perturbation(self, a_air):
        # no preamble: the adversary is not a cooperative frame source
        self.prepare(a_air, preamble=False)


class Receiver:
    def __init__(self, uri, args):
        assert adi is not None, 'pyadi-iio not available (run on the Pi)'
        self.args = args
        sdr = adi.Pluto(uri)
        sdr.rx_lo = args.carrier_freq
        sdr.sample_rate = args.sample_rate
        n_payload = getattr(args, 'n_air_symbol', args.n_symbol)
        sdr.rx_buffer_size = (n_payload + args.n_preamble) * args.n_sps * 8
        sdr.rx_rf_bandwidth = args.sample_rate
        sdr.gain_control_mode_chan0 = args.gain_control_mode
        if args.gain_control_mode == 'manual':
            sdr.rx_hardwaregain_chan0 = args.rx_gain
        self.sdr = sdr

    def receive(self):
        args = self.args
        sdr = self.sdr
        preamble = util.get_preamble()
        for _ in range(args.n_clear_buffer):
            sdr.rx()
        samples = sdr.rx()
        samples = samples / (np.max(np.abs(samples)) + 1e-12)
        return util.ofdm_frame_sync(samples, preamble, args)


def create(args):
    """Return (transmitter, adversary, receiver) hardware radios."""
    tx = Transmitter(args.tx_uri, args)
    adv = Adversary(args.attacker_uri, args)
    rx = Receiver(args.rx_uri, args)
    return tx, adv, rx

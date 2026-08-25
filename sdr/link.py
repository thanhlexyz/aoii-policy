"""The radio side of one IRSA frame: transmit each slot, capture it, then let
sdr.irsa peel the collisions apart."""
import numpy as np

from sdr import irsa

class RadioIrsaLink:
    """Three ADALM-Plutos wired into a combiner at the receiver, in the usual
    mMTC testbed arrangement:

      tx_uri     one device under test, on its own propagation path
      crowd_uri  the aggregate of the other N-1 contending devices
      rx_uri     the base station

    Every replica really is on the air, so the base station decodes and cancels
    the crowd exactly as it does the device under test; what the split buys is
    a device under test whose path is genuinely independent of the interference
    it contends against. The crowd shares one path between its members, which
    is the honest limit of a three-radio bench.
    """

    def __init__(self, args):
        from pysdr.radios import Receiver, Transmitter
        self.args  = args
        self.tx    = Transmitter(args.tx_uri, args)
        # one bench radio can stand in for both roles, in which case it carries
        # the superposition of everything; opening the same pluto twice would
        # give two conflicting contexts onto one device
        self.shared = args.crowd_uri == args.tx_uri
        self.crowd = self.tx if self.shared else Transmitter(args.crowd_uri, args)
        self.rx    = Receiver(args.rx_uri, args)

    def _emit(self, radio, devices, seqs, gains, power):
        radio.stop()
        if len(devices) == 0:
            return 0.0
        z = irsa.encode_slot(devices, seqs, gains, self.args)
        radio.prepare(z)
        # prepare() peak-scales the buffer into the DAC, so emulating several
        # devices on one radio would quietly shrink each of them. Undo that, so
        # every emulated device leaves the antenna at the power a device of its
        # own would have used.
        power += irsa.aggregation_gain(z, devices[:1], seqs, gains, self.args)
        lo, hi = radio.TX_GAIN_RANGE
        clipped = min(max(power, lo), hi)
        radio.set_power(clipped)
        radio.transmit()
        return power - clipped

    def slot(self, devices, focus, seqs, gains, tx_power, crowd_power):
        if self.shared:
            self.headroom_shortfall = self._emit(self.tx, list(devices), seqs,
                                                 gains, tx_power)
            y = self.rx.receive()
            self.tx.stop()
            return y
        self._emit(self.tx, [k for k in devices if k == focus],
                   seqs, gains, tx_power)
        # the crowd radio runs out of headroom once too many devices share a
        # slot, which caps how large a crowd one radio can stand in for
        self.headroom_shortfall = self._emit(
            self.crowd, [k for k in devices if k != focus],
            seqs, gains, crowd_power)
        y = self.rx.receive()
        self.tx.stop()
        self.crowd.stop()
        return y

    def frame(self, pattern, seqs, gains, tx_power, crowd_power=None, focus=0):
        crowd_power = tx_power if crowd_power is None else crowd_power
        received = []
        for j in range(pattern.shape[1]):
            devices = np.flatnonzero(pattern[:, j])
            received.append(self.slot(devices, focus, seqs, gains,
                                      tx_power, crowd_power))
        return irsa.decode_frame(received, pattern, seqs, self.args)

    def close(self):
        self.tx.stop()
        if not self.shared:
            self.crowd.stop()

class SoftwareIrsaLink:
    """Loopback through a software channel. Exercises the same encoder, the
    same frame format and the same cancellation as the radios, so the protocol
    can be tested without hardware.

    tx_power is a dB SNR here and a Pluto transmit gain in dB on the radios;
    the calibration sweep is what maps one onto the other."""

    def __init__(self, args, seed=0):
        self.args = args
        self.rng = np.random.default_rng(seed)

    def frame(self, pattern, seqs, gains, tx_power, crowd_power=None, focus=0):
        received = []
        for j in range(pattern.shape[1]):
            devices = np.flatnonzero(pattern[:, j])
            z = irsa.encode_slot(devices, seqs, gains, self.args)
            received.append(irsa.software_channel(z, tx_power, self.rng))
        return irsa.decode_frame(received, pattern, seqs, self.args)

    def close(self):
        pass

def create_link(args, software):
    return SoftwareIrsaLink(args) if software else RadioIrsaLink(args)

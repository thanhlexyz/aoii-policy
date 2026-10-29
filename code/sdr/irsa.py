"""One IRSA frame on the air.

A frame is n_slot OFDM blocks. Every active device draws a replication
pattern, and each replica is one status-update frame carrying that device's
id, so a slot occupied by exactly one replica decodes and a slot occupied by
several does not. The receiver then runs successive interference
cancellation: a decoded device's waveform is regenerated, its channel in each
of its other slots is estimated by least squares, and the reconstruction is
subtracted before the next sweep.

This mirrors simulator.irsa.IdealIrsa exactly, except that "decodes" here
means a CRC that checks on a real -- or software-channel -- waveform rather
than a slot occupancy count.
"""
import numpy as np

from sdr import frame

def encode_slot(devices, seqs, gains, args):
    """Superpose one status-update frame per device onto a single slot.

    Only the phase differences between the devices sharing a slot are physical
    -- they are what makes a collision a collision, and what the canceller has
    to estimate. The slot's common phase is not: the transmitter prepends a
    preamble that carries no such rotation, and the receiver's sync window
    reaches back into it, so a rotated payload behind an unrotated preamble
    leaves a discontinuity the pilot phase-detrend cannot fit. Referencing
    every slot to its first device keeps the relative geometry and drops the
    common part, which the real channel supplies anyway.
    """
    devices = list(devices)
    z = np.zeros(args.n_air_symbol, dtype=np.complex64)
    if not devices:
        return z
    reference = np.exp(-1j * np.angle(gains[devices[0]]))
    for k in devices:
        z_k, _ = frame.encode(int(k), int(seqs[k]), args)
        z[:len(z_k)] += np.complex64(reference * gains[k]) * z_k
    return z

def aggregation_gain(z, one_device, seqs, gains, args):
    """dB by which a superposed slot must be boosted so each device it stands
    in for arrives at the power a single device would have used, given that
    the transmitter peak-scales whatever buffer it is handed."""
    z1 = encode_slot(one_device, seqs, gains, args)
    peak, peak1 = np.max(np.abs(z)), np.max(np.abs(z1))
    if peak1 < 1e-12 or peak < 1e-12:
        return 0.0
    return float(20 * np.log10(peak / peak1))

def project(y, z):
    """Least-squares complex gain of a known waveform inside a received slot.
    The payloads are pseudo-random and mutually uncorrelated, so an unrelated
    interferer contributes only O(1/sqrt(N)) to this estimate."""
    energy = np.vdot(z, z)
    return np.vdot(z, y) / energy if abs(energy) > 1e-12 else 0.0

def cancel(y, devices, seqs, args):
    """Regenerate each named device's waveform and subtract it from y."""
    for k in devices:
        z_k, _ = frame.encode(int(k), int(seqs[k]), args)
        y[:len(z_k)] -= project(y[:len(z_k)], z_k) * z_k
    return y

def decode_frame(received, pattern, seqs, args):
    """Successive interference cancellation over the received slots.

    received: list of n_slot complex blocks, as captured.
    pattern : (n_device, n_slot) 0/1 replication pattern.
    Returns the 0/1 success vector, plus per-slot diagnostics.
    """
    n_device, n_slot = pattern.shape
    slots   = [np.array(y, dtype=np.complex64, copy=True) for y in received]
    success = np.zeros(n_device)
    evm     = [np.nan] * n_slot
    resolved = set()
    while True:
        decoded = []
        for j in range(n_slot):
            if j in resolved:
                continue
            rx = frame.decode(slots[j], args)
            evm[j] = rx['evm_db']
            k = rx['client']
            # a slot counts as resolved only when the crc checks and the
            # header names a device that really did transmit in this slot
            if not rx['crc_ok'] or k >= n_device or pattern[k, j] == 0:
                continue
            if rx['seq'] != int(seqs[k]):
                continue
            resolved.add(j)
            if not success[k]:
                success[k] = 1
                decoded.append(k)
        if not decoded:
            return success, evm
        # peel every newly decoded device out of the slots it repeated in
        for k in decoded:
            for j in range(n_slot):
                if pattern[k, j] and j not in resolved:
                    slots[j] = cancel(slots[j], [k], seqs, args)

def software_channel(z, snr_db, rng):
    """Flat phase rotation plus AWGN, at unit mean sample power."""
    power = float(np.mean(np.abs(z) ** 2))
    if power < 1e-12:
        power = 1.0
    sigma = np.sqrt(power / (2 * 10 ** (snr_db / 10)))
    noise = rng.normal(0, sigma, z.shape) + 1j * rng.normal(0, sigma, z.shape)
    phase = np.exp(1j * rng.uniform(-np.pi / 32, np.pi / 32))
    return (phase * z + noise).astype(np.complex64)

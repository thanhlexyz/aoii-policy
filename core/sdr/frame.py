"""Status-update frame: header (client id, sequence) + CRC-16 + PRBS filler,
QPSK-mapped onto the OFDM data tones of ``pysdr.ofdm``.

A frame is *delivered* iff the decoded header matches the transmitted one and
the CRC over the whole payload checks — this is the "success" event whose
probability is p_i in Kadota et al. 2018.
"""
from __future__ import annotations

import numpy as np

from pysdr import ofdm

HEADER_BITS = 8 + 16          # client id, sequence number
CRC_BITS = 16


def crc16(bits: np.ndarray) -> int:
    """CRC-16/CCITT-FALSE over a bit array (MSB first)."""
    reg = 0xFFFF
    for b in bits:
        reg ^= int(b) << 15
        reg = ((reg << 1) ^ 0x1021) & 0xFFFF if reg & 0x8000 else (reg << 1) & 0xFFFF
    return reg


def _int_bits(x: int, n: int) -> np.ndarray:
    return np.array([(x >> (n - 1 - k)) & 1 for k in range(n)], dtype=np.uint8)


def _bits_int(bits: np.ndarray) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def payload_bits(client: int, seq: int, n_bits: int) -> np.ndarray:
    n_fill = n_bits - HEADER_BITS - CRC_BITS
    assert n_fill >= 0
    rng = np.random.RandomState((client * 65537 + seq) & 0x7FFFFFFF)
    body = np.concatenate([_int_bits(client, 8), _int_bits(seq, 16),
                           rng.randint(0, 2, size=n_fill).astype(np.uint8)])
    return np.concatenate([body, _int_bits(crc16(body), CRC_BITS)])


def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(-1, 2).astype(np.float32)
    return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])).astype(np.complex64) / np.sqrt(2)


def qpsk_demod(sym: np.ndarray) -> np.ndarray:
    bits = np.empty(2 * sym.size, dtype=np.uint8)
    bits[0::2] = (np.real(sym) < 0)
    bits[1::2] = (np.imag(sym) < 0)
    return bits


def encode(client: int, seq: int, args) -> tuple[np.ndarray, np.ndarray]:
    """Return (time-domain OFDM block, transmitted bits)."""
    bits = payload_bits(client, seq, 2 * args.n_data)
    z = ofdm.modulate(qpsk_mod(bits), args.n_fft, args.n_cp, args.pilot_spacing,
                      pilot_seed=args.seed)
    return z, bits


def decode(z_hat: np.ndarray, args) -> dict:
    """Demodulate one received block; return header, CRC verdict and EVM."""
    d_hat = ofdm.demodulate(z_hat, args.n_fft, args.n_cp, args.pilot_spacing,
                            args.n_data, pilot_seed=args.seed)
    bits = qpsk_demod(d_hat)
    body, crc = bits[:-CRC_BITS], bits[-CRC_BITS:]
    ok = crc16(body) == _bits_int(crc)
    ref = qpsk_mod(bits)
    evm = float(np.mean(np.abs(d_hat - ref) ** 2) / np.mean(np.abs(ref) ** 2))
    return dict(client=_bits_int(body[:8]), seq=_bits_int(body[8:24]), crc_ok=ok,
                evm_db=10 * np.log10(evm + 1e-12), bits=bits)


def delivered(rx: dict, client: int, seq: int) -> bool:
    return bool(rx["crc_ok"] and rx["client"] == client and rx["seq"] == seq)

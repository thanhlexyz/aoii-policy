"""Software model of the 3-radio adversarial channel.

The over-the-air testbed superimposes the transmitter and the adversary at the
receiver:  y = h * z + a + n,  where z is the victim's channel-symbol vector,
a is the adversary's injected perturbation, n is AWGN and h is a residual CSI
phase (slow Rayleigh, |h|=1 after amplitude equalization). Everything is done
on the k complex data tones, so the software channel and the real OFDM link are
described by the same equation (Sec. II of the paper).

The adversary's on-air power is measured relative to the victim's by the
perturbation-to-signal ratio  PSR = ||a||^2 / ||z||^2. Because the receiver
normalizes to unit mean tone power, we keep ||z||^2 = k and scale a to the
requested PSR just before superposition.
"""
import math

import torch


def unit_power(z, eps=1e-12):
    """Scale a complex tensor to unit mean tone power (E|z_i|^2 = 1)."""
    p = (z.abs() ** 2).mean(dim=-1, keepdim=True).clamp_min(eps).sqrt()
    return z / p


def awgn_sigma(snr_db):
    """Per-real-axis noise std for a unit-power complex symbol at snr_db."""
    snr = 10 ** (snr_db / 10.0)
    return math.sqrt(1.0 / (2.0 * snr))


def add_awgn(z, snr_db):
    sigma = awgn_sigma(snr_db)
    noise = torch.complex(torch.randn_like(z.real), torch.randn_like(z.imag)) * sigma
    return z + noise


def apply_phase(z, angle_std_deg):
    """Multiply by h = e^{j theta}, theta ~ U[-delta, delta], one per row."""
    if angle_std_deg <= 0:
        return z
    delta = math.radians(angle_std_deg)
    shape = (z.shape[0],) + (1,) * (z.ndim - 1)
    theta = (torch.rand(shape, device=z.device) * 2 - 1) * delta
    h = torch.complex(torch.cos(theta), torch.sin(theta))
    return h * z


class SoftwareChannel:
    """y = h*z + a + n on the complex data tones.

    z, a are (..., k) complex tensors already at unit mean tone power.
    """

    def __init__(self, snr_db=27.0, angle_std_deg=5.4, seed=None):
        self.snr_db = snr_db
        self.angle_std_deg = angle_std_deg
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, z, a=None):
        y = apply_phase(z, self.angle_std_deg)
        if a is not None:
            y = y + a
        y = add_awgn(y, self.snr_db)
        return y

'''Unit-power + known-TX common-phase correction (measure and OTA).'''
import numpy as np

# BEGIN: AI code


def unit(x):
    # unit *mean tone power* (not unit ℓ2-norm: that made per-tone power
    # 1/n_data and inflated the reported SNR by 10·log10(n_data/2) dB)
    x = np.asarray(x, dtype=np.complex64).reshape(-1)
    return x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)


def common_phase(d, d_hat):
    # φ = ∠ Σ d̂ conj(d) on unit-power vectors
    d = unit(d)
    d_hat = unit(d_hat)
    return float(np.angle(np.sum(d_hat * np.conj(d))))


def correct(d, d_hat):
    # unit-power both, wipe common phase → (d_u, d̂_corr, φ)
    d_u = unit(d)
    d_hat_u = unit(d_hat)
    phi = float(np.angle(np.sum(d_hat_u * np.conj(d_u))))
    d_hat_corr = (d_hat_u * np.exp(-1j * phi)).astype(np.complex64)
    return d_u, d_hat_corr, phi


def residual_tone_phases(d_u, d_hat_corr):
    # per-tone residual phase after common-phase wipe
    d_u = np.asarray(d_u, dtype=np.complex64).reshape(-1)
    d_hat_corr = np.asarray(d_hat_corr, dtype=np.complex64).reshape(-1)
    return np.angle(d_hat_corr * np.conj(d_u)).astype(np.float64)
# END: AI code

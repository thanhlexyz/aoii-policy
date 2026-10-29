import numpy as np

# BEGIN: AI code

# sync window starts this many samples early (inside the CP) so timing
# jitter shows up as a circular shift, never as a late/ICI window
TIMING_BACKOFF = 16
# CP samples to skip in the CFO estimator (may hold pre-frame content)
CFO_SKIP = TIMING_BACKOFF + 8


def pilot_indices(n_fft, pilot_spacing):
    # comb pilots on bins 0, spacing, 2*spacing, ...
    return np.arange(0, n_fft, pilot_spacing, dtype=np.int64)


def data_indices(n_fft, pilot_spacing, n_data):
    # first n_data non-pilot bins (large k=256)
    pilots = set(pilot_indices(n_fft, pilot_spacing).tolist())
    idx = [i for i in range(n_fft) if i not in pilots]
    assert len(idx) >= n_data, f'need {n_data} data bins, have {len(idx)}'
    return np.asarray(idx[:n_data], dtype=np.int64)


def known_pilots(n_pilot, seed=0):
    # deterministic unit BPSK pilots
    rng = np.random.RandomState(seed)
    bits = rng.randint(0, 2, size=n_pilot)
    return (2 * bits - 1).astype(np.complex64)


def modulate(data, n_fft, n_cp, pilot_spacing, pilot_seed=0):
    '''Map n_data complex symbols + comb pilots → time block (n_fft+n_cp,).'''
    data = np.asarray(data, dtype=np.complex64).reshape(-1)
    p_idx = pilot_indices(n_fft, pilot_spacing)
    d_idx = data_indices(n_fft, pilot_spacing, data.size)
    grid = np.zeros(n_fft, dtype=np.complex64)
    grid[p_idx] = known_pilots(p_idx.size, seed=pilot_seed)
    grid[d_idx] = data
    # unitary IFFT so Parseval stays stable
    time = np.fft.ifft(grid, n=n_fft).astype(np.complex64)
    # cyclic prefix
    block = np.concatenate([time[-n_cp:], time])
    # unit *mean* power so |z|≈1 matches Barker ±1 (‖z‖₂=1 made OFDM ~24× quieter)
    block = block / (np.sqrt(np.mean(np.abs(block) ** 2)) + 1e-12)
    return block.astype(np.complex64)


def _trim_block(time_block, n_fft, n_cp):
    # pad/truncate to one OFDM symbol with CP
    time_block = np.asarray(time_block, dtype=np.complex64).reshape(-1)
    need = n_fft + n_cp
    if time_block.size < need:
        time_block = np.pad(time_block, (0, need - time_block.size))
    elif time_block.size > need:
        time_block = time_block[:need]
    return time_block


def fft_grid(time_block, n_fft, n_cp):
    # strip CP then FFT → frequency grid
    time_block = _trim_block(time_block, n_fft, n_cp)
    time = time_block[n_cp:n_cp + n_fft]
    return np.fft.fft(time, n=n_fft).astype(np.complex64)


def _natural_freq(bins, n_fft):
    # FFT bin index → signed subcarrier index (…−2,−1,0,1,2,…)
    bins = np.asarray(bins)
    return np.where(bins >= n_fft // 2, bins - n_fft, bins)


def _interp_channel(h_pilot, p_idx, n_fft):
    # complex linear interp in *natural* frequency order (no Nyquist wrap),
    # with linear-phase detrend so timing offsets interpolate exactly
    nat_p = _natural_freq(p_idx, n_fft)
    order = np.argsort(nat_p)
    x = nat_p[order].astype(np.float64)
    h_p = h_pilot[order]
    phase = np.unwrap(np.angle(h_p))
    a = np.vstack([x, np.ones_like(x)]).T
    slope = np.linalg.lstsq(a, phase, rcond=None)[0][0]
    h_flat = h_p * np.exp(-1j * slope * x)
    nat_all = _natural_freq(np.arange(n_fft), n_fft).astype(np.float64)
    re = np.interp(nat_all, x, np.real(h_flat))
    im = np.interp(nat_all, x, np.imag(h_flat))
    return ((re + 1j * im) * np.exp(1j * slope * nat_all)).astype(np.complex64)


def equalize(y, n_fft, pilot_spacing, pilot_seed=0):
    # LS channel on pilots → complex per-bin EQ (amplitude + phase together)
    p_idx = pilot_indices(n_fft, pilot_spacing)
    x_p = known_pilots(p_idx.size, seed=pilot_seed)
    h_p = y[p_idx] / (x_p + 1e-12)
    h = _interp_channel(h_p, p_idx, n_fft)
    return (y / (h + 1e-12)).astype(np.complex64)


def integer_cfo_shift(y, n_fft, pilot_spacing, pilot_seed=0, max_shift=8):
    '''Detect integer-subcarrier CFO left after fractional (CP) correction.

    Differential pilot correlation: v_m = y[p_m+s] conj(x_m); under a timing
    slope v_m rotates by a constant per pilot, so |Σ v_{m+1} conj(v_m)| stays
    coherent only at the true shift s.
    '''
    p_idx = pilot_indices(n_fft, pilot_spacing)
    x_p = known_pilots(p_idx.size, seed=pilot_seed)
    best_shift, best_score = 0, -1.0
    for shift in range(-max_shift, max_shift + 1):
        v = y[(p_idx + shift) % n_fft] * np.conj(x_p)
        score = float(np.abs(np.sum(v[1:] * np.conj(v[:-1]))))
        if score > best_score:
            best_score, best_shift = score, shift
    return best_shift


def extract_data(y_sync, n_fft, pilot_spacing, n_data):
    # pull equalized data tones after amp EQ + phase sync
    d_idx = data_indices(n_fft, pilot_spacing, n_data)
    return y_sync[d_idx].astype(np.complex64)


def estimate_cfo_cp(time_block, n_fft, n_cp, skip=0):
    # CP correlation: ∠Σ r[cp] conj(r[tail]) / n_fft → rad/sample
    # skip: ignore the first CP samples (may hold pre-frame junk when the
    # sync window is biased early by TIMING_BACKOFF)
    time_block = _trim_block(time_block, n_fft, n_cp)
    corr = np.vdot(
        time_block[skip:n_cp],
        time_block[n_fft + skip:n_fft + n_cp],
    )
    return float(np.angle(corr) / n_fft)


def correct_cfo(time_block, cfo_rad):
    # derotate r[n] *= exp(-j * cfo * n)
    n = np.arange(time_block.size, dtype=np.float64)
    return (time_block * np.exp(-1j * cfo_rad * n)).astype(np.complex64)


def demodulate(time_block, n_fft, n_cp, pilot_spacing, n_data, pilot_seed=0):
    '''Fractional CFO (CP), FFT, integer CFO (pilots), complex pilot EQ.

    Timing offsets ≤ TIMING_BACKOFF land inside the CP (sync window is
    biased early), appear as a linear phase across bins, and are removed
    exactly by the detrended pilot interpolation — no time-domain search.
    '''
    time_block = _trim_block(time_block, n_fft, n_cp)
    cfo = estimate_cfo_cp(time_block, n_fft, n_cp, skip=CFO_SKIP)
    time_block = correct_cfo(time_block, cfo)
    y = fft_grid(time_block, n_fft, n_cp)
    shift = integer_cfo_shift(y, n_fft, pilot_spacing, pilot_seed=pilot_seed)
    if shift:
        y = np.roll(y, -shift)
    y = equalize(y, n_fft, pilot_spacing, pilot_seed=pilot_seed)
    return extract_data(y, n_fft, pilot_spacing, n_data)


def air_length(n_fft, n_cp):
    return int(n_fft + n_cp)
# END: AI code


def modulate_data_only(data, n_fft, n_cp, pilot_spacing):
    '''Adversary block: perturbation on the victim's DATA tones, no pilots.

    The adversary has no cooperative pilots, so it places its perturbation on the
    same data subcarriers the victim uses and leaves the pilot bins empty. After
    the receiver equalizes with the victim's pilots (flat combiner channel h~=1),
    this lands as an additive perturbation on the data tones -- matching the
    software channel y = h z + a + n.
    '''
    data = np.asarray(data, dtype=np.complex64).reshape(-1)
    d_idx = data_indices(n_fft, pilot_spacing, data.size)
    grid = np.zeros(n_fft, dtype=np.complex64)
    grid[d_idx] = data
    time = np.fft.ifft(grid, n=n_fft).astype(np.complex64)
    block = np.concatenate([time[-n_cp:], time])
    block = block / (np.sqrt(np.mean(np.abs(block) ** 2)) + 1e-12)
    return block.astype(np.complex64)

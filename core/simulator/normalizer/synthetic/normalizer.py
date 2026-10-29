import pandas as pd
import numpy as np
import os

# One slot per second; the run replays the first N_STEP slots and the later
# block starts at LATER_BLOCK. Every population of 30 streams has the same
# mix, so the three populations the subsets study replays are comparable.
N_SLOT       = 45000
N_POPULATION = 3
SEED         = 0
REGIME       = 10000   # slot at which the sources change behaviour
NOISE        = 0.002   # measurement noise, below the trigger threshold

# the mix of one population: (kind, count). Each kind encodes one property
# the real traces exhibit but cannot be varied on:
#   quiet  -- piecewise constant, rare heavy-tailed step changes: prolonged desynchronization
#   active -- bounded random walk crossing epsilon every few slots: most of the load
#   drift  -- slow seasonal swing plus a trend that leaves the initial range:
#             causal scaling and nonstationarity
#   burst  -- piecewise constant with transitions at times common to the group:
#             correlated triggers that collide together
MIX = [('quiet', 10), ('active', 5), ('drift', 7), ('burst', 8)]

class Normalizer:
    """Writes data/data/synthetic/ from the generator below; nothing is read."""

    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        d = os.path.join(args.data_dir, args.dataset)
        os.system(f'rm -rf "{d}"')
        os.makedirs(d)
        rng = np.random.default_rng(SEED)
        t = np.arange(N_SLOT, dtype=np.float64)
        index = {'sensor_id': [], 'n_event': [], 'description': []}
        sensor_id = 0
        for p in range(N_POPULATION):
            # one event clock per population for the burst group
            burst_times = np.flatnonzero(rng.random(N_SLOT) < rate(1 / 300, t))
            for kind, count in MIX:
                for i in range(count):
                    value = GENERATORS[kind](rng, t, burst_times)
                    value = value + NOISE * rng.standard_normal(N_SLOT)
                    pd.DataFrame({'timestamp': t, 'value': value}).to_csv(os.path.join(d, f'{sensor_id}.csv'), index=None)
                    index['sensor_id'].append(sensor_id); index['n_event'].append(N_SLOT)
                    index['description'].append(f'{kind}_{p}_{i}')
                    sensor_id += 1
        pd.DataFrame(index).to_csv(os.path.join(d, 'index.csv'), index=None)
        with open(os.path.join(d, 'min_timestamp.txt'), 'w') as fp:
            fp.write('0.0')
        print(f'[+] {sensor_id} synthetic streams of {N_SLOT} slots under {d}')

def rate(base, t):
    """Per-slot event rate, doubled from the regime change on."""
    return np.where(t < REGIME, base, 2 * base)

def jumps(rng, t, times, scale):
    """Piecewise-constant path: at each time a heavy-tailed jump of random
    sign (Pareto with tail index 1.5, scaled, clipped), reflected into [0, 1]."""
    value = np.full(len(t), 0.5)
    level = 0.5
    size  = scale * np.minimum(rng.pareto(1.5, len(times)) + 1, 20) * rng.choice([-1, 1], len(times))
    for time, s in zip(times, size):
        level = fold(level + s)
        value[time:] = level
    return value

def fold(x):
    """Reflect x into [0, 1]."""
    return 1 - abs((x % 2) - 1)

def quiet(rng, t, burst_times):
    base  = 1 / rng.uniform(300, 2000)
    times = np.flatnonzero(rng.random(len(t)) < rate(base, t))
    return jumps(rng, t, times, scale=0.05)

def active(rng, t, burst_times):
    # steps below epsilon, so the reading leaves the record every few slots
    # and the trigger load of the population stays under the IRSA budget
    sigma = rng.uniform(0.003, 0.008)
    walk  = np.cumsum(sigma * rng.standard_normal(len(t)) * np.where(t < REGIME, 1.0, 1.5))
    # fold the walk into [0, 1]
    return np.abs(((walk + 0.5) % 2) - 1)

def drift(rng, t, burst_times):
    period = rng.uniform(3000, 8000)
    phase  = rng.uniform(0, 2 * np.pi)
    trend  = rng.uniform(0.5, 1.5) / len(t)
    return 0.3 + 0.2 * np.sin(2 * np.pi * t / period + phase) + trend * t

def burst(rng, t, burst_times):
    # the group's common event times, each stream responding to most of them
    times = burst_times[rng.random(len(burst_times)) < 0.8]
    return jumps(rng, t, times, scale=0.03)

GENERATORS = {'quiet': quiet, 'active': active, 'drift': drift, 'burst': burst}

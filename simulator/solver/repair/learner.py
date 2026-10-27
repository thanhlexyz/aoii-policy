import itertools
import numpy as np
import torch
from PyXAB.algos.VHCT import VHCT

# the repair schedule theta = (q, n_r, x_bg): the probability of a re-send
# while the repeat budget lasts, the budget, and the background rate after it
PARAMS = ['repair_q', 'repair_n', 'repair_bg']

def grids(args):
    """The candidate values of each parameter, from the comma-separated
    --repair_*_grid arguments; a single value fixes that parameter."""
    return [np.array([float(v) for v in getattr(args, p + '_grid').split(',')]) for p in PARAMS]

class Learner:
    """Chooses the repair schedule of every device once per round and is told
    the per-device reward of eq. (3) at the end of it. A shared learner
    (shared = True, the base station) pools the rewards into their mean and
    broadcasts one schedule; a per-device learner keeps one instance per
    device. Freezing stops both the updates and the arm changes."""

    def __init__(self, args, shared=True):
        self.args     = args
        self.n        = 1 if shared else args.n_device
        self.learning = True
        self.grids    = grids(args)
        # before the first pull: the first value of every grid
        self.theta    = torch.tensor([g[0] for g in self.grids], dtype=torch.float32).expand(args.n_device, 3).clone()

    def pull(self, t):
        if self.learning:
            self.theta = self.choose(t).expand(self.args.n_device, 3).clone()
        return self.theta

    def feedback(self, t, reward):
        if self.learning:
            self.observe(t, reward.mean(dim=0, keepdim=True) if self.n == 1 else reward)

    def choose(self, t):
        return self.theta[:self.n]

    def observe(self, t, reward):
        pass

class Fixed(Learner):
    """The first value of every grid, never changed: the ablation's fixed
    schedule."""

class GridUCB(Learner):
    """UCB1 over the log-spaced GRID with a scale-free bonus: the exploration
    term is scaled by the pooled within-arm standard deviation of the rewards
    seen so far, the noise an arm mean carries, which keeps the bonus
    comparable to the reward gaps whatever the reward scale of the trace."""

    def __init__(self, args, shared=True):
        super().__init__(args, shared)
        # one arm per combination of the grids, in lexicographic order
        self.arms   = torch.tensor(list(itertools.product(*self.grids)), dtype=torch.float32)
        # statistics in double precision: the argmax below must not flip on
        # float32 ties
        self.n_pull = torch.zeros(self.n, len(self.arms), dtype=torch.float64)
        self.sum_r  = torch.zeros(self.n, len(self.arms), dtype=torch.float64)
        self.sum_r2 = torch.zeros(self.n, len(self.arms), dtype=torch.float64)
        self.choice = torch.zeros(self.n, dtype=torch.long)

    def noise(self):
        n = self.n_pull
        var = (self.sum_r2 - self.sum_r ** 2 / n.clamp(min=1)) / (n - 1).clamp(min=1)
        seen = n > 1
        if not seen.any():
            return torch.ones(self.n, 1, dtype=torch.float64)
        pooled = (var * seen).sum(dim=1, keepdim=True) / seen.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled.sqrt().where(seen.any(dim=1, keepdim=True), torch.ones_like(pooled))

    def choose(self, t):
        mean  = self.sum_r / self.n_pull.clamp(min=1)
        bonus = self.noise() * torch.sqrt(2 * np.log(t + 1) / self.n_pull.clamp(min=1))
        score = torch.where(self.n_pull == 0, torch.full_like(mean, float('inf')), mean + bonus)
        # ties among unpulled arms go to the lowest index, so every arm is tried once first
        self.choice = score.argmax(dim=1)
        return self.arms[self.choice]

    def observe(self, t, reward):
        k = torch.arange(self.n)
        reward = reward.to(torch.float64)
        self.n_pull[k, self.choice] += 1
        self.sum_r[k, self.choice]  += reward
        self.sum_r2[k, self.choice] += reward ** 2

class XArmed(Learner):
    """VHCT over the unit cube of the parameters whose grid spans a range, the
    X-armed alternative: an arm u in [0, 1] maps to the grid's span, on a log
    scale with repair_log (a zero lower end maps to one thousandth of the top).
    Its smoothness scale nu is set to the spread of the rewards seen so far."""

    def __init__(self, args, shared=True):
        super().__init__(args, shared)
        self.free    = [i for i, g in enumerate(self.grids) if len(g) > 1]
        self.bandits = [VHCT(domain=[[0, 1]] * len(self.free), nu=1.0, rho=0.5) for _ in range(self.n)]
        self.rewards = []

    def level(self, i, u):
        lo, hi = self.grids[i].min(), self.grids[i].max()
        if self.args.repair_log:
            lo = max(lo, hi / 1000)
            return hi * (lo / hi) ** (1 - u)
        return lo + (hi - lo) * u

    def choose(self, t):
        nu = max(float(np.std(self.rewards)) if len(self.rewards) > 1 else 1.0, 1e-4)
        theta = self.theta[:self.n].clone()
        for k, b in enumerate(self.bandits):
            b.nu = nu
            b.c1 = np.power(b.rho / (3 * nu), 1.0 / 8)
            for i, u in zip(self.free, b.pull(t)):
                theta[k, i] = self.level(i, float(u))
        return theta

    def observe(self, t, reward):
        self.rewards.append(float(reward.mean()))
        for b, r in zip(self.bandits, reward.tolist()):
            b.receive_reward(t, r)

class Thompson(GridUCB):
    """Multiple-play Thompson sampling over the same grid: the base station
    keeps one posterior per arm (Gaussian, the pooled within-arm noise as its
    scale) and draws an arm for every device from it, so a round of K devices
    yields K attributed samples, exploration is diverse while the posteriors
    are wide and concentrates on the best arm as they sharpen. The assignment
    is the broadcast; the learner has no exploration constant."""

    def __init__(self, args, shared=True):
        super().__init__(args, shared=True)
        self.n = args.n_device   # one reward stream per device, shared statistics
        self.choice = torch.zeros(self.n, dtype=torch.long)
        self.residuals = []

    def noise(self):
        """Robust scale of the per-device rewards: 1.4826 times the median
        absolute deviation of the within-arm residuals. Per-device rewards are
        a tight cluster plus the rare lockup spikes, which the standard
        deviation would let dominate and keep every posterior wide."""
        if self.args.repair_scale == 'std':
            return super().noise()
        if len(self.residuals) < 2:
            return torch.ones(1, 1, dtype=torch.float64)
        r = torch.cat(self.residuals)
        return (1.4826 * (r - r.median()).abs().median()).clamp(min=1e-6).reshape(1, 1)

    def pull(self, t):
        # once learning stops, every device exploits the posterior-best arm
        if not self.learning:
            best = (self.sum_r / self.n_pull.clamp(min=1)).squeeze(0).argmax()
            self.theta = self.arms[best].expand(self.args.n_device, 3).clone()
        return super().pull(t)

    def choose(self, t):
        mean  = (self.sum_r / self.n_pull.clamp(min=1)).squeeze(0)
        scale = (self.noise() / self.n_pull.clamp(min=1).sqrt()).squeeze(0)
        prior = self.n_pull.squeeze(0) == 0
        sample = mean + scale * torch.randn(self.n, len(self.arms), dtype=torch.float64)
        # an arm never played is drawn first, by every device that draws it
        sample = torch.where(prior, torch.full_like(sample, float('inf')) - torch.arange(len(self.arms)), sample)
        self.choice = sample.argmax(dim=1)
        return self.arms[self.choice]

    def observe(self, t, reward):
        reward = reward.to(torch.float64)
        for a in self.choice.unique():
            r = reward[self.choice == a]
            self.residuals.append(r - r.mean())
            self.n_pull[0, a] += len(r)
            self.sum_r[0, a]  += r.sum()
            self.sum_r2[0, a] += (r ** 2).sum()

LEARNERS = {'none':  lambda args: Fixed(args),
            'ts':    lambda args: Thompson(args),
            'cgrid': lambda args: GridUCB(args, shared=True),
            'grid':  lambda args: GridUCB(args, shared=False),
            'cvhct': lambda args: XArmed(args, shared=True),
            'vhct':  lambda args: XArmed(args, shared=False)}

def create_learner(args):
    return LEARNERS[args.repair_learn](args)

import itertools
import numpy as np
import torch

# the retransmission schedule theta = (q, n_r, x_bg): the probability of a re-send
# while the repeat budget lasts, the budget, and the background rate after it
PARAMS = ['mpgts_q', 'mpgts_n', 'mpgts_bg']

def grids(args):
    """The candidate values of each parameter, from the comma-separated
    --mpgts_*_grid arguments; a single value fixes that parameter."""
    return [np.array([float(v) for v in getattr(args, p + '_grid').split(',')]) for p in PARAMS]

class Learner:
    """Chooses the retransmission schedule of every device once per round and observes
    the per-device reward combining AoII and transmission cost at the end of it.
    A network-mean learner (shared = True) pools rewards into their mean and
    broadcasts one schedule; a multiple-play or decentralized learner maintains
    per-device reward attribution. Freezing stops both updates and arm switches."""

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
        """reward: this round's per-device reward."""
        if self.learning:
            self.observe(t, reward.mean(dim=0, keepdim=True) if self.n == 1 else reward)

    def choose(self, t):
        return self.theta[:self.n]

    def observe(self, t, reward):
        pass

class Fixed(Learner):
    """The first value of every grid, never changed: the ablation's fixed
    schedule."""

class UCB(Learner):
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

class Thompson(UCB):
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
        """Robust scale of per-device rewards: 1.4826 times the median absolute
        deviation of within-arm residuals. Per-device rewards exhibit a heavy-tailed
        distribution with outlier penalties from rare desynchronization events, which
        would artificially inflate standard deviation and keep posteriors overly wide."""
        if len(self.residuals) < 2:
            return torch.ones(1, 1, dtype=torch.float64)
        r = torch.cat(self.residuals)
        return (1.4826 * (r - r.median()).abs().median()).clamp(min=1e-6).reshape(1, 1)

    def pull(self, t):
        # once learning stops, every device exploits the posterior-best arm
        if not self.learning:
            best = (self.sum_r / self.n_pull.clamp(min=1)).squeeze(0).argmax()
            self.theta = self.arms[best].expand(self.args.n_device, 3).clone()
            return self.theta
        return super().pull(t)

    def choose(self, t):
        mean  = (self.sum_r / self.n_pull.clamp(min=1)).squeeze(0)
        scale = (self.noise() / self.n_pull.clamp(min=1).sqrt()).squeeze(0)
        # warm-up: an arm with fewer than mpgts_warm credited device-rounds is
        # still drawn first, spread over the devices
        prior = self.n_pull.squeeze(0) < self.args.mpgts_warm
        sample = mean + scale * torch.randn(self.n, len(self.arms), dtype=torch.float64)
        if prior.any():
            # the arms never played are drawn first and spread over the
            # devices (device k takes the k-th unplayed arm, cyclically), so
            # that one round of K devices covers up to K unplayed arms
            unplayed = prior.nonzero().flatten()
            pick = unplayed[torch.arange(self.n) % len(unplayed)]
            sample = torch.where(prior, torch.full_like(sample, float('-inf')), sample)
            sample[torch.arange(self.n), pick] = float('inf')
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
            'cucb': lambda args: UCB(args, shared=True),
            'ucb':  lambda args: UCB(args, shared=False)}

def create_learner(args):
    return LEARNERS[args.mpgts_learn](args)

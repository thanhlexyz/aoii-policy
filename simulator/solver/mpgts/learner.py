import itertools
import numpy as np
import torch
from PyXAB.algos.VHCT import VHCT

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

    def pull(self, t, stale=None):
        """stale: which devices are in the erroneous state at the start of the
        round (used by the hold credit)."""
        if self.learning:
            self.theta = self.choose(t).expand(self.args.n_device, 3).clone()
        return self.theta

    def feedback(self, t, reward, deferred=None, stats=None):
        """reward: this round's per-device reward; deferred: {(round, device):
        penalty} of AoII incurred this round by error periods that began in an
        earlier round, to be charged to the arm that device held then; stats:
        per-device transition counts of the AoII chain over the round."""
        if self.learning:
            self.observe(t, reward.mean(dim=0, keepdim=True) if self.n == 1 else reward)
            if deferred:
                self.observe_deferred(deferred)
            if stats is not None:
                self.observe_stats(t, stats)

    def choose(self, t):
        return self.theta[:self.n]

    def observe(self, t, reward):
        pass

    def observe_deferred(self, deferred):
        pass

    def observe_stats(self, t, stats):
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
        # the arm every device held in each round, for the deferred credit
        self.history = {}

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
        self.history[t] = self.choice.clone()

    def observe_deferred(self, deferred):
        # a penalty adds to the running sum of the arm the device held at the
        # onset of its error period, without counting as a new play
        for (t, k), penalty in deferred.items():
            if t in self.history:
                row = k if self.n > 1 else 0
                self.sum_r[row, self.history[t][row]] += penalty

class XArmed(Learner):
    """VHCT over the unit cube of the parameters whose grid spans a range, the
    X-armed alternative: an arm u in [0, 1] maps to the grid's span, on a log
    scale with mpgts_log (a zero lower end maps to one thousandth of the top).
    Its smoothness scale nu is set to the spread of the rewards seen so far.
    The tree has no arm to charge a deferred penalty to, so it only sees the
    per-round rewards."""

    def __init__(self, args, shared=True):
        super().__init__(args, shared)
        self.free    = [i for i, g in enumerate(self.grids) if len(g) > 1]
        self.bandits = [VHCT(domain=[[0, 1]] * len(self.free), nu=1.0, rho=0.5) for _ in range(self.n)]
        self.rewards = []

    def level(self, i, u):
        lo, hi = self.grids[i].min(), self.grids[i].max()
        if self.args.mpgts_log:
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
        # truncated mean: every reward seen so far, for the running quantile
        self.seen = []
        # model credit: per-arm counts of the two-state chain, columns as in
        # Solver.stats, and the network-wide onset rate and mismatch they share
        self.counts = torch.zeros(len(self.arms), 6, dtype=torch.float64)
        # hold credit: rounds for which each device has kept its schedule while stale
        self.held = torch.zeros(self.n, dtype=torch.long)

    def noise(self):
        """Robust scale of per-device rewards: 1.4826 times the median absolute
        deviation of within-arm residuals. Per-device rewards exhibit a heavy-tailed
        distribution with outlier penalties from rare desynchronization events, which
        would artificially inflate standard deviation and keep posteriors overly wide."""
        if self.args.mpgts_scale == 'std':
            return super().noise()
        if len(self.residuals) < 2:
            return torch.ones(1, 1, dtype=torch.float64)
        r = torch.cat(self.residuals)
        return (1.4826 * (r - r.median()).abs().median()).clamp(min=1e-6).reshape(1, 1)

    def pull(self, t, stale=None):
        # once learning stops, every device exploits the posterior-best arm
        if not self.learning:
            if self.args.mpgts_credit == 'model':
                best = self.model_sample(mean=True)[0].argmax()
            else:
                best = (self.sum_r / self.n_pull.clamp(min=1)).squeeze(0).argmax()
            self.theta = self.arms[best].expand(self.args.n_device, 3).clone()
            return self.theta
        previous = self.choice.clone()
        theta = super().pull(t)
        if self.args.mpgts_credit == 'hold' and stale is not None and t > 0:
            # a device in error keeps the schedule under which the error began
            # (for at most mpgts_hold rounds, 0 = until it recovers), so that one
            # schedule governs, and is charged for, the whole error period
            keep = stale & ((self.held < self.args.mpgts_hold) | (self.args.mpgts_hold == 0))
            self.choice = torch.where(keep, previous, self.choice)
            self.held = torch.where(keep, self.held + 1, torch.zeros_like(self.held))
            self.theta = self.arms[self.choice]
        return self.theta

    def model_sample(self, mean=False):
        """A posterior sample of the expected reward of every arm for every
        device under the two-state AoII chain: the onset rate and the mismatch
        are properties of the traffic, pooled over arms; the recovery rate and
        the transmission cost are what a schedule controls, estimated from the
        device-slots it held. Each count is tempered to one observation per
        round, and the rates get Beta posteriors."""
        c = self.counts / self.args.mpgts_temper
        tot = c.sum(dim=0)
        def beta(a, b, shape):
            a = (a + 1).expand(shape); b = (b + 1).expand(shape)
            return a / (a + b) if mean else torch.distributions.Beta(a, b).sample()
        shape = (self.n, len(self.arms))
        onset    = beta(tot[1], tot[0] - tot[1], (self.n, 1))
        recovery = beta(c[:, 3], c[:, 2] - c[:, 3], shape)
        cost     = beta(c[:, 4], c[:, 0] + c[:, 2] - c[:, 4], shape)
        delta    = tot[5] / max(float(tot[2] * self.args.mpgts_temper), 1.0)
        aoii = delta * onset / (recovery * (onset + recovery))
        return -(aoii + self.args.alpha * cost)

    def choose(self, t):
        mean  = (self.sum_r / self.n_pull.clamp(min=1)).squeeze(0)
        scale = (self.noise() / self.n_pull.clamp(min=1).sqrt()).squeeze(0)
        # warm-up: an arm with fewer than mpgts_warm credited device-rounds is
        # still drawn first, spread over the devices
        prior = self.n_pull.squeeze(0) < self.args.mpgts_warm
        if self.args.mpgts_credit == 'model':
            sample = self.model_sample()
        else:
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
        if self.args.mpgts_baseline:
            # the round's mean over the devices is a common shock (an epoch
            # of correlated losses hits every schedule at once); arms are
            # compared on their deviation from it
            reward = reward - reward.mean()
        if self.args.mpgts_clip > 0:
            # truncated mean (Bubeck et al. 2013): a reward below the running
            # mpgts_clip quantile of all rewards counts as that quantile, so a
            # rare catastrophic round stays a very bad round without burying
            # the arm's estimate under one sample
            self.seen.extend(reward.tolist())
            if len(self.seen) >= 100:
                floor = float(np.quantile(self.seen, self.args.mpgts_clip))
                reward = reward.clamp(min=floor)
        for a in self.choice.unique():
            r = reward[self.choice == a]
            self.residuals.append(r - r.mean())
            self.n_pull[0, a] += len(r)
            self.sum_r[0, a]  += r.sum()
            self.sum_r2[0, a] += (r ** 2).sum()
        self.history[t] = self.choice.clone()

    def observe_deferred(self, deferred):
        for (t, k), penalty in deferred.items():
            if t in self.history:
                self.sum_r[0, self.history[t][k]] += penalty

    def observe_stats(self, t, stats):
        self.counts.index_add_(0, self.choice, stats.to(torch.float64))

LEARNERS = {'none':  lambda args: Fixed(args),
            'ts':    lambda args: Thompson(args),
            'cucb': lambda args: UCB(args, shared=True),
            'ucb':  lambda args: UCB(args, shared=False),
            'cvhct': lambda args: XArmed(args, shared=True),
            'vhct':  lambda args: XArmed(args, shared=False)}

def create_learner(args):
    return LEARNERS[args.mpgts_learn](args)

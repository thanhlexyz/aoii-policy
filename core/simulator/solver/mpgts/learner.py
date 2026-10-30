import itertools
import numpy as np
import torch

SCHEDULE_GRIDS = ['repeat_probability_grid', 'repeat_budget_grid', 'background_rate_grid']

def grids(args):
    return [np.array([float(v) for v in getattr(args, grid).split(',')]) for grid in SCHEDULE_GRIDS]

class Learner:

    def __init__(self, args, shared=True):
        self.args     = args
        self.n        = 1 if shared else args.n_sensor
        self.learning = True
        self.grids    = grids(args)
        self.schedule    = torch.tensor([g[0] for g in self.grids], dtype=torch.float32).expand(args.n_sensor, 3).clone()

    def pull(self, t):
        if self.learning:
            self.schedule = self.choose(t).expand(self.args.n_sensor, 3).clone()
        return self.schedule

    def feedback(self, t, reward):
        if self.learning:
            self.observe(t, reward.mean(dim=0, keepdim=True) if self.n == 1 else reward)

    def choose(self, t):
        return self.schedule[:self.n]

    def observe(self, t, reward):
        pass

class FixedLearner(Learner):
    pass

class UcbLearner(Learner):

    def __init__(self, args, shared=True):
        super().__init__(args, shared)
        self.candidate_schedules   = torch.tensor(list(itertools.product(*self.grids)), dtype=torch.float32)
        self.n_credited = torch.zeros(self.n, len(self.candidate_schedules), dtype=torch.float64)
        self.reward_sum  = torch.zeros(self.n, len(self.candidate_schedules), dtype=torch.float64)
        self.reward_square_sum = torch.zeros(self.n, len(self.candidate_schedules), dtype=torch.float64)
        self.assigned = torch.zeros(self.n, dtype=torch.long)

    def noise(self):
        n = self.n_credited
        var = (self.reward_square_sum - self.reward_sum ** 2 / n.clamp(min=1)) / (n - 1).clamp(min=1)
        seen = n > 1
        if not seen.any():
            return torch.ones(self.n, 1, dtype=torch.float64)
        pooled = (var * seen).sum(dim=1, keepdim=True) / seen.sum(dim=1, keepdim=True).clamp(min=1)
        return pooled.sqrt().where(seen.any(dim=1, keepdim=True), torch.ones_like(pooled))

    def choose(self, t):
        mean  = self.reward_sum / self.n_credited.clamp(min=1)
        bonus = self.noise() * torch.sqrt(2 * np.log(t + 1) / self.n_credited.clamp(min=1))
        score = torch.where(self.n_credited == 0, torch.full_like(mean, float('inf')), mean + bonus)
        self.assigned = score.argmax(dim=1)
        return self.candidate_schedules[self.assigned]

    def observe(self, t, reward):
        k = torch.arange(self.n)
        reward = reward.to(torch.float64)
        self.n_credited[k, self.assigned] += 1
        self.reward_sum[k, self.assigned]  += reward
        self.reward_square_sum[k, self.assigned] += reward ** 2

class ThompsonLearner(UcbLearner):

    def __init__(self, args, shared=True):
        super().__init__(args, shared=True)
        self.n = args.n_sensor
        self.assigned = torch.zeros(self.n, dtype=torch.long)
        self.residuals = []

    def noise(self):
        if len(self.residuals) < 2:
            return torch.ones(1, 1, dtype=torch.float64)
        r = torch.cat(self.residuals)
        return (1.4826 * (r - r.median()).abs().median()).clamp(min=1e-6).reshape(1, 1)

    def pull(self, t):
        if not self.learning:
            best = (self.reward_sum / self.n_credited.clamp(min=1)).squeeze(0).argmax()
            self.schedule = self.candidate_schedules[best].expand(self.args.n_sensor, 3).clone()
            return self.schedule
        return super().pull(t)

    def choose(self, t):
        mean  = (self.reward_sum / self.n_credited.clamp(min=1)).squeeze(0)
        scale = (self.noise() / self.n_credited.clamp(min=1).sqrt()).squeeze(0)
        prior = self.n_credited.squeeze(0) < self.args.warm_up
        sample = mean + scale * torch.randn(self.n, len(self.candidate_schedules), dtype=torch.float64)
        if prior.any():
            unplayed = prior.nonzero().flatten()
            pick = unplayed[torch.arange(self.n) % len(unplayed)]
            sample = torch.where(prior, torch.full_like(sample, float('-inf')), sample)
            sample[torch.arange(self.n), pick] = float('inf')
        self.assigned = sample.argmax(dim=1)
        return self.candidate_schedules[self.assigned]

    def observe(self, t, reward):
        reward = reward.to(torch.float64)
        for a in self.assigned.unique():
            r = reward[self.assigned == a]
            self.residuals.append(r - r.mean())
            self.n_credited[0, a] += len(r)
            self.reward_sum[0, a]  += r.sum()
            self.reward_square_sum[0, a] += (r ** 2).sum()

LEARNERS = {'fixed':          lambda args: FixedLearner(args),
            'thompson':       lambda args: ThompsonLearner(args),
            'ucb_server':     lambda args: UcbLearner(args, shared=True),
            'ucb_per_sensor': lambda args: UcbLearner(args, shared=False)}

def create_learner(args):
    return LEARNERS[args.learner](args)

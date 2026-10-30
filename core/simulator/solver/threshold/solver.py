import torch

from ..base import BaseSolver

class ThresholdSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.local_estimate = torch.zeros(args.n_sensor)
        self.local_error_age    = torch.zeros(args.n_sensor)

    def select_action(self, observation):
        args = self.args
        env  = self.env
        local_error  = torch.abs(env.source_value - self.local_estimate)
        stale = local_error > args.error_tolerance
        self.local_error_age = (self.local_error_age + 1) * stale
        proxy = local_error * self.local_error_age
        eligible = proxy >= args.threshold_tau
        return torch.logical_and(eligible, torch.rand(args.n_sensor) < args.threshold_rho)

    def update(self):
        env    = self.env
        transmitting = env.transmitting
        self.local_estimate[transmitting] = env.source_value[transmitting]
        self.local_error_age[transmitting]    = 0

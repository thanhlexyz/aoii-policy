import torch

from ..base import BaseSolver

class HybridSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.local_estimate = torch.zeros(args.n_sensor)
        self.event     = torch.zeros(args.n_sensor, dtype=torch.bool)
        self.n_event   = 0
        self.n_total   = 0

    def supplement(self):
        args    = self.args
        p_event = self.n_event / self.n_total if self.n_total > 0 else 0
        return (args.throughput_rate - p_event) * args.supplement_scale

    def select_action(self, observation):
        args = self.args
        env  = self.env
        self.event = torch.abs(env.source_value - self.local_estimate) > args.error_tolerance
        throughput_rate = self.supplement()
        self.info_extra = {'supplement': max(throughput_rate, 0.0),
                           'p_event': self.n_event / self.n_total if self.n_total else 0.0}
        if throughput_rate <= 0:
            return self.event
        return torch.logical_or(self.event, torch.rand(args.n_sensor) < throughput_rate)

    def update(self):
        args   = self.args
        env    = self.env
        transmitting = env.transmitting
        self.local_estimate[transmitting] = env.source_value[transmitting]
        self.n_event += self.event.sum().item()
        self.n_total += args.n_sensor

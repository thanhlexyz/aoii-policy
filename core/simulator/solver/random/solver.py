import torch

from ..base import BaseSolver

class ThroughputSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        args = self.args
        action = torch.rand(args.n_sensor) < args.throughput_rate
        return action

    def update(self):
        pass

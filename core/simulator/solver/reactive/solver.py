import numpy as np
import torch

from ..base import BaseSolver

class ReactiveSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.local_estimate = self.env.source_value.clone()

    def select_action(self, observation):
        args = self.args
        env  = self.env
        action = torch.abs(env.source_value - self.local_estimate) > args.error_tolerance
        return action

    def update(self):
        env    = self.env
        transmitting = env.transmitting
        self.local_estimate[transmitting] = env.source_value[transmitting]

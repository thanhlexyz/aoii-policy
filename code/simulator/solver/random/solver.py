import torch

from ..base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def select_action(self, observation):
        # extract args
        args = self.args
        # get random action from uniform sending probability
        action = torch.rand(args.n_device) < args.p_send
        return action

    def update(self):
        pass

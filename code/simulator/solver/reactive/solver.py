import numpy as np
import torch

from ..base import BaseSolver

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.X_hat_hat = self.env.X.clone()

    def select_action(self, observation):
        # extract args
        args = self.args
        env  = self.env
        # get random action from uniform sending probability
        action = torch.abs(env.X - self.X_hat_hat) > args.error_threshold
        return action

    def update(self):
        # extract args
        env    = self.env
        active = env.active
        # update sensor estimate of server
        self.X_hat_hat[active] = env.X[active]

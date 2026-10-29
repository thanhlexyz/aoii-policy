import numpy as np
import torch

from ..base import BaseSolver
from ..hybrid.solver import Solver as Hybrid

class Solver(Hybrid):
    """The simplest learner that fits the same feedback budget as VHCT: the
    arm space [0, p_send] is cut into n_ucb_arm equispaced probabilities and
    each device runs UCB1 over them, one pull per feedback round, on the same
    per-slot reward VHCT receives. Rides on the event trigger like VHCT."""

    def __init__(self, args):
        super().__init__(args)
        self.arms    = np.linspace(0, args.p_send, args.n_ucb_arm)
        self.n_pull  = np.zeros((args.n_device, args.n_ucb_arm))
        self.sum_r   = np.zeros((args.n_device, args.n_ucb_arm))
        self.choice  = np.zeros(args.n_device, dtype=int)
        self.round   = 0
        self.p_send_learned    = torch.zeros(args.n_device)
        self.cumulative_aoii   = torch.zeros(args.n_device)
        self.cumulative_active = torch.zeros(args.n_device)
        self.info_extra = {'p_learned': 0.0}

    def pull(self):
        for k in range(self.args.n_device):
            unpulled = np.flatnonzero(self.n_pull[k] == 0)
            if len(unpulled):
                self.choice[k] = unpulled[0]
                continue
            mean  = self.sum_r[k] / self.n_pull[k]
            bonus = np.sqrt(2 * np.log(self.round + 1) / self.n_pull[k])
            self.choice[k] = int(np.argmax(mean + bonus))
        self.p_send_learned = torch.tensor(self.arms[self.choice], dtype=torch.float32)
        self.info_extra = {'p_learned': float(self.p_send_learned.mean())}

    def select_action(self, observation):
        args = self.args
        env  = self.env
        mismatch   = torch.abs(env.X - self.X_hat_hat) > args.error_threshold
        self.event = mismatch if args.ucb_reactive else torch.zeros_like(mismatch)
        # like vhct, never gated by the fixed load budget
        frozen = args.freeze_after > 0 and env.global_step >= args.freeze_after * args.n_step
        if env.global_step % args.round_interval == 0 and not frozen:
            self.pull()
        return torch.logical_or(self.event, torch.rand(args.n_device) < self.p_send_learned)

    def update(self):
        args = self.args
        env  = self.env
        super().update()
        err = torch.abs(env.X - env.X_hat)
        self.cumulative_aoii   += err * env.age
        self.cumulative_active += env.active
        if env.global_step % args.round_interval:
            return
        if args.freeze_after > 0 and env.global_step > args.freeze_after * args.n_step:
            return
        # per-slot reward of eq. (3), fed back at the end of the round
        reward = -(self.cumulative_aoii + args.alpha * self.cumulative_active) / args.round_interval
        for k in range(args.n_device):
            self.n_pull[k, self.choice[k]] += 1
            self.sum_r[k, self.choice[k]] += float(reward[k])
        self.round += 1
        self.cumulative_aoii[...]   = 0
        self.cumulative_active[...] = 0

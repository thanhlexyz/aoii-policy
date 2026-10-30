import numpy as np
import torch

from ..base import BaseSolver
from ..hybrid.solver import HybridSolver

class UcbSolver(HybridSolver):

    def __init__(self, args):
        super().__init__(args)
        self.candidate_schedules    = np.linspace(0, args.throughput_rate, args.n_ucb_arm)
        self.n_credited  = np.zeros((args.n_sensor, args.n_ucb_arm))
        self.reward_sum   = np.zeros((args.n_sensor, args.n_ucb_arm))
        self.assigned  = np.zeros(args.n_sensor, dtype=int)
        self.round   = 0
        self.p_send_learned    = torch.zeros(args.n_sensor)
        self.round_aoii   = torch.zeros(args.n_sensor)
        self.round_transmissions = torch.zeros(args.n_sensor)
        self.info_extra = {'p_learned': 0.0}

    def pull(self):
        for k in range(self.args.n_sensor):
            unpulled = np.flatnonzero(self.n_credited[k] == 0)
            if len(unpulled):
                self.assigned[k] = unpulled[0]
                continue
            mean  = self.reward_sum[k] / self.n_credited[k]
            bonus = np.sqrt(2 * np.log(self.round + 1) / self.n_credited[k])
            self.assigned[k] = int(np.argmax(mean + bonus))
        self.p_send_learned = torch.tensor(self.candidate_schedules[self.assigned], dtype=torch.float32)
        self.info_extra = {'p_learned': float(self.p_send_learned.mean())}

    def select_action(self, observation):
        args = self.args
        env  = self.env
        self.event = torch.abs(env.source_value - self.local_estimate) > args.error_tolerance
        frozen = args.freeze_after > 0 and env.global_step >= args.freeze_after * args.horizon
        if env.global_step % args.round_length == 0 and not frozen:
            self.pull()
        return torch.logical_or(self.event, torch.rand(args.n_sensor) < self.p_send_learned)

    def update(self):
        args = self.args
        env  = self.env
        super().update()
        estimation_error = torch.abs(env.source_value - env.server_estimate)
        self.round_aoii   += estimation_error * env.error_age
        self.round_transmissions += env.transmitting
        if env.global_step % args.round_length:
            return
        if args.freeze_after > 0 and env.global_step > args.freeze_after * args.horizon:
            return
        reward = -(self.round_aoii + args.cost_weight * self.round_transmissions) / args.round_length
        for k in range(args.n_sensor):
            self.n_credited[k, self.assigned[k]] += 1
            self.reward_sum[k, self.assigned[k]] += float(reward[k])
        self.round += 1
        self.round_aoii[...]   = 0
        self.round_transmissions[...] = 0

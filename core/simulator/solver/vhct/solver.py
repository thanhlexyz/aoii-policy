from PyXAB.algos.VHCT import VHCT
import numpy as np
import torch

from ..base import BaseSolver

MAX_DEPTH = 10
RHO       = 0.5

def aoii_model(x, p_event, p_correct, p_loss, delta):
    q_random  = x * (1 - p_loss)
    q_trigger = p_event * (1 - p_loss)
    r = 1 - (1 - p_correct) * (1 - q_random) * (1 - q_trigger)
    r = max(r, 1e-6)
    breaks = p_event * p_loss
    return delta * breaks / (r * (breaks + r))

def model_nu(args, p_event, p_loss, delta):
    x_bar = args.throughput_rate
    nu = 0.0
    for h in range(MAX_DEPTH + 1):
        width = x_bar * RHO ** h
        variation = (aoii_model(0.0, p_event, 0.0, p_loss, delta)
                     - aoii_model(width, p_event, 0.0, p_loss, delta)
                     + args.cost_weight * (1 - p_event) * width)
        nu = max(nu, variation / RHO ** h)
    return nu

class VhctSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.bandits = [VHCT(domain=[[0, args.throughput_rate]], nu=1.0, rho=RHO)
                        for _ in range(args.n_sensor)]
        self.round   = 0
        self.local_estimate = torch.zeros(args.n_sensor)
        self.event     = torch.zeros(args.n_sensor, dtype=torch.bool)
        self.n_event   = torch.zeros(args.n_sensor)
        self.n_total   = 0
        self.sum_delta      = torch.zeros(args.n_sensor)
        self.n_attempt      = torch.zeros(args.n_sensor)
        self.n_success      = torch.zeros(args.n_sensor)
        self.round_attempt  = torch.zeros(args.n_sensor)
        self.round_success  = torch.zeros(args.n_sensor)
        self.frozen         = False
        self.round_aoii   = torch.zeros(args.n_sensor)
        self.round_transmissions = torch.zeros(args.n_sensor)
        self.p_send_learned    = torch.zeros(args.n_sensor)
        self.info_extra = {'p_learned': 0.0, 'nu': 1.0}

    def supplement(self):
        args    = self.args
        p_event = (self.n_event.sum().item() / (self.n_total * args.n_sensor)
                   if self.n_total > 0 else 0)
        return (args.throughput_rate - p_event) * args.supplement_scale

    def set_descriptor(self):
        args = self.args
        if self.n_total == 0:
            return
        nus = []
        for k, bandit in enumerate(self.bandits):
            p_event = float(self.n_event[k]) / self.n_total
            p_loss  = 1 - float(self.n_success[k]) / max(float(self.n_attempt[k]), 1)
            delta   = float(self.sum_delta[k]) / max(float(self.n_event[k]), 1)
            if p_event <= 0 or delta <= 0:
                continue
            nu = max(model_nu(args, p_event, p_loss, delta), 1e-3)
            bandit.nu = nu
            bandit.c1 = np.power(bandit.rho / (3 * nu), 1.0 / 8)
            nus.append(nu)
        if nus:
            self.info_extra['nu'] = float(np.mean(nus))

    def select_action(self, observation):
        args = self.args
        env  = self.env
        gap        = torch.abs(env.source_value - self.local_estimate)
        mismatch   = gap > args.error_tolerance
        self.event = mismatch
        self.n_event   += mismatch
        self.sum_delta += gap * mismatch
        self.n_total   += 1
        self.frozen = args.freeze_after > 0 and env.global_step >= args.freeze_after * args.horizon
        if env.global_step % args.round_length == 0 and not self.frozen:
            self.set_descriptor()
            pulls = [bandit.pull(self.round) for bandit in self.bandits]
            self.p_send_learned = torch.tensor(pulls).reshape(args.n_sensor)
            self.info_extra['p_learned'] = float(self.p_send_learned.mean())
        return torch.logical_or(self.event, torch.rand(args.n_sensor) < self.p_send_learned)

    def update(self):
        args   = self.args
        env    = self.env
        transmitting = env.transmitting
        self.round_attempt += transmitting
        self.round_success += env.decoded
        self.local_estimate[transmitting] = env.source_value[transmitting]
        estimation_error = torch.abs(env.source_value - env.server_estimate)
        self.round_aoii   += estimation_error * env.error_age
        self.round_transmissions += transmitting
        if env.global_step % args.round_length:
            return
        self.n_attempt += self.round_attempt; self.round_attempt[...] = 0
        self.n_success += self.round_success; self.round_success[...] = 0
        if self.frozen:
            self.round_aoii[...] = 0; self.round_transmissions[...] = 0
            return
        reward = -(self.round_aoii + args.cost_weight * self.round_transmissions) / args.round_length
        for k, bandit in enumerate(self.bandits):
            bandit.receive_reward(self.round, float(reward[k]))
        self.round                 += 1
        self.round_aoii[...]   = 0
        self.round_transmissions[...] = 0

import torch

from ..base import BaseSolver
from .learner import create_learner

class MpgtsSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)
        self.learner   = create_learner(args)
        self.local_estimate = torch.zeros(args.n_sensor)
        self.repeats_left   = torch.zeros(args.n_sensor)
        self.schedule     = self.learner.schedule
        self.round     = 0
        self.round_aoii   = torch.zeros(args.n_sensor)
        self.round_transmissions = torch.zeros(args.n_sensor)
        self.info_extra = self.info()

    def info(self, n_retx=0.0):
        repeat_probability, repeat_budget, background_rate = self.schedule.mean(dim=0).tolist()
        agree = self.schedule.unique(dim=0, return_counts=True)[1].max().item() / self.args.n_sensor
        return {'n_retx': n_retx, 'q': repeat_probability, 'n_repeat': repeat_budget, 'bg': background_rate, 'agree': agree}

    def select_action(self, observation):
        args = self.args
        env  = self.env
        if env.global_step % args.round_length == 0:
            self.schedule = self.learner.pull(self.round)
        repeat_probability, repeat_budget, background_rate = self.schedule.unbind(dim=1)
        event  = torch.abs(env.source_value - self.local_estimate) > args.error_tolerance
        repeat = torch.logical_and(self.repeats_left > 0, torch.rand(args.n_sensor) < repeat_probability)
        retx = torch.logical_and(torch.logical_or(repeat, torch.rand(args.n_sensor) < background_rate), ~event)
        self.repeats_left = torch.where(event, repeat_budget, self.repeats_left)
        self.repeats_left = torch.clamp(self.repeats_left - retx.to(self.repeats_left), min=0)
        self.info_extra = self.info(float(retx.sum()))
        return torch.logical_or(event, retx)

    def update(self):
        args   = self.args
        env    = self.env
        transmitting = env.transmitting
        self.local_estimate[transmitting] = env.source_value[transmitting]
        estimation_error = torch.abs(env.source_value - env.server_estimate)
        self.round_aoii   += estimation_error * env.error_age
        self.round_transmissions += transmitting
        if env.global_step % args.round_length:
            return
        round_reward = -(self.round_aoii + args.cost_weight * self.round_transmissions) / args.round_length
        self.learner.learning = not (0 < args.freeze_after * args.horizon <= env.global_step)
        self.learner.feedback(self.round, round_reward)
        self.round += 1
        self.round_aoii[...] = 0; self.round_transmissions[...] = 0

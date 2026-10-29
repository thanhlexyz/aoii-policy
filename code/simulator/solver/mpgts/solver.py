import torch

from ..base import BaseSolver
from .learner import create_learner

class Solver(BaseSolver):
    """Event trigger with proactive retransmission and a learned background rate.

    Device side: A device transmits when its reading differs from its local
    record by more than epsilon (the event trigger). Lacking acknowledgements,
    it cannot detect collisions directly, so it executes proactive retransmission:
    it re-sends the record up to mpgts_n times, each with probability mpgts_q
    per slot. Spreading repeats across slots prevents simultaneous triggers from
    re-colliding in the next slot, while quiet devices incur no channel usage.
    Afterward, a background rate x_bg (transmitted with probability x_bg per slot)
    bounds the desynchronization duration surviving all repeats.

    The triple (mpgts_q, mpgts_n, x_bg) defines the retransmission schedule theta;
    a device transmits with probability pi_theta(m) depending on its count m
    of re-sends since the latest trigger: mpgts_q while m < mpgts_n, and x_bg
    afterward.

    Base station side: Every round_interval slots, the base station evaluates
    the per-device reward combining AoII and transmission cost, credits each
    observation to the assigned candidate schedule, and selects schedules for
    the next round via the configured bandit learner (by default, multiple-play
    Thompson sampling with a robust posterior scale over the grid candidates).
    Broadcasting the assigned schedule indices once per round serves as the sole
    downlink signaling."""

    def __init__(self, args):
        super().__init__(args)
        self.learner   = create_learner(args)
        self.X_hat_hat = torch.zeros(args.n_device)
        self.pending   = torch.zeros(args.n_device)
        self.theta     = self.learner.theta
        self.round     = 0
        self.cumulative_aoii   = torch.zeros(args.n_device)
        self.cumulative_active = torch.zeros(args.n_device)
        self.info_extra = self.info()

    def info(self, n_retx=0.0):
        q, n, bg = self.theta.mean(dim=0).tolist()
        # how far the devices agree on one schedule: the share on the modal arm
        agree = self.theta.unique(dim=0, return_counts=True)[1].max().item() / self.args.n_device
        return {'n_retx': n_retx, 'q': q, 'n_repeat': n, 'bg': bg, 'agree': agree}

    def select_action(self, observation):
        args = self.args
        env  = self.env
        if env.global_step % args.round_interval == 0:
            self.theta = self.learner.pull(self.round)
        q, n_r, bg = self.theta.unbind(dim=1)
        event  = torch.abs(env.X - self.X_hat_hat) > args.error_threshold
        repeat = torch.logical_and(self.pending > 0, torch.rand(args.n_device) < q)
        retx = torch.logical_and(torch.logical_or(repeat, torch.rand(args.n_device) < bg), ~event)
        # a trigger restarts the repeat budget; any re-send of the record,
        # scheduled repeat or background, spends one
        self.pending = torch.where(event, n_r, self.pending)
        self.pending = torch.clamp(self.pending - retx.to(self.pending), min=0)
        self.info_extra = self.info(float(retx.sum()))
        return torch.logical_or(event, retx)

    def update(self):
        args   = self.args
        env    = self.env
        active = env.active
        # the device assumes its own transmissions arrived
        self.X_hat_hat[active] = env.X[active]
        # the base station accumulates this round's reward of eq. (3)
        self.cumulative_aoii   += torch.abs(env.X - env.X_hat) * env.age
        self.cumulative_active += active
        if env.global_step % args.round_interval:
            return
        reward = -(self.cumulative_aoii + args.alpha * self.cumulative_active) / args.round_interval
        # learners stop adapting past the freeze point (0 = never)
        self.learner.learning = not (0 < args.freeze_after * args.n_step <= env.global_step)
        self.learner.feedback(self.round, reward)
        self.round += 1
        self.cumulative_aoii[...] = 0; self.cumulative_active[...] = 0

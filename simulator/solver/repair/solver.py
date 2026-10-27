import torch

from ..base import BaseSolver
from .learner import create_learner

class Solver(BaseSolver):
    """Event trigger with blind repair and a learned background rate.

    Device side. A device transmits when its reading differs from its local
    record by more than epsilon (the event trigger). With no acknowledgement it
    cannot know whether that packet collided, so it repairs blindly: it re-sends
    the record up to repair_n times, each with probability repair_q per slot,
    which spreads the repeats so a burst of simultaneous triggers does not
    collide again in the next slot. A quiet device spends no airtime. On top, a
    background rate x_bg (sent w.p. x_bg in every slot) bounds the lockup that
    survives every repeat; it is the one quantity the policy has to size per
    deployment.

    The three numbers (repair_q, repair_n, x_bg) form the repair schedule
    theta; a device transmits with probability pi_theta(m) that depends only on
    its count m of re-sends since the last trigger: repair_q while m < repair_n,
    x_bg after. Base station side. Every round_interval slots the base station
    scores the network-mean reward of eq. (3) for the schedule it broadcast and
    pulls the next one from its learner (learner.py; by default UCB1 over the
    product of the --repair_*_grid candidates): one broadcast per round is the
    only downlink signalling."""

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

    def info(self, n_repair=0.0):
        q, n, bg = self.theta.mean(dim=0).tolist()
        return {'n_repair': n_repair, 'q': q, 'n_repeat': n, 'bg': bg}

    def select_action(self, observation):
        args = self.args
        env  = self.env
        if env.global_step % args.round_interval == 0:
            self.theta = self.learner.pull(self.round)
        q, n_r, bg = self.theta.unbind(dim=1)
        event  = torch.abs(env.X - self.X_hat_hat) > args.error_threshold
        repeat = torch.logical_and(self.pending > 0, torch.rand(args.n_device) < q)
        repair = torch.logical_and(torch.logical_or(repeat, torch.rand(args.n_device) < bg), ~event)
        # a trigger restarts the repeat budget; any re-send of the record,
        # scheduled repeat or background, spends one
        self.pending = torch.where(event, n_r, self.pending)
        self.pending = torch.clamp(self.pending - repair.to(self.pending), min=0)
        self.info_extra = self.info(float(repair.sum()))
        return torch.logical_or(event, repair)

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

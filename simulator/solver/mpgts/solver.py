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
        # the round in which the current error period of every device began
        # (-1: in the correct state); the AoII of an error period is charged to
        # the schedule the device held in that round, see update()
        self.onset    = torch.full((args.n_device,), -1, dtype=torch.long)
        self.deferred = {}
        # per-device transition counts of the two-state chain over the round,
        # for the model credit: correct device-slots and onsets among them,
        # stale device-slots and recoveries among them, transmissions, and the
        # mismatch summed over stale slots
        self.stats      = torch.zeros(args.n_device, 6)
        self.prev_stale = torch.zeros(args.n_device, dtype=torch.bool)
        # local credit: the age counted from the start of the current round,
        # so that a round's score is bounded and reflects what the schedule
        # did during the round rather than the error it inherited
        self.age_local  = torch.zeros(args.n_device)
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
            self.theta = self.learner.pull(self.round, env.age > 0)
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
        # the base station accumulates this round's reward of eq. (3). The
        # transmission cost is charged to the schedule a device holds now; the
        # AoII of an error period is charged to the schedule it held when the
        # period began (onset credit), because a lost trigger under one
        # schedule leaves the device stale for many rounds, over which it is
        # re-assigned, and a per-round credit would spread the penalty over
        # schedules that merely inherited the error.
        err   = torch.abs(env.X - env.X_hat)
        stale = env.age > 0
        if env.global_step % args.round_interval == 0:
            self.age_local[...] = 0
        self.age_local = (self.age_local + 1) * stale
        aoii  = err * (self.age_local if args.mpgts_credit == 'local' else env.age)
        was   = self.prev_stale
        self.stats += torch.stack([(~was).float(), (~was & stale).float(), was.float(), (was & ~stale).float(),
                                   active.float(), err * stale], dim=1)
        self.prev_stale = stale
        self.onset = torch.where(stale, self.onset, torch.full_like(self.onset, -1))
        self.onset = torch.where(stale & (self.onset < 0), torch.full_like(self.onset, self.round), self.onset)
        if args.mpgts_credit == 'onset':
            earlier = stale & (self.onset < self.round)
            for k in torch.nonzero(earlier).flatten().tolist():
                key = (int(self.onset[k]), k)
                self.deferred[key] = self.deferred.get(key, 0.0) + float(aoii[k])
            aoii = torch.where(earlier, torch.zeros_like(aoii), aoii)
        self.cumulative_aoii   += aoii
        self.cumulative_active += active
        if env.global_step % args.round_interval:
            return
        reward   = -(self.cumulative_aoii + args.alpha * self.cumulative_active) / args.round_interval
        deferred = {key: -v / args.round_interval for key, v in self.deferred.items()}
        # learners stop adapting past the freeze point (0 = never)
        self.learner.learning = not (0 < args.freeze_after * args.n_step <= env.global_step)
        self.learner.feedback(self.round, reward, deferred, self.stats)
        self.round += 1
        self.deferred = {}
        self.stats = torch.zeros_like(self.stats)
        self.cumulative_aoii[...] = 0; self.cumulative_active[...] = 0

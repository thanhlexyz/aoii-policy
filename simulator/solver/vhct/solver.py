from PyXAB.algos.VHCT import VHCT
import numpy as np
import torch

from ..base import BaseSolver

# depth to which the model descriptor is made to dominate the per-node bound
MAX_DEPTH = 10
RHO       = 0.5   # a binary partition halves the node width at every depth

def aoii_model(x, p_event, p_correct, p_loss, delta):
    """Mean AoII per slot for the two-state chain of Sec. IV: the estimate goes
    wrong with probability p_event per slot and is repaired with probability
    r = p_correct + a(x)(1 - p_loss), where a(x) is the probability the device
    transmits in a slot under the event trigger plus a random supplement x.
    Within a wrong sojourn of geometric length, linear age at penalty delta
    accumulates delta * W(W+1)/2, which per slot averages to delta / r."""
    a = p_event + (1 - p_event) * x
    r = max(p_correct + a * (1 - p_loss), 1e-6)
    return delta * p_event / (r * (p_event + r))

def model_nu(args, p_event, p_loss, delta):
    """Resolution descriptor from the model. Both reward terms are monotone in
    x -- AoII falls and airtime rises -- so the reward variation over a node is
    the difference of the two terms' values at its endpoints. The steepest
    node at each depth is the one at x = 0; nu is the smallest constant whose
    nu * rho^h dominates that variation at every depth up to MAX_DEPTH."""
    x_bar = args.p_send
    nu = 0.0
    for h in range(MAX_DEPTH + 1):
        width = x_bar * RHO ** h
        variation = (aoii_model(0.0, p_event, 0.0, p_loss, delta)
                     - aoii_model(width, p_event, 0.0, p_loss, delta)
                     + args.alpha * (1 - p_event) * width)
        nu = max(nu, variation / RHO ** h)
    return nu

class Solver(BaseSolver):
    """Each device runs one X-armed bandit over the random supplement
    x in [0, p_send], pulled once per feedback round on the per-slot reward
    of eq. (3). With --vhct_reactive the supplement rides on top of
    event-triggered transmissions; with --vhct_descriptor=model the tree's
    resolution descriptor is set from the AoII model each round, otherwise
    PyXAB's generic nu = 1, rho = 1/2 is kept. Unlike the hybrid rule the
    learner is never gated by the fixed load budget."""

    def __init__(self, args):
        super().__init__(args)
        self.bandits = [VHCT(domain=[[0, args.p_send]], nu=1.0, rho=RHO)
                        for _ in range(args.n_device)]
        self.round   = 0
        # the device knows the server starts uninformed, and -- with no
        # acknowledgement -- can only assume its own transmissions arrived
        self.X_hat_hat = torch.zeros(args.n_device)
        self.event     = torch.zeros(args.n_device, dtype=torch.bool)
        self.n_event   = torch.zeros(args.n_device)
        self.n_total   = 0
        # what the model descriptor is estimated from, per device
        self.sum_delta  = torch.zeros(args.n_device)
        self.n_attempt  = torch.zeros(args.n_device)
        self.n_success  = torch.zeros(args.n_device)
        # per-round bandit feedback
        self.cumulative_aoii   = torch.zeros(args.n_device)
        self.cumulative_active = torch.zeros(args.n_device)
        self.p_send_learned    = torch.zeros(args.n_device)
        self.info_extra = {'p_learned': 0.0, 'nu': 1.0}

    def supplement(self):
        """Probability left over once the measured event-driven rate is
        subtracted from the target load."""
        args    = self.args
        # mean per-device event rate: n_event is per device, n_total per slot
        p_event = (self.n_event.sum().item() / (self.n_total * args.n_device)
                   if self.n_total > 0 else 0)
        return (args.p_send - p_event) * args.supplement_scale

    def set_descriptor(self):
        args = self.args
        if args.vhct_descriptor != 'model' or self.n_total == 0:
            return
        nus = []
        for k, bandit in enumerate(self.bandits):
            p_event = float(self.n_event[k]) / self.n_total
            p_loss  = 1 - float(self.n_success[k]) / max(float(self.n_attempt[k]), 1)
            delta   = float(self.sum_delta[k]) / max(float(self.n_event[k]), 1)
            # a device that has never seen an event has nothing to model yet
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
        # transmit on error, if the event trigger is on
        gap        = torch.abs(env.X - self.X_hat_hat)
        mismatch   = gap > args.error_threshold
        self.event = mismatch if args.vhct_reactive else torch.zeros_like(mismatch)
        # what the descriptor and the load budget are estimated from: the
        # trigger rate and the size of the move that fired it
        self.n_event   += mismatch
        self.sum_delta += gap * mismatch
        self.n_total   += 1
        # the learner is not gated by the fixed load budget the hybrid rule
        # uses: deciding how much airtime to add on top of the trigger is its
        # job, and the alpha term of the reward is what charges it for that
        # pull one arm per device at the start of each learning round
        if env.global_step % args.round_interval == 0:
            self.set_descriptor()
            pulls = [bandit.pull(self.round) for bandit in self.bandits]
            self.p_send_learned = torch.tensor(pulls).reshape(args.n_device)
            self.info_extra['p_learned'] = float(self.p_send_learned.mean())
        return torch.logical_or(self.event, torch.rand(args.n_device) < self.p_send_learned)

    def update(self):
        args   = self.args
        env    = self.env
        active = env.active
        # the delivery rate the base station reports in its batch feedback
        self.n_attempt += active
        self.n_success += env.success
        # update the device-side replica of the server estimate
        self.X_hat_hat[active] = env.X[active]
        # accumulate this round's reward. The device cannot see the server
        # estimate, so the aoii term is what the base station measures and
        # returns in the delayed batch feedback at the end of the round.
        err = torch.abs(env.X - env.X_hat)
        self.cumulative_aoii   += err * env.age
        self.cumulative_active += active
        if env.global_step % args.round_interval:
            return
        # per-slot reward of eq. (3)
        reward = -(self.cumulative_aoii + args.alpha * self.cumulative_active) / args.round_interval
        for k, bandit in enumerate(self.bandits):
            bandit.receive_reward(self.round, float(reward[k]))
        self.round                 += 1
        self.cumulative_aoii[...]   = 0
        self.cumulative_active[...] = 0

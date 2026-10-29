import torch

from ..base import BaseSolver

class Solver(BaseSolver):
    """Reactive sampling plus a random supplement sized to whatever load budget
    the event-driven traffic leaves unused."""

    def __init__(self, args):
        super().__init__(args)
        # the device knows the server starts uninformed, and -- with no
        # acknowledgement -- can only assume its own transmissions arrived
        self.X_hat_hat = torch.zeros(args.n_device)
        self.event     = torch.zeros(args.n_device, dtype=torch.bool)
        self.n_event   = 0
        self.n_total   = 0

    def supplement(self):
        """Probability left over once the measured event-driven rate is
        subtracted from the target load."""
        args    = self.args
        p_event = self.n_event / self.n_total if self.n_total > 0 else 0
        return (args.p_send - p_event) * args.supplement_scale

    def select_action(self, observation):
        # extract args
        args = self.args
        env  = self.env
        # transmit on error
        self.event = torch.abs(env.X - self.X_hat_hat) > args.error_threshold
        p_send = self.supplement()
        self.info_extra = {'supplement': max(p_send, 0.0),
                           'p_event': self.n_event / self.n_total if self.n_total else 0.0}
        if p_send <= 0:
            # the event traffic alone already fills the target load
            return self.event
        return torch.logical_or(self.event, torch.rand(args.n_device) < p_send)

    def update(self):
        # extract args
        args   = self.args
        env    = self.env
        active = env.active
        # update the device-side replica of the server estimate
        self.X_hat_hat[active] = env.X[active]
        # the budget is charged against the event-driven traffic only, so the
        # supplement does not crowd itself out
        self.n_event += self.event.sum().item()
        self.n_total += args.n_device

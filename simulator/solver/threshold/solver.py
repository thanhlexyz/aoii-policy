import torch

from ..base import BaseSolver

class Solver(BaseSolver):
    """Structured AoII baseline: transmit with probability rho once a local
    AoII proxy reaches tau, in the manner of the threshold policies derived
    for slotted ALOHA and IRSA remote estimation.

    The device cannot see the base station's estimate, so its proxy is built
    from its own last transmitted value: the mismatch against that value,
    times the slots elapsed since the mismatch first exceeded epsilon. With no
    acknowledgement the device assumes its transmissions arrived, exactly as
    the hybrid and learned policies do."""

    def __init__(self, args):
        super().__init__(args)
        self.X_sent = torch.zeros(args.n_device)
        self.age    = torch.zeros(args.n_device)

    def select_action(self, observation):
        args = self.args
        env  = self.env
        err  = torch.abs(env.X - self.X_sent)
        stale = err > args.error_threshold
        # local aoii proxy: age of the current mismatch, weighted by its size
        self.age = (self.age + 1) * stale
        proxy = err * self.age
        eligible = proxy >= args.threshold_tau
        return torch.logical_and(eligible, torch.rand(args.n_device) < args.threshold_rho)

    def update(self):
        env    = self.env
        active = env.active
        self.X_sent[active] = env.X[active]
        self.age[active]    = 0

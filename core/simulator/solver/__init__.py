from . import random, reactive, hybrid, threshold, ucb, vhct, mpgts

SOLVERS = {'random': random.ThroughputSolver, 'reactive': reactive.ReactiveSolver, 'hybrid': hybrid.HybridSolver,
           'threshold': threshold.ThresholdSolver, 'ucb': ucb.UcbSolver, 'vhct': vhct.VhctSolver,
           'mpgts': mpgts.MpgtsSolver}

def create_solver(args):
    return SOLVERS[args.solver](args)

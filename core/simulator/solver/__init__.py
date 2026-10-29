from . import random, reactive, hybrid, threshold, ucb, vhct, mpgts

SOLVERS = {'random': random, 'reactive': reactive, 'hybrid': hybrid,
           'threshold': threshold, 'ucb': ucb, 'vhct': vhct, 'mpgts': mpgts}

def create_solver(args):
    return SOLVERS[args.solver].Solver(args)

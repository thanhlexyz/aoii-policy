from . import random, reactive, hybrid, threshold, grid, vhct

SOLVERS = {'random': random, 'reactive': reactive, 'hybrid': hybrid,
           'threshold': threshold, 'grid': grid, 'vhct': vhct}

def create_solver(args):
    return SOLVERS[args.solver].Solver(args)

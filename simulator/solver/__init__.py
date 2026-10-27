from . import random, reactive, hybrid, threshold, grid, vhct, repair

SOLVERS = {'random': random, 'reactive': reactive, 'hybrid': hybrid,
           'threshold': threshold, 'grid': grid, 'vhct': vhct, 'repair': repair}

def create_solver(args):
    return SOLVERS[args.solver].Solver(args)

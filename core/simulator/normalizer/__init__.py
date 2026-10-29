from . import bts, berkeley, sindit, synthetic

NORMALIZERS = {'bts': bts, 'berkeley': berkeley, 'sindit': sindit, 'synthetic': synthetic}

def create_normalizer(args):
    return NORMALIZERS[args.dataset].Normalizer(args)

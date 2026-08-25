from . import bts, berkeley, sindit

NORMALIZERS = {'bts': bts, 'berkeley': berkeley, 'sindit': sindit}

def create_normalizer(args):
    return NORMALIZERS[args.dataset].Normalizer(args)

import torch

class IdealIrsa:

    def __init__(self, args):
        self.args = args

    def decode(self, pattern):
        pattern = pattern.clone()
        decoded = torch.zeros(pattern.shape[0])
        while True:
            singleton = torch.where(pattern.sum(dim=0) == 1)[0]
            if not len(singleton):
                return decoded
            j = singleton[0]
            i = torch.where(pattern[:, j] == 1)[0]
            decoded[i] = 1
            pattern[i, :] = 0

def sample_pattern(args, transmitting):
    p_replica = args.avg_n_replica / args.n_irsa_slot
    pattern   = torch.rand(args.n_sensor, args.n_irsa_slot) < p_replica
    return (transmitting[:, None] * pattern).to(torch.float32)

def create_channel(args):
    return IdealIrsa(args)

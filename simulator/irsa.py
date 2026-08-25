import torch

class IdealIrsa:
    """Collision channel with ideal successive interference cancellation.

    A slot carrying exactly one replica is decoded; that device's remaining
    replicas are then erased and the sweep repeats. Slots that stay collided
    when no singleton is left are lost -- there is no retransmission and no
    per-packet acknowledgement.
    """

    def __init__(self, args):
        self.args = args

    def decode(self, pattern):
        pattern = pattern.clone()
        success = torch.zeros(pattern.shape[0])
        while True:
            singleton = torch.where(pattern.sum(dim=0) == 1)[0]
            if not len(singleton):
                return success
            j = singleton[0]
            i = torch.where(pattern[:, j] == 1)[0]
            success[i] = 1
            pattern[i, :] = 0

def sample_pattern(args, active):
    """Draw one IRSA replication pattern per active device: each device picks
    each of the n_irsa_slot slots independently, so it sends
    avg_n_irsa_replica replicas on average."""
    p_replica = args.avg_n_irsa_replica / args.n_irsa_slot
    pattern   = torch.rand(args.n_device, args.n_irsa_slot) < p_replica
    return (active[:, None] * pattern).to(torch.float32)

def create_channel(args):
    if args.channel == 'ideal':
        return IdealIrsa(args)
    # the waveform backends pull in the sdr package, which the ideal path does
    # not need -- and which the ideal path must not require torch-free hosts to
    # install
    from sdr.channel import OtaIrsa
    return OtaIrsa(args, software=args.channel == 'software')

"""Simulator-side client for the slot server.

Env asks this for the same thing simulator.irsa.IdealIrsa provides -- which
devices got through this slot -- but the answer comes from a waveform that
went through a software channel or a real radio pair.
"""
import numpy as np
import subprocess
import torch
import json
import sys

class SlotServer:
    """One JSON-lines pipe, either to a local --software process or over a
    single ssh connection to the machine holding the radios."""

    def __init__(self, args, software):
        if software:
            cmd = [sys.executable, '-m', 'sdr', '--software',
                   '--n_irsa_slot', str(args.n_irsa_slot)]
        else:
            remote = (f'cd {args.ota_dir} && {args.ota_python} -m sdr '
                      f'--n_irsa_slot {args.n_irsa_slot} '
                      f'--tx_uri {args.ota_tx_uri} '
                      f'--crowd_uri {args.ota_crowd_uri} '
                      f'--rx_uri {args.ota_rx_uri}')
            cmd = ['ssh', args.ota_host, remote]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        line = self.proc.stdout.readline().strip()
        if line != 'READY':
            raise RuntimeError(f'slot server did not start: {line!r}')

    def call(self, **req):
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        try:
            self.call(cmd='quit')
        except Exception:
            pass
        self.proc.wait(timeout=30)

class OtaIrsa:
    """Drop-in replacement for simulator.irsa.IdealIrsa backed by a waveform."""

    def __init__(self, args, software=True):
        self.args   = args
        self.server = SlotServer(args, software)
        self.seq    = np.zeros(args.n_device, dtype=int)
        # per-device complex gain: the two radios cannot give thirty devices
        # thirty propagation paths, so the per-device path is synthesized here
        rng = np.random.default_rng(args.seed)
        phase = rng.uniform(-np.pi, np.pi, args.n_device)
        self.gain = np.exp(1j * phase)
        self.frame_index = 0

    def decode(self, pattern):
        p = pattern.cpu().numpy()
        # only one radio carries the device under test, so rotate the role and
        # every device is eventually measured on the clean path
        focus = (self.args.ota_focus if self.args.ota_focus >= 0
                 else self.frame_index % self.args.n_device)
        r = self.server.call(cmd='frame', pattern=p.tolist(),
                             seq=self.seq.tolist(),
                             gain_re=self.gain.real.tolist(),
                             gain_im=self.gain.imag.tolist(),
                             tx_power=self.args.ota_tx_power,
                             crowd_power=self.args.ota_crowd_power,
                             focus=focus)
        self.frame_index += 1
        # a fresh reading gets a fresh sequence number
        self.seq = (self.seq + p.max(axis=1).astype(int)) % 65536
        return torch.tensor(r['success'], dtype=torch.float32)

    def close(self):
        self.server.close()

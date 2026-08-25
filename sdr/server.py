"""Slot server, run on the Pi that owns the radios (or locally with
--software).

Protocol: JSON lines on stdin -> JSON lines on stdout.
  {"cmd":"frame","pattern":[[0/1]...],"seq":[..],"gain_re":[..],"gain_im":[..],
   "tx_power":g,"crowd_power":g,"focus":k}
      -> {"success":[0/1..],"evm_db":[..]}
  {"cmd":"calib","powers":[..],"n":N,"n_device":K}
      -> {"powers":[..],"rate":[..],"evm_db":[..]}
  {"cmd":"quit"}
Prints READY once the radios are configured.
"""
import numpy as np
import json
import sys

from sdr.args import parse_args
from sdr.link import create_link

def calibrate(link, powers, n, n_device, args):
    """Delivery rate against transmit power for the device under test,
    transmitting alone -- the collision-free reference every IRSA number is
    read against, and what maps a Pluto gain onto a software dB SNR."""
    rates, evms = [], []
    gains = np.ones(n_device)
    for power in powers:
        ok, ev = [], []
        for seq in range(n):
            pattern = np.zeros((n_device, args.n_irsa_slot))
            pattern[0, seq % args.n_irsa_slot] = 1
            seqs = np.full(n_device, seq % 65536)
            success, evm = link.frame(pattern, seqs, gains, power, power, 0)
            ok.append(float(success[0]))
            ev.append(float(np.nanmean(evm)))
        rates.append(sum(ok) / len(ok))
        evms.append(sum(ev) / len(ev))
    return dict(powers=list(powers), rate=rates, evm_db=evms)

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    software = '--software' in argv
    if software:
        argv.remove('--software')
    args = parse_args(argv)
    link = create_link(args, software)
    out = sys.stdout
    print('READY', flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req['cmd']
        if cmd == 'quit':
            break
        if cmd == 'frame':
            gain = np.array(req['gain_re']) + 1j * np.array(req['gain_im'])
            success, evm = link.frame(np.array(req['pattern'], dtype=float),
                                      np.array(req['seq'], dtype=int),
                                      gain, float(req['tx_power']),
                                      float(req.get('crowd_power', req['tx_power'])),
                                      int(req.get('focus', 0)))
            r = dict(success=[float(s) for s in success],
                     evm_db=[None if np.isnan(e) else float(e) for e in evm])
        elif cmd == 'calib':
            r = calibrate(link, req['powers'], int(req['n']),
                          int(req['n_device']), args)
        else:
            r = dict(error=f'unknown cmd {cmd}')
        out.write(json.dumps(r) + '\n')
        out.flush()
    link.close()

if __name__ == '__main__':
    main()

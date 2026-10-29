import matplotlib.pyplot as plt
import pandas as pd
import os

from sdr.channel import SlotServer

# tx_power means different things on the two backends, which is the whole
# point of calibrating: a dB SNR in the software channel, and the Pluto
# transmit hardware gain (an attenuation, so never positive) on the radios.
POWERS  = {'software': [-6, -3, 0, 3, 6, 9, 12, 15, 18, 21],
           'ota'     : [-60, -50, -40, -35, -30, -25, -20, -15, -10, -5]}
N_FRAME = 40

def ota_calib(args):
    """Delivery rate for a lone device against transmit power: the
    collision-free reference every IRSA number is read against, and the curve
    that maps a Pluto transmit gain onto the software channel's dB SNR."""
    server = SlotServer(args, software=args.channel == 'software')
    try:
        r = server.call(cmd='calib', powers=POWERS[args.channel], n=N_FRAME,
                        n_device=args.n_device)
    finally:
        server.close()
    df   = pd.DataFrame({'tx_power_db': r['powers'], 'rate': r['rate'],
                         'evm_db': r['evm_db']})
    path = os.path.join(args.csv_dir, f'ota_calib_{args.channel}.csv')
    df.to_csv(path, index=None, float_format='%.4f')
    print(df.to_string(index=False))
    # and the figure the paper shows
    fig, ax = plt.subplots()
    ax.plot(df['tx_power_db'], df['rate'], '-o', color='tab:blue')
    ax.set_xlabel('Transmit power (dB)')
    ax.set_ylabel('Delivery rate', color='tab:blue')
    ax.set_ylim(-0.05, 1.05)
    ax2 = ax.twinx()
    ax2.plot(df['tx_power_db'], df['evm_db'], '--s', color='tab:red')
    ax2.set_ylabel('EVM (dB)', color='tab:red')
    fig.tight_layout()
    out = os.path.join(args.paper_figure_dir, 'simulation',
                       f'plot_ota_calib_{args.channel}.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f'[+] wrote {path} and {out}')

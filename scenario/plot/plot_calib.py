import matplotlib.pyplot as plt
import pandas as pd
import os

from .plot_panel import COLUMN_IN, STYLE

LABEL = {'software': 'software channel', 'ota': 'ADALM-Pluto'}

def plot_calib(args):
    """Both calibration sweeps on one axes. Transmit power means a dB SNR in
    software and a Pluto transmit gain on the radios, so the two curves share
    an axis only in the sense that each locates its own operating plateau."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.15))
        for ax, channel in zip(axes, ['software', 'ota']):
            path = os.path.join(args.csv_dir, f'ota_calib_{channel}.csv')
            if not os.path.exists(path):
                print(f'[!] missing {path}')
                continue
            df = pd.read_csv(path)
            ax.plot(df['tx_power_db'], df['rate'], '-o', markersize=2,
                    color='tab:blue')
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('Transmit power (dB)', labelpad=1)
            ax.set_title(LABEL[channel], pad=2)
            ax.locator_params(axis='x', nbins=4)
            ax2 = ax.twinx()
            ax2.plot(df['tx_power_db'], df['evm_db'], '--s', markersize=2,
                     color='tab:red')
            ax2.tick_params(axis='y', colors='tab:red')
            ax2.locator_params(axis='y', nbins=4)
        axes[0].set_ylabel('Delivery', color='tab:blue', labelpad=1)
        fig.text(0.995, 0.55, 'EVM (dB)', color='tab:red', rotation=90,
                 va='center', ha='right', fontsize=6)
        fig.tight_layout(pad=0.2, w_pad=1.4, rect=(0, 0, 0.97, 1))
        out = os.path.join(args.paper_figure_dir, 'simulation', 'plot_calib.pdf')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
    print(f'[+] wrote {out}')

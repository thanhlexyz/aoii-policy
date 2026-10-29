import pandas as pd
import os

pd.set_option('display.float_format', '{:.2f}'.format)

def stat_autocorr_per_dataset(args):
    path = os.path.join(args.csv_dir, 'stat', f'autocorr_{args.dataset}.csv')
    df   = pd.read_csv(path)
    print(f'[{args.dataset}] {len(df)} sensors')
    print(df.agg(['min', 'max', 'mean', 'std', 'median']))

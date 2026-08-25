import pandas as pd
import glob
import os

import util

from .stat_final import tail_metrics

# (tau, rho) is chosen on the first CALIB fraction of the run and reported,
# like every other policy, on the last 1000 slots -- disjoint windows, so the
# choice never sees the interval it is scored on
CALIB = 0.4

def calib_reward(path):
    df = pd.read_csv(path)
    return df['reward'].iloc[:int(CALIB * len(df))].mean()

def stat_threshold(args):
    """The threshold baseline is tuned per dataset over (tau, rho) on a
    calibration window disjoint from the reporting window."""
    rows = []
    for dataset in util.DATASETS:
        pattern = os.path.join(args.csv_dir, args.mode, f'{dataset}_threshold_t*_s[0-9]*.csv')
        for path in sorted(glob.glob(pattern)):
            stem = os.path.basename(path).replace('.csv', '')
            cfg  = stem.split('_threshold_')[1].rsplit('_s', 1)[0]      # t<tau>_r<rho>
            tau  = float(cfg.split('_r')[0][1:]); rho = float(cfg.split('_r')[1])
            seed = int(stem.rsplit('_s', 1)[1])
            rows.append({'dataset': dataset, 'tau': tau, 'rho': rho, 'seed': seed,
                         'calib_reward': calib_reward(path),
                         **tail_metrics(path, args.n_irsa_slot)})
    if not rows:
        print('[!] no threshold sweep on disk'); return
    df  = pd.DataFrame(rows)
    agg = (df.groupby(['dataset', 'tau', 'rho'], sort=False)
             .agg(n_seed=('seed', 'count'), calib_reward=('calib_reward', 'mean'),
                  avg_aoii=('avg_aoii', 'mean'),
                  load=('utilization', 'mean'), reward=('reward', 'mean'),
                  reward_std=('reward', 'std')).reset_index())
    best = agg.loc[agg.groupby('dataset')['calib_reward'].idxmax()]
    path = os.path.join(args.csv_dir, 'threshold_sweep.csv')
    agg.to_csv(path, index=None, float_format='%.6f')
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(agg.to_string(index=False)); print('\nbest per dataset:'); print(best.to_string(index=False))
    print(f'[+] wrote {path}')
    # promote the best configuration to the name the main tables read
    import shutil
    for _, row in best.iterrows():
        cfg = f"_t{row.tau:g}_r{row.rho:g}"
        for src in glob.glob(os.path.join(args.csv_dir, args.mode, f'{row.dataset}_threshold{cfg}_s[0-9]*.csv')):
            dst = src.replace(f'_threshold{cfg}_', '_threshold_')
            shutil.copyfile(src, dst)
        print(f'[+] {row.dataset}: threshold tau={row.tau:g} rho={row.rho:g} promoted')

import pandas as pd
import os

MIN_EVENT = 100

def stat_autocorr_per_sensor(args):
    # load index
    d     = os.path.join(args.data_dir, args.dataset)
    index = pd.read_csv(os.path.join(d, 'index.csv'))
    lags  = range(1, 10)
    print(f'sensor_id,{",".join("autocorr_"+str(lag) for lag in lags)}')
    for sensor_id in index['sensor_id']:
        # load normalized sensor data
        df = pd.read_csv(os.path.join(d, f'{sensor_id}.csv'))
        if len(df) < MIN_EVENT:
            continue
        autocorrs = [f'{df["value"].autocorr(lag):0.2f}' for lag in lags]
        print(f'{sensor_id},{",".join(autocorrs)}')

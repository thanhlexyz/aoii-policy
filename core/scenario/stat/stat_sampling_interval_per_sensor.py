import pandas as pd
import os

MIN_EVENT = 100

def stat_sampling_interval_per_sensor(args):
    d     = os.path.join(args.data_dir, args.dataset)
    index = pd.read_csv(os.path.join(d, 'index.csv'))
    print('sensor_id,min_diff,max_diff,mean_diff,std_diff,median_diff')
    for sensor_id in index['sensor_id']:
        df = pd.read_csv(os.path.join(d, f'{sensor_id}.csv'))
        if len(df) < MIN_EVENT:
            continue
        diff = df['timestamp'].diff()
        print(f'{sensor_id},{diff.min():0.2f},{diff.max():0.2f},'
              f'{diff.mean():0.2f},{diff.std():0.2f},{diff.median():0.2f}')

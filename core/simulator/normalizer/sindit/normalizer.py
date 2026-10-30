import itertools as it
import pandas as pd
import numpy as np
import tqdm
import os

class Normalizer:

    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        raw_d = os.path.join(args.raw_dir, args.dataset)
        index_data = {'sensor_id': [],
                      'n_event': [],
                      'description': []}
        d = os.path.join(args.data_dir, args.dataset)
        os.system(f'rm -rf "{d}"')
        os.makedirs(d)
        sensor_id = 0
        min_timestamp = np.inf
        print(f'{len(os.listdir(raw_d))} sensors')
        for name in tqdm.tqdm(os.listdir(raw_d)):
            path = os.path.join(raw_d, name)
            df1 = pd.read_csv(path)
            description = name.replace('.csv', '')
            df2 = pd.DataFrame({'timestamp': df1['time'], 'value': df1['value'].astype(float)})
            df2 = df2.dropna(subset=['value'])
            df2['timestamp'] = pd.to_datetime(df2['timestamp'], format='mixed')
            df2 = df2.sort_values(by='timestamp')
            if len(df2) > 0:
                df2['timestamp'] = df2['timestamp'].apply(lambda x: x.timestamp())
                path = os.path.join(d, f'{sensor_id}.csv')
                df2.to_csv(path, index=None)
                index_data['sensor_id'].append(sensor_id)
                index_data['n_event'].append(len(df2))
                index_data['description'].append(description)
                sensor_id += 1
                if df2['timestamp'].min() < min_timestamp:
                    min_timestamp = df2['timestamp'].min()
        df = pd.DataFrame(index_data)
        path = os.path.join(d, 'index.csv')
        df.to_csv(path, index=None)
        path = os.path.join(d, 'min_timestamp.txt')
        with open(path, 'w') as fp:
            fp.write(f'{min_timestamp}')

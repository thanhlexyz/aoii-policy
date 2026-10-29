import itertools as it
import pandas as pd
import numpy as np
import tqdm
import os

class Normalizer:

    def __init__(self, args):
        # save args
        self.args = args

    def run(self):
        # extract args
        args = self.args
        # load raw data
        path = os.path.join(args.raw_dir, args.dataset, 'raw.csv')
        df = pd.read_csv(path)
        # get list of sensors
        mote_ids = df['moteid'].unique()
        columns  = ['temperature', 'humidity', 'light', 'voltage']
        print(f'{len(mote_ids)}x{len(columns)}={len(mote_ids)*len(columns)} sensors')
        # initialize global data
        index_data = {'sensor_id': [],
                      'mote_id': [],
                      'n_event': [],
                      'description': []}
        # reset data dir
        d = os.path.join(args.data_dir, args.dataset)
        os.system(f'rm -rf "{d}"')
        os.makedirs(d)
        # loop
        sensor_id = 0
        min_timestamp = np.inf
        for mote_id, column in tqdm.tqdm(it.product(mote_ids, columns)):
            # extract and save subframe
            df1 = df[df['moteid'] == mote_id]
            df2 = pd.DataFrame({'timestamp': df1['timestamp'], 'value': df1[column].astype(float)})
            df2 = df2.dropna(subset=['value'])
            df2['timestamp'] = pd.to_datetime(df2['timestamp'], format='mixed')
            df2 = df2.sort_values(by='timestamp')
            if len(df2) > 0:
                # convert timestamp to float
                df2['timestamp'] = df2['timestamp'].apply(lambda x: x.timestamp())
                # save subframe to csv
                path = os.path.join(d, f'{sensor_id}.csv')
                df2.to_csv(path, index=None)
                # add global data to index.csv
                index_data['sensor_id'].append(sensor_id)
                index_data['mote_id'].append(mote_id)
                index_data['n_event'].append(len(df2))
                index_data['description'].append(column)
                sensor_id += 1
                # compute min timestamp
                if df2['timestamp'].min() < min_timestamp:
                    min_timestamp = df2['timestamp'].min()
        # save global data
        df = pd.DataFrame(index_data)
        path = os.path.join(d, 'index.csv')
        df.to_csv(path, index=None)
        # save min timestamp
        path = os.path.join(d, 'min_timestamp.txt')
        with open(path, 'w') as fp:
            fp.write(f'{min_timestamp}')

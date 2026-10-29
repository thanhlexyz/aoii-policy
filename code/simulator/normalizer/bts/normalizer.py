import itertools as it
import pandas as pd
import numpy as np
import tqdm
import os

PARAMETER_DESCRIPTION = {
    111: 	'Room Temperature',
    112: 	'Temperature of Airconditioner 1',
    126: 	'Frequency of Power Generator',
    123: 	'Frequency of Power Grid',
    124: 	'Voltage of Power Generator',
    114: 	'Outdoor temperature',
    121: 	'Voltage of Power Grid',
    113: 	'Temperature of Airconditioner 2',
    162: 	'Runtime of Airconditioner 1',
    163: 	'Runtime of Airconditioner 2',
    164: 	'Runtime of AC',
    165: 	'Runtime of Power Generator',
    153: 	'Motion sensor',
    151: 	'Smoke sensor',
    152: 	'Door sensor',
    154: 	'Water leak sensor',
    125: 	'Load of power generator',
    122: 	'Load of Power Grid',
    161: 	'Capacity',
    155: 	'Heat increase',
    141: 	'Total Battery Voltage',
    115: 	'Humidity',
    116: 	'Battery temperature',
    143: 	'Voltage of Battery 1',
    144: 	'Voltage of Battery 2',
    142: 	'Load of Battery 1',
    145: 	'Load of Battery 1',
    1  : 	'Unknown',
}

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
        station_ids   = df['station_id'].unique()
        parameter_ids = df['parameter_id'].unique()
        # initialize global data
        index_data = {'sensor_id': [],
                      'station_id': [],
                      'parameter_id': [],
                      'n_event': [],
                      'description': []}
        # reset data dir
        d = os.path.join(args.data_dir, args.dataset)
        os.system(f'rm -rf "{d}"')
        os.makedirs(d)
        # loop
        sensor_id = 0
        min_timestamp = np.inf
        for station_id, parameter_id in tqdm.tqdm(it.product(station_ids, parameter_ids)):
            # extract and save subframe
            df1 = df[(df['station_id'] == station_id) & (df['parameter_id'] == parameter_id)]
            if parameter_id in [1, 126, 153, 154, 155, 162, 163, 164, 165]:
                df1.loc[:, 'value'] -= 999990
            if parameter_id in [151, 152]:
                df1.loc[:, 'value'] -= 999999
            if len(df1) > 0:
                path = os.path.join(d, f'{sensor_id}.csv')
                df2 = pd.DataFrame({'timestamp': df1['reading_time'], 'value': df1['value'].astype(float)})
                df2 = df2.dropna(subset=['value'])
                df2['timestamp'] = pd.to_datetime(df2['timestamp'], format='mixed')
                df2 = df2.sort_values(by='timestamp')
                # convert timestamp to float
                df2['timestamp'] = df2['timestamp'].apply(lambda x: x.timestamp())
                df2.to_csv(path, index=None)
                # add global data
                index_data['sensor_id'].append(sensor_id)
                index_data['station_id'].append(station_id)
                index_data['parameter_id'].append(parameter_id)
                index_data['n_event'].append(len(df2))
                index_data['description'].append(PARAMETER_DESCRIPTION[parameter_id])
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

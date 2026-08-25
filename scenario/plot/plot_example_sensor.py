import matplotlib.pyplot as plt
import pandas as pd
import os

N_EVENT = 100

def plot_example_sensor(args):
    # load index
    d     = os.path.join(args.data_dir, args.dataset)
    index = pd.read_csv(os.path.join(d, 'index.csv'))
    # one figure per parameter, using the sensor with the most events
    for parameter_id, group in index.groupby('parameter_id'):
        row = group.loc[group['n_event'].idxmax()]
        df  = pd.read_csv(os.path.join(d, f'{row["sensor_id"]}.csv'))[:N_EVENT]
        # elapsed second since the first reading
        x = df['timestamp'] - df['timestamp'].iloc[0]
        plt.plot(x, df['value'], '-o')
        # decorate and save
        plt.title(row['description'])
        plt.xlabel('Reading time (s)')
        plt.ylabel('Value')
        plt.tight_layout()
        path = os.path.join(args.figure_dir, f'{args.scenario}_{parameter_id}.pdf')
        plt.savefig(path)
        plt.cla()
        plt.clf()

import pandas as pd
import numpy as np
import os

class TimeSeries:

    def __init__(self, i, args):
        self.args = args
        self.i    = i
        d    = os.path.join(args.data_dir, args.dataset)
        N    = len(pd.read_csv(os.path.join(d, 'index.csv')))
        self.stream = (i + args.stream_offset) % N
        df   = pd.read_csv(os.path.join(d, f'{self.stream}.csv'))
        min_timestamp = float(open(os.path.join(d, 'min_timestamp.txt')).read())
        self.timestamp = (df['timestamp'].to_numpy() - min_timestamp
                          - args.start_slot * args.slot_interval)
        value = df['value'].to_numpy(dtype=np.float64)
        if args.causal_norm > 0:
            horizon = args.causal_norm * args.horizon * args.slot_interval
            seen = value[(self.timestamp >= 0) & (self.timestamp < horizon)]
            seen = seen if len(seen) else value[:1]
            min_value, max_value = seen.min(), seen.max()
        else:
            min_value, max_value = value.min(), value.max()
        if max_value > min_value:
            self.value = (value - min_value) / (max_value - min_value)
        elif args.causal_norm > 0:
            self.value = value - min_value + 1.0
        else:
            self.value = np.ones_like(value)

    def timeslot_to_timestamp(self, timeslot):
        return timeslot * self.args.slot_interval

    def sample(self, timeslot):
        timestamp = self.timeslot_to_timestamp(timeslot)
        j = np.searchsorted(self.timestamp, timestamp, side='right') - 1
        return 0.0 if j < 0 else float(self.value[j])

    @property
    def max_timestamp(self):
        return self.timestamp[-1]

import pandas as pd
import numpy as np
import os

class TimeSeries:
    """One sensor trace, replayed as a zero-order hold: the value at a slot is
    the latest reading whose timestamp does not exceed that slot."""

    def __init__(self, i, args):
        # save args
        self.args = args
        self.i    = i
        # load time series, cycling through the sensors when the device count
        # exceeds the trace count
        d    = os.path.join(args.data_dir, args.dataset)
        N    = len(pd.read_csv(os.path.join(d, 'index.csv')))
        df   = pd.read_csv(os.path.join(d, f'{i % N}.csv'))
        # zero-base the timestamp against the dataset-wide first reading
        min_timestamp = float(open(os.path.join(d, 'min_timestamp.txt')).read())
        self.timestamp = df['timestamp'].to_numpy() - min_timestamp
        # min-max scale the value; a constant sensor carries no information
        value     = df['value'].to_numpy(dtype=np.float64)
        min_value = value.min()
        max_value = value.max()
        if min_value == max_value:
            self.value = np.ones_like(value)
        else:
            self.value = (value - min_value) / (max_value - min_value)

    def timeslot_to_timestamp(self, timeslot):
        return timeslot * self.args.slot_interval

    def sample(self, timeslot):
        timestamp = self.timeslot_to_timestamp(timeslot)
        j = np.searchsorted(self.timestamp, timestamp, side='right') - 1
        # nothing has been read yet at this slot
        return 0.0 if j < 0 else float(self.value[j])

    @property
    def max_timestamp(self):
        return self.timestamp[-1]

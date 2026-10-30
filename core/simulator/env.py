import numpy as np
import torch
import os

from .time_series import TimeSeries
from . import irsa

class Env:

    def __init__(self, args):
        self.args = args
        self.global_step  = 0
        self.channel = irsa.create_channel(args)
        self.load_dataset()
        self.reset()

    def reset(self):
        args = self.args
        self.episode_step = 0
        self.error_age     = torch.zeros(args.n_sensor)
        self.source_value       = self.sample_sources(0)
        self.server_estimate   = torch.zeros(args.n_sensor)
        self.decoded = torch.ones(args.n_sensor)
        self.transmitting  = torch.zeros(args.n_sensor)
        observation = self.get_observation()
        return observation

    def step(self, action):
        args = self.args
        self.decode(action)
        next_observation = self.get_observation()
        reward, cost, info = self.get_reward(action)
        self.global_step += 1
        self.episode_step += 1
        done = self.episode_step >= args.horizon
        return next_observation, reward, cost, done, info

    def load_dataset(self):
        args = self.args
        self.time_series = [TimeSeries(i, args) for i in range(args.n_sensor)]
        self.max_timestamp = max(ts.max_timestamp for ts in self.time_series)
        n_slot = min(ts.max_timestamp for ts in self.time_series) / args.slot_interval
        if n_slot < args.horizon:
            print(f'[!] {args.dataset}: shortest trace ends at slot {n_slot:.0f}, '
                  f'run is {args.horizon} slots')

    def decode(self, action):
        args = self.args
        transmitting  = action
        pattern = irsa.sample_pattern(args, transmitting)
        self.source_value = self.sample_sources(self.episode_step)
        decoded = self.channel.decode(pattern)
        self.error_age = self.error_age * (1 - decoded)
        self.server_estimate = self.server_estimate * (1 - decoded) + self.source_value * decoded
        self.decoded = decoded
        self.transmitting  = transmitting
        mask = (torch.abs(self.source_value - self.server_estimate) > args.error_tolerance).to(self.error_age)
        self.error_age = (self.error_age + 1) * mask + 0 * (1 - mask)

    def close(self):
        if hasattr(self.channel, 'close'):
            self.channel.close()

    def sample_sources(self, timeslot):
        values = [ts.sample(timeslot) for ts in self.time_series]
        return torch.tensor(values, dtype=torch.float32)

    def get_observation(self):
        args = self.args
        error_age     = self.error_age
        avg_age = torch.full_like(error_age, error_age.mean())
        estimation_error     = torch.abs(self.source_value - self.server_estimate)
        avg_err = torch.full_like(error_age, estimation_error.mean())
        decoded = self.decoded
        if self.transmitting.sum() > 0:
            avg_success = torch.full_like(error_age, self.decoded.sum() / self.transmitting.sum())
        else:
            avg_success = torch.ones_like(error_age)
        observation = torch.stack([error_age, avg_age, estimation_error, avg_err, decoded, avg_success], dim=1)
        return observation

    def get_reward(self, action):
        args   = self.args
        estimation_error    = torch.abs(self.source_value - self.server_estimate)
        error_age    = self.error_age
        aoii   = estimation_error * error_age
        transmitting = self.transmitting
        reward = - (aoii.mean() + args.cost_weight * transmitting.sum() / args.n_sensor)
        cost   = transmitting.sum() / args.n_irsa_slot
        info = {'reward' : reward.item(),
                'cost'   : cost.item(),
                'avg_age': torch.mean(error_age).item(),
                'avg_err': torch.mean(estimation_error).item(),
                'max_err': torch.max(estimation_error).item(),
                'avg_aoii': torch.mean(aoii).item(),
                'max_aoii': torch.max(aoii).item(),
                'n_send': torch.sum(self.transmitting).item(),
                'n_success': torch.sum(self.decoded).item(),
                'n_error': torch.sum(self.transmitting).item() - torch.sum(self.decoded).item(),
        }
        return reward, cost, info

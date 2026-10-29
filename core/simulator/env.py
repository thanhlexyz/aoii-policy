import numpy as np
import torch
import os

from .time_series import TimeSeries
from . import irsa

class Env:

    def __init__(self, args):
        # save args
        self.args = args
        # initialize
        self.global_step  = 0
        # create the irsa uplink
        self.channel = irsa.create_channel(args)
        # load data
        self.load_dataset()
        self.reset()

    def reset(self):
        # extract args
        args = self.args
        # initialize state
        self.episode_step = 0
        self.age     = torch.zeros(args.n_device)
        self.X       = self.sample_sources(0)
        self.X_hat   = torch.zeros(args.n_device)
        self.success = torch.ones(args.n_device)
        self.active  = torch.zeros(args.n_device)
        # get observation
        observation = self.get_observation()
        return observation

    def step(self, action):
        # extract args
        args = self.args
        # decode
        self.decode(action)
        # extract next observation
        next_observation = self.get_observation()
        # extract reward
        reward, cost, info = self.get_reward(action)
        # extract done
        self.global_step += 1
        self.episode_step += 1
        done = self.episode_step >= args.n_step
        return next_observation, reward, cost, done, info

    def load_dataset(self):
        # extract args
        args = self.args
        #
        self.time_series = [TimeSeries(i, args) for i in range(args.n_device)]
        self.max_timestamp = max(ts.max_timestamp for ts in self.time_series)
        # past the end of a trace the zero-order hold freezes, which scores a
        # degenerate zero rather than a real result
        n_slot = min(ts.max_timestamp for ts in self.time_series) / args.slot_interval
        if n_slot < args.n_step:
            print(f'[!] {args.dataset}: shortest trace ends at slot {n_slot:.0f}, '
                  f'run is {args.n_step} slots')

    def decode(self, action):
        # extract args
        args = self.args
        # devices choose irsa pattern
        active  = action
        pattern = irsa.sample_pattern(args, active)
        # extract new value from source
        self.X = self.sample_sources(self.episode_step)
        # contend for the uplink
        success = self.channel.decode(pattern)
        # handle success
        self.age = self.age * (1 - success)
        self.X_hat = self.X_hat * (1 - success) + self.X * success
        self.success = success
        self.active  = active
        # get devices with error larger than threshold
        mask = (torch.abs(self.X - self.X_hat) > args.error_threshold).to(self.age)
        # devices and server increase age counter
        self.age = (self.age + 1) * mask + 0 * (1 - mask)

    def close(self):
        if hasattr(self.channel, 'close'):
            self.channel.close()

    def sample_sources(self, timeslot):
        values = [ts.sample(timeslot) for ts in self.time_series]
        return torch.tensor(values, dtype=torch.float32)

    def get_observation(self):
        # extract args
        args = self.args
        age     = self.age
        avg_age = torch.full_like(age, age.mean())
        err     = torch.abs(self.X - self.X_hat)
        avg_err = torch.full_like(age, err.mean())
        success = self.success
        if self.active.sum() > 0:
            avg_success = torch.full_like(age, self.success.sum() / self.active.sum())
        else:
            avg_success = torch.ones_like(age)
        observation = torch.stack([age, avg_age, err, avg_err, success, avg_success], dim=1)
        return observation

    def get_reward(self, action):
        args   = self.args
        err    = torch.abs(self.X - self.X_hat)
        age    = self.age
        aoii   = err * age
        active = self.active
        # the objective of eq. (2): aoii traded off against the transmission cost each
        # device spends, both averaged per device
        reward = - (aoii.mean() + args.alpha * active.sum() / args.n_device)
        cost   = active.sum() / args.n_irsa_slot
        # extract info
        info = {'reward' : reward.item(),
                'cost'   : cost.item(),
                'avg_age': torch.mean(age).item(),
                'avg_err': torch.mean(err).item(),
                'max_err': torch.max(err).item(),
                'avg_aoii': torch.mean(aoii).item(),
                'max_aoii': torch.max(aoii).item(),
                'n_send': torch.sum(self.active).item(),
                'n_success': torch.sum(self.success).item(),
                'n_error': torch.sum(self.active).item() - torch.sum(self.success).item(),
        }
        return reward, cost, info

import simulator

class BaseSolver:

    def __init__(self, args):
        simulator.set_random_seed(args)
        self.args = args
        self.env = simulator.Env(args)
        self.monitor = simulator.Monitor(args)

    def solve(self):
        simulator.set_random_seed(self.args)
        self.test()

    def select_action(self, observation):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def add_info(self, info):
        args                = self.args
        info['step']        = self.step
        info['episode']     = self.episode
        info['utilization'] = info['n_send'] / args.n_irsa_slot
        info['p_success']   = info['n_success'] / info['n_send'] if info['n_send'] > 0 else 1.0
        info.update(getattr(self, 'info_extra', {}))
        return info

    def test_episode(self):
        env     = self.env
        monitor = self.monitor
        observation = env.reset()
        while True:
            action = self.select_action(observation)
            next_observation, reward, cost, done, info = env.step(action)
            self.update()
            monitor.step(self.add_info(info))
            self.step += 1
            if done:
                break
            observation = next_observation

    def test(self):
        args         = self.args
        self.step    = 0
        self.episode = 0
        for _ in range(args.n_test_episode):
            self.test_episode()
            self.episode += 1
        self.monitor.export_csv()
        self.env.close()

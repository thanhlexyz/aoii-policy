import simulator

class BaseSolver:

    def __init__(self, args):
        # seed before anything is built: a learner may draw at construction
        # (PyXAB's VHCT picks a split dimension with numpy), and solve()
        # reseeds before the run itself
        simulator.set_random_seed(args)
        # save args
        self.args = args
        # create env
        self.env = simulator.Env(args)
        # load monitor
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
        # offered load: transmission attempts per irsa slot, summed over devices
        info['utilization'] = info['n_send'] / args.n_irsa_slot
        info['p_success']   = info['n_success'] / info['n_send'] if info['n_send'] > 0 else 1.0
        info.update(getattr(self, 'info_extra', {}))
        return info

    def test_episode(self):
        # extract args
        env     = self.env
        monitor = self.monitor
        # reset
        observation = env.reset()
        while True:
            # select action
            action = self.select_action(observation)
            # send to env
            next_observation, reward, cost, done, info = env.step(action)
            # update solver
            self.update()
            # log every step, including the one that ends the episode
            monitor.step(self.add_info(info))
            self.step += 1
            if done:
                break
            # update observation
            observation = next_observation

    def test(self):
        # extract args
        args         = self.args
        self.step    = 0
        self.episode = 0
        # loop
        for _ in range(args.n_test_episode):
            self.test_episode()
            self.episode += 1
        # export csv
        self.monitor.export_csv()
        self.env.close()

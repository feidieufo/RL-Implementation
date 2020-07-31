import gym
import numpy as np
from multiprocessing import Process, Pipe

import cloudpickle


class CloudpickleWrapper(object):

    def __init__(self, data):
        self.data = data

    def __getstate__(self):
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data):
        self.data = cloudpickle.loads(data)

class RunningStat:  # for class AutoNormalization
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        # assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            pre_memo = self._M.copy()
            self._M[...] = pre_memo + (x - pre_memo) / self._n
            self._S[...] = self._S + (x - pre_memo) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class Identity:
    def __call__(self, x):
        return x


class RewardFilter:
    def __init__(self, pre_filter, shape, center=True, scale=True, clip=10.0, gamma=0.99):
        self.pre_filter = pre_filter
        self.center = center
        self.scale = scale
        self.clip = clip
        self.gamma = gamma

        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)

    def __call__(self, x):
        x = self.pre_filter(x)
        self.ret = self.ret*self.gamma + x
        self.rs.push(self.ret)
        x = self.ret/self.rs.std
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x


class AutoNormalization:
    def __init__(self, pre_filter, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.pre_filter = pre_filter

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        x = self.pre_filter(x)
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    @staticmethod
    def output_shape(input_space):
        return input_space.shape



class Env(gym.Wrapper):
    def __init__(self, task, norm_state=False, norm_rewards=False):
        env = gym.make(task)
        super().__init__(env)
        self.total_reward = 0
        self.episode = 0
        self.total_step = 0

        self.state_norm = Identity()
        self.reward_norm = Identity()
        if norm_state:
            self.state_norm = AutoNormalization(self.state_norm, shape=env.observation_space.shape, clip=5)
        if norm_rewards:
            self.reward_norm = AutoNormalization(self.reward_norm, shape=(), clip=5)

    def reset(self, **kwargs):
        self.total_reward = 0
        self.episode = 0
        self.total_step = 0

        s = self.env.reset()
        return self.state_norm(s)

    def step(self, action):
        s_, r, d, info = self.env.step(action)

        self.total_reward += r
        self.total_step += 1
        if d:
            self.episode += 1
            s_ = self.env.reset()

        return self.state_norm(s_), self.reward_norm(r), d, info

    def statistics(self):
        return self.episode, self.total_reward


class VectorEnv:
    def __init__(self, env_fn):
        self.envs = env_fn
        self.env_num = len(env_fn)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        self._obs = np.stack([e.reset() for e in self.envs], axis=0)
        return self._obs

    def step(self, action):
        result = [e.step(a) for e, a in zip(self.envs, action)]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def statistics(self):
        result = [env.statistics() for env in self.envs]
        epi, total_reward = zip(*result)
        return np.sum(epi, axis=0), np.sum(total_reward, axis=0)


def worker(parent, p, env_fn_wrapper):
    parent.close()
    # env = env_fn_wrapper.data()
    env = env_fn_wrapper
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'step':
                p.send(env.step(data))
            elif cmd == 'reset':
                p.send(env.reset())
            elif cmd == 'close':
                p.send(env.close())
                p.close()
                break
            elif cmd == "statistics":
                p.send(env.statistics())
            elif cmd == 'render':
                p.send(env.render(**data) if hasattr(env, 'render') else None)
            elif cmd == 'seed':
                p.send(env.seed(data) if hasattr(env, 'seed') else None)
            elif cmd == 'getattr':
                p.send(getattr(env, data) if hasattr(env, data) else None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubVectorEnv:
    def __init__(self, env_fn):
        self.envs = env_fn
        self.env_num = len(env_fn)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.parent_remote, self.child_remote = \
            zip(*[Pipe() for _ in range(self.env_num)])

        self.processes = [
            Process(target=worker, args=(
                parent, child, env_fn), daemon=True)
            for (parent, child, env_fn) in zip(
                self.parent_remote, self.child_remote, self.envs)
        ]

        for p in self.processes:
            p.start()
        for c in self.child_remote:
            c.close()

    def reset(self):
        for p in self.parent_remote:
            p.send(['reset', None])
        self._obs = np.stack([p.recv() for p in self.parent_remote])
        return self._obs

    def step(self, action):
        for p, a in zip(self.parent_remote, action):
            p.send(['step', a])
        result = [p.recv() for p in self.parent_remote]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def statistics(self):
        for p in self.parent_remote:
            p.send(['statistics', None])
        result = [p.recv() for p in self.parent_remote]

        epi, total_reward = zip(*result)
        return np.sum(epi), np.sum(total_reward)

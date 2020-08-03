try:
    import algos.ppo.core_cnn_torch as core
except Exception:
    import core_cnn_torch as core

import numpy as np
import gym
import argparse
import scipy
from scipy import signal

import os
from utils.logx import EpochLogger
import torch
import dmc2gym
from collections import deque

class DMCFrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--encoder_type', default='pixel', type=str)

    parser.add_argument('--exp_name', default="ppo_cheetah_run_clipv_maxgrad_anneallr2.5e-3_stack3_normal_state01")
    parser.add_argument('--seed', default=10)
    parser.add_argument('--norm_state', default=False)
    parser.add_argument('--norm_rewards', default=False)
    args = parser.parse_args()

    # env = gym.make("Hopper-v2")
    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat
    )
    if args.encoder_type == 'pixel':
        env = DMCFrameStack(env, k=args.frame_stack)
    state_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    actor = core.Actor(state_dim, act_dim).to(device)
    checkpoint = torch.load(os.path.join(logger_kwargs["output_dir"], "checkpoints", str(900) + '.pth'))
    actor.load_state_dict(checkpoint["actor"])

    state_norm = Identity()
    reward_norm = Identity()
    if args.norm_state:
        state_norm = AutoNormalization(state_norm, state_dim, clip=5.0)
    if args.norm_rewards:
        reward_norm = AutoNormalization(reward_norm, (), clip=5.0)

    obs = env.reset()
    obs = state_norm(obs)
    rew = 0
    rew_list = []
    epi = 0
    while True:
        # env.render()
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a, var = actor(state_tensor)
        logpi = actor.log_pi(state_tensor, a)
        a = torch.squeeze(a, 0).detach().cpu().numpy()
        obs, r, d, _ = env.step(a)

        rew += r
        if d:
            rew_list.append(rew)
            epi += 1
            # print("reward", rew)

            if epi % 10 == 0:
                print("teset_", np.mean(rew_list))
                rew_list = []
            obs = env.reset()
            rew = 0

        obs = state_norm(obs)

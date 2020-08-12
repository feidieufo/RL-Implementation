try:
    import algos.ppo.core_cnn_torch as core
except Exception:
    import core_cnn_torch as core

import numpy as np
import gym
import argparse
import scipy
from scipy import signal
import pickle

import os
from utils.logx import EpochLogger
import torch
from torch.utils.tensorboard import SummaryWriter

from env.atari_lib import make_atari, wrap_deepmind
from utils.normalization import *
import json
from algos.ppo.utils import discount_path, get_path_indices

class ReplayBuffer:
    def __init__(self, size, state_dim, act_dim, gamma=0.99, lam=0.95, is_gae=True):
        self.size = size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.is_gae = is_gae
        self.reset()

    def reset(self):
        self.state = np.zeros((self.size,) + self.state_dim, np.float32)
        if type(self.act_dim) == np.int64 or type(self.act_dim) == np.int:
            self.action = np.zeros((self.size, ), np.int32)
        else:
            self.action = np.zeros((self.size,) + self.act_dim, np.float32)
        self.v = np.zeros((self.size, ), np.float32)
        self.reward = np.zeros((self.size, ), np.float32)
        self.adv = np.zeros((self.size, ), np.float32)
        self.mask = np.zeros((self.size, ), np.float32)
        self.ptr, self.path_start = 0, 0

    def add(self, s, a, r, mask):
        if self.ptr < self.size:
            self.state[self.ptr] = s
            self.action[self.ptr] = a
            self.reward[self.ptr] = r
            self.mask[self.ptr] = mask
            self.ptr += 1

    def update_v(self, v, pos):
        self.v[pos] = v

    def finish_path(self, last_v):
        """
          Calculate GAE advantage, discounted returns, and
          true reward (average reward per trajectory)

          GAE: delta_t^V = r_t + discount * V(s_{t+1}) - V(s_t)
          using formula from John Schulman's code:
          V(s_t+1) = {0 if s_t is terminal
                   {v_s_{t+1} if s_t not terminal and t != T (last step)
                   {v_s if s_t not terminal and t == T
          """
        v_ = np.concatenate([self.v[1:], [last_v]], axis=0) * self.mask
        adv = self.reward + self.gamma * v_ - self.v

        indices = get_path_indices(self.mask)

        for (start, end) in indices:
            self.adv[start:end] = discount_path(adv[start:end], self.gamma * self.lam)
            if not self.is_gae:
                self.reward[start:end] = discount_path(self.reward[start:end], self.gamma)
        if self.is_gae:
            self.reward = self.adv + self.v

        self.adv = (self.adv - np.mean(self.adv))/(np.std(self.adv) + 1e-8)

    def get_batch(self, batch=100, shuffle=True):
        if shuffle:
            indices = np.random.permutation(self.size)
        else:
            indices = np.arange(self.size)

        for idx in np.arange(0, self.size, batch):
            pos = indices[idx:(idx + batch)]
            yield (self.state[pos], self.action[pos], self.reward[pos], self.adv[pos], self.v[pos])

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.array(observation).astype(np.float32) / 255.0
        return np.transpose(obs, axes=(2, 0, 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lam', default=0.95, type=float)
    parser.add_argument('--a_update', default=10, type=int)
    parser.add_argument('--lr', default=2.5e-4, type=float)
    parser.add_argument('--log', type=str, default="logs")
    parser.add_argument('--steps', default=3000, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--env', default="BreakoutNoFrameskip-v4")
    parser.add_argument('--env_num', default=4, type=int)
    parser.add_argument('--exp_name', default="ppo_Pong")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch', default=50, type=int)
    parser.add_argument('--norm_state', default=False)
    parser.add_argument('--norm_rewards', default=False)
    parser.add_argument('--clip_coef', type=float, default=0.2)
    parser.add_argument('--is_clip_v', default=True)
    parser.add_argument('--is_gae', default=True)
    parser.add_argument('--max_grad_norm', default=False)
    parser.add_argument('--anneal_lr', default=False)
    parser.add_argument('--debug', default=True)
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--target_kl', default=0.03, type=float)
    parser.add_argument('--test_epoch', default=10, type=int)
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
    writer = SummaryWriter(os.path.join(logger.output_dir, "logs"))
    with open(os.path.join(logger.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    env = make_atari(args.env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = wrap_deepmind(env, frame_stack=True)
    env = ImageToPyTorch(env)
    # test_env = make_atari(args.env)
    # test_env = gym.wrappers.RecordEpisodeStatistics(test_env)
    # test_env = wrap_deepmind(test_env, frame_stack=True)
    # test_env = ImageToPyTorch(test_env)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    state_dim = env.observation_space.shape
    act_dim = env.action_space.n
    ppo = core.PPO(state_dim, act_dim, 1, args.clip_coef, device, lr_a=args.lr,
                   max_grad_norm=args.max_grad_norm,
                   anneal_lr=args.anneal_lr, train_steps=args.iteration)
    replay = ReplayBuffer(args.steps, state_dim, act_dim, is_gae=args.is_gae)

    state_norm = Identity()
    reward_norm = Identity()
    if args.norm_state:
        state_norm = AutoNormalization(state_norm, state_dim, clip=10.0)
    if args.norm_rewards == "rewards":
        reward_norm = AutoNormalization(reward_norm, (), clip=10.0)
    elif args.norm_rewards == "returns":
        reward_norm = RewardFilter(reward_norm, (), clip=10.0)

    obs = env.reset()
    obs = state_norm(obs)
    rew = 0
    for iter in range(args.iteration):
        ppo.train()
        replay.reset()
        flag = 0

        for step in range(args.steps):
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_tensor = ppo.actor.select_action(state_tensor)
            a = a_tensor.detach().cpu().numpy()
            obs_, r, done, info = env.step(a)
            rew += r
            r = reward_norm(r)
            mask = 1-done

            replay.add(obs, a, r, mask)

            obs = obs_
            if done:
                if 'episode' in info.keys():
                    logger.store(reward=info['episode']['r'])
                    flag = 1
                rew = 0
                obs = env.reset()
            obs = state_norm(obs)
        if flag == 0:
            logger.store(reward=0)
            print("null reward")

        state = replay.state
        for idx in np.arange(0, state.shape[0], args.batch):
            if idx + args.batch <= state.shape[0]:
                pos = np.arange(idx, idx + args.batch)
            else:
                pos = np.arange(idx, state.shape[0])
            s = torch.tensor(state[pos], dtype=torch.float32).to(device)
            v = ppo.getV(s).detach().cpu().numpy()
            replay.update_v(v, pos)
        s_tensor = torch.tensor(obs_, dtype=torch.float32).to(device).unsqueeze(0)
        last_v = ppo.getV(s_tensor).detach().cpu().numpy()
        replay.finish_path(last_v)

        ppo.update_a()

        for i in range(args.a_update):
            for (s, a, r, adv, v) in replay.get_batch(batch=args.batch):
                s_tensor = torch.tensor(s, dtype=torch.float32, device=device)
                a_tensor = torch.tensor(a, dtype=torch.float32, device=device)
                adv_tensor = torch.tensor(adv, dtype=torch.float32, device=device)
                r_tensor = torch.tensor(r, dtype=torch.float32, device=device)
                v_tensor = torch.tensor(v, dtype=torch.float32, device=device)

                info = ppo.train_ac(s_tensor, a_tensor, adv_tensor, r_tensor, v_tensor, is_clip_v=args.is_clip_v)

                if args.debug:
                    logger.store(aloss=info["aloss"])
                    logger.store(vloss=info["vloss"])
                    logger.store(entropy=info["entropy"])
                    logger.store(kl=info["kl"])
            
            if logger.get_stats("kl")[0] > args.target_kl:
                print("stop at:", str(i))
                break

        if args.anneal_lr:
            ppo.lr_scheduler()


        # writer.add_scalar("test_reward", logger.get_stats("test_reward")[0], global_step=iter) 
        writer.add_scalar("reward", logger.get_stats("reward")[0], global_step=iter)
        writer.add_histogram("action", np.array(replay.action), global_step=iter)
        if args.debug:
            writer.add_scalar("aloss", logger.get_stats("aloss")[0], global_step=iter)
            writer.add_scalar("vloss", logger.get_stats("vloss")[0], global_step=iter)
            writer.add_scalar("entropy", logger.get_stats("entropy")[0], global_step=iter)
            writer.add_scalar("kl", logger.get_stats("kl")[0], global_step=iter)

        logger.log_tabular('Epoch', iter)
        logger.log_tabular("reward", with_min_and_max=True)
        # logger.log_tabular("test_reward", with_min_and_max=True)
        if args.debug:
            logger.log_tabular("aloss", with_min_and_max=True)
            logger.log_tabular("vloss", with_min_and_max=True)
            logger.log_tabular("entropy", with_min_and_max=True)
            logger.log_tabular("kl", with_min_and_max=True)
        logger.dump_tabular()

        if not os.path.exists(os.path.join(logger.output_dir, "checkpoints")):
            os.makedirs(os.path.join(logger.output_dir, "checkpoints"))
        if iter % args.log_every == 0:
            state = {
                "actor": ppo.actor.state_dict(),
                "critic": ppo.critic.state_dict(),

            }
            torch.save(state, os.path.join(logger.output_dir, "checkpoints", str(iter) + '.pth'))
            norm = {"state": state_norm, "reward": reward_norm}
            with open(os.path.join(logger.output_dir, "checkpoints", str(iter) + '.pkl'), "wb") as f:
                pickle.dump(norm, f)



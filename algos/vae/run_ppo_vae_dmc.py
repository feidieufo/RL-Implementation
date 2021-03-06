import numpy as np
import gym
import argparse
import scipy
from scipy import signal
import pickle
from collections import deque

import os
from utils.logx import EpochLogger
import torch
from torch.utils.tensorboard import SummaryWriter
import dmc2gym
import env.atari_lib as atari
from env.dmc_env import DMCFrameStack
from utils.normalization import *
from algos.ppo.utils import discount_path, get_path_indices
import json

from algos.vae.encoder import PixelEncoder
from algos.vae.decoder import PixelDecoder

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
        self.state = np.zeros((self.size, self.state_dim), np.float32)
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

    def finish_path(self):
        """
          Calculate GAE advantage, discounted returns, and
          true reward (average reward per trajectory)

          GAE: delta_t^V = r_t + discount * V(s_{t+1}) - V(s_t)
          using formula from John Schulman's code:
          V(s_t+1) = {0 if s_t is terminal
                   {v_s_{t+1} if s_t not terminal and t != T (last step)
                   {v_s if s_t not terminal and t == T
          """
        v_ = np.concatenate([self.v[1:], self.v[-1:]], axis=0) * self.mask
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


class ImageEncodeProcess:
    def __init__(self, pre_filter):
        self.pre_filter = pre_filter
    def __call__(self, x, update=True):
        x = self.pre_filter(x)
        x = np.array(x).astype(np.float32)
        x = torch.tensor(x, device=device).unsqueeze(0)
        x = encoder(x).detach()
        return x
    
    def reset(self):
        self.pre_filter.reset() 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--encoder_type', default='pixel', type=str)

    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lam', default=0.95, type=float)
    parser.add_argument('--a_update', default=10, type=int)
    parser.add_argument('--lr_a', default=2.5e-4, type=float)
    parser.add_argument('--c_en', default=0.01, type=float)    
    parser.add_argument('--c_vf', default=0.5, type=float)
    parser.add_argument('--log', type=str, default="logs")
    parser.add_argument('--steps', default=3000, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--env_num', default=4, type=int)
    parser.add_argument('--exp_name', default="ppo_vae_cheetah_run_test")
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--batch', default=50, type=int)
    parser.add_argument('--norm_state', action="store_true")
    parser.add_argument('--norm_rewards', default=False)
    parser.add_argument('--is_clip_v', action="store_true")
    parser.add_argument('--max_grad_norm', default=-1, type=float)
    parser.add_argument('--anneal_lr', action="store_true")
    parser.add_argument('--debug', action="store_false")
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--network', default="cnn")
    parser.add_argument('--feature_dim', default=50, type=int)
    parser.add_argument('--target_kl', default=0.03, type=float)
    parser.add_argument('--encoder_dir', default="vae_2")
    parser.add_argument('--encoder_check', default=300, type=int)
    parser.add_argument('--test_epoch', default=10, type=int) 
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
    with open(os.path.join(logger.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    writer = SummaryWriter(os.path.join(logger.output_dir, "logs"))

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
    test_env = dmc2gym.make(
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
        test_env = DMCFrameStack(test_env, k=args.frame_stack)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    test_env.seed(args.seed)

    state_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    action_max = env.action_space.high[0]
    if args.network == "cnn":
        import algos.vae.core_vae as core
        ppo = core.PPO(state_dim, act_dim, action_max, 0.2, device, lr_a=args.lr_a,
                       max_grad_norm=args.max_grad_norm,
                       anneal_lr=args.anneal_lr, train_steps=args.iteration, emb_dim=args.feature_dim,
                       c_en=args.c_en, c_vf=args.c_vf)

    replay = ReplayBuffer(args.steps, args.feature_dim, act_dim)
    encoder = PixelEncoder(state_dim, args.feature_dim, num_layers=4).to(device)
    encoder_kwargs = setup_logger_kwargs(args.encoder_dir, args.seed)
    encoder_file = os.path.join(encoder_kwargs["output_dir"], "checkpoints", str(args.encoder_check) + ".pth")
    check = torch.load(encoder_file)
    encoder.load_state_dict(check["encoder"])

    state_norm = Identity()
    state_norm = ImageEncodeProcess(state_norm)
    reward_norm = Identity()
    if args.norm_state:
        state_norm = AutoNormalization(state_norm, state_dim, clip=5.0)
    if args.norm_rewards == "rewards":
        reward_norm = AutoNormalization(reward_norm, (), clip=5.0)
    elif args.norm_rewards == "returns":
        reward_norm = RewardFilter(reward_norm, (), clip=5.0)

    state_norm.reset()
    reward_norm.reset()
    obs = env.reset()
    obs = state_norm(obs)
    rew = 0
    for iter in range(args.iteration):
        ppo.train()
        replay.reset()

        for step in range(args.steps):
            a_tensor = ppo.actor.select_action(obs)
            a = a_tensor.detach().cpu().numpy()
            a = np.clip(a, -1, 1)
            obs_, r, done, _ = env.step(a)
            obs_ = state_norm(obs_)
            rew += r
            r = reward_norm(r)

            mask = 1-done
            replay.add(obs.cpu().numpy()[0], a, r, mask)

            obs = obs_
            if done:
                logger.store(reward=rew)
                rew = 0
                state_norm.reset()
                reward_norm.reset()
                obs = env.reset()
                obs = state_norm(obs)

        state = replay.state
        for idx in np.arange(0, state.shape[0], args.batch):
            if idx + args.batch <= state.shape[0]:
                pos = np.arange(idx, idx + args.batch)
            else:
                pos = np.arange(idx, state.shape[0])
            s = torch.tensor(state[pos], dtype=torch.float32).to(device)
            v = ppo.getV(s).detach().cpu().numpy()
            replay.update_v(v, pos)
        if args.last_v:
            s_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            last_v = ppo.getV(s_tensor).detach().cpu().numpy() 
            replay.finish_path(last_v=last_v) 
        else:      
            replay.finish_path()
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

            if logger.get_stats("kl", with_min_and_max=True)[3] > args.target_kl:
                print("stop at:", str(i))
                break

        if args.anneal_lr:
            ppo.lr_scheduler()
        
        ppo.eval()
        for i in range(args.test_epoch):
            test_obs = test_env.reset()
            test_obs = state_norm(test_obs, update=False)
            test_rew = 0

            while True:
                a_tensor, var = ppo.actor(test_obs)
                a_tensor = torch.squeeze(a_tensor, dim=0)
                a = a_tensor.detach().cpu().numpy()
                test_obs, r, done, _ = test_env.step(np.clip(a, -1, 1))
                test_rew += r

                if done:
                    logger.store(test_reward=test_rew)
                    break
                test_obs = state_norm(test_obs, update=False)

        writer.add_scalar("test_reward", logger.get_stats("test_reward")[0], global_step=iter)  
        writer.add_scalar("reward", logger.get_stats("reward")[0], global_step=iter)
        writer.add_histogram("action", np.array(replay.action), global_step=iter)
        if args.debug:
            writer.add_scalar("aloss", logger.get_stats("aloss")[0], global_step=iter)
            writer.add_scalar("vloss", logger.get_stats("vloss")[0], global_step=iter)
            writer.add_scalar("entropy", logger.get_stats("entropy")[0], global_step=iter)
            writer.add_scalar("kl", logger.get_stats("kl")[0], global_step=iter)

        logger.log_tabular('Epoch', iter)
        logger.log_tabular("reward", with_min_and_max=True)
        logger.log_tabular("test_reward", with_min_and_max=True)
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



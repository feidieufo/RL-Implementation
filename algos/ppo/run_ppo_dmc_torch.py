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

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


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

class ImageProcess:
    def __init__(self, pre_filter):
        self.pre_filter = pre_filter
    def __call__(self, x):
        x = self.pre_filter(x)
        x = np.array(x).astype(np.float32) / 255.0
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
        x = self.ret/(self.rs.std + 1e-8)
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


class ReplayBuffer:
    def __init__(self, size, gamma=0.99, lam=0.95):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.state = []
        self.action = []
        self.v = np.zeros((self.size, ), np.float32)
        self.reward = np.zeros((self.size, ), np.float32)
        self.adv = np.zeros((self.size, ), np.float32)
        self.ptr, self.path_start = 0, 0

    def add(self, s, a, v, r):
        if self.ptr < self.size:
            self.state.append(s)
            self.action.append(a)
            self.v[self.ptr] = v
            self.reward[self.ptr] = r
            self.ptr += 1

    def finish_path(self, last_v):
        path_slice = slice(self.path_start, self.ptr)
        reward = self.reward[path_slice]

        v = self.v[path_slice]
        v_ = np.append(self.v[self.path_start + 1:self.ptr], last_v)
        adv = reward + self.gamma * v_ - v
        adv = discount_cumsum(adv, self.gamma * self.lam)

        # reward[-1] = reward[-1] + self.gamma * last_v
        # reward = discount_cumsum(reward, self.gamma)
        self.reward[path_slice] = adv + v 

        self.adv[path_slice] = adv
        self.path_start = self.ptr

    def get(self):
        self.adv = (self.adv - np.mean(self.adv))/(np.std(self.adv) + 1e-8)
        # self.reward = (self.reward - np.mean(self.reward))/np.std(self.reward)


    def get_batch(self, batch=100, shuffle=True):
        if shuffle:
            indices = np.random.permutation(self.size)
        else:
            indices = np.arange(self.size)

        state = np.array(self.state)
        action = np.array(self.action)
        for idx in np.arange(0, self.size, batch):
            pos = indices[idx:(idx + batch)]
            yield (state[pos], action[pos], self.reward[pos], self.adv[pos], self.v[pos])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=4, type=int)
    parser.add_argument('--encoder_type', default='pixel', type=str)

    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--lam', default=0.95)
    parser.add_argument('--a_update', default=10)
    parser.add_argument('--c_update', default=10)
    parser.add_argument('--lr_a', default=2.5e-4)
    parser.add_argument('--lr_c', default=1e-3)
    parser.add_argument('--log', type=str, default="logs")
    parser.add_argument('--steps', default=3000, type=int)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--env_num', default=4, type=int)
    parser.add_argument('--exp_name', default="ppo_cheetah_run")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch', default=50)
    parser.add_argument('--norm_state', default=False)
    parser.add_argument('--norm_rewards', default=False)
    parser.add_argument('--is_clip_v', default=True)
    parser.add_argument('--max_grad_norm', default=False)
    parser.add_argument('--anneal_lr', default=False)
    parser.add_argument('--debug', default=True)
    parser.add_argument('--log_every', default=10)
    parser.add_argument('--network', default="cnn")
    parser.add_argument('--target_kl', default=0.03, type=float)
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
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
    if args.encoder_type == 'pixel':
        env = DMCFrameStack(env, k=args.frame_stack)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    state_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    action_max = env.action_space.high[0]
    if args.network == "cnn":
        import algos.ppo.core_cnn_torch as core
        ppo = core.PPO(state_dim, act_dim, action_max, 0.2, device, lr_a=args.lr_a,
                       max_grad_norm=args.max_grad_norm,
                       anneal_lr=args.anneal_lr, train_steps=args.iteration)
    elif args.network == "resnet":
        import algos.ppo.core_resnet_simple_torch as core
    elif args.network == "mlp" or args.encoder_type == 'state':
        import algos.ppo.core_torch as core
        ppo = core.PPO(state_dim, act_dim, action_max, 0.2, device, lr_a=args.lr_a,
                    lr_c=args.lr_c, max_grad_norm=args.max_grad_norm,
                    anneal_lr=args.anneal_lr, train_steps=args.iteration)
    replay = ReplayBuffer(args.steps)

    state_norm = Identity()
    state_norm = ImageProcess(state_norm)
    reward_norm = Identity()
    if args.norm_state:
        state_norm = AutoNormalization(state_norm, state_dim, clip=5.0)
    if args.norm_rewards == "rewards":
        reward_norm = AutoNormalization(reward_norm, (), clip=5.0)
    elif args.norm_rewards == "returns":
        reward_norm = RewardFilter(reward_norm, (), clip=5.0)

    for iter in range(args.iteration):
        replay.reset()
        rewards = []
        obs = env.reset()
        obs = state_norm(obs)
        rew = 0

        for step in range(args.steps):
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_tensor = ppo.actor.select_action(state_tensor)
            a = a_tensor.detach().cpu().numpy()
            a = np.clip(a, -1, 1)
            obs_, r, done, _ = env.step(a)
            obs_ = state_norm(obs_)
            rew += r
            r = reward_norm(r)


            v_pred = ppo.getV(state_tensor)
            replay.add(obs, a, v_pred.detach().cpu().numpy(), r)

            obs = obs_
            if done or step == args.steps-1:
                if done:
                    replay.finish_path(0)
                else:
                    state_tensor = torch.tensor(obs_, dtype=torch.float32, device=device).unsqueeze(0)
                    last_v = ppo.getV(state_tensor)
                    replay.finish_path(last_v.detach().cpu().numpy())

                rewards.append(rew)
                logger.store(reward=rew)
                rew = 0
                obs = env.reset()
                obs = state_norm(obs)

        ppo.update_a()
        replay.get()
        writer.add_scalar("reward", logger.get_stats("reward")[0], global_step=iter)
        writer.add_histogram("action", np.array(replay.action), global_step=iter)

        for i in range(args.a_update):
            for (s, a, r, adv, v) in replay.get_batch(batch=args.batch):
                s_tensor = torch.tensor(s, dtype=torch.float32, device=device)
                a_tensor = torch.tensor(a, dtype=torch.float32, device=device)
                adv_tensor = torch.tensor(adv, dtype=torch.float32, device=device)
                r_tensor = torch.tensor(r, dtype=torch.float32, device=device)
                v_tensor = torch.tensor(v, dtype=torch.float32, device=device)

                info = ppo.train(s_tensor, a_tensor, adv_tensor, r_tensor, v_tensor, is_clip_v=args.is_clip_v)

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
        if args.debug:
            writer.add_scalar("aloss", logger.get_stats("aloss")[0], global_step=iter)
            writer.add_scalar("vloss", logger.get_stats("vloss")[0], global_step=iter)
            writer.add_scalar("entropy", logger.get_stats("entropy")[0], global_step=iter)
            writer.add_scalar("kl", logger.get_stats("kl")[0], global_step=iter)

        logger.log_tabular('Epoch', iter)
        logger.log_tabular("reward", with_min_and_max=True)
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



try:
    import algos.gail.pytorch.core_torch as core
except Exception:
    import core_torch as core

import numpy as np
import gym
import argparse
import scipy
from scipy import signal

import os
from utils.logx import EpochLogger
import torch
from torch.utils.tensorboard import SummaryWriter
from env.car_racing import CarRacing
from env.vecenv import SubVectorEnv, Env, VectorEnv
import pickle


# Can be used to convert rewards into discounted returns:
# ret[i] = sum of t = i to T of gamma^(t-i) * rew[t]
def discount_path(path, gamma):
    '''
    Given a "path" of items x_1, x_2, ... x_n, return the discounted
    path, i.e.
    X_1 = x_1 + h*x_2 + h^2 x_3 + h^3 x_4
    X_2 = x_2 + h*x_3 + h^2 x_4 + h^3 x_5
    etc.
    Can do (more efficiently?) w SciPy. Python here for readability
    Inputs:
    - path, list/tensor of floats
    - h, discount rate
    Outputs:
    - Discounted path, as above
    '''
    curr = 0
    rets = []
    for i in range(len(path)):
        curr = curr*gamma + path[-1-i]
        rets.append(curr)
    rets =  np.stack(list(reversed(rets)), 0)
    return rets


def get_path_indices(not_dones):
    """
    Returns list of tuples of the form:
        (agent index, time index start, time index end + 1)
    For each path seen in the not_dones array of shape (# agents, # time steps)
    E.g. if we have an not_dones of composition:
    tensor([[1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]], dtype=torch.uint8)
    Then we would return:
    [(0, 0, 3), (0, 3, 10), (1, 0, 3), (1, 3, 5), (1, 5, 9), (1, 9, 10)]
    """
    indices = []
    num_timesteps = not_dones.shape[1]
    for actor in range(not_dones.shape[0]):
        last_index = 0
        for i in range(num_timesteps):
            if not_dones[actor, i] == 0.:
                indices.append((actor, last_index, i + 1))
                last_index = i + 1
        if last_index != num_timesteps:
            indices.append((actor, last_index, num_timesteps))
    return indices


class ReplayBuffer:
    def __init__(self, env_num, size, state_dim, action_dim, gamma=0.99, lam=0.95):
        self.env_num = env_num
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.state = np.zeros((self.env_num, self.size, self.state_dim), np.float32)
        self.action = np.zeros((self.env_num, self.size, self.action_dim), np.float32)
        self.mask = np.zeros((self.env_num, self.size), np.int32)
        self.v = np.zeros((self.env_num, self.size), np.float32)
        self.reward = np.zeros((self.env_num, self.size, ), np.float32)
        self.adv = np.zeros((self.env_num, self.size, ), np.float32)
        self.ptr, self.path_start = 0, 0

    def add(self, s, a, v, r, mask):
        if self.ptr < self.size:
            self.state[:, self.ptr, :] = s
            self.action[:, self.ptr, :] = a
            self.v[:, self.ptr] = v
            self.reward[:, self.ptr] = r
            self.mask[:, self.ptr] = mask
            self.ptr += 1

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
        v_ = np.concatenate([self.v[:, :-1], self.v[:, -1:]], axis=1)*self.mask
        adv = self.reward + self.gamma*v_ -self.v

        indices = get_path_indices(self.mask)

        for (num, start, end) in indices:
            self.adv[num, start:end] = discount_path(adv[num, start:end], self.gamma*self.lam)
            self.reward[num, start:end] = discount_path(self.reward[num, start:end], self.gamma)


    def get(self):
        self.state = np.concatenate([state for state in self.state], axis=0)
        self.action = np.concatenate([action for action in self.action], axis=0)
        self.v = np.concatenate([v for v in self.v], axis=0)
        self.adv = np.concatenate([adv for adv in self.adv], axis=0)
        self.reward = np.concatenate([r for r in self.reward], axis=0)
        self.adv = (self.adv - np.mean(self.adv))/np.std(self.adv)

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
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--lam', default=0.95)
    parser.add_argument('--a_update', default=10)
    parser.add_argument('--c_update', default=10)
    parser.add_argument('--lr_a', default=4e-4)
    parser.add_argument('--lr_c', default=1e-3)
    parser.add_argument('--log', type=str, default="logs")
    parser.add_argument('--steps', default=3000, type=int)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--env', default="Pendulum-v0")
    parser.add_argument('--env_num', default=12, type=int)
    parser.add_argument('--exp_name', default="ppo_Pendulum_test")
    parser.add_argument('--seed', default=0)
    parser.add_argument('--batch', default=50)
    parser.add_argument('--norm_state', default=True)
    parser.add_argument('--norm_rewards', default=True)
    parser.add_argument('--is_clip_v', default=True)
    parser.add_argument('--max_grad_norm', default=False)
    parser.add_argument('--anneal_lr', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--log_every', default=10)
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
    writer = SummaryWriter(os.path.join(logger.output_dir, "logs"))

    env = gym.make(args.env)
    if args.env_num > 1:
        env = [Env(args.env, norm_state=args.norm_state, norm_rewards=args.norm_rewards)
               for _ in range(args.env_num)]
        env = SubVectorEnv(env)
    # env = CarRacing()
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape
    action_max = env.action_space.high[0]
    ppo = core.PPO(state_dim, act_dim, action_max, 0.2, device, lr_a=args.lr_a,
                   lr_c=args.lr_c, max_grad_norm=args.max_grad_norm,
                   anneal_lr=args.anneal_lr, train_steps=args.iteration)
    replay = ReplayBuffer(args.env_num, args.steps, state_dim, act_dim)

    for iter in range(args.iteration):
        replay.reset()
        rewards = []
        obs = env.reset()
        rew = 0

        for step in range(args.steps):
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            a_tensor = ppo.actor.select_action(state_tensor)
            a = a_tensor.detach().cpu().numpy()
            obs_, r, done, _ = env.step(a)
            mask = 1-done

            v_pred = ppo.getV(state_tensor)
            replay.add(obs, a, v_pred.detach().cpu().numpy(), r, mask)

            obs = obs_
        replay.finish_path()
        epi, total_reward = env.statistics()

        ppo.update_a()
        replay.get()
        writer.add_scalar("reward", total_reward/epi, global_step=iter)
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

        if args.anneal_lr:
            ppo.lr_scheduler()
        if args.debug:
            writer.add_scalar("aloss", logger.get_stats("aloss")[0], global_step=iter)
            writer.add_scalar("vloss", logger.get_stats("vloss")[0], global_step=iter)
            writer.add_scalar("entropy", logger.get_stats("entropy")[0], global_step=iter)
            writer.add_scalar("kl", logger.get_stats("kl")[0], global_step=iter)

        logger.log_tabular('Epoch', iter)
        logger.log_tabular("reward", total_reward/epi)
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
                "critic": ppo.critic.state_dict()
            }
            torch.save(state, os.path.join(logger.output_dir, "checkpoints", str(iter) + '.pth'))

            norm = {"state": env.envs[0].state_norm, "reward": env.envs[0].reward_norm}
            with open(os.path.join(logger.output_dir, "checkpoints", str(iter) + '.pkl'), "wb") as f:
                pickle.dump(norm, f)




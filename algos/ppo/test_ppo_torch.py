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
from env.dmc_env import DMCFrameStack
from utils.normalization import *


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
    parser.add_argument('--check_num', default=900, type=int)
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
    checkpoint = torch.load(os.path.join(logger_kwargs["output_dir"], "checkpoints", str(args.check_num) + '.pth'))
    actor.load_state_dict(checkpoint["actor"])

    state_norm = Identity()
    state_norm = ImageProcess(state_norm)
    reward_norm = Identity()
    file = os.path.join(logger_kwargs["output_dir"], "checkpoints", str(args.check_num) + '.pkl')
    with open(file, "rb") as f:
        if args.norm_state:
            state_norm = pickle.load(f)["state"]
        if args.norm_rewards:
            reward_norm = pickle.load(f)["reward"]

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

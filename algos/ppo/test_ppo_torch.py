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
from utils.video import VideoRecorder
from user_config import DEFAULT_VIDEO_DIR
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--encoder_type', default='pixel', type=str)

    parser.add_argument('--exp_name', default="ppo_test_cheetah_run_clipv_maxgrad_anneallr3e-4_normal_maxkl0.05_gae_norm-state_return_steps2048_batch256_notdone_lastv_4_entropy_update30")
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--norm_state', default=True, type=bool)
    parser.add_argument('--norm_rewards', default=True, type=bool)
    parser.add_argument('--check_num', default=900, type=int)
    parser.add_argument('--test_num', default=10, type=int)
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
        import algos.ppo.core_cnn_torch as core
    else:
        import algos.ppo.core_torch as core

    state_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    action_max = env.action_space.high[0]
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    actor = core.Actor(state_dim, act_dim, action_max).to(device)
    checkpoint = torch.load(os.path.join(logger_kwargs["output_dir"], "checkpoints", str(args.check_num) + '.pth'))
    actor.load_state_dict(checkpoint["actor"])

    state_norm = Identity()
    state_norm = ImageProcess(state_norm)
    reward_norm = Identity()
    file = os.path.join(logger_kwargs["output_dir"], "checkpoints", str(args.check_num) + '.pkl')
    with open(file, "rb") as f:
        norm = pickle.load(f)
        if args.norm_state:
            state_norm = norm["state"]
        if args.norm_rewards:
            reward_norm = norm["reward"]

    out_file = os.path.join(DEFAULT_VIDEO_DIR, args.exp_name)
    if not os.path.exists(DEFAULT_VIDEO_DIR):
        os.mkdir(DEFAULT_VIDEO_DIR)
    if not os.path.exists(out_file):
        os.mkdir(out_file)

    video = VideoRecorder(out_file)
    rew_list = []
    for i in range(args.test_num):
        video.init()
        obs = env.reset()
        obs = state_norm(obs)
        rew = 0

        while True:
            # env.render()
            video.record(env)
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a, var = actor(state_tensor)
            logpi = actor.log_pi(state_tensor, a)
            a = torch.squeeze(a, 0).detach().cpu().numpy()
            obs, r, d, _ = env.step(a)

            rew += r
            obs = state_norm(obs)
            if d:
                rew_list.append(rew)
                print("reward", rew)
                video.save(str(i) + ".mp4")

                if (i+1) % 10 == 0:
                    print("teset_", np.mean(rew_list))
                    rew_list = []

                break




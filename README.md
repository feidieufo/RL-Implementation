# RL-Implementation
simple code to reinforcement learning

- pendulum
  - python -um algos.ppo.run_ppo_torch --exp_name xxx --max_grad_norm 0.5 --steps 2048 --anneal_lr  --is_clip_v   --seed 20 --env Pendulum-v0  --is_gae --target_kl 0.03  --lr 0.0003 --norm_state --iteration 1000 --batch 64 --last_v  --norm_rewards returns --a_update 10

- mujoco
  - python -um algos.ppo.run_ppo_torch --exp_name xxx --max_grad_norm 0.5 --steps 2048 --anneal_lr  --is_clip_v   --seed 30 --env HalfCheetah-v2  --is_gae --target_kl 0.07  --lr 0.0003 --norm_state --iteration 2000 --batch 64 --last_v  --norm_rewards returns --a_update 10

- dmc
  - python -um algos.ppo.run_ppo_dmc_torch --exp_name xxx --max_grad_norm 0.5 --steps 2048 --anneal_lr  --is_clip_v   --seed 10 --domain_name cheetah --task_name run --network mlp --target_kl 0.05 --lr 0.0003 --iteration 2000 --batch 64  --encoder_type state --last_v   --is_gae  --a_update 10 --norm_state --norm_rewards returns --c_en 0

- if you want to test dmc and see video
  - python -um algos.ppo.test_ppo_torch --exp_name  xxx --encoder_type state

- if you want to plot results (mujoco)
  - please modify taskname in plot_seed.py
  - then python -um algos.ppo.plot_seed

- output results in data/* when you  train experiments
  - you can find logs(tensorboard),checkpoints(pytorch),args.json and process.txt

- if you know chinese, you can read 
  - 强化学习中的调参经验与编程技巧(on policy 篇)
https://blog.csdn.net/qq_27008079/article/details/108313137

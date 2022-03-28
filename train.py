from wandb.integration.sb3 import WandbCallback
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import matplotlib.pyplot as plt
import os
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 250000,
    "env_name": "MultiCrossingKey-v1",
}

run = wandb.init(
    project="MultiCrossingKey",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True)

env = gym.make('MiniGrid-MultiCrossingKey-v1')
env = Monitor(env)
logdir = "logs"
models_dir = "models/PPO"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

model = PPO("CnnPolicy", env, gamma=0.96, verbose=1, ent_coef=0.01, tensorboard_log=logdir)
model.learn(total_timesteps=250000,
    callback=WandbCallback(
        gradient_save_freq=100,
        verbose=2,
    ))
model.save(f"{models_dir}/PPO")

num_episodes = 5

for eps in range(num_episodes):
    current_reward = []
    obs = env.reset()
    for i in range(100):
        action, _states  = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        current_reward.append(rewards)
        env.render()
        if dones:
            obs = env.reset()
    print(f"Episode: {eps},  Average reward: {np.mean(current_reward)}")

env.close()

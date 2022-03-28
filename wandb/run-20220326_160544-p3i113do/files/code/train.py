import wandb
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

config = {
    "policy": 'CnnPolicy',
    "total_timesteps": 250000
}

wandb.init(
    config=config,
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="MiniGrid-MultiCrossingKey-v1",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
)

env = gym.make('MiniGrid-MultiCrossingKey-v1')
logdir = "logs"
models_dir = "models/PPO"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

model = PPO("CnnPolicy", env, gamma=0.99, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=250000)
wandb.finish()
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

import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import matplotlib.pyplot as plt
import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

args = argparse.ArgumentParser(description='Train a PPO agent')
parser.add_argument('--test', required=True, type=bool, help='Test or Train')

args = parser.pars_args()
env = gym.make('MiniGrid-MultiCrossingKey-v1')
logdir = "logs"
models_dir = "models/PPO"
os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

if not args.test:
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=300000)
    model.save(f"{models_dir}/PPO")
else:
    model = PPO.load(f"{models_dir}/PPO")
    num_episodes = 10
    for eps in range(num_episodes):
        current_reward = []
        obs = env.reset()
        for i in range(1000):
            action, _states  = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            current_reward.append(rewards)
            env.render()
            if dones:
                obs = env.reset()
        print(f"Episode: {eps},  Average reward: {np.mean(current_reward)}")

env.close()

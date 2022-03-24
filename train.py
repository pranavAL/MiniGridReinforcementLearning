import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('MiniGrid-MultiCrossingKey-v1')
logdir = "logs"
os.makedirs(logdir, exist_ok=True)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=250000)
num_episodes = 1

for eps in range(num_episodes):
    current_reward = []
    obs = env.reset()
    for i in range(1000):
        action, _states  = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        current_reward.append(rewards)
        #env.render()
        if dones:
            obs = env.reset()
    print(f"Episode: {eps},  Average reward: {np.mean(current_reward)}")

env.close()

#!/usr/bin/env python3
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

env = gym.make('MiniGrid-MultiCrossingKey-v1')
#check_env(env)

print(f"Observation Space: {env.observation_space}")
print(f"Shape: {env.observation_space.shape}")
print(f"Action space: {env.action_space}")

obs = env.reset()
print(type(obs))
img = env.render('rgb_array')
action = env.action_space.sample()
print(f"Sampled action: {action}")
obs, reward, done, info = env.step(action)
plt.imshow(obs)

print(obs.shape, reward, done, info)

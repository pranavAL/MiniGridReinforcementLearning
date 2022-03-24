#!/usr/bin/env python3
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import matplotlib.pyplot as plt

env = gym.make('MiniGrid-MultiCrossingKey-v1')

print(f"Observation Space: {env.observation_space}")
print(f"Shape: {env.observation_space.shape}")
print(f"Action space: {env.action_space}")

obs = env.reset()
img = env.render('rgb_array')
action = env.action_space.sample()
print(f"Sampled action: {action}")
obs, reward, done, info = env.step(action)
plt.imshow(obs)

print(obs.shape, reward, done, info)

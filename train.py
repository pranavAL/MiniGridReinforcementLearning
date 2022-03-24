import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from stable_baselines3 import DQN

env = gym.make('MiniGrid-MultiCrossingKey-v1')

model = DQN("CnnPolicy", env, buffer_size=5000, verbose=1)
model.learn(total_timesteps=10000)
num_episodes = 200

for eps in range(num_episodes):
    current_reward = []
    state = env.reset()
    for i in range(1000):
        action, state = model.predict(state, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        current_reward.append(rewards)
        env.render()
        if dones:
            state = env.reset()
    print(f"Episode: {eps},  Average reward: {np.mean(current_reward)}")
env.close()

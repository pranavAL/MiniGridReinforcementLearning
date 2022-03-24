import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('MiniGrid-MultiCrossingKey-v1')

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
num_episodes = 1

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

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward}")
env.close()

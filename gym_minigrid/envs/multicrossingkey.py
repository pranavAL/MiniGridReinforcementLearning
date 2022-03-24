import cv2
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.window import Window
from math import dist

import itertools as itt

IMG_HEIGHT = 64
IMG_WIDTH = 64

class MultiKeyCrossingEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=11, num_crossings=2, obstacle_type=Wall,
                        obs1=2, obs2=3, seed=None):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        self.path1_obstacles = obs1
        self.path2_obstacles = obs2

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

        self.window = Window('gym_minigrid - MultiCrossingKey-v1')

    def reset(self):
        obs = MiniGridEnv.reset(self)
        image = self.render('rgb_array')
        obs = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        return obs

    def step(self, action):
        penalty = 0
        reward = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type == 'ball'

        # Update obstacle positions
        obst_loc = []
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            new_pos = [old_pos[0], max((old_pos[1]+1)%10,1)]
            self.put_obj(self.obstacles[i_obst], *new_pos)
            self.grid.set(*old_pos, None)
            obst_loc.append(new_pos)

        # Update the agent's position/direction
        obs, _, done, info = MiniGridEnv.step(self, action)
        image = self.render('rgb_array')
        obs = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        dist_from_goal = dist(self.agent_pos, self.goal_pos)
        reward = 1 - (dist_from_goal/12.0)

        if list(self.agent_pos) in obst_loc:
            penalty = -1
            done = True
            reward = reward + penalty
            return obs, reward, done, info
        return obs, reward, done, info

    def render(self, mode='human'):
        img = MiniGridEnv.render(self, mode)
        if mode=='human':
            self.window.show_img(img)
        return img

class MultiCrossingKeyEnv(MultiKeyCrossingEnv):
    def __init__(self):
        super().__init__(size=11, num_crossings=1, obstacle_type=Wall,
                         obs1=2, obs2=3)

register(
    id='MiniGrid-MultiCrossingKey-v1',
    entry_point='gym_minigrid.envs:MultiCrossingKeyEnv'
)

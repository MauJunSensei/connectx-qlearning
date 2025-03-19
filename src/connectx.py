import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import gymnasium as gym
from kaggle_environments import make
from src import config as cfg

class ConnectX(gym.Env):
    def __init__(self):
        self.env = make("connectx", debug=True)
        self.trainer = self.env.train([None, cfg.AGENT])

        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

    def step(self, action):
        return self.trainer.step(action)

    def reset(self, **kwargs):
        return self.trainer.reset()

    def render(self, **kwargs):
        return self.env.render(**kwargs)
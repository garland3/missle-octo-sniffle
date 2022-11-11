import gym
from gym import spaces
from missle_defense import SCREEN_WIDTH, SCREEN_HEIGHT
import numpy as np

N_DISCRETE_ACTIONS = 3

class Missle_Env(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

#   def __init__(self, arg1, arg2, ...):
  def __init__(self, arg1, arg2):
      
    super(Missle_Env, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    ...
    return observation, reward, done, info
  def reset(self):
    ...
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    ...
  def close (self):
    ...
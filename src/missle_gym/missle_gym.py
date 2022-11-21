import os
import gym
from gym import spaces
import pygame
from missle_defense.missle_game import SCREEN_WIDTH, SCREEN_HEIGHT, MissleGame
import numpy as np
import matplotlib.pylab as plt

N_DISCRETE_ACTIONS = 3

N_CHANNELS = 5


class Missle_Env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    #   def __init__(self, arg1, arg2, ...):
    def __init__(self):

        super(Missle_Env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self.game = MissleGame(True, False)

        data_per_missle = 3
        # self.observation_space = spaces.Box(low=-10,high=SCREEN_WIDTH, shape(data_per_missle, dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, N_CHANNELS), dtype=np.uint8)
        self.observation = np.random.randint(0, 255, size=(
            SCREEN_HEIGHT, SCREEN_WIDTH, N_CHANNELS), dtype=np.uint8)
        # self.observation_space = spaces.Dict()
        self.screen_buffer = np.zeros(
            (SCREEN_HEIGHT, SCREEN_WIDTH, N_CHANNELS), dtype=np.uint8)

    def get_screen(self):
        temp = pygame.surfarray.array3d(self.game.screen).transpose((1, 0, 2))

        # I think that this is preventing a segfault crash
        # Sometimes, temp would cause a segfault crash when trying to calculate the mean
        if temp is not None:
            
            # get the mean of the 3 channels, basically convert to grayscale
            self.np_screen = np.mean(temp, axis=2).astype(np.uint8)  # convert to uint8
            self.np_screen = self.np_screen[:,:,None]  # add a channel dimension
            
            # move the data back onto the screen buffer
            self.screen_buffer = np.roll(self.screen_buffer, -1, axis=2)
            
            # print(f"screen_buffer.shape: {self.screen_buffer.shape}")
            # print(f"np_screen.shape: {self.np_screen.shape}")
            self.screen_buffer[:, :, 0] = self.np_screen[:, :,0]

        return self.np_screen

    def step(self, action):
        # ...
        self.game.step(action)
        self.observation = self.get_screen()  # pygame.surfarray.array3d(self.game.screen)
        reward = self.game.score
        done = not self.game.running
        info = {}

        return self.observation, reward, done, info

    def reset(self):
        # self.game.end_the_game()
        del self.game
        self.game = MissleGame(True, False)
        self.game.step()
        self.observation = self.get_screen()  # pygame.surfarray.array3d(self.game.screen)
        # print(f"reset. dtype: {observation.dtype}, shape: {observation.shape}, max: {observation.max()}, min: {observation.min()}")
        # plt.imshow(observation)
        # plt.savefig("images/reset.png")
        # plt.close('all')
        return self.observation  # reward, done, info can't be included

    def render(self, mode='human'):
        os.makedirs("images", exist_ok=True)
        filename = f"images/{self.game.cnt}.png"
        plt.imshow(self.observation)
        plt.savefig(filename)
        plt.close('all')
        pass

    def close(self):
        pass

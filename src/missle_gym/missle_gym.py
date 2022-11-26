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
        del self.game
        self.game = MissleGame(True, False)
        self.game.step()
        self.observation = self.get_screen()  # pygame.surfarray.array3d(self.game.screen)
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
    
from missle_defense.missle_game import SCREEN_WIDTH, SCREEN_HEIGHT, MissleGame
class Missle_Env_ObAsVector(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        super(Missle_Env_ObAsVector, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_space.low = 0
        self.action_space.high = 3
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        self.game = MissleGame(True, False)
        # self.observation_space = spaces.Box(low=-10,high=SCREEN_WIDTH, shape(data_per_missle, dtype=np.uint8)
        
        high = max(SCREEN_WIDTH, SCREEN_HEIGHT)
        # self.observation_dtype = np.int16
        self.observation_dtype = float
        
        self.observe_n_missles = 60
        self.previous_n_frames = 5
        # need the, x and y position of each missle. 
        # also, need the last 5 frames of game state
        self.ob_size = (2, self.observe_n_missles,self.previous_n_frames)
        self.observation_space = spaces.Box(low=0, high=high,
                                            shape=self.ob_size, dtype=self.observation_dtype)
        # self.observation = np.random.randint(0, high,size=ob_size, dtype=np.uint16)
        self.observation = np.zeros(self.ob_size, dtype=self.observation_dtype)
        
    # def init(self, **kwargs):
    #     self.
       

    def update_observation(self):
        self.observation = np.roll(self.observation, -1, axis=2)
        for i,m in enumerate(self.game.missles):
            if i>=self.observe_n_missles-1:
                break
            self.observation[0, i] = m.center[0]
            self.observation[1, i] = m.center[1]
        
        # fig, axs = plt.subplots(1,self.previous_n_frames, figsize=(20, 5))
        # for i in range(self.previous_n_frames):
        #     for j in range(self.observe_n_missles):
        #         axs[i].scatter(self.observation[0,j,i], self.observation[1,j,i])
        #     axs[i].set_xlim(0, SCREEN_WIDTH)
        #     axs[i].set_ylim(0, SCREEN_HEIGHT)
            
        #     # axs[i].imshow(self.observation[:,:,i])
        # os.makedirs("images", exist_ok=True)
        # plt.savefig(f"images/observation{self.game.cnt}.png")
        return self.observation
            
        

    def step(self, action):
        # ...
        self.game.step(action)
        self.observation = self.update_observation()  # pygame.surfarray.array3d(self.game.screen)
        reward = self.game.score
        done = not self.game.running
        info = {}
        return_tuple =  self.observation, reward, done, info
        # print(return_tuple)
        return return_tuple
    
    def reset(self):
        # del self.game
        self.game = MissleGame(True, False)
        self.game.step()
        self.observation = self.update_observation()  # pygame.surfarray.array3d(self.game.screen)
        # print(f"resetting observation: {self.observation}")
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

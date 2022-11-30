import math
import os
from pathlib import Path
import random
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
        self.action_space.high = 2
        self.action_range = self.action_space.high - self.action_space.low
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        self.game = MissleGame(True, False)
        # self.observation_space = spaces.Box(low=-10,high=SCREEN_WIDTH, shape(data_per_missle, dtype=np.uint8)
        
        self.high = max(SCREEN_WIDTH, SCREEN_HEIGHT, 360) # 360 is the max angle
        self.high_position_only = max(SCREEN_WIDTH, SCREEN_HEIGHT)
        # self.observation_dtype = np.int16
        self.observation_dtype = float
        
        self.observe_n_missles_or_bullets = 5
        # self.observe_n_bullets = 60
        self.previous_n_frames = 3

        # the 3 is for the missles and bullets and then the angle of the gun
        # the 20 is for the x and y position (update, try forier embedding like in the paper)
        self.embedding_size = 1
        # the 5 is for the last 5 frames
        self.ob_size = (3, 2*self.embedding_size, self.observe_n_missles_or_bullets,self.previous_n_frames)
        print(f"Observation size: {self.ob_size}")
        self.observation_space = spaces.Box(low=0, high=self.high,
                                            shape=self.ob_size, dtype=self.observation_dtype)
        # self.observation = np.random.randint(0, high,size=ob_size, dtype=np.uint16)
        self.observation = self.norm_position(np.zeros(self.ob_size, dtype=self.observation_dtype))
        
        self.games_played = 0
        
    # def init(self, **kwargs):
    #     self.
       
    def norm_position(self, position):
        return (position - self.high_position_only) / self.high_position_only
    
    def de_norm_position(self, position):
        return position * self.high_position_only + self.high_position_only

    def update_observation(self):
        self.observation = np.roll(self.observation, 1, axis=3)
        all_but_last = self.ob_size[0:-1]
        # clear the first frame
        self.observation[:,:,:,0]=self.norm_position(np.zeros(all_but_last, dtype=self.observation_dtype))
        for i,m in enumerate(self.game.missles):
            if i>=self.observe_n_missles_or_bullets-1:
                break
            self.observation[0,0 , i,0] = self.norm_position(m.center[0])
            self.observation[0,1, i,0] =self.norm_position(m.center[1])
            
            # self.observation[0,0:self.embedding_size , i,0] = np.array([np.sin(i*np.pi*m.center[0]/ self.high) for i in range(self.embedding_size)])
            # self.observation[0, self.embedding_size:, i,0] =np.array( [np.sin(i*np.pi*m.center[1]/ self.high) for i in range(self.embedding_size)])
            
        for i,b in enumerate(self.game.bullets):
            if i>=self.observe_n_missles_or_bullets-1:
                break
            self.observation[1,0, i,0] =self.norm_position(b.center[0])
            self.observation[1,1, i,0] =self.norm_position(b.center[1])
        
        # record the angle of the gun
        k,i = 2,0
        self.observation[k,i, 0,0] =np.sin( (math.pi / 180) * self.game.rotation)
        self.observation[k,i, 1,0] =np.cos(  (math.pi / 180) *self.game.rotation)
        
        if self.games_played% 200==100:
            # print(f"observation: {self.observation}")
            print(f"Sample observation to img and np array: {self.games_played}_{self.game.cnt}")
            os.makedirs(f"images/{self.games_played}", exist_ok=True)
            # os.makedirs("images", exist_ok=True)
            
            # images_dir = Path("images")
            # pngs_cnt = len(list(images_dir.glob("*.png")))
            fig, axs = plt.subplots(1,self.previous_n_frames, figsize=(20, 5))
            for i in range(self.previous_n_frames):
                for j in range(self.observe_n_missles_or_bullets):
                    axs[i].scatter(self.de_norm_position(self.observation[0,0,j,i]), self.de_norm_position(self.observation[0,1,j,i]), c='r')
                for j in range(self.observe_n_missles_or_bullets):
                    axs[i].scatter(self.de_norm_position(self.observation[1,0,j,i]), self.de_norm_position(self.observation[1,1,j,i]), c='b')
                axs[i].set_xlim(0, SCREEN_WIDTH)
                axs[i].set_ylim(0, SCREEN_HEIGHT)
                rotation = self.observation[2,0,0,i]
                axs[i].set_title(f"frame {i} cnt would be: {self.game.cnt-i} rotation: {rotation}.\n games played: {self.games_played}")
                axs[i].invert_yaxis()
            plt.tight_layout()
            filename = f"images/{self.games_played}/{self.games_played}_{self.game.cnt}.png"
            plt.savefig(filename)
            plt.close('all')
            os.makedirs("numpyarrays", exist_ok=True)
            # numpyarrays_dir = Path("numpyarrays")
            # number_of_numpyarrays = len(list(numpyarrays_dir.glob("*.npy")))            
            
            np.save(f"numpyarrays/{self.games_played}_{self.game.cnt}.npy", self.observation)
                
                # axs[i].imshow(self.observation[:,:,i])
            # plt.savefig(f"images/observation{self.game.cnt}.png")
        return self.observation
            
        

    def step(self, action):
        # ...
        # print(f"step action {action}, games played: {self.games_played} and cnt: {self.game.cnt}")
        # action-=1
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
        self.games_played += 1
        self.game = MissleGame(True, False)
        # self.observation = np.zeros(self.ob_size, dtype=self.observation_dtype)
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

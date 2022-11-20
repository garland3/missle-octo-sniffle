# import gym

# from stable_baselines3 import A2C

# env = gym.make("CartPole-v1")

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()

import numpy as np
from stable_baselines3.common.env_checker import check_env
from missle_gym.missle_gym import Missle_Env

env = Missle_Env()
# Box(4,) means that it is a Vector with 4 components
# print("Observation space:", env.observation_space)
# print("Shape:", env.observation_space.shape)
# # Discrete(2) means that there is two discrete actions
print("Action space:", env.action_space)


# It will check your custom environment and output additional warnings if needed


#  $env:KMP_DUPLICATE_LIB_OK="TRUE"
# check_env(env)

# obs = env.reset()
# env.render()
# # print(env.observation_space)
# print(env.action_space)
# print(env.action_space.sample())

# turn_left = 2
# # Hardcoded best agent: always go left!
# n_steps = 200
# for step in range(n_steps):
#     # print("Step {}".format(step + 1))
#     if step<10:
#         action_to_take=1
#     else:
#         action_to_take=2
#     obs, reward, done, info = env.step(action_to_take)
#     #   print('obs=', obs, 'reward=', reward, 'done=', done)
#     # print(f"Step {step+1}. dtype: {obs.dtype}, shape: {obs.shape}, max: {obs.max()}, min: {obs.min()}")
#     env.render()
#     if done:
#         print("Goal reached!", "reward=", reward)
#         break

from stable_baselines3 import DQN #, PPO2, A2C, ACKTR
# from stable_baselines3.common.c
# from stable_baselines3.common.cmd_util import make_vec_env

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

attempt1 = 1
if attempt1==0:
    # Instantiate the env
    env = Missle_Env()
    # Train the agent
    model = DQN('MlpPolicy', env, verbose=1, buffer_size=5000)# 
    model.learn(total_timesteps=1000)
    model.save("dqn_missle")
    
if attempt1==1:
    
    # class VecExtractDictObs(VecEnvWrapper):
    #     """
    #     A vectorized wrapper for filtering a specific key from dictionary observations.
    #     Similar to Gym's FilterObservation wrapper:
    #         https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

    #     :param venv: The vectorized environment
    #     :param key: The key of the dictionary observation
    #     """

    #     def __init__(self, venv: VecEnv, key: str):
    #         self.key = key
    #         super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    #     def reset(self) -> np.ndarray:
    #         obs = self.venv.reset()
    #         return obs[self.key]

    #     def step_async(self, actions: np.ndarray) -> None:
    #         self.venv.step_async(actions)

    #     def step_wait(self) -> VecEnvStepReturn:
    #         obs, reward, done, info = self.venv.step_wait()
    #         return obs[self.key], reward, done, info
        
    def make_env():
        env = Missle_Env()
        # Important: use a wrapper that flattens the output of `step()`
        # env = VecExtractDictObs(env, "observation")
        return env
    
    # env = DummyVecEnv([lambda: make_env(),lambda: make_env()])
    env = DummyVecEnv([ Missle_Env for i in range(3)])
    
    # Wrap the VecEnv
    # env = VecExtractDictObs(envdummy, key="observation")


    # Instantiate the env
    # env = Missle_Env()
    # # Train the agent
    model = DQN('MlpPolicy', env, verbose=1, buffer_size=10000)# 
    model.learn(total_timesteps=10000, progress_bar=True)
    model.save("dqn_missle")
from stable_baselines3.common.env_checker import check_env
from missle_gym.missle_gym import Missle_Env, Missle_Env_ObAsVector
from stable_baselines3 import DQN #, PPO2, A2C, ACKTR

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
    
MyEnvClass = Missle_Env_ObAsVector

attempt1 = 1
if attempt1==0:
    # Instantiate the env
    env = MyEnvClass()
    # Train the agent
    model = DQN('MlpPolicy', env, verbose=1, buffer_size=5000)# 
    model.learn(total_timesteps=1000)
    model.save("dqn_missle")
    
if attempt1==1:
    
    from stable_baselines3 import PPO
    n_envs = 4
    env = DummyVecEnv([ MyEnvClass for i in range(n_envs)])
    # model = PPO('CnnPolicy', env, verbose=1, n_steps=1000, tensorboard_log="./tensor_board/") # 
    
    # model = PPO('MlpPolicy', env, verbose=2, n_steps=1000,batch_size=1000*n_envs, tensorboard_log="./tensor_board/")# 
    model = DQN('MlpPolicy', env, verbose=1, buffer_size=5000, batch_size=5000*n_envs)# 
    
    model.learn(total_timesteps=500000, progress_bar=True, log_interval=5)
    model.save("ppo_missle")
    
# tensorboard --logdir ./tensor_board/
# https://stackoverflow.com/a/36608933/1319433
# Xvfb :3 -screen 0 1920x1080x24+32 -fbdir /var/tmp &
# export DISPLAY=:3
import copy
from pathlib import Path
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.memories.torch import Memory
from skrl.memories.torch import RandomMemory, PrioritizedMemory
import numpy as np
# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin


from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
# from skrl.memories.torch.base import RandomMemory, Memory
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.trainers.torch.parallel import PARALLEL_TRAINER_DEFAULT_CONFIG
# from skrl.resources.preprocessors.torch import Normalizer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.envs.torch import wrap_env
from missle_gym.missle_gym import Missle_Env, Missle_Env_ObAsVector
from   skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG
# from  skrl.agents.torch.q_learning.q_learning import Q_LEARNING_DEFAULT_CONFIG, Q_LEARNING
import os
#  skrl.memory.torch.Memory

#  DISPLAY=:3
os.environ["DISPLAY"] = ":3"

models = {}
env1 = Missle_Env_ObAsVector()
# env_multiple = gym.vector.make("Pendulum-v1", num_envs=10, asynchronous=False)
env = wrap_env(env1,  wrapper="gym")
env.device = 'cuda'
device = env.device

# images_dir = Path("images")
if  os.path.exists("images"):
    shutil.rmtree("images")
    print("images dir deleted")
if os.path.exists("numpyarrays"):
    shutil.rmtree("numpyarrays")
    print("numpyarrays dir deleted")
    



# class G_Actor(GaussianMixin, Model):

#     def __init__(self, observation_space, action_space, device, clip_actions=True):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions)
class D_Actor(DeterministicMixin, Model):

    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.device  = device      
        self.make_model()
        # self.log_std_model = self.make_model()    

    # def make_model(self, hidden_size=64):
    #     n_layers = 2
    #     layer_list = []
    #     self.linear_layer_1 = nn.Linear(self.num_observations, hidden_size)
    #     layer_list.append(self.linear_layer_1)
    #     for i in range(n_layers):
    #         # layer_list.append(nn.BatchNorm1d(hidden_size))
    #         layer_list.append(nn.Linear(hidden_size, hidden_size))
    #         layer_list.append(nn.ReLU())
    #     layer_list.append(nn.Linear(hidden_size, self.num_actions))
    #     return nn.Sequential(*layer_list)
    
    def make_model(self, hidden_size=64):
        embedding_size = 64
        self.key_value_query_list = [ nn.Linear(self.num_observations, embedding_size).to(self.device) for i in range(3)]
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=4, dropout=0.1).to(self.device)
        
        self.key_value_query_list2 = [ nn.Linear(embedding_size, embedding_size).to(self.device) for i in range(3)]
        self.multi_head_attention2 = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=4, dropout=0.1).to(self.device)
        
        self.final_layer = nn.Linear(embedding_size, self.num_actions).to(self.device)
        # n_layers = 2
        # layer_list = []
        # self.linear_layer_1 = nn.Linear(self.num_observations, hidden_size)
        # layer_list.append(self.linear_layer_1)
        # for i in range(n_layers):
        #     # layer_list.append(nn.BatchNorm1d(hidden_size))
        #     layer_list.append(nn.Linear(hidden_size, hidden_size))
        #     layer_list.append(nn.ReLU())
        # layer_list.append(nn.Linear(hidden_size, self.num_actions))
        # return nn.Sequential(*layer_list)
        
    def compute(self, states, taken_actions, role):
        # print("states.shape", states.shape)
        key, value, query = [ key_value_query(states) for key_value_query in self.key_value_query_list]
        attn_output, attn_output_weights = self.multi_head_attention(query, key, value)
        
        key, value, query = [ key_value_query(attn_output) for key_value_query in self.key_value_query_list2]
        attn_output2, attn_output_weights2 = self.multi_head_attention2(query, key, value)
        
        
        means = self.final_layer(attn_output2)
        means = 1+means*3
        # print("attn_output.shape", attn_output.shape)
        # means = []
        # means  = self.mean_model(states)
        # std = self.log_std_model(states)
        return means
        # x = F.relu(self.linear_layer_2(x))
        # return 3 * torch.tanh(self.action_layer(x)) , self.log_std_parameter  # Pendulum-v1 action_space is -2 to 2

# class PreprocessStates(skrl.resources.torch.preprocessors.torch.Preprocessor):

# memory = RandomMemory(memory_size=100000, num_envs=env.num_envs, device=device)
memory = PrioritizedMemory(memory_size=100_000, num_envs=env.num_envs, device=device)


models['q_network'] = D_Actor(env.observation_space, env.action_space, device, clip_actions=True)
models['target_q_network'] = D_Actor(env.observation_space, env.action_space, device, clip_actions=True)

config = copy.deepcopy(DDQN_DEFAULT_CONFIG)
# config.update({})
config['experiment']['write_interval'] = 100
config['experiment']['directory'] = 'missle_ddqn'
config['experiment']['name'] = 'missle_ddqn'
config['experiment']['device'] = device
config['experiment']['checkpoint_interval'] = 5000
config['experiment']['random_timesteps'] = 10000
config['exploration']['time_steps'] = 90000
config['update_interval'] = 1000
config['target_update_interval'] =config['update_interval'] * 100
config['learning_rate'] = 1e-3

# n = env.observation_space
config['state_preprocessor'] = RunningStandardScaler
config['state_preprocessor_kwargs']={'size' : env.ob_size, 'device' :  device}
config["value_preprocessor"] = RunningStandardScaler
config['value_preprocessor_kwargs']={'size' : 1, 'device' :  device}

# config['value_preprocessor'] = RunningStandardScaler(env.action_space, device)

class myDDQN(DDQN):
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            sampled_states = self._state_preprocessor(sampled_states, train=not gradient_step)
            sampled_next_states = self._state_preprocessor(sampled_next_states)
            

            # compute target values
            with torch.no_grad():
                self.track_data("sampled_states / mean_value", torch.mean(sampled_states).item())
                # hist, _  = np.histogram(sampled_actions.cpu().numpy(), bins = 3, density=True )
                # self.writer.add_histogram("sampled_actions / histogram",hist , timestep)
                self.writer.add_histogram("sampled_actions / histogram",sampled_actions , timestep)
                
                
                next_q_values, _, _ = self.target_q_network.act(states=sampled_next_states, taken_actions=None, role="target_q_network")
                
                target_q_values = torch.gather(next_q_values, dim=1, index=torch.argmax(self.q_network.act(states=sampled_next_states, \
                    taken_actions=None, role="q_network")[0], dim=1, keepdim=True))
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute Q-network loss
            q_values = torch.gather(self.q_network.act(states=sampled_states, taken_actions=None, role="q_network")[0], 
                                    dim=1, index=sampled_actions.long())

            q_network_loss = F.mse_loss(q_values, target_values)
            
            # optimize Q-network
            self.optimizer.zero_grad()
            q_network_loss.backward()
            self.optimizer.step()

            # update target network
            if not timestep % self._target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

agent =myDDQN(models,memory, observation_space = env.observation_space, action_space=env.action_space, cfg = config)#
print(type(agent), "agent type. ")


cfg_trainer = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG )

cfg_trainer = {"timesteps": 1500000, "headless": True}
trainer = SequentialTrainer(env,agent,  config, cfg_trainer)

trainer.train()

# # tensorboard --logdir ./missle_ddqn/
# # https://stackoverflow.com/a/36608933/1319433
# # Xvfb :3 -screen 0 1920x1080x24+32 -fbdir /var/tmp &
# # export DISPLAY=:3
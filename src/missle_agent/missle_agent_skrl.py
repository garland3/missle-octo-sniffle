import copy
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
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
import wandb

wandb.init(project="missle_ddqn")

#  skrl.memory.torch.Memory

#  DISPLAY=:3
os.environ["DISPLAY"] = ":3"

def get_wandb_config_value_if_exists(key, default_value):
    if key in wandb.config:
        return wandb.config[key]
    else:
        print(f"WARNING: {key} not found in wandb.config, using default value {default_value}")
        return default_value

models = {}
previous_n_frames = get_wandb_config_value_if_exists("previous_n_frames", 3)
env1 = Missle_Env_ObAsVector(previous_n_frames=previous_n_frames)

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
    
if os.path.exists("memories"):
    shutil.rmtree("memories")
    print("memories dir deleted")


# class G_Actor(GaussianMixin, Model):

#     def __init__(self, observation_space, action_space, device, clip_actions=True):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions)
class D_Actor(DeterministicMixin, Model):

    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.device  = device     
        
        self.attention_model = True
        if self.attention_model:     
            self.make_attention_model()
        else:
            self.make_mlp_model()
        # self.log_std_model = self.make_model()    

    def make_mlp_model(self, hidden_size=64):
        n_layers = 3
        layer_list = []
        self.linear_layer_1 = nn.Linear(self.num_observations, hidden_size)
        layer_list.append(self.linear_layer_1)
        for i in range(n_layers):
            # layer_list.append(nn.BatchNorm1d(hidden_size))
            layer_list.append(nn.Linear(hidden_size, hidden_size))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(hidden_size, self.num_actions))
        self.mlp_model =  nn.Sequential(*layer_list)
    
    def compute_mlp(self,states, taken_actions, role):
        x = self.mlp_model(states)
       
        return x
    
    def make_attention_model(self, embedding_size=8):
        self.embedding_size = embedding_size
        self.key_value_query_list = [ nn.Linear(self.num_observations, embedding_size).to(self.device) for i in range(3)]
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=4, dropout=0.1).to(self.device)
        
        # self.key_value_query_list2 = [ nn.Linear(embedding_size, embedding_size).to(self.device) for i in range(3)]
        # self.multi_head_attention2 = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=4, dropout=0.1).to(self.device)
        
        # self.key_value_query_list3 = [ nn.Linear(embedding_size, embedding_size).to(self.device) for i in range(3)]
        # self.multi_head_attention3 = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=4, dropout=0.1).to(self.device)
        
        self.value_and_adv_pred_layer = nn.Linear(embedding_size, self.embedding_size*2).to(self.device)
        self.value_layer = nn.Linear(self.embedding_size,self.num_actions).to(self.device)
        self.advantage_layer = nn.Linear(self.embedding_size,self.num_actions).to(self.device)
      
    def compute_attention(self,states, taken_actions, role):
        """
        use multi head attention to compute the value and advantage
        then compute the q value
        
        """
        key, value, query = [ key_value_query(states) for key_value_query in self.key_value_query_list]
        attention_output, attention_weights = self.multi_head_attention(query, key, value)
        
        # key, value, query = [ key_value_query(attention_output) for key_value_query in self.key_value_query_list2]
        # attention_output, attention_weights = self.multi_head_attention2(query, key, value)
        
        # key, value, query = [ key_value_query(attention_output) for key_value_query in self.key_value_query_list3]
        # attention_output, attention_weights = self.multi_head_attention3(query, key, value)
        
        x  =  self.value_and_adv_pred_layer(attention_output)
        value_pred = self.value_layer(x[:,0:self.embedding_size])
        advantage_pred = self.advantage_layer(x[:,self.embedding_size:])
        means = value_pred + advantage_pred - advantage_pred.mean(dim=1, keepdim=True)
      
        return means
    
    
        
    def compute(self, states, taken_actions, role):
        # print("compute, states.shape", states.shape)
        if self.attention_model:
            return self.compute_attention(states, taken_actions, role)
        elif self.attention_model == False:
            return self.compute_mlp(states, taken_actions, role)
        else:
            raise ValueError("attention_model should be True or False")

# class PreprocessStates(skrl.resources.torch.preprocessors.torch.Preprocessor):

# memory = RandomMemory(memory_size=100000, num_envs=env.num_envs, device=device)
# memory = PrioritizedMemory(memory_size=100_000, num_envs=env.num_envs, device=device)

models['q_network'] = D_Actor(env.observation_space, env.action_space, device, clip_actions=True)
models['target_q_network'] = D_Actor(env.observation_space, env.action_space, device, clip_actions=True)

config = copy.deepcopy(DDQN_DEFAULT_CONFIG)
# config.update({})
config['experiment']['write_interval'] = 100
config['experiment']['directory'] = 'missle_ddqn'
config['experiment']['name'] = 'missle_ddqn'
config['experiment']['device'] = device
config['experiment']['checkpoint_interval'] = 5000
# config['experiment']['random_timesteps'] = 10000
# config['exploration']['time_steps'] = 200_000
config["exploration"]["final_epsilon"]=0.2
config['update_interval'] = 1
config['target_update_interval'] =get_wandb_config_value_if_exists("target_update_interval", 10)
config['learning_rate'] =  get_wandb_config_value_if_exists("learning_rate", 1e-3)
config['memorysize'] = 100_000
config['batch_size'] = get_wandb_config_value_if_exists("batch_size", 64*4)


memory = PrioritizedMemory(memory_size=config['memorysize'] , num_envs=env.num_envs, device=device)

wandb.config.update(config)
# wandb.init(project="missle_ddqn", config=config)
# n = env.observation_space
# config['state_preprocessor'] = RunningStandardScaler
# config['state_preprocessor_kwargs']={'size' : env.ob_size, 'device' :  device}


# config['value_preprocessor'] = RunningStandardScaler(env.action_space, device)


def normalize_rewards(rewards):
    # return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return torch.log(rewards+100)/ 1000


class myDDQN(DDQN):
    moving_avg_reward = 0
    average_episode_reward_update_rate = 0.001
    
    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        """Write tracking data to TensorBoard

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # print("write_tracking_data in myDDQN")
        new_data = {}
        for k, v in self.tracking_data.items():
            if k.endswith("(min)"):
                # self.writer.add_scalar(k, np.min(v), timestep)
                value = np.min(v)
            elif k.endswith("(max)"):
                value = np.max(v)
            else:
                value = np.mean(v)
            # print(f"key {k} value {value} at timestep {timestep}")
            new_data[k] = value    
        # wandb.log(new_data, step=timestep)
        wandb.log(new_data)
        
        # print(f"new_data {new_data} at timestep {timestep}")
        # reset data containers for next iteration
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()
        
        
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
            
        avg_reward = sampled_rewards.mean().cpu()
        ee = self.average_episode_reward_update_rate
        self.moving_avg_reward = ee * self.moving_avg_reward + (1-ee)* avg_reward
        self.track_data("Reward / moving_avg_reward", self.moving_avg_reward)
            
        # gradient steps
        for gradient_step in range(self._gradient_steps):
            sampled_states = self._state_preprocessor(sampled_states, train=not gradient_step)
            sampled_next_states = self._state_preprocessor(sampled_next_states)

            # compute target values
            with torch.no_grad():
                self.track_data("sampled_states / mean_value", torch.mean(sampled_states).item())
                # wandb.log({"sampled_states / mean_value": torch.mean(sampled_states.cpu()).item()})
                
                # hist, _  = np.histogram(sampled_actions.cpu().numpy(), bins = 3, density=True )
                # self.writer.add_histogram("sampled_actions / histogram",hist , timestep)
                wandb.log({"sampled_actions / histogram": wandb.Histogram(sampled_actions.cpu().numpy())}, step=timestep)
                
                # log sampled sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones a every 1000 steps
                if timestep % 1000 == 0:
                    wandb.log({"sampled_states / histogram": wandb.Histogram(sampled_states.cpu())})
                    wandb.log({"sampled_next_states / histogram": wandb.Histogram(sampled_next_states.cpu())})
                    wandb.log({"sampled_rewards / histogram": wandb.Histogram(sampled_rewards.cpu())})
                    wandb.log({"sampled_dones / histogram": wandb.Histogram(sampled_dones.cpu())})
                    wandb.log({"sampled_actions / histogram": wandb.Histogram(sampled_actions.cpu())})
                    
                    fig = plt.figure()
                    print("sampled_states.shape", sampled_states.shape)
                    plt.plot(np.transpose( sampled_states.cpu().numpy(), (1,0)))
                    wandb.log({"sampled_states / figure": wandb.Image(fig)})
                    plt.close(fig)
                    
                sampled_rewards = normalize_rewards(sampled_rewards)
                
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
            self.track_data("Loss / Q-network loss divided by target_values", q_network_loss.item()/target_values.mean().cpu())
            

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())
           

            # if self._learning_rate_scheduler:
                # self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
                # wandb.log({"Learning / Learning rate": self.scheduler.get_last_lr()[0]})
            
            # log optimizer learning rate
            wandb.log({"Learning / Learning rate": self.optimizer.param_groups[0]['lr']})
            

agent =myDDQN(models,memory, observation_space = env.observation_space, action_space=env.action_space, cfg = config)#
print(type(agent), "agent type. ")


cfg_trainer = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG )

timesteps = 120_000
wandb.config.update({"timesteps": timesteps})
cfg_trainer = {"timesteps": timesteps, "headless": True}
trainer = SequentialTrainer(env,agent,  config, cfg_trainer)

print(f"Config is {config}")
print("---------------------------------------")
print("Wandb config is ", wandb.config)
trainer.train()

# # tensorboard --logdir ./missle_ddqn/
# # https://stackoverflow.com/a/36608933/1319433
# # Xvfb :3 -screen 0 1920x1080x24+32 -fbdir /var/tmp &
# # export DISPLAY=:3

# solved a werid bug by adding this line
# https://askubuntu.com/a/1405450/199696
# DISPLAY=:3 LIBGL_DEBUG=verbose python src/missle_agent/missle_agent_skrl.py 
#  mv /home/garlan/miniconda3/envs/missle2/bin/../lib/libstdc++.so.6 ~/

from typing import Dict
from matplotlib import pyplot as plt
from   skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG
import torch

from typing import Union, Tuple, Dict, Any

import gym
import copy
import math

import torch
import torch.nn.functional as F

from skrl.memories.torch import Memory
from skrl.memories.torch import RandomMemory, PrioritizedMemory

from skrl.models.torch import Model



def normalize_rewards(rewards):
    # return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return torch.log(rewards+1)/ 1000

class myDDQN(DDQN):
    
    # init, call super and then add another network. 
    def __init__(self, 
                 models: Dict[str, Model], 
                 memory: Union[Memory, Tuple[Memory], None] = None, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0", 
                 cfg: dict = {}) -> None:
        _cfg = copy.deepcopy(DDQN_DEFAULT_CONFIG)
        _cfg.update(cfg)
        super().__init__(models=models, 
                         memory=memory, 
                         observation_space=observation_space, 
                         action_space=action_space, 
                         device=device, 
                         cfg=_cfg)

        # models
        # self.q_network = self.models.get("q_network", None)
        # self.target_q_network = self.models.get("target_q_network", None)
        self.two_out_q_network = self.models.get("two_out_q_network", None)

        # checkpoint models
        self.checkpoint_modules["two_out_q_network"] = self.two_out_q_network
        # self.checkpoint_modules["target_q_network"] = self.target_q_network

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]
        
        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        
        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._update_interval = self.cfg["update_interval"]
        self._target_update_interval = self.cfg["target_update_interval"]

        self._exploration_initial_epsilon = self.cfg["exploration"]["initial_epsilon"]
        self._exploration_final_epsilon = self.cfg["exploration"]["final_epsilon"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        
        # set up optimizer and learning rate scheduler
        if self.two_out_q_network is not None:
            self.optimizer = torch.optim.Adam(self.two_out_q_network.parameters(), lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

        self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
    
    
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
            
        
        sampled_rewards = normalize_rewards(sampled_rewards)

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
                
                # log sampled sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones a every 1000 steps
                if timestep % 1000 == 0:
                    self.writer.add_histogram("sampled_states / histogram",sampled_states , timestep)
                    self.writer.add_histogram("sampled_next_states / histogram",sampled_next_states , timestep)
                    self.writer.add_histogram("sampled_rewards / histogram",sampled_rewards , timestep)
                    self.writer.add_histogram("sampled_dones / histogram",sampled_dones , timestep)
                    self.writer.add_histogram("sampled_actions / histogram",sampled_actions , timestep)
                    
                    fig = plt.figure()
                    print("sampled_states.shape", sampled_states.shape)
                    plt.plot(np.transpose( sampled_states.cpu().numpy(), (1,0)))
                    self.writer.add_figure("sampled_states / figure", fig, timestep)
                    plt.close(fig)
                
                
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
import random
from typing import Optional
from importlib_metadata import NullFinder
from numpy import ndarray, size, squeeze
import numpy as np
from dqn.replay_buffer import ReplayBuffer
from model import QModel
import torch
import torch.nn as nn
import copy
import gym

class DQNAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self,
                 obs_dim: int,
                 num_actions: int,
                 learning_rate: float,
                 gamma: float,
                 epsilon_max: Optional[float] = None,
                 epsilon_min: Optional[float] = None,
                 epsilon_decay: Optional[float] = None,
                 replay_size: int = None,
                 state_shape: tuple = None,
                 sample_size: int = None,
                 reset_network_every: int = None):
        """
        :param num_states: Number of states.
        :param num_actions: Number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.nn = QModel(obs_dim, num_actions)
        self.target_nn = copy.copy(self.nn)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer(replay_size, state_shape)
        self.sample_size = sample_size
        self.reset_network_every = reset_network_every
        self.reset_network_counter = 0

    def greedy_action(self, observation) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        return torch.argmax(self.nn(observation)).item()

    def act(self, observation, training: bool = True) -> int:
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        if (not training):
            return self.greedy_action(observation)
        elif(random.uniform(0,1) < self.epsilon):
            return random.randint(0, self.num_actions-1) ;" chooses a random action "
        else: return self.greedy_action(observation)

    def learn(self, obs, act, rew, done, next_obs) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """
        
        self.buffer.add_transition(obs, act, rew,done, next_obs)                        # add the new transition to the buffer
        if (self.sample_size < self.buffer.i):                                          # if the amount of transitions in the buffer is smaller than the sample_size, do not sample
            states, actions, rewards, dones, next_states = self.buffer.sample(self.sample_size)  # get the values of the sampled transitions
            states = torch.tensor(states)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)
            dones = torch.ByteTensor(dones)
            next_states = torch.tensor(next_states)
            q_values = self.nn(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                targets = self.target_nn(next_states).gather(1, self.nn(next_states).max(1)[1].unsqueeze(-1)).squeeze(-1)
                targets[dones] = 0.0
                target_q_values = rewards + self.gamma * targets
            
            lossfunction = nn.MSELoss()
            loss = lossfunction(q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.reset_network_counter += 1                                  # update the target-network to the current network every 'reset_network_every' often
            if (self.reset_network_counter == self.reset_network_every):
                self.reset_network_counter = 0
                self.target_nn.load_state_dict(self.nn.state_dict())
        
            

    def update_epsilon(self):
        if (self.epsilon > self.epsilon_min):
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)


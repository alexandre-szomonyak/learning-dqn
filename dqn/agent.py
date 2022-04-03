import random
from typing import Optional
from numpy import ndarray
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
                 sample_size: int = None):
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
        # Compute the loss !
        self.buffer.add_transition(obs, act, rew,done, next_obs)
        states, actions, rewards, dones, next_actions = self.buffer.sample(self.sample_size)
        with torch.no_grad():
            target = rew + self.gamma * (1-done) * max(self.target_nn(next_obs))
        prediction = self.nn(obs)[act]
        lossfunction = nn.MSELoss()
        loss = lossfunction(prediction, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (done):
            if (self.epsilon > self.epsilon_min):
                self.epsilon = self.epsilon*self.epsilon_decay
            if (self.epsilon < self.epsilon_min):
                self.epsilon = self.epsilon_min



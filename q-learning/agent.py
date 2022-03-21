from re import X
from typing import Optional
from numpy import ndarray, size, zeros
import random


def create_q_table(num_states: int, num_actions: int) -> ndarray:
    """
    Function that returns a q_table as an array of shape (num_states, num_actions) filled with zeros.

    :param num_states: Number of states.
    :param num_actions: Number of actions.
    :return: q_table: Initial q_table.
    """
    # TODO: complete.

    return zeros(shape=(num_states, num_actions))

class QLearnerAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 learning_rate: float,
                 gamma: float,
                 epsilon_max: Optional[float] = None,
                 epsilon_min: Optional[float] = None,
                 epsilon_decay: Optional[float] = None):
        """
        :param num_states: Number of states.
        :param num_actions: Number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = create_q_table(num_states, num_actions)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.num_actions = num_actions

    def greedy_action(self, observation: int) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        # TODO: complete.

        return self.q_table.argmax(axis=1)[observation]

    def act(self, observation: int, training: bool = True) -> int:
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        # TODO: complete.

        if (not training):
            return self.greedy_action(observation)
        elif(random.uniform(0,1) < self.epsilon):
            return random.randint(0, self.num_actions-1) ;" chooses a random action "
        else: return self.greedy_action(observation)

    def learn(self, obs: int, act: int, rew: float, done: bool, next_obs: int) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """
        # TODO: complete.

        self.q_table[obs, act] += self.learning_rate * (rew + (1-done) * self.gamma * self.q_table.max(axis=1)[next_obs] - self.q_table[obs, act])
        if (done):
            if (self.epsilon > self.epsilon_min):
                self.epsilon = self.epsilon*self.epsilon_decay
            if (self.epsilon < self.epsilon_min):
                self.epsilon = self.epsilon_min


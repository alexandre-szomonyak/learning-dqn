from hashlib import new
from lib2to3.pgen2.token import NEWLINE
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import gym
import numpy as np
from gym import Env
from numpy import ndarray
import math
import random

from agent import DQNAgent


def run_episode(env: Env, agent: DQNAgent, training: bool, gamma) -> float:
    """
    Interact with the environment for one episode using actions derived from the q_table and the action_selector.

    :param env: The gym environment.
    :param agent: The agent.
    :param training: If true the q_table will be updated using q-learning. The flag is also passed to the action selector.
    :param gamma: The discount factor.
    :return: The cumulative discounted reward.
    """
    done = False
    obs = env.reset()
    cum_reward = 0.
    t = 0
    while not done:
        action = agent.act(obs, training)
        new_obs, reward, done, _ = env.step(action)
        if training:
            agent.learn(obs, action, reward, done, new_obs)
        obs = new_obs
        cum_reward += gamma ** t * reward
        t += 1
    return cum_reward

return_from_all_iterations = []
evaluation_x_values = []

def train(env: Env, gamma: float, num_episodes: int, evaluate_every: int, num_evaluation_episodes: int,
          alpha: float, epsilon_max: Optional[float] = None, epsilon_min: Optional[float] = None,
          epsilon_decay: Optional[float] = None, replay_size: int = 10000, reset_network_every: int = 50, sample_size: int = 30) -> Tuple[DQNAgent, ndarray, ndarray]:
    """
    Training loop.

    :param env: The gym environment.
    :param gamma: The discount factor.
    :param num_episodes: Number of episodes to train.
    :param evaluate_every: Evaluation frequency.
    :param num_evaluation_episodes: Number of episodes for evaluation.
    :param alpha: Learning rate.
    :param epsilon_max: The maximum epsilon of epsilon-greedy.
    :param epsilon_min: The minimum epsilon of epsilon-greedy.
    :param epsilon_decay: The decay factor of epsilon-greedy.
    :return: Tuple containing the agent, the returns of all training episodes and averaged evaluation return of
            each evaluation.
    """
    digits = len(str(num_episodes))
    agent = DQNAgent(4, 2, alpha, gamma, epsilon_max,
                          epsilon_min, epsilon_decay, replay_size, env.observation_space.shape, sample_size, reset_network_every)
    evaluation_returns = np.zeros(num_episodes // evaluate_every)
    episode_evaluations = np.zeros(num_episodes // evaluate_every)
    returns = np.zeros(num_episodes)
    for episode in range(num_episodes):
        returns[episode] = run_episode(env, agent, True, gamma)

        if (episode + 1) % evaluate_every == 0:
            evaluation_step = episode // evaluate_every
            cum_rewards_eval = np.zeros(num_evaluation_episodes)
            for eval_episode in range(num_evaluation_episodes):
                cum_rewards_eval[eval_episode] = run_episode(env, agent, False, gamma)
            evaluation_returns[evaluation_step] = np.mean(cum_rewards_eval)
            episode_evaluations[evaluation_step] = episode + 1
            print(f"Episode {(episode + 1): >{digits}}/{num_episodes:0{digits}}:\t"
                  f"Averaged evaluation return {evaluation_returns[evaluation_step]:0.3}")
    global evaluation_x_values 
    evaluation_x_values= episode_evaluations
    plt.plot(episode_evaluations, evaluation_returns)
    return_from_all_iterations.append(evaluation_returns)
    return agent, returns, evaluation_returns

def show_variation_from_iterations(iteration_returns):
    number_of_evaluations = len(iteration_returns[0])
    lower_bound = np.zeros(number_of_evaluations)
    upper_bound = np.zeros(number_of_evaluations)
    mean = np.zeros(number_of_evaluations)
    std_dev = np.zeros(number_of_evaluations)
    for curr in range(number_of_evaluations):
        sum_of_returns = 0
        for iteration in range(len(iteration_returns)):
            sum_of_returns += iteration_returns[iteration][curr]
        mean[curr] = sum_of_returns/len(iteration_returns)

        sum_deviations = 0
        for iteration in range(len(iteration_returns)):
            sum_deviations += (iteration_returns[iteration][curr] - mean[curr]) ** 2
        std_dev[curr] = math.sqrt(sum_deviations / number_of_evaluations)
        lower_bound[curr] = mean[curr] - std_dev[curr]
        upper_bound[curr] = mean[curr] + std_dev[curr]
    plt.fill_between(evaluation_x_values, lower_bound, upper_bound, color='yellow', alpha=0.8)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    plt.figure()
    plt.xlabel("Number of episodes")
    plt.ylabel("Averaged evaluation return")
    random.seed(56)
    train(env, 0.99, 3000, 50, 32, 0.01, 1.0, 0.05, 0.999, 10000, 1000, 30)
    """"
    random.seed(98)
    train(env, 0.99, 3000, 50, 32, 0.01, 1.0, 0.05, 0.999, 10000, 1000, 30)
    random.seed(126)
    train(env, 0.99, 3000, 50, 32, 0.01, 1.0, 0.05, 0.999, 10000, 1000, 30)
    random.seed(541)
    train(env, 0.99, 3000, 50, 32, 0.01, 1.0, 0.05, 0.999, 10000, 1000, 30)
    random.seed(352)
    train(env, 0.99, 3000, 50, 32, 0.01, 1.0, 0.05, 0.999, 10000, 1000, 30)
    show_variation_from_iterations(return_from_all_iterations)
    plt.show()
    """

    


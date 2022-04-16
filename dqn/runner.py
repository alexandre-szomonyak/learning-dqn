from cProfile import label
from hashlib import new
from lib2to3.pgen2.token import NEWLINE
from statistics import mean
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import gym
import numpy as np
from gym import Env
from numpy import ndarray
import math

from agent import DQNAgent
""" env.render to render the example
 1) flow: tutorial
 2)  

"""

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
    agent.update_epsilon()
    return cum_reward

return_from_all_iterations = []
evaluation_x_values = []

def train(env: Env, gamma: float, num_episodes: int, evaluate_every: int, num_evaluation_episodes: int,
          alpha: float, epsilon_max: Optional[float] = None, epsilon_min: Optional[float] = None,
          epsilon_decay: Optional[float] = None, replay_size: int = 10000, reset_network_every: int = 1000, sample_size: int = 100) -> Tuple[DQNAgent, ndarray, ndarray]:
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
    for x_value in range(episode_evaluations.size):
        episode_evaluations[x_value] = evaluate_every*x_value
    global evaluation_x_values 
    evaluation_x_values= episode_evaluations
    returns = np.zeros(num_episodes)
    for episode in range(num_episodes):
        returns[episode] = run_episode(env, agent, True, gamma)

        if (episode + 1) % evaluate_every == 0:
            evaluation_step = episode // evaluate_every
            cum_rewards_eval = np.zeros(num_evaluation_episodes)
            for eval_episode in range(num_evaluation_episodes):
                cum_rewards_eval[eval_episode] = run_episode(env, agent, False, gamma)
            evaluation_returns[evaluation_step] = np.mean(cum_rewards_eval)
            print(f"Episode {(episode + 1): >{digits}}/{num_episodes:0{digits}}:\t"
                  f"Averaged evaluation return {evaluation_returns[evaluation_step]:0.3}")
    return_from_all_iterations.append(evaluation_returns)
    return agent, returns, evaluation_returns


def show_variation_from_iterations(iteration_returns):
    std_dev = np.std(iteration_returns, axis=0)
    mean_of_returns = np.mean(iteration_returns, axis=0)
    lower_bound = mean_of_returns - std_dev
    upper_bound = mean_of_returns + std_dev
    plt.fill_between(evaluation_x_values, lower_bound, upper_bound, color='cyan', alpha=0.8)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    plt.figure()
    plt.xlabel("Number of episodes")
    plt.ylabel("Averaged evaluation return")
    plt.ylim(0, 100)
    train(env, 0.99, 2000, 50, 32, 0.001, 1.0, 0.05, 0.99, 500000, 20000, 128)
    train(env, 0.99, 2000, 50, 32, 0.001, 1.0, 0.05, 0.99, 500000, 20000, 128)
    train(env, 0.99, 2000, 50, 32, 0.001, 1.0, 0.05, 0.99, 500000, 20000, 128)
    train(env, 0.99, 2000, 50, 32, 0.001, 1.0, 0.05, 0.99, 500000, 20000, 128)
    train(env, 0.99, 2000, 50, 32, 0.001, 1.0, 0.05, 0.99, 500000, 20000, 128)

    plt.plot(evaluation_x_values, np.mean(return_from_all_iterations, axis=0))
    show_variation_from_iterations(return_from_all_iterations)
    plt.legend()
    plt.show()
    

    


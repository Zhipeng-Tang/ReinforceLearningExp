import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

def make_epsilon_greedy_policy(Q1, Q2, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q1[observation]+Q2[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def choose_action(A):
    prob = np.random.rand()
    distribute = np.zeros(len(A)+1)
    for i in range(1, len(distribute)):
        distribute[i] = distribute[i-1]+A[i-1]
    distribute[len(A)] = 1
    for i in range(len(distribute)-1):
        if distribute[i] <= prob < distribute[i+1]:
            return i

def double_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
    Q2 = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q1, Q2, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
 ########################################Implement your code here##########################################################################       

            
            # step 1 : Take a step
            action = choose_action(policy(state))
            new_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # step 2 : TD Update
            if np.random.rand() < 0.5:
                Q1[state][action] += alpha * (reward + discount_factor * Q2[new_state][np.argmax(Q1[new_state])] - Q1[state][action])
            else:
                Q2[state][action] += alpha * (reward + discount_factor * Q1[new_state][np.argmax(Q2[new_state])] - Q2[state][action])
            policy = make_epsilon_greedy_policy(Q1, Q2, epsilon, env.action_space.n)
            state = new_state
            if done:
                break

#######################################Imlement your code end###########################################################################
    return Q1, Q2, stats


Q1, Q2, stats = double_q_learning(env, 1000)

plotting.plot_episode_stats(stats)
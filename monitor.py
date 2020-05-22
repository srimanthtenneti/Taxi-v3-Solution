# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:23:03 2020

@author: Srimanth Tenneti
"""

from collections import deque
import sys
import math
import numpy as np
import agent as a


def interact(env, agent, num_episodes=1000000, window=100,alpha = 0.6):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        score = 0
        state = env.reset()
        
        eps = 0.6/ i_episode
        action = agent.select_action(state,eps,env)
        
        while True:
            # agent selects an action
            action = agent.select_action(state,eps,env)
            next_state , reward , done , info = env.step(action)
            score += reward
            agent.Q[state][action] = agent.update_Q_sarsa(0.5 ,0.6 , agent.Q , state , action , reward , next_state)
            state = next_state
            if done:
                samp_rewards.append(score)
                break
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.4:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    return avg_rewards, best_avg_reward
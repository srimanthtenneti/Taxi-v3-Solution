# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:23:33 2020

@author: Srimanth Tenneti
"""

from agent import Agent
from monitor import interact
import gym
import numpy as np
import  matplotlib.pyplot as plt

env = gym.make('Taxi-v3') # Loading the environment
agent = Agent()  # Creating an agent instance
avg_rewards, best_avg_reward = interact(env, agent) # Training the agent

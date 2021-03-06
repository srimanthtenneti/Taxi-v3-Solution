# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:13:33 2020

@author: Srimanth Tenneti
"""

import gym
import numpy as np
import  matplotlib.pyplot as plt

env = gym.make('Taxi-v3')

state = env.reset()
for t in range(100):
    action = env.action_space.sample() #Selecting a random action from the action space
    plt.axis('off')
    state, reward, done, _ = env.step(action) # Executing the action
    env.render() # Rendering environment 
    if done:
        print('Score: ', t+1)
        break
        
env.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:32:22 2018

@author: avalon
"""

# Playing with OpenAI Gym: CartPole-v0

import time
import gym
import numpy as np

##################################################################################################
# policies

def naive_policy(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1

def random_policy(obs):
	angle = obs[2]
	return 0 if np.random.uniform() < 0.5 else 1

##################################################################################################

def naive_main( policy ):
    debug = True
    env = gym.make("CartPole-v0")
    obs = env.reset()

    # episodic reinforcement learning
    totals = []
    for episode in range(100):
        episode_rewards = 0
        obs = env.reset()
        for step in range(10000):
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                #print ("Game over. Number of steps = ", step)
                break
            totals.append(episode_rewards)
    return np.mean(totals)
##################################################################################################

if __name__ == "__main__": 
    output = naive_main(naive_policy)
    
    print(f"The average score achieved by naive method is {output}")

##################################################################################################

if __name__ == "__main__": 
    output = naive_main(random_policy)
    
    print(f"The average score achieved by random method is {output}")
    
print("Obviously, the average score achieved by random method is way",
      "better than the score achieved by naive method")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:45:49 2018

@author: avalon
"""

import gym
import random
from statistics import mean, median
import numpy as np
import xgboost as xgb
env = gym.make("CartPole-v0")
env.reset()
gameNum = 100000
iterTimes = 500
scoreTarget = 50

def init_training() :
    trainingData = []
    highScoreRecord = []
    scoreSet = []
    actionSet = []
    
    for _ in range(gameNum) :
        
        score = 0
        gameMemory = []
        preObservation = []
        
        for _ in range(iterTimes) :
            
            action = random.choice([0, 1])
            observation, reward, done, info = env.step(action)
            if len(preObservation) > 0 :
                gameMemory.append([preObservation, action])
            preObservation = observation
            score += reward
            if done :
                break
        
        if score >= 50 :
            highScoreRecord.append(score)
            for memory in gameMemory :
                trainingData.append(memory[0])
                actionSet.append(memory[1])
                
        env.reset()
        scoreSet.append(score)
    
    print(f"The mean of the value who is greater than 50 is {mean(highScoreRecord)}")
    print(f"The median of the value who is greater than 50 is {median(highScoreRecord)}")
    print(f"The max value of the value who is greater than 50 is {max(highScoreRecord)}")
    
    trainingData = np.array([i for i in trainingData])
    labels = np.array(actionSet)
    labels = labels.reshape((len(labels), 1))
    trainingData = np.concatenate((trainingData, labels), axis = 1)
    return trainingData

theData = init_training()

'''
For the part below, we initiate our training process in order to learn a 
better policy.
'''

dataTrain = xgb.DMatrix(theData[:, 0:4], label = theData[:, 4])
xgbParam = {"tree_method" : "exact",
            "max_depth" : 2,
            "objective" : "binary:logistic"}
nRound = 10
model = xgb.train(xgbParam, dataTrain, nRound)

'''
For the part below, we will test our policy on the cartpole environment.
'''

def testPerformance() :
    highScoreRecord = []
    scoreSet = []
    actionSet = []
    
    for _ in range(100) :
        
        score = 0
        gameMemory = []
        preObservation = []
        
        for _ in range(iterTimes) :
            
            if len(preObservation) > 0 :
                lastObservation = np.array(preObservation).reshape((1, -1))
                lastObservation = xgb.DMatrix(lastObservation)
                action = model.predict(lastObservation)
                if action >= 0.5 :
                    action = 1
                else :
                    action = 0
            else :
                action = random.choice([0, 1])
            observation, reward, done, info = env.step(action)
            if len(preObservation) > 0 :
                gameMemory.append([preObservation, action])
            preObservation = observation
            score += reward
            if done :
                break
        
        if score >= 50 :
            highScoreRecord.append(score)
            for memory in gameMemory :
                actionSet.append(memory[1])
        env.reset()
        scoreSet.append(score)
    
    print(f"The mean of the value who is greater than 50 is {mean(highScoreRecord)}")
    print(f"The median of the value who is greater than 50 is {median(highScoreRecord)}")
    print(f"The max value of the value who is greater than 50 is {max(highScoreRecord)}")
    
testPerformance()
    
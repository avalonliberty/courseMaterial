#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 23:25:13 2018

@author: avalon
"""

# import required modules
import numpy as np
import pandas as pd

# Reading data and cleaning data
learningData = pd.read_csv("train-a1-449.txt", header = None, sep = " ")
labels = learningData[1024]
labels.replace(["Y", "N"], [1, -1], inplace = True)
del learningData[1024], learningData[1025]

# creaing module for running PLA
class PLA(object) :
    def __init__(self, learningData, labels):
        self.learningData = learningData
        self.weights = np.random.rand(len(learningData.columns))
        self.labels = np.array(labels)
        
    def PLAtraining(self):
        vecIndicator = np.vectorize(lambda x : 1 if x > 0 else -1)
        while True:
            valueOutput = np.dot(self.learningData, self.weights)
            predictedLabels = vecIndicator(valueOutput)
            if np.all(self.labels == predictedLabels):
                return self.weights
                break
            Index = np.random.choice(np.where(self.labels != predictedLabels)[0])
            self.weights += self.labels[Index] * self.learningData.iloc[Index]

# Executing prebuilt module            
Solution = PLA(learningData = learningData, labels = labels)
weights = Solution.PLAtraining()
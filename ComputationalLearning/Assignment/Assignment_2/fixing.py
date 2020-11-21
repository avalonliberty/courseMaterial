#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:46:48 2018

@author: avalon
"""
import numpy as np
import pandas as pd
## reading in data and cleaning it
myData = pd.read_csv("a2-train-data.txt", header = None,
                     delimiter = " ")
labels = pd.read_csv("a2-train-label.txt", header = None,
                     delimiter = " ").values
myData.drop([1000], axis = 1, inplace = True)
myData = myData.values

class DL(object) :
    
    def __init__(self, trainingSet, labels, nNodes, epoch = 20, eta = 0.0001) :
        self.trainingSet = trainingSet 
        self.labels = labels
        self.nrow = trainingSet.shape[0]
        self.ncol = trainingSet.shape[1]
        self.bias1 = np.random.normal(0, 0.01, nNodes)
        self.bias2 = np.random.normal(0, 0.01, 1)
        self.inToHidden = np.random.normal(0, 0.01, (self.ncol, nNodes))
        self.hiddenToOutput = np.random.normal(0, 0.01, (nNodes, 1))
        self.epoch = epoch
        self.eta = eta
    
    def deriTanh(self, x) :
        return 1 - (x ** 2)
    
    def sigmoid(self, x) :
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def deriSigmoid(self, x) :
        return x * (1 - x)
    
    def squaredError(self, yHat, y) :
        return np.sum(np.subtract(yHat, y) ** 2) / 2
    
    def deriSquaredError(self, yHat, y) :
        return yHat - y
    
    def crossEntropy(self, yHat, y):
        return (np.sum(-(y * np.log(yHat) + (1 - y) * np.log(1 - yHat)))) / y.Hat.shape[0]
    
    def normalize(self) :
        self.trainingSet = np.apply_along_axis(
                lambda x : (x - np.mean(x)) / np.std(x), 1, self.trainingSet)
        
    def fit(self) :

        #epoch
        for epoch in range(self.epoch) :
            executedOrder = np.random.choice(np.arange(self.nrow), self.nrow, False)
            
            for i, runningIndex in enumerate(executedOrder) :
                
                #forward propagate
                # first layer                    
                hiddenLayerOutput = np.dot(self.trainingSet[runningIndex,], self.inToHidden) + self.bias1
                hiddenLayerActi = np.tanh(hiddenLayerOutput)
                
                # second layer
                outputLayerOutput = np.dot(hiddenLayerActi, self.hiddenToOutput) + self.bias2
                outputLayerActi = self.sigmoid(outputLayerOutput)
                
                # back propagate
                holdingPart = np.subtract(outputLayerActi, self.labels[runningIndex]) \
                              * self.deriSigmoid(outputLayerActi)
                hiddenToOutputUpdate = (holdingPart * hiddenLayerActi).reshape((self.hiddenToOutput.shape[0], 1))
                bias2Update = holdingPart
                holdingPart2 = np.dot(self.hiddenToOutput, holdingPart) \
                                      * self.deriTanh(hiddenLayerActi)
                bias1Update = holdingPart2
                inputToHiddenUpdate = np.outer(self.trainingSet[runningIndex], holdingPart2)
                
                # weights update
                self.inToHidden = self.inToHidden - self.eta * inputToHiddenUpdate
                self.bias1 = self.bias1 - self.eta * bias1Update
                self.hiddenToOutput = self.hiddenToOutput - self.eta * hiddenToOutputUpdate
                self.bias2 = self.bias2 - self.eta * bias2Update
            
            # error evaluation
            Layer1 = np.dot(self.trainingSet, self.inToHidden) + self.bias1
            actiLayer1 = np.tanh(Layer1)
            Layer2 = np.dot(actiLayer1, self.hiddenToOutput) + self.bias2
            actiLayer2 = self.sigmoid(Layer2)
            outputLen = len(actiLayer2)
            output = np.where(actiLayer2 > 0, np.ones(outputLen), -np.ones(outputLen))
            totalError = np.mean(output != self.labels)
            #totalError = np.mean(np.subtract(actiLayer2, self.labels) ** 2) / 2
            print(f"The total error at {epoch} epoch is {totalError}")
model = DL(myData, labels, 10, epoch = 200)
model.normalize()
model.fit()
    
                
                
        
        
        
    
        
    
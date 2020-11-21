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
    '''
    The architecture aims to fit the assignment2 of the course, computational
    learning; therefore, the only hyperparameter you can tune is the number
    of nodes in the first and the only hidden layer in the architecture.
    '''
    def __init__(self, trainingSet, labels, nNodes, epoch = 20, eta = 0.1) :
        self.trainingSet = trainingSet 
        self.labels = labels
        self.nrow = trainingSet.shape[0]
        self.ncol = trainingSet.shape[1]
        self.bias1 = np.random.randn(nNodes)
        self.bias2 = np.random.randn(1)
        self.inToHidden = np.random.randn(self.ncol, nNodes)
        self.hiddenToOutput = np.random.randn(nNodes, 1)
        self.epoch = epoch
        self.eta = eta
        
            
    def relu(self, x) :
        return max(0, x)
    
    def deriRelu(self, x) :
        if x > 0 : 
            return 1
        elif x < 0 :
            return 0
    
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
        
    '''   
    def train(self) :
        relu = np.vectorize(self.relu, otypes = [np.float64])
        sigmoid = np.vectorize(self.sigmoid, otypes = [np.float64])
        # epoch 
        for epoch in range(self.epoch) :
            executedBatch = np.array([])
            while executedBatch.size != self.trainingSet.shape[0] :
                # batch
                diffIndex = np.setdiff1d(np.arange(0, self.trainingSet.shape[0]), executedBatch)
                batchIndex = np.random.choice(diffIndex, self.batchSize, False)
                executedBatch = np.concatenate((executedBatch, batchIndex))
                # foward propagate
                hiddenLayerOutput = np.dot(self.trainingSet[batchIndex, ], self.inToHidden)
                hiddenLayerActi = sigmoid(hiddenLayerOutput)
                outputLayerOutput = np.dot(hiddenLayerActi, self.hiddenToOutput)
                outputLayerActi = sigmoid(outputLayerOutput)
                if self.errorFunction == "squaredError" :
                    totalError = self.squaredError(outputLayerActi, self.labels[batchIndex])
                elif self.errorFunction == "crossEntropy" :
                    totalError = self.crossEntropy(outputLayerActi, self.labels[batchIndex])
                # back propagate
                errorToOutput = np.subtract(outputLayerActi, )
    '''
    
    def fit(self) :
        relu = np.vectorize(self.relu, otypes = [np.float64])
        sigmoid = np.vectorize(self.sigmoid, otypes = [np.float64])
        deriRelu = np.vectorize(self.deriRelu, otypes = [np.float64])
        deriSigmoid = np.vectorize(self.deriSigmoid, otypes = [np.float64])
        #epoch
        for epoch in range(self.epoch) :
            executedOrder = np.random.choice(np.arange(self.nrow), self.nrow, False)
            for runningIndex in executedOrder :
                #forward propagate
                # first layer
                hiddenLayerOutput = np.dot(self.trainingSet[runningIndex,], self.inToHidden) + self.bias1
                hiddenLayerActi = relu(hiddenLayerOutput)
                
                # second layer
                outputLayerOutput = np.dot(hiddenLayerActi, self.hiddenToOutput) + self.bias2
                outputLayerActi = sigmoid(outputLayerOutput)
                    
                # back propagate
                holdingPart = np.subtract(outputLayerActi, self.labels[runningIndex]) \
                              * deriSigmoid(outputLayerActi)
                hiddenToOutputUpdate = holdingPart * hiddenLayerActi
                bias2Update = holdingPart
                holdingPart2 = np.dot(self.hiddenToOutput, holdingPart) \
                                      * deriRelu(hiddenLayerActi)
                bias1Update = holdingPart2
                inputToHiddenUpdate = np.outer(self.trainingSet[runningIndex], holdingPart2)
                
                # weights update
                self.inToHidden = self.inToHidden - self.eta * inputToHiddenUpdate
                self.bias1 = self.bias1 - self.eta * bias1Update
                self.hiddenToOutput = self.hiddenToOutput - self.eta * hiddenToOutputUpdate
                self.bias2 = self.bias2 - self.eta * bias2Update
            Layer1 = np.dot(self.trainingSet, self.inToHidden) + self.bias1
            actiLayer1 = relu(Layer1)
            Layer2 = np.dot(actiLayer1, self.hiddenToOutput) + self.bias2
            actiLayer2 = sigmoid(Layer2)
            totalError = np.sum(np.subtract(actiLayer2, self.labels) ** 2) / 2
            print(f"The total error at {epoch} epoch is {totalError}")
    
                
                
        
        
        
    
        
    
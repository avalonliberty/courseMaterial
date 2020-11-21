#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:01:58 2018

@author: avalon
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data = datasets.load_digits()
myData = data.data
labels = data.target
class Adaboost(object) :
    '''
    '''
    
    def __init__(self, dataset, labels, nrounds, learner = "decisionTree") :
        self.trainingset = dataset
        self.labels = labels
        self.nrounds = nrounds
        self.learner = learner
        self.predictionSet = []
        self.alpha = []
        
    def fit(self) :
        Weights = np.ones(self.trainingset.shape[0]) / self.trainingset.shape[0]
        for i in range(self.nrounds) :
            if self.learner == "decisionTree" :   
                learner = DecisionTreeClassifier(max_leaf_nodes = 10)
            elif self.learner == "naiveBayes" :
                learner = MultinomialNB()
            learner.fit(self.trainingset, self.labels, sample_weight = Weights)
            prediction = learner.predict(self.trainingset)
            self.predictionSet.append(prediction)
            error = np.sum(Weights[np.where(prediction != self.labels)[0]]) / np.sum(Weights)
            theAlpha = np.log((1 - error) / error) + np.log(9)
            self.alpha.append(theAlpha)
            Weights = Weights * np.exp(theAlpha * (prediction != self.labels))
            Weights = Weights / np.sum(Weights)
    
    def predict(self, dataset) :
        answer = []
        for i in range(self.trainingset.shape[0]) :
            vote = np.zeros(10)
            for j, k in enumerate(self.predictionSet) :
                vote[k[i]] += self.alpha[j]
            answer.append(np.argmax(vote))
        return np.array(answer)
            
myTreeLearner = []
for i in range(1, 31) :
    data = {}
    theLearner = Adaboost(myData, labels, i)
    theLearner.fit()
    output = theLearner.predict(myData)
    error = np.mean(labels != output)
    data["iteration"] = i
    data["error"] = error
    myTreeLearner.append(data)       
    
bayesLearner = []
for i in range(1, 301) :
    data = {}
    theLearner = Adaboost(myData, labels, i, "naiveBayes")
    theLearner.fit()
    output = theLearner.predict(myData)
    error = np.mean(labels != output)
    data["iteration"] = i
    data["error"] = error
    bayesLearner.append(data)
    
skLearner = []
for i in range(1, 31) :
    data = {}
    theLearner = AdaBoostClassifier(base_estimator =
                                    DecisionTreeClassifier(max_leaf_nodes = 10),
                                    n_estimators = i)
    theLearner.fit(myData, labels)
    output = theLearner.predict(myData)
    error = np.mean(labels != output)
    data["iteration"] = i
    data["error"] = error
    skLearner.append(data)


myTreeLearner = pd.DataFrame(myTreeLearner)
bayesLearner = pd.DataFrame(bayesLearner)
skLearner = pd.DataFrame(skLearner)

theFigure = plt.figure(figsize = (20, 5))
plt.subplot(1, 3, 1)
sns.lineplot(x = "iteration", y = "error", data = myTreeLearner)
plt.title("decisionTreeImplementation")

plt.subplot(1, 3, 2)
sns.lineplot(x = "iteration", y = "error", data = bayesLearner)
plt.title("naiveBayesImplementation")

plt.subplot(1, 3, 3)
sns.lineplot(x = "iteration", y = "error", data = skLearner, )
plt.title("skLearnImplementation")

theFigure.savefig("comparasion.png",  dpi = 300)
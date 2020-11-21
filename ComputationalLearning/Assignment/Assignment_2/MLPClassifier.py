#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:42:32 2018

@author: avalon
"""

import sklearn
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
## reading in data and cleaning it
myData = pd.read_csv("a2-train-data.txt", header = None,
                     delimiter = " ")
labels = pd.read_csv("a2-train-label.txt", header = None,
                     delimiter = " ").values
myData.drop([1000], axis = 1, inplace = True)
myData = myData.values

model = MLPClassifier(solver = "lbfgs")
model.fit(myData, labels)
accuracy = model.score(myData, labels)
print(f"The accuracy of current model is {accuracy * 100}%")


# ============================================================================
testingData = pd.read_csv("a2-test-data.txt", header = None,
                     delimiter = " ")
temp = open("a2-test-label.txt")
testinglabels = temp.read()
testinglabels = testinglabels[1:len(testinglabels) - 1].replace(" ", "").split(",")
testinglabels = [float(i) for i in testinglabels]
testinglabels = np.array(testinglabels)

model = MLPClassifier(solver = "lbfgs")
model.fit(myData, labels)
testPrediction = model.predict(testingData)
errorEvaluation = testPrediction != testinglabels
print(f"The model predict {sum(errorEvaluation)} errors on the testing set")
print(f"The error rate of the model on prediction set is {np.mean(errorEvaluation * 100)}%")
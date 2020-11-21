#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 16:58:19 2018

@author: avalon
"""

import pandas as pd
from sklearn.linear_model import Perceptron

myData = pd.read_csv("train-a1-449.txt", sep = " ", header = None)
label = myData[1024]
label.replace({"Y" : 1, "N" : -1}, inplace = True)
myData.drop([1024, 1025], axis = 1, inplace = True)
PLA = Perceptron()
PLA.fit(myData, label)

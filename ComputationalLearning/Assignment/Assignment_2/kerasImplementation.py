#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:11:03 2018

@author: avalon
"""

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import initializers
## reading in data and cleaning it

def normalize(data) :
    return np.apply_along_axis(
            lambda x : (x - np.mean(x)) / np.std(x), 1, data)
    
myData = pd.read_csv("a2-train-data.txt", header = None,
                     delimiter = " ")
labels = pd.read_csv("a2-train-label.txt", header = None,
                     delimiter = " ").values
myData.drop([1000], axis = 1, inplace = True)
myData = normalize(myData.values)

model = Sequential()
adam = optimizers.SGD(lr = .0001)
normal = initializers.random_normal(stddev = .01)
model.add(Dense(units = 1000, activation = 'tanh', input_shape = (1000, ),
                kernel_initializer = normal))
model.add(Dense(units = 1, activation = 'softmax'))
model.compile(optimizer = adam, loss = 'binary_crossentropy',
              metrics = ['acc', 'mse'])
model.fit(myData, labels, batch_size = 20, epochs = 5000)
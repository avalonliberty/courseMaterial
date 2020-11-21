#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:08:02 2018

@author: avalon
"""
import tensorflow as tf
import numpy as np
import pandas as pd

pyData = pd.read_csv("train-a1-449.txt", sep = " ", header = None)
labels = pyData[1024].replace({"Y" : 1., "N" : -1.})
del pyData[1024], pyData[1025]
data = tf.convert_to_tensor(pyData.values, dtype = tf.float64)
weights = tf.convert_to_tensor(np.random.randn(1024), dtype = tf.float64)
labels = tf.convert_to_tensor(labels)
sess = tf.Session()
while True :
    dotOutput = tf.tensordot(data, weights, 1)
    checkingSeq = tf.where(dotOutput > 0,
                           tf.ones(792, dtype = tf.float64),
                           -tf.ones(792, dtype = tf.float64))
    checkingList = sess.run(tf.equal(labels, checkingSeq))
    if np.all(checkingList) :
        break
    checkingIndex = np.where(checkingList == False)[0]
    updateIndex = np.random.choice(checkingIndex)
    weights += tf.multiply(labels[updateIndex], data[updateIndex])

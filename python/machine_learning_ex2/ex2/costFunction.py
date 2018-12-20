# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:11:15 2018

@author: Jingjing
"""
import numpy as np
import sigmoid

def costFunction(theta, X, y):
    m, n = X.shape
    hypothesis = sigmoid.sigmoid(X.dot(theta))
    J = -y.dot(np.log(hypothesis)) - (1 - y).dot(np.log(1 - hypothesis))
    J = J / m
    #grad = X.T.dot(hypothesis - y) / m
    return J

def gradient(theta, X, y):
    m, n = X.shape
    hypothesis = sigmoid.sigmoid(X.dot(theta))
    grad = X.T.dot(hypothesis - y) / m
    return grad
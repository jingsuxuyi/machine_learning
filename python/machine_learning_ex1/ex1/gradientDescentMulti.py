# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:50:50 2018

@author: Jingjing
"""
import numpy as np
import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    
    for i in np.arange(num_iters):
        theta = theta - alpha / m * X.T.dot(X.dot(theta) - y)
        J_history[i] = computeCostMulti.computeCostMulti(X, y, theta)
    return theta, J_history
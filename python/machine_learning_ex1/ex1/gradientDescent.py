# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:54:03 2018

@author: Jingjing
"""
import numpy as np
import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    m = max(y.shape)
    J_history = np.zeros((num_iters, 1))
    for i in np.arange(num_iters):
        theta = theta - alpha / m * X.T.dot(X.dot(theta) - y)
        J_history[i] = computeCost.computeCost(X, y, theta)
    return theta
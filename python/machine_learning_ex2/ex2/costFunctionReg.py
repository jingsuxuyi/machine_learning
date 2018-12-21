# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:49:57 2018

@author: Jingjing
"""
import sigmoid
import numpy as np

def costFunctionReg(theta, X, y, lam):
    m, n = X.shape
    hypothesis = sigmoid.sigmoid(X.dot(theta))
    theta_temp = theta.copy()
    theta_temp[0] = 0  
    J = -y.dot(np.log(hypothesis)) - (1 - y).dot(np.log(1 - hypothesis)) + (theta_temp).dot(theta_temp) * lam / 2
    J = J / m
    #grad = X.T.dot(hypothesis - y) / m
    return J

def gradientReg(theta, X, y, lam):
    m, n = X.shape
    theta_temp = theta.copy()
    theta_temp[0] = 0
    hypothesis = sigmoid.sigmoid(X.dot(theta))
    grad = X.T.dot((hypothesis - y)) / m + lam * theta_temp / m
    return grad
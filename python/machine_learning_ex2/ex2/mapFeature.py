# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 22:04:37 2018

@author: Jingjing
"""
import numpy as np
def mapFeature(X1, X2):
# MAPFEATURE Feature mapping function to polynomial features
#
#   MAPFEATURE(X1, X2) maps the two input features
#   to quadratic features used in the regularization exercise.
#
#   Returns a new feature array with more features, comprising of 
#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#
#   Inputs X1, X2 must be the same size
#
    degree = 6
    out = np.ones((X1.size, 1))
    for i in np.arange(degree) + 1:
        for j in np.arange(i + 1):
            temp = X1 ** (i - j) * (X2 ** j)
            out = np.c_[out, temp]
    return out


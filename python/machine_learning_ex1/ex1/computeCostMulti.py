# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:07:55 2018

@author: Jingjing
"""

def computeCostMulti(X, y, theta):
    m = y.size
    
    temp = X.dot(theta) - y
    J = temp.T.dot(temp) / 2 / m
    return J
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:15:47 2018

@author: Jingjing
"""
import sigmoid

def predict(theta, X):
    return sigmoid.sigmoid(X.dot(theta)) > 0.5
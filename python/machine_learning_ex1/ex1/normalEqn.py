# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:48:16 2018

@author: Jingjing
"""
import numpy.linalg as nlg

def normalEqn(X, y):
    return nlg.pinv(X.T.dot(X)).dot(X.T).dot(y)
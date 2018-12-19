# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 11:49:37 2018

@author: Jingjing
"""
import numpy as np

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
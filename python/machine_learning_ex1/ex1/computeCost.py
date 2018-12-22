# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 22:16:01 2018

@author: Jingjing
"""

import numpy as np

def computeCost(X, y, theta):
    m = y.size
    return sum((np.dot(X, theta) - y) ** 2) / 2 / m
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:11:54 2018

@author: Jingjing
"""
import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
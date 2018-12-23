# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:17:20 2018

@author: Jingjing
"""

## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io as sio
import displayData

# Setup the parameters you will use for this part of the exercise
input_layer_size  = 400             # 20x20 Input Images of Digits
num_labels = 10                     # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print("Loading and Visualizing Data ...")

mat_contents = sio.loadmat("ex3data1.mat")

X, y = mat_contents['X'], mat_contents['y']

m = y.size

# Randomly select 100 data points to display

rand_indices = np.random.randint(0, m, size=100)
sel = X[rand_indices, :]

displayData.displayData(sel)
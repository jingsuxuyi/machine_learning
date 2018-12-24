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
import lrCostFunction
import oneVsAll
import predictOneVsAll

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
y = y.flatten()
m = y.size

# Randomly select 100 data points to display

rand_indices = np.random.randint(0, m, size=100)
sel = X[rand_indices, :]

displayData.displayData(sel)

input("Program paused. Press enter to continue.")

## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

# Test case for lrCostFunction
print("Testing lrCostFunction() with regularization")

theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[np.ones((5, 1)), (np.arange(1,16) / 10).reshape((5, 3), order='F')]
y_t = np.array([1,0,1,0,1])
lambda_t = 3
J = lrCostFunction.lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = lrCostFunction.gradient(theta_t, X_t, y_t, lambda_t)

print("Cost: {:.6f}".format(J))
print("Expected cost: 2.534819")
print("Gradients:")
print(grad)
print("Expected gradients:")
print("0.146561, -0.548558, 0.724722, 1.398003")

input("Program paused. Press enter to continue.")

# ============ Part 2b: One-vs-All Training ============
print("Training One-vs-All Logistic Regression...")

lambda_n = 0.1

all_theta = oneVsAll.oneVsAll(X, y, num_labels, lambda_n)

input("Program paused. Press enter to continue.")

# ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll.predictOneVsAll(all_theta, X)

print("Training Set Accuracy: {:f}% ".format(np.mean((pred == y).astype(np.float64)) * 100))

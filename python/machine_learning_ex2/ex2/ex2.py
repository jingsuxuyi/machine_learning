# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:06:30 2018

@author: Jingjing
"""

## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import plotData
import costFunction

## Initialization
data = np.loadtxt("ex2data1.txt", delimiter=",")
#col = data.shape[1]
#data = np.hsplit(data, (col-1, col))
X, y = data[:,0:2], data[:,2]


print("'Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")

plotData.plotData(X, y)

input("Program paused. Press enter to continue.")

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

X = np.c_[np.ones((m, 1)), X]
inital_theta = np.zeros(n+1)

cost, grid = costFunction.costFunction(inital_theta, X, y)

print("Cost at initial theta (zeros): {:.3f}".format(cost))
print("Expected cost (approx): 0.693")
print("Gradient at initial theta (zeros): ")
print(grid)
print("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628")

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grid = costFunction.costFunction(test_theta, X, y)

print("Cost at initial theta: {:.3f}".format(cost))
print("Expected cost (approx): 0.218")
print("Gradient at test theta:")
print(grid)
print("Expected gradients (approx):\n 0.043\n 2.566\n 2.647")

input("Program paused. Press enter to continue.")

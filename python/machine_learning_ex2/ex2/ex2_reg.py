# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:53:41 2018

@author: Jingjing
"""

## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#


## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

import numpy as np
import plotData
import mapFeature
import costFunctionReg
import scipy.optimize as opt
import plotDecisionBoundary
import predict
import matplotlib.pyplot as plt

data = np.loadtxt("ex2data2.txt", delimiter=",")
X, y = data[:, :2], data[:, 2]

plotData.plotData(X, y, "y = 1", "y = 0")
plt.xlabel( "Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend()
plt.show()

###########modify the xlabel and legend############

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled

X = mapFeature.mapFeature(X[:, 0], X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lam to 1
lam = 1

# Compute and display initial cost and gradient for regularized logistic regression
cost = costFunctionReg.costFunctionReg(initial_theta, X, y, lam)
grad = costFunctionReg.gradientReg(initial_theta, X, y, lam)

print("Cost at initial theta (zeros): ", cost)
print("Expected cost (approx): 0.693")
print("Gradient at initial theta (zeros) - first five values only:")
print(grad[0:5])
print("Expected gradients (approx) - first five values only:")
print(" 0.0085 0.0188 0.0001 0.0503 0.0115")

input("Program paused. Press enter to continue.")

# Compute and display cost and gradient
# with all-ones theta and lam = 10
test_theta = np.ones(X.shape[1])
cost = costFunctionReg.costFunctionReg(test_theta, X, y, 10)
grad = costFunctionReg.gradientReg(test_theta, X, y, 10)

print("Cost at initial theta (with lam = 10): ", cost)
print("Expected cost (approx): 3.16")
print("Gradient at test theta - first five values only:")
print(grad[0:5])
print("Expected gradients (approx) - first five values only:")
print(" 0.3460 0.1614 0.1948 0.2269 0.0922")

input("Program paused. Press enter to continue.")

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lam and
#  see how regularization affects the decision coundart
#
#  Try the following values of lam (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lam? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lam to 1 (you should vary this)
lam = 1

result = opt.minimize(fun=costFunctionReg.costFunctionReg, x0=initial_theta,
                   args=(X, y, lam), method="TNC", jac=costFunctionReg.gradientReg)

# Plot Boundary
plotDecisionBoundary.plotDecisionBoundary(result.x, X, y)

# Compute accuracy on our training set
p = predict.predict(result.x, X)

print("Train Accuracy: {:f}".format(np.mean(p == y) * 100))
print("Expected accuracy (approx): 83.1")
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 11:27:36 2018

@author: Jingjing
"""

import numpy as np
import featureNormalize
import matplotlib.pyplot as plt
import gradientDescentMulti
import normalEqn

# ================ Part 1: Feature Normalization ================

print("Loading data...")

# Load Data
data = np.loadtxt("ex1data2.txt", delimiter=',')
X = data[:, :2]
# convert 1-D array to 2-D for matrix operation 
y = data[:, 2][:, np.newaxis]
m = y.size

# Print out some data points
print('First 10 examples from the dataset:')
print(X[0:10, :], "\n ", y[:10])

input('Program paused. Press enter to continue.')

print("Normalizing Features ...")
X, mu, sigma = featureNormalize.featureNormalize(X)

# Add intercept term to X
X = np.hstack((np.ones((m, 1)), X))

# ================ Part 2: Gradient Descent ================
print("Running gradient descent ...")

alpha = 0.01
num_iters = 400

theta = np.zeros((3,1))

theta, J_history = gradientDescentMulti.gradientDescentMulti(X, y, theta, alpha, num_iters)

plt.figure()
plt.plot(np.arange(400), J_history.ravel(), c='b', linewidth=2)
plt.xlabel("Number of iterations")
plt.ylabel("Cost J")
plt.show()

print("Theta computed from gradient descent:")
print(theta)


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

price = np.hstack((np.ones(1), (np.array([1650, 3]) - mu) / sigma))[np.newaxis, :].dot(theta)

print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ", price)
input("Program paused. Press enter to continue.")


# ================ Part 3: Normal Equations ================
print("Solving with normal equations...")

# Load Data
data = np.loadtxt("ex1data2.txt", delimiter=',')
X = data[:, :2]
# convert 1-D array to 2-D for matrix operation 
y = data[:, 2][:, np.newaxis]
m = y.size
X = np.hstack((np.ones((m, 1)), X))
theta=normalEqn.normalEqn(X, y)
print("Theta computed from the normal equations:")
print(theta, theta.shape)

price = np.hstack((np.ones(1), np.array([1650, 3])))[np.newaxis, :].dot(theta)

print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ", price)

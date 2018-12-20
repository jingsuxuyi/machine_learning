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
import scipy.optimize as opt
import plotDecisionBoundary
import sigmoid
import predict

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

cost = costFunction.costFunction(inital_theta, X, y)
grad = costFunction.gradient(inital_theta, X, y)

print("Cost at initial theta (zeros): {:.3f}".format(cost))
print("Expected cost (approx): 0.693")
print("Gradient at initial theta (zeros): ")
print(grad)
print("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628")

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost = costFunction.costFunction(test_theta, X, y)
grad = costFunction.gradient(test_theta, X, y)

print("Cost at initial theta: {:.3f}".format(cost))
print("Expected cost (approx): 0.218")
print("Gradient at test theta:")
print(grad)
print("Expected gradients (approx):\n 0.043\n 2.566\n 2.647")

input("Program paused. Press enter to continue.")

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

result = opt.minimize(fun=costFunction.costFunction, x0=inital_theta, 
                      args=(X, y), method="TNC", jac=costFunction.gradient)
print("Cost at theta found by fminunc: {:.3f}".format(result.fun))
print("Expected cost (approx): 0.203")
print("theta: ")
print(result.x)
print("Expected theta (approx):")
print(" -25.161\n 0.206\n 0.201")

plotDecisionBoundary.plotDecisionBoundary(result.x, X, y)

input("Program paused. Press enter to continue.")

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid.sigmoid(np.array([1, 45, 85]).dot(result.x))
print("For a student with scores 45 and 85, we predict an admission probability of {:.3f}".format(prob))
print("Expected value: 0.775 +/- 0.002")

p = predict.predict(result.x, X)

print("Train Accuracy: {:f}".format(np.mean(p == y) * 100))
print("Expected accuracy (approx): 89.0")
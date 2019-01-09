## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1: 
# X, y, Xval, yval, Xtest, ytest can be extracted from mat_content
mat_content = sio.loadmat("ex5data1.mat")
X, y = mat_content['X'], mat_content['y'].flatten(order='F')
Xval, yval = mat_content['Xval'], mat_content['yval'].flatten(order='F')
Xtest, ytest = mat_content['Xtest'], mat_content['ytest'].flatten(order='F')

# m = Number of examples
m = X.shape[0]

# Plot training data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

input('Program paused. Press enter to continue.')


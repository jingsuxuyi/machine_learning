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
import linearRegCostFunction
import trainLinearReg


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


## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.array([1, 1])

J = linearRegCostFunction.linearRegCostFunction( theta, np.c_[np.ones((m, 1)), X], y, 1)

print('Cost at theta = [1 ; 1]: {:.6f} (this value should be about 303.993192)'.format(J))

input('Program paused. Press enter to continue.\n')



## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#
theta = np.array([1, 1])

grad = linearRegCostFunction.linearRegGradient(theta, np.c_[np.ones((m, 1)), X], y, 1)

print('Cost at theta = [1 ; 1]: [{:.6f} {:.6f}] (this value should be about [-15.303016, 598.250744])'.format(grad[0], grad[1]))

input('Program paused. Press enter to continue.\n')


## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0

lam = 0

theta = trainLinearReg.trainLinearReg(np.c_[np.ones((m, 1)), X], y, lam)

# Plot fit over the data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, np.c_[np.ones((m, 1)), X].dot(theta), '--', linewidth=2)
input('Program paused. Press enter to continue.\n')
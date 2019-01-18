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
import learningCurve
import polyFeatures
import featureNormalize
import plotFit


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


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#
lam = 0

error_train, error_val = learningCurve.learningCurve(np.c_[np.ones((m, 1)), X], 
							y, np.c_[np.ones((yval.size, 1)), Xval], yval, lam)

plt.figure()							
plt.plot(np.arange(m)+1, error_train, label='Train')
plt.plot(np.arange(m)+1, error_val,  label='Cross Validation')
plt.title('Learning curve for linear regression')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in np.arange(m):
	print("\t{:d}\t\t{:f}\t{:f}".format(i, error_train[i], error_val[i]))
	
input("Program paused. Press enter to continue.\n")


## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures.polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize.featureNormalize(X_poly)
X_poly = np.c_[np.ones((m, 1)), X_poly]

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures.polyFeatures(Xtest, p)
X_poly_test = (X_poly_test - mu) / sigma
X_poly_test = np.c_[np.ones((X_poly_test.shape[0], 1)), X_poly_test]

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures.polyFeatures(Xval, p)
X_poly_val = (X_poly_val - mu) / sigma
X_poly_val = np.c_[np.ones((X_poly_val.shape[0], 1)), X_poly_val]

print('Normalized Training Example 1:\n')
print(X_poly[0, :])

input('Program paused. Press enter to continue.\n')


## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lam = 1

theta = trainLinearReg.trainLinearReg(X_poly, y, lam)

# Plot training data and fit
plt.figure()
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit.plotFit(np.min(X, axis=0), np.max(X, axis=0), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
title_label = 'Polynomial Regression Learning Curve (lambda = {})'.format(lam)
plt.title(title_label)

plt.figure()
error_train, error_val = learningCurve.learningCurve(X_poly, y, X_poly_val, yval, lam)
plt.plot(np.arange(m)+1, error_train, label='Train')
plt.plot(np.arange(m)+1, error_val,  label='Cross Validation')
title_label = 'Polynomial Regression Learning Curve (lambda = {})'.format(lam)
plt.title(title_label)
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])

print('Polynomial Regression (lambda = {})'.format(lam))
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in np.arange(m):
	print("\t{:d}\t\t{:f}\t{:f}".format(i, error_train[i], error_val[i]))
	
input("Program paused. Press enter to continue.\n")


     # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import warmUpExercise
import numpy as np
import plotData
import computeCost
import gradientDescent
import matplotlib.pyplot as plt

# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ...')
print('5x5 Identity Matrix: ')

warmUpExercise.warmUpExercise()

print('Program paused. Press enter to continue.')
input() #for pause

# ======================= Part 2: Plotting =======================
print('Plotting Data ...')

data = np.loadtxt('ex1data1.txt', delimiter=',')
#split the 2-D array to three array, the last is empty array
data = np.hsplit(data, ((data.shape)[1]-1, data.shape[1]))
X, y = data[0], data[1]
plotData.plotData(X, y)

print('Program paused. Press enter to continue.')
input() #for pause

# =================== Part 3: Cost and Gradient descent ===================
m = max(y.shape)

##reshape to convert row vector to column vector
##besides, np.ones is row vector, not column vector
#X = np.insert(X.reshape(-1, 1), 0, np.ones((1, m)), axis=1)
#y = y.reshape(-1, 1) #np.extend_dims(y, axis=1)
#add all one before X
X = np.hstack((np.ones((m, 1)), X))

theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01

print('Testing the cost function ...')

J = computeCost.computeCost(X, y, theta)

print('With theta = [0 ; 0]\nCost computed = %f' % J)
print('Expected cost value (approx) 32.07')

#further testing of the cost function
J = computeCost.computeCost(X, y, np.array([-1, 2]).reshape(2,1))
print('With theta = [-1 ; 2]\nCost computed = %f' % J)
print('Expected cost value (approx) 54.24')

print('Program paused. Press enter to continue.')
input() #for pause

print("Running Gradient Descent ...")

theta =gradientDescent.gradientDescent(X, y, theta, alpha, iterations)

print("Theta found by gradient descent:")
print(theta.ravel())
print("Expected theta values (approx)")
print("[-3.6303  1.1664]")

plt.plot(X[:,1], X.dot(theta).ravel(), label="Linear regression")
plt.legend()

#print(theta)
# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([[1, 3.5]]).dot(theta)
print("For population = 35,000, we predict a profit of %f" % (predict1[0,0] * 10000))
predict2 = np.array([[1, 7]]).dot(theta)
print("For population = 70,000, we predict a profit of %f" % (predict2[0,0] * 10000))

print('Program paused. Press enter to continue.')
input() #for pause

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print("Visualizing J(theta_0, theta_1) ...")

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
len_theta0_vals = len(theta0_vals)
len_theta1_vals = len(theta1_vals)

#initialize J_vals to a matrix of 0's
J_vals = np.zeros((len_theta0_vals, len_theta1_vals))

# Fill out J_vals
for i in range(len_theta0_vals):
    for j in range(len_theta1_vals):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = computeCost.computeCost(X, y, t)
        
xx, yy = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')

ax3d.plot_surface(xx, yy, J_vals.T)
ax3d.set_xlabel('theta_0')
ax3d.set_ylabel('theta_1')


fig = plt.figure()
plt.contour(xx, yy, J_vals.T, np.logspace(-2, 3, 20))
plt.xlabel("theta_0")
plt.ylabel("theta_1")

plt.plot(theta[0], theta[1], 'rx', linewidth=2)
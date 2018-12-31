## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io as sio
import displayData
import nnCostFunction
import sigmoidGradient
import randInitializeWeights
import checkNNGradients


## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print("Loading and Visualizing Data ...")

mat_contents = sio.loadmat("ex4data1.mat")

X, y = mat_contents['X'], mat_contents['y']
y = y.flatten(order='F')
m = y.size

# Randomly select 100 data points to display

rand_indices = np.random.randint(0, m, size=100)
sel = X[rand_indices, :]

displayData.displayData(sel)

input("Program paused. Press enter to continue.")

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print("Loading Saved Neural Network Parameters ...")

# Load the weights into variables Theta1 and Theta2
mat_theta = sio.loadmat("ex4weights.mat")
Theta1, Theta2 = mat_theta['Theta1'], mat_theta['Theta2']

# Unroll parameters 
nn_params = np.r_[Theta1.flatten(order='F'), Theta2.flatten(order='F')]

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print("Loading Saved Neural Network Parameters ...")

# Load the weights into variables Theta1 and Theta2
mat_theta = sio.loadmat("ex4weights.mat")
Theta1, Theta2 = mat_theta['Theta1'], mat_theta['Theta2']

# Unroll parameters 
nn_params = np.r_[Theta1.flatten(order='F'), Theta2.flatten(order='F')]

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print("Feedforward Using Neural Network ...")

lam = 0

J = nnCostFunction.nnCostFunction(nn_params, input_layer_size, 
							hidden_layer_size, num_labels, X, y, lam)
							
print('Cost at parameters (loaded from ex4weights): {:.6f} \n(this value should be about 0.287629)\n'.format(J))

input("Program paused. Press enter to continue.")

# =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#
print("Checking Cost Function (w/ Regularization) ... ")

lam = 1

J = nnCostFunction.nnCostFunction(nn_params, input_layer_size, 
							hidden_layer_size, num_labels, X, y, lam)

print('Cost at parameters (loaded from ex4weights): {:.6f} \n(this value should be about 0.383770)\n'.format(J))
		
input("Program paused. Press enter to continue.")		


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print("Evaluating sigmoid gradient...")

g = sigmoidGradient.sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print("Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ")
print(g)

input("Program paused. Press enter to continue.")

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print("Initializing Neural Network Parameters ...")

initial_Theta1 = randInitializeWeights.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights.randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.r_[initial_Theta1.flatten(order='F'), initial_Theta2.flatten(order='F')]

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#

print("Checking Backpropagation... ")

# Check gradients by running checkNNGradients
checkNNGradients.checkNNGradients()

input("Program paused. Press enter to continue.")



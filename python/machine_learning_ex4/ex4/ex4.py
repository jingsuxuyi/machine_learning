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


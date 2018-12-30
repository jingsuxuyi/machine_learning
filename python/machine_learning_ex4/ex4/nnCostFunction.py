# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:33:53 2018

@author: Jingjing
"""

import numpy as np
import sigmoid 

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lam):
	#NNCOSTFUNCTION Implements the neural network cost function for a two layer
	#neural network which performs classification
	#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
	#   X, y, lambda) computes the cost and gradient of the neural network. The
	#   parameters for the neural network are "unrolled" into the vector
	#   nn_params and need to be converted back into the weight matrices. 
	# 
	#   The returned parameter grad should be a "unrolled" vector of the
	#   partial derivatives of the neural network.
	#

	# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	# for our 2 layer neural network	
	
	Theta1 = nn_params[0: hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size, input_layer_size+1, order='F')
	Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels, hidden_layer_size + 1, order='F')

	# Setup some useful variables
	m = X.shape[0]	

	# X shape is 5000x401
	X = np.c_[np.ones((m, 1)), X]
	# z_2 shape is 5000x25
	z_2 = X.dot(Theta1.T)
	a_2 = sigmoid.sigmoid(z_2)

	# a_2 shape is 5000x26
	a_2 = np.c_[np.ones((m, 1)), a_2]
	# z_3 shape is 5000x10
	z_3 = a_2.dot(Theta2.T)
	a_3 = sigmoid.sigmoid(z_3)

	# recode y to vector, shape is 5000x10
	row_index = np.arange(m)
	y_mat = np.zeros((m, num_labels))
	y_mat[row_index, y-1] = 1 #y = 10 indicates 0
	'''y_mat = np.zeros((m, num_labels))
	for i in np.arange(m):
		y_mat[i, y[i]%num_labels] = 1'''

	J = np.sum(-y_mat * np.log(a_3) - (1 - y_mat) * np.log(1 - a_3)) / m

	# calc the regularization
	theta1_r = Theta1[:, 1:]
	theta2_r = Theta2[:, 1:]
	theta_sum = np.sum(theta1_r ** 2) + np.sum(theta2_r ** 2)
	J = J + lam * theta_sum / 2 / m
	
	return J
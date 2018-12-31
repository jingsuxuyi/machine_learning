import numpy as np
import debugInitializeWeights
import scipy.optimize as opt
import nnCostFunction
import computeNumericalGradient
import numpy.linalg as lin

def checkNNGradients(lam=0):
	#CHECKNNGRADIENTS Creates a small neural network to check the
	#backpropagation gradients
	#   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
	#   backpropagation gradients, it will output the analytical gradients
	#   produced by your backprop code and the numerical gradients (computed
	#   using computeNumericalGradient). These two gradient computations should
	#   result in very similar values.
	#
	
	input_layer_size = 3
	hidden_layer_size = 5
	num_labels = 3
	m = 5
	
	# We generate some 'random' test data
	Theta1 = debugInitializeWeights.debugInitializeWeights(hidden_layer_size, input_layer_size)
	Theta2 = debugInitializeWeights.debugInitializeWeights(num_labels, hidden_layer_size)
	# Reusing debugInitializeWeights to generate X
	X  = debugInitializeWeights.debugInitializeWeights(m, input_layer_size - 1)
	y  = 1 + np.arange(1, m+1) % num_labels
	
	# Unroll parameters
	nn_params = np.r_[Theta1.flatten(order='F'), Theta2.flatten(order='F')]
	
	# define lambda expression
	costFunc = lambda p: nnCostFunction.nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lam)
	
	gradFunc = lambda p: nnCostFunction.nnGradient(p, input_layer_size, hidden_layer_size, num_labels, X, y, lam)
	
	grad = gradFunc(nn_params)
	
	numgrad = computeNumericalGradient.computeNumericalGradient(costFunc, nn_params)
	
	# Visually examine the two gradient computations.  The two columns
	# you get should be very similar. 
	print(np.c_[grad, numgrad])
	
	print("The above two columns you get should be very similar \
			(Left-Your Numerical Gradient, Right-Analytical Gradient)\n")
			
	# Evaluate the norm of the difference between two solutions.  
	# If you have a correct implementation, and assuming you used EPSILON = 0.0001 
	# in computeNumericalGradient.m, then diff below should be less than 1e-9	
	diff = lin.norm(numgrad - grad) / lin.norm(numgrad + grad)
	
	print("If your backpropagation implementation is correct, then \n \
	the relative difference will be small (less than 1e-9). \n \
	elative Difference:", diff)
	
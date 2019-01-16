import numpy as np
import trainLinearReg
import linearRegCostFunction

def learningCurve(X, y, Xval, yval, lam):
	#LEARNINGCURVE Generates the train and cross validation set errors needed 
	#to plot a learning curve
	#   [error_train, error_val] = ...
	#       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
	#       cross validation set errors for a learning curve. In particular, 
	#       it returns two vectors of the same length - error_train and 
	#       error_val. Then, error_train(i) contains the training error for
	#       i examples (and similarly for error_val(i)).
	#
	#   In this function, you will compute the train and test errors for
	#   dataset sizes from 1 up to m. In practice, when working with larger
	#   datasets, you might want to do this in larger intervals.
	#
	
	# Number of training examples
	m = y.size
	
	# You need to return these values correctly
	error_train = np.zeros(m)
	error_val = np.zeros(m)

	for i in np.arange(m):
		theta = trainLinearReg.trainLinearReg(X[:i+1, :], y[:i+1], lam)
		J_train = linearRegCostFunction.linearRegCostFunction(theta, X[:i+1, :], y[:i+1], 0)
		J_val = linearRegCostFunction.linearRegCostFunction(theta, Xval, yval, 0)
		error_train[i] =  J_train
		error_val[i] = J_val
		
	return error_train, error_val
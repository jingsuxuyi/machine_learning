import numpy as np
import scipy.optimize as opt
import lrCostFunction

def oneVsAll(X, y, num_labels, lambda_n):
	"""
	ONEVSALL trains multiple logistic regression classifiers and returns all
	the classifiers in a matrix all_theta, where the i-th row of all_theta 
	corresponds to the classifier for label i
	[all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
	logistic regression classifiers and returns each of these classifiers
	in a matrix all_theta, where the i-th row of all_theta corresponds 
	to the classifier for label i
	"""
	
	#Some useful variables
	m, n = X.shape
	
	# You need to return the following variables correctly 
	all_theta = np.zeros((num_labels, n + 1))
	
	# Add ones to the X data matrix
	X = np.c_[np.ones((m, 1)), X]
	initial_theta = np.zeros(n + 1)
	
	for ii in np.arange(num_labels):
		print("label {} is runing...".format(ii+1))
		# note the args, not y, but y == ii+1
		result = opt.minimize(fun=lrCostFunction.lrCostFunction, x0=initial_theta,\
							 args=(X, (y == (ii+1)).astype(np.float64), lambda_n), method="TNC", jac=lrCostFunction.gradient, options={'maxiter': 50})
		all_theta[ii, :] = (result.x)[np.newaxis, :]
		#print(result.x.shape)
		
	return all_theta
	
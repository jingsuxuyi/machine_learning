import numpy as np

def computeNumericalGradient(J, theta):
	#COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
	#and gives us a numerical estimate of the gradient.
	#   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
	#   gradient of the function J around theta. Calling y = J(theta) should
	#   return the function value at theta.

	# Notes: The following code implements numerical gradient checking, and 
	#        returns the numerical gradient.It sets numgrad(i) to (a numerical 
	#        approximation of) the partial derivative of J with respect to the 
	#        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
	#        be the (approximately) the partial derivative of J with respect 
	#        to theta(i).)
	#                
	numgrad = np.zeros(theta.shape)
	perturb = np.zeros(theta.shape)
	delta = 1e-4
	for i in np.arange(theta.size):
		perturb[i] = delta
		loss1 = J(theta - perturb)
		loss2 = J(theta + perturb)
		numgrad[i] = (loss2 - loss1) / 2 / delta
		perturb[i] = 0
		
	return numgrad
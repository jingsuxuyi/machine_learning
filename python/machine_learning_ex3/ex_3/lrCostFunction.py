import sigmoid
import numpy as np

def lrCostFunction(theta, X, y, lambda_n):
	"""LRCOSTFUNCTION Compute cost and gradient for logistic regression with regularization
    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

	# Initialize some useful values
	m = y.size
	hypothesis = sigmoid.sigmoid(X.dot(theta))
	#copy must be used!!!
	theta_temp = theta.copy()
	theta_temp[0] = 0
	J = -y.dot(np.log(hypothesis)) - (1 - y).dot(np.log(1 - hypothesis)) + theta_temp.dot(theta_temp) * lambda_n / 2
	J = J / m
	return J
	
def gradient(theta, X, y, lambda_n):

	# Initialize some useful values
	m = y.size
	hypothesis = sigmoid.sigmoid(X.dot(theta))
	#copy must be used!!!
	theta_temp = theta.copy()
	theta_temp[0] = 0
	grad = X.T.dot(hypothesis - y) / m + lambda_n * theta_temp / m
	return grad
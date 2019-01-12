def linearRegCostFunction(theta, X, y, lam):
	'''LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
	regression with multiple variables
	   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
	   cost of using theta as the parameter for linear regression to fit the 
	   data points in X and y. Returns the cost in J and the gradient in grad'''
	
	# Initialize some useful values
	m = y.size
	
	temp_var =  X.dot(theta) - y
	J = temp_var.T.dot(temp_var) / (2 * m)
	theta_temp = theta.copy()
	theta_temp[0] = 0
	J = J + theta_temp.T.dot(theta_temp) * lam / (2 * m)
	
	return J
	
def linearRegGradient(theta, X, y, lam):
	'''LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
	regression with multiple variables
	   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
	   cost of using theta as the parameter for linear regression to fit the 
	   data points in X and y. Returns the cost in J and the gradient in grad'''
	
	# Initialize some useful values
	m = y.size
	
	temp_var =  X.dot(theta) - y
	theta_temp = theta.copy()
	theta_temp[0] = 0
	grad = X.T.dot(temp_var) / m + lam * theta_temp / m
	
	return grad
import numpy as np
import linearRegCostFunction
import scipy.optimize as opt
#test
def trainLinearReg(X, y, lam):
	#TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
	#regularization parameter lambda
	#   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
	#   the dataset (X, y) and regularization parameter lambda. Returns the
	#   trained parameters theta.
	#
	if len(X.shape) != 2:
		X = X.reshape(-1, 1)
	# Initialize Theta
	initial_theta = np.zeros(X.shape[1])

	# Create "short hand" for the cost function to be minimized
	costFunc = lambda p: linearRegCostFunction.linearRegCostFunction(p, X, y, lam)
	gradFunc = lambda p: linearRegCostFunction.linearRegGradient(p, X, y, lam)

	#minimize the target function
	result = opt.minimize(fun=costFunc, x0=initial_theta, method="TNC", jac=gradFunc, options={'maxiter': 200})
	
	return result.x

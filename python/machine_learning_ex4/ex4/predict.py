import numpy as np
import sigmoid

def predict(Theta1, Theta2, X):
	"""PREDICT Predict the label of an input given a trained neural network
	   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	   trained weights of a neural network (Theta1, Theta2)"""
	# Useful values
	m = X.shape[0]
	num_labels = Theta2.shape[1]
	
	X = np.c_[np.ones((m, 1)), X]
	
	a_2 = sigmoid.sigmoid(Theta1.dot(X.T))
	a_2 = np.r_[np.ones((1, m)), a_2]
	a_3 = sigmoid.sigmoid(Theta2.dot(a_2))
	
	p = np.argmax(a_3, axis=0) + 1
	
	return p.T
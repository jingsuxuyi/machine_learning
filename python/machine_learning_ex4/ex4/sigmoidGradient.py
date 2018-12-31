import sigmoid

def sigmoidGradient(z):
	temp = sigmoid.sigmoid(z)
	return temp * (1 - temp)
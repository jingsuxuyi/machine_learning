import numpy as np

def debugInitializeWeights(fan_out, fan_in):
	#DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
	#incoming connections and fan_out outgoing connections using a fixed
	#strategy, this will help you later in debugging
	#   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
	#   of a layer with fan_in incoming connections and fan_out outgoing 
	#   connections using a fix set of values
	#
	#   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
	#   the first row of W handles the "bias" terms
	#
	size_r = fan_out * (1 + fan_in)
	return np.sin(np.arange(1, size_r+1)).reshape((fan_out, 1+fan_in), order='F') / 10 
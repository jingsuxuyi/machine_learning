## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy as np
import scipy.io as sio
import displayData
import predict

# Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
mat_contents = sio.loadmat('ex3data1.mat')

X, y = mat_contents['X'], mat_contents['y']
y = y.flatten()
m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.randint(0, m, size=100)
sel = X[rand_indices, :]

displayData.displayData(sel)

input("Program paused. Press enter to continue.")

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print("Loading Saved Neural Network Parameters ...")
mat_contents = sio.loadmat('ex3weights.mat')
Theta1, Theta2 = mat_contents['Theta1'], mat_contents['Theta2']

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict.predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy: {:f}'.format(np.mean((pred == y).astype(np.float64)) * 100))

input("Program paused. Press enter to continue.")

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.


#  Randomly permute examples
rp = np.random.randint(0, m, size=m)
print(rp)
for i in np.arange(m):
	#display
	print("Displaying Example Image")
	displayData.displayData(X[rp[i], :].reshape(1,-1))
	pred = predict.predict(Theta1, Theta2, X[rp[i],:].reshape(1,-1))
	print('Neural Network Prediction: {:d} (digit {:d})\n'.format(pred[0], np.mod(pred, 10)[0]))
	
	#Pause with quit option
	s = input("Paused - press enter to continue, q to exit: ")
	if s == 'q':
		break
	
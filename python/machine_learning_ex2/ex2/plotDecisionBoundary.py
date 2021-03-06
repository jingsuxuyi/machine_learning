# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 20:05:28 2018

@author: Jingjing
"""
import matplotlib.pyplot as plt
import plotData
import numpy as np
import mapFeature

def plotDecisionBoundary(theta, X, y):
#PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
#   positive examples and o for the negative examples. X is assumed to be 
#   a either 
#   1) Mx3 matrix, where the first column is an all-ones column for the 
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones    
    
    #plot Data
    plotData.plotData(X[:, 1:], y, "Admited", "Not admited")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")  
    
    m, n = X.shape
    if n <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        #the min-2, max+2 are only used to indice the X axis, i.e. x1
        #boundary function is theta0+theta1*x1+theta2*x2=0, x2 is Y axis
        plot_x = np.array([np.min(X[:, 1])-2, np.max(X[:, 1])+2])
        plot_y = -1 / theta[2] * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y, label="Decision Boundary")
        plt.legend()
        plt.axis([30, 100, 30, 100])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((np.size(u), np.size(v)))
        
        for i in np.arange(np.size(u)):
            for j in np.arange(np.size(v)):
                z[i,j] = mapFeature.mapFeature(u[i], v[j]).dot(theta)
        plt.contour(u, v, z.T, levels=[0], linewidths=2)
        plt.legend()
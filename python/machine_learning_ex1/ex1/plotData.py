# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:04:32 2018

@author: Jingjing
"""

import matplotlib.pyplot as plt

def plotData(x, y):
    plt.figure()
    plt.scatter(x, y, s=15, c='r', marker='x', linewidths=1, label="Training data")
    plt.show()
    
    
if __name__ == "__main__":
    plotData([1,2,3,4], [1,2,3,4])
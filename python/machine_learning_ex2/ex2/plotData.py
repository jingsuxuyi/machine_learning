# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:27:25 2018

@author: Jingjing
"""

import matplotlib.pyplot as plt

def plotData(X, y, xlabel, ylabel):
    fig, ax = plt.subplots(1,1)
    ax.plot(X[y==1, 0], X[y==1, 1], 'k+', linewidth=2, markersize=7, label=xlabel)
    ax.plot(X[y==0, 0], X[y==0, 1], 'ko', markerfacecolor='y', markersize=7, label=ylabel)

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 21:48:24 2018
@author: Jingjing
"""

import numpy as np
import matplotlib.pyplot as plt

def displayData(x, examle_width=0):
#DISPLAYDATA Display 2D data in a nice grid
#   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
#   stored in X in a nice grid. It returns the figure handle h and the 
#   displayed array if requested.
    m, n = x.shape
    # Set example_width automatically if not passed in
    if examle_width == 0:
        example_width = np.round(np.sqrt(n)).astype(np.int32)
    example_height = (n / example_width).astype(np.int32)
    
    #  Compute number of items to display
    display_rows = np.floor(np.sqrt(m)).astype(np.int32)
    display_cols = np.ceil(m / display_rows).astype(np.int32)
    
    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_height + pad)))  
    
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_rows):
            if curr_ex > m:
                break
            #copy the patch
            
            # get the max values of the patch
            max_val = np.max(x[curr_ex, :])
            display_array[pad+j*(example_height+pad)+np.arange(example_height),\
                          pad+i*(example_width+pad)+np.arange(example_width)[:,np.newaxis]]=\
                          x[curr_ex].reshape((example_height, example_width)) / max_val
                       
            #print(pad+i*(example_width+pad)+np.arange(example_width))
            curr_ex += 1
            if curr_ex > m:
                break
            
    # Display image
    plt.figure()
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.axis('off') 
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:35:25 2017

@author: Jingjing Guo
"""

import numpy as np
from random import randint
import matplotlib.pyplot as plt
#import pandas as pd

def row2m28(row_num):
    with open("digits-raw.csv") as f:
        for i, line in enumerate(f):
            if i == (row_num) :
                rem_punc = line.maketrans('', '')
                m784 = line.translate(rem_punc).split(',')
            elif i > row_num:
                break
    output = np.array(m784, dtype='int16')[2:786].reshape(28,28)
    plt.imshow(output,cmap=plt.gray())

def visualizeData(sampleSize):
    samples = np.random.choice(range(20000), sampleSize, replace = False).tolist()

    sample1000Data = []
    counter = 0
    iis = []
    with open("digits-embedding.csv") as f:
        for i, line in enumerate(f):
            if i in samples:
                iis.append(i)
                counter += 1
                rem_punc = line.maketrans('', '')
                sample1000Data.append(line.translate(rem_punc).split(','))
                if counter > sampleSize:
                    break
                
    data = np.array(sample1000Data)
#    index = np.array(data[:,0], dtype = 'int16')
    label = np.array(data[:,1], dtype = 'int16')
    xy = np.array(data[:,2:4], dtype = 'float32')
    
    plt.scatter(xy[:,0], xy[:,1], c=label, cmap=plt.cm.jet)
    plt.show()

'''
Part A: 1. randomly pick a number and show its digit
'''
plt.figure(0)
i = randint(1, 19999)
row2m28(i)

'''
Part A: 2. visualize 1000 
'''
plt.figure(1)
visualizeData(1000)
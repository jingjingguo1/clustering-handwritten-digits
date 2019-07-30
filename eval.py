# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 19:05:26 2017

@author: Jingjing Guo
"""
#import numpy as np
from numpy import ma,zeros,multiply,log, array

# NMI
def computeNMI(CG, n_G, n_C):
    #CG should be a two column np array with Column 1 for Cluster and Column 2 for Class
    pcg = zeros([n_G,n_C])
    for line in CG:
        pcg[line[0], line[1]] += 1 # column 1 is cluster and thus row
    pcg = pcg/pcg.sum()
    
    pc = pcg.sum(axis=0)
    pg = pcg.sum(axis=1)
    
#    construct pcpg, where pcpg[i,j]:= p(c=i) and p(g=j)
    pcpg = zeros([n_G, n_C])
    for i in range(n_G):
        for j in range(n_C):
            print(i,j)
            pcpg[i,j] = 1/(pc[j]*pg[i])
    
    
    forLog0 = multiply(pcg, pcpg)
    forLog0 = ma.log(forLog0)
    forLog = forLog0.filled(0)
    
    numerator = multiply(pcg, forLog).sum()
    denominatorM = -multiply(pc, log(pc)).sum() - multiply(pg, log(pg))
    NMI = numerator/denominatorM.sum()
    return NMI

''' TEST
CG = array([[0,1], [1,1], [0,0], [1,2], [1,0]])
n_C = 3
n_G = 2

NMI = computeNMI(CG, n_G, n_C)
'''
#NMI = computeNMI(cg, n_G, n_C)
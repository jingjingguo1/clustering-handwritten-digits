# -*- coding: utf-8 -*-
"""
@author: Jingjing Guo
"""

#import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from kmean import d2

def sample10x10():
    #create samples 
    df = pd.read_csv('digits-embedding.csv', header=None, names=["label","x","y" ])
    n_d = df.label.value_counts()
    
    '''
    define variable P100 to store 100 points np.array
    for i in range(10):
        class_data = extracted points from pd
        indices = np.random.choice(range(n_d[i]), 10, replace = False).tolist()
        P100.append(class_data[indices,:], axis=0)
    '''
    
    P100 = np.empty([0,2])
    for i in range(10):
        class_data = np.array(df[df.label==i])
        indices = np.random.choice(range(n_d[i]), 10, replace = False).tolist()
        P100 = np.concatenate((P100, class_data[indices, 1:3]), axis = 0)
    #k0_i = np.random.choice(range(n_df), K, replace = False).tolist()
    #
    #
    #sp.cluster.hierarchy()
    return P100

def DPnxn(pn,n):
    """
    find the upper triangle distance matrix
    """
    dijU = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j >= i:
                dijU[i][j] = d2(pn[i], pn[j])
    return dijU

def hc(P100, method):
    """
    take int the P100 
    """
    
    Input = DPnxn(P100,100)
    
    Z = linkage(Input, method)
    
    
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()
    
    
# run cases
P100 = sample10x10()

plt.figure(0)
Z0 = hc(P100, 'single')
i = 0

P2use = np.empty([100,4]) # column 1: X, column 2: Y, column 3: cluster, column 4: distance to cluster centroid
P2use[:,[0,1]] = P100
     
wck = []
sck = []

for K  in range(5):
    c = fcluster(Z0, K+1,'maxclust')
    realK = len(set(c))
    
    centroids = np.empty([realK, 2])
    for i in range(realK):
        cii = np.where(c==i) #cluster i indices
        P2use[cii,2] = i # use cluster label given in c to update P2use cluster column
    # Find centroids
        iXY= P2use[cii,[0,1]]
        centroids[i,0], centroids[i,1] = iXY.mean(axis=1)

    # loop through all points and attach d in P2use[:,3]    
    for i,row in enumerate(P2use):
        p1 = row[0:2]
        p2 = centroids[row[2]]
    
        P2use[i,3] = d2(p1,p2)
    
   
    #calculate WC-SSD
    WC_SSD = sum(P2use[:,3]**2)
    
    wck.append(WC_SSD)
    #calculate SC
    pn = P100
    dij = DPnxn(pn,100)
    S = [] # K
    # find index sets for K clusters
    for i in range(realK):
        S_i = np.where(c==i)
        S.append(S_i)
    SCi = []
    # 
    for i in range(100):
        ci = np.where(c==i)
        a = dij[i,:][S[ci]].mean()
        b = dij[i,:][list(set(range(100))-set(S[ci]))].mean()
        SCi.append( (b-a)/max(a,b) )
    SC = np.array(SCi).mean()
    sck.append(SC)    

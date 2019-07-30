# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:15:45 2017

@author: guo49
"""

from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
import sys
import numpy as np
import pandas as pd
import math

def d2(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def pick_k(px, cks):
    minD = 1e10
    for i in range(len(cks)):
        Dxk = d2(px, cks[i])
        if Dxk < minD:
            minC, minD = i, Dxk
    return minC, minD

def DPnxn(pn,n):
    dij = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dij[i][j] = d2(pn[i], pn[j])
    return dij

def importData(rows):
    df = pd.read_csv('digits-embedding.csv', header=None, names=["label","x","y" ])
    df['c'] = 0
    df['dc']= 0.0
    # delete what is not in
    toDelete = set(range(10)) - set(rows)
    df.ix[df.label.isin(toDelete)]
    
#    df = df[df['label'].isin(rows)]
    return df


#df = pd.read_csv('digits-embedding.csv', header=None, names=["label","x","y" ])
#df['c'] = 0
#df['dc']= 0.0
#subset = df.loc[df['label'].isin([2,4,6,7])]

#K = 5

def kmeans(datafile, K):

    df = pd.read_csv(datafile, header=None, names=["label","x","y" ])
    
    df['c'] = 0
    df['dc']= 0.0
    
    n_df = df.shape[0]
    df = df.reset_index(drop=True)
    
    
    # Intial K points
    k0_i = np.random.choice(range(n_df), K, replace = False).tolist()
    k0 = np.array(df.iloc[k0_i, 1:3])
    centroids_old = np.zeros([K,2])
    centroids_new = k0
    #    classes = df['label'].iloc[k0_i]
    
    #xy = np.array(k0[['x','y']])
    
    # check if no changes or if counter has reached 50
    counter = 0
    
      
    while sum(sum((np.array(centroids_new)-np.array(centroids_old))**2)) > 1e-3:
    #while list(centroids_new) != list(centroids_old):
        counter += 1
        # assign labels
        for i in range(n_df):
            px = np.array(df.iloc[i,1:3])
    #        c,dc = pick_k(px, centroids_new)
            df.loc[i,'c'], df.loc[i,'dc'] =  pick_k(px, centroids_new)
        centroids_old = np.array(centroids_new)
    #        print(list(centroids_old))
        
        for j in range(K):
            k_cluster = df.loc[df['c']==j]
            xc = k_cluster['x'].mean()
            yc = k_cluster['y'].mean()
            
            centroids_new[j] = [xc,yc]
        if counter > 50:
            break
    
    
    
    #calculate WC-SSD
    WC_SSD = sum(np.array(df.dc)**2)
    
    #calculate SC
    pn = np.array(df[['x','y']])
    dij = DPnxn(pn,n_df)
    S = [] # K
    # find index sets for K clusters
    for i in range(K):
        S_i = df[df.c==i].index.values
        S.append(S_i)
    SCi = []
    # 
    for i in range(n_df):
        ci = df.c[i]
        a = dij[i,:][S[ci]].mean()
        b = dij[i,:][list(set(range(n_df))-set(S[ci]))].mean()
        SCi.append( (b-a)/max(a,b) )
    SC = np.array(SCi).mean()
    
    SC = metrics.silhouette_score(pn, df.label)
    
    NMI = normalized_mutual_info_score(df.label,df.c)      
    
    return WC_SSD, SC, NMI
    

if len(sys.argv) == 3:
    data_file = sys.argv[1]
    K = sys.argv[2]
    
    WC_SSD, SC, NMI= kmeans(data_file, K)
    
    print('WC-SSD:', WC_SSD)
    print('SC:', SC)
    print('NMI:', NMI)


#K = 5
#
#WC_SSD, SC = kmeans(K)
#
#print('WC-SSD', WC_SSD)
#print('SC', SC)


import numpy as np
import math
from scipy.special import comb
from itertools import combinations

def get_k(shape):
    m,n = shape
    k = 0
    sum = 0
    while(True):
        nCr = comb(n+k-1,k,exact = True) #calculates nCr
        if(nCr == 0):
            break
        sum += nCr
        if(sum > m-1):
            break
        
        k += 1
    
    return k
    
def featureEngineer(dataset):
    dataset = np.asarray(dataset)
    m,n = dataset.shape
    k = get_k(dataset.shape)
    a = np.arange(n)    
    
    for i in range(2,k+1):
        new_cols = np.asarray([])

        b = list(combinations(a,i))
        #b = np.asarray(b)

        for element in b:
            subset = dataset[:,element]
            subset = np.prod(subset, axis = 1)
            #print(subset.shape)
            subset = np.reshape(subset,(subset.shape[0],1))
            
            if(new_cols.size):
                new_cols = np.concatenate((new_cols,subset),axis =1)
            else:
                new_cols = subset 

        dataset = np.concatenate((dataset,new_cols),axis =1)

    bias = np.ones((m,1))
    dataset = np.concatenate((bias,dataset),axis =1)
    
    return dataset
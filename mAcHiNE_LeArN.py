# ### Import stuff

# In[ ]:


import numpy as np
import pandas as pd
import h5py
from featureEngineer import featureEngineer
from readData import readData


# ### Set global variables
# 
# Note: The first N entries in the dataset should have labels. The rest will be used for testing. The very last column should contain the labels 

# In[ ]:


FILE_PATH = 'dataset.csv'
OUT_FILE = 'run1.h5'


# ### Clean and split data into arrays

# In[ ]:


X,y = readData(FILE_PATH)
X,k = featureEngineer(X)

A = np.matmul(X.T,X)
B = np.matmul(X.T,y)

A = np.linalg.pinv(A)

W = np.matmul(A,B)#These are the learned weights

with h5py.File(OUT_FILE,'w') as file:
    file.create_dataset('weights', W.shape)
    file['weights'][...] = W
    file.create_dataset('k',(1,), data = k)
    


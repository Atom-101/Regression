# ### Import stuff

# In[ ]:


import numpy as np
import pandas as pd
import h5py
from featureEngineer import featureEngineer
from readData import readData


with h5py.File('run1.h5', 'r') as f:
    W = f['weights'][...]
    k = f['k'][...][0]

dataset = readData('dataset_test.csv', train=False)
dataset,_ = featureEngineer(dataset, k)


predict = np.matmul(dataset,W)
print(predict)

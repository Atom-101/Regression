import pandas as pd
import numpy as np

def readData(file):
  df = pd.read_csv(file)
  for column in df:
    if (df[column].dtype != np.number):
      df[column] = pd.factorize(df[column])[0]
  
  dataset = df.values
  
  return dataset

  
  

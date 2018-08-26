import pandas as pd
import numpy as np

def readData(file, train = True):
    df = pd.read_csv(file)

    if(train):    
        #y = df.pop(df.columns[-1]) #select all rows and last column
        #y = y.replace('',np.nan)
        for column in df:#iterate over all columns except y
            df[column].replace('', np.nan, inplace=True) #make empty strings as NaN 
            df.dropna(subset=[column], inplace=True) #drop rows with NaNs

            if (df[column].dtype != np.number):
                df[column] = pd.factorize(df[column], sort = True)[0]

        #Get y and remove rows for test
        y = df.ix[:,-1]
        y = y.values
        y = np.reshape(y,(y.shape[0],1))
        print(y.shape)
        #delete y from main dataframe
        df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
        dataset = df.values
        print(dataset.shape)
  
        return dataset,y
    
    else:
        for column in df:
            if (df[column].dtype != np.number):
                df[column] = pd.factorize(df[column], sort = True)[0]
        
        dataset = df.values

        return dataset


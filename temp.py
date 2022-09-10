from fileinput import close
import pandas as pd
import numpy as np
from pandas import HDFStore
#hdf = HDFStore('hdf_file.h5')

#df = pd.read_csv("iris.csv")  # read Iris file
#hdf.put('key1', df, format='table', data_columns=True) #put data in hdf file

#df2 = pd.DataFrame(np.random.rand(5,3),columns=['X','Y','Z']) #dataframe df2
#hdf.put('key2',df2) # to add a dataframe to the hdf file
#df3= pd.DataFrame(np.random.rand(10,2),columns=['X','Y'])
#hdf.put('/group1/key3',df3) # to add a group with df3 in the hdf file
#hdf =HDFStore('hdf_file.h5', mode='r')
#data = hdf.get('/key1')
#hdf.close()


hdf =HDFStore('hdf_file.h5', mode='r')
data = hdf.get('/key1')

new = data.loc[data['class'] == 'Iris-setosa']
print(new)

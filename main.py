from audioop import lin2adpcm
from cProfile import label
from matplotlib.dates import SecondLocator
from sympy import GoldenRatio
from EMFFeko import *
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from EMFIXUS import IXUSField
import time
import mayavi.mlab as mlab





#df = GetField('IEC-62232-panel-antenna (3)_NearField3.efe','IEC-62232-panel-antenna (3)_NearField3.hfe').df


xrange = np.arange(0.1, 20, 0.15)
yrange = np.arange(-20, 20, 0.15)
zrange = np.arange(-20, 20, 0.15)

x,y,z = np.meshgrid(xrange, yrange, zrange)

data = np.array([x,y,z]).reshape(3,-1).T
df = pd.DataFrame(data,columns=['X','Y','Z'])

df['R'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
df['phi'] = np.arccos(df['X']/df['R'])
df['theta'] = np.arccos(df['Z']/df['R'])
df['S'] = PeakCylindricalSector(df['phi'], df['R'])
#test_mesh(df, error = 0.5)

df = GetField('IEC-62232-panel-antenna (3)_NearField1.efe','IEC-62232-panel-antenna (3)_NearField1.hfe').df

#plotByCartesian(df,1)

plotSZones(df,0.345, error = 0.01)

#Validationtest1()
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







xrange = np.arange(0.1, 20, 2)
yrange = np.arange(-20, 20, 2)
zrange = np.arange(-20, 20, 2)

x,y,z = np.meshgrid(xrange, yrange, zrange)

data = np.array([x,y,z]).reshape(3,-1).T
df = pd.DataFrame(data,columns=['X','Y','Z'])

df['R'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
df['phi'] = np.arccos(df['X']/df['R'])
df['theta'] = np.arccos(df['Z']/df['R'])
df['S'] = PeakCylindricalSector(df['phi'], df['R'])
df['Gain'] = GetGain(df['phi'],df['theta'])

print(df['Gain'])
plotByCartesian(df)
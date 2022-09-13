from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import HDFStore
import math
from Field import Field

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)

class Antenna:
    def __init__(self,P,G,f,x=0,y=0,z=0):
        self.x = x
        self.y = y
        self.z = z
        self.P = P
        self.G = G
        self.f = f

class Surface:
    def __init__(self,xMin,xStep,xMax,yMin,yStep,yMax,z=0): 
        self.xMin = xMin
        self.xMax = xMax
        self.xStep = xStep
        self.yMin = yMin
        self.yMax = yMax
        self.yStep = yStep   

        x = np.linspace(xMin,xMax,xStep)
        y = np.linspace(yMin,yMax,yStep)
        xv,yv= np.meshgrid(x,y)
        
        self.df = pd.DataFrame({
            'X':xv.ravel(),
            'Y':yv.ravel(),
            'Z':z
        })
        self.df = self.df.astype(float)

class Field(Field):
    def __init__(self,antenna,space):
        space.df['R'] = np.sqrt((space.df['X']-antenna.x)**2 + (space.df['Y']-antenna.y)**2 + (space.df['Z']-antenna.z)**2)
        space.df.loc[space.df['R'] < 0.5,'R'] = 100
        space.df['S(E)'] = (antenna.P*antenna.G)/(4*np.pi*(space.df['R'])**2)
        super().__init__(space.df,antenna.f)
        self.antenna = antenna
        self.space = space

        i = 0
        xSamples = self.space.xStep
        ySamples = self.space.xStep
        #zSamples = self.space.xStep

        if xSamples > 1: i+=1
        if ySamples > 1: i+=1
        #if zSamples > 1: i+=1
        self.dimentions = i
        if self.dimentions == 2:
            if xSamples > 1 and ySamples > 1:
                self.axis = ['X','Y'] 
            #elif xSamples > 1 and zSamples > 1:
            #    self.axis = ['X','Z']
            #elif ySamples > 1 and zSamples > 1:
            #    self.axis = ['Y','Z']
        



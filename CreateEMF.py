from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import HDFStore
import math
from Field import Field
import mayavi.mlab as mlab
from mayavi.mlab import *


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
        self.xMesh,self.yMesh = np.mgrid[xMin:xMax:(xStep*1j) , yMin:yMax:(yStep*1j)]
        self.df = self.df.astype(float)

class Field(Field):
    def __init__(self,antenna,space,spaceMin = 3,sMax = 10):
        space.df['R'] = np.sqrt((space.df['X']-antenna.x)**2 + (space.df['Y']-antenna.y)**2 + (space.df['Z']-antenna.z)**2)
        space.df['S(E)'] = (antenna.P*antenna.G)/(4*np.pi*(space.df['R'])**2)
        space.df.loc[space.df['S(E)'] > sMax ,'S(E)'] = space.df['R']
        print(space.df.sort_values(by=['S(E)']))
        super().__init__(space.df,antenna.f)
        self.antenna = antenna
        self.space = space
        self.df = space.df
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
        
    def plot2D(self, field='S(E)', color='Reds', method='mayavi',show = True):
            if method == 'cadfeko':
                fig, ax = plt.subplots(1)
                ax1 =ax.scatter(x =self.df[self.axis[0]],y= self.df[self.axis[1]],c =self.df[field],cmap = color)
                plt.colorbar(ax1)
                ax.set_xlabel(self.axis[0])
                ax.set_ylabel(self.axis[1])
                ax.set_title("{} over {}{} plane".format(field,self.axis[0],self.axis[1]))
                if show:
                    plt.show()
            elif method == "mayavi":
                #mlab.points3d(self.df['X'],self.df['Y'],self.df[field],self.df[field],colormap = 'Reds',scale_mode='none')
                #mlab.points3d(self.df['X'],self.df['Y'],self.df[field],self.df['R'],colormap = 'Reds',scale_mode='none')
                self.df = self.df.sort_values(by=['Z','Y','X'])
                arr = self.df[field].to_numpy()
                arr = arr.reshape(self.space.xStep,self.space.yStep)
                mlab.surf(arr,warp_scale = 'auto')
                mlab.show()





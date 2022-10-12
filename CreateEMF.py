from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import *
from Field import *
import mayavi.mlab as mlab
from mayavi.mlab import *


import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)


class AntennaSurface:
    def __init__(self,xMin,xMax, xStep,yMin,yMax ,yStep, zMin, zMax ,zStep,P = 80, f = 900): 
        self.xMin = xMin
        self.xMax = xMax
        self.xStep = xStep
        self.yMin = yMin
        self.yMax = yMax
        self.yStep = yStep   
        self.zMin = zMin
        self.zMax = zMax
        self.zStep = zStep   
        self.P = P
        self.f = f

    xrange = np.arange(0.1, 20, 2)
    yrange = np.arange(-20, 20, 2)
    zrange = np.arange(-20, 20, 2)
    x,y,z = np.meshgrid(xrange, yrange, zrange)

    data = np.array([x,y,z]).reshape(3,-1).T
    df = pd.DataFrame(data,columns=['X','Y','Z'])

    df['R'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    df['phi'] = np.arccos(df['X']/df['R'])
    df['theta'] = np.arccos(df['Z']/df['R'])
    df['Gain'] = GetGain(df['phi'],df['theta'])
    df['ICNIRP'] = ICNIRPmeshAverage(df['phi'], df['R'],df['theta'], f = )
    df['']
            

class Field(Field):
    def __init__(self,antenna,space,spaceMin = 3,sMax = 10):
        space.df['R'] = np.sqrt((space.df['X']-antenna.x)**2 + (space.df['Y']-antenna.y)**2 + (space.df['Z']-antenna.z)**2)
        space.df['S'] = (antenna.P*antenna.G)/(4*np.pi*(space.df['R'])**2)
        space.df.loc[space.df['S'] > sMax ,'S'] = space.df['R']
        print(space.df.sort_values(by=['S']))
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
            #   self.axis = ['Y','Z']
        
    def plot2D(self, field='S', color='Reds', method='mayavi',show = True):
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





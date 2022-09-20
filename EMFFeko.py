from fileinput import filename
from platform import freedesktop_os_release
from unittest import result
from kiwisolver import Solver
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import HDFStore
import math
from Field import Field
import mayavi.mlab as mlab
import time

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)



class Fekofield(Field):
    def __init__(self,source,frequency,coordSystem, xSamples, ySamples, zSamples,df):
        super().__init__(df,frequency*(10**-6),type = 'Feko')
        self.source = source
        self.frequency = frequency
        self.coordSystem= coordSystem
        self.xSamples = xSamples
        self.ySamples = ySamples
        self.zSamples = zSamples

        i = 0
        if xSamples > 1: i+=1
        if ySamples > 1: i+=1
        if zSamples > 1: i+=1
        self.dimentions = i
        if self.dimentions == 2:
            if xSamples > 1 and ySamples > 1:
                self.axis = ['X','Y'] 
            elif xSamples > 1 and zSamples > 1:
                self.axis = ['X','Z']
            elif ySamples > 1 and zSamples > 1:
                self.axis = ['Y','Z']
    
    def plot2D(self, c='S', color='Reds',show = True,method = 'matplotlib'):
        if method == 'matplotlib':
            fig, ax = plt.subplots(1)
            ax1 =ax.scatter(x =self.df[self.axis[0]],y= self.df[self.axis[1]],c =self.df[c],cmap = color)
            plt.colorbar(ax1)
            ax.set_xlabel(self.axis[0])
            ax.set_ylabel(self.axis[1])
            ax.set_title("{} over {}{} plane".format(c,self.axis[0],self.axis[1]))
            if show:
                plt.show()
        elif method == "mayavi":
            self.df = self.df.sort_values(by=['Z','Y','X'])
            arr = self.df[c].to_numpy()
            arr = arr.reshape(self.xSamples,self.ySamples)
            mlab.surf(arr,warp_scale = 'auto')
            mlab.show()


                
def GetField(filenameE,filenameH,S = 'S(E)',compress = True):
    source= ''
    frequency= 0
    coordSystem= ''
    xSamples= 0
    ySamples= 0
    zSamples= 0
    global i
    i = 0
    filenameE = 'venv/Include/CADFeko/{}'.format(filenameE)
    filenameH = 'venv/Include/CADFeko/{}'.format(filenameH)

    with open(filenameE, 'r') as file:
        for line in file:
            if '##Source: ' in line:
                source = line[:-1].split("##Source: ",1)[1]
            elif "#Frequency: " in line:
                frequency = int(float(line[:-1].split("#Frequency:   ",1)[1]))  
            elif "#Coordinate System: " in line:
                coordSystem = line[:-1].split("#Coordinate System: ",1)[1]
            elif "#No. of X Samples: " in line:
                xSamples = int(line[:-1].split("No. of X Samples: ",1)[1])
            elif "#No. of Y Samples: " in line:
                ySamples = int(line[:-1].split("No. of Y Samples: ",1)[1])
            elif "#No. of Z Samples: " in line:
                zSamples = int(line[:-1].split("No. of Z Samples: ",1)[1])

                global dataT
                dataT = np.zeros((ySamples*xSamples*zSamples,9)) 
            if line[0] != '#' and line[0] != '*' and line[0] != '\n':
                dataT[i] = line[4:-1].split('   ')
                i+=1
        df = pd.DataFrame(dataT,columns=['X','Y','Z','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)'])
        df = df.astype(float)
        #df['R'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        #df['phi'] = np.arccos(df['Z']/df['R'])
        #df['theta'] = np.arccos(df['X']/(df['R']*np.sin(df['phi'])))
    file.close()

    with open(filenameH, 'r') as file:
        dataH = np.zeros((ySamples*xSamples*zSamples,6))
        i = 0
        for line in file:
            if line[0] != '#' and line[0] != '*' and line[0] != '\n':
                dataH[i] = line[4:-1].split('   ')[3:]
                i+=1
    file.close()
    df['Re(Hx)'] = dataH[:,0]
    df['Im(Hx)'] = dataH[:,1]
    df['Re(Hy)'] = dataH[:,2]
    df['Im(Hy)'] = dataH[:,3]
    df['Re(Hz)'] = dataH[:,4]
    df['Im(Hz)'] = dataH[:,5]
    df['|E|'] = np.sqrt(df['Re(Ex)']**2  + df['Im(Ex)']**2 + df['Re(Ey)']**2  + df['Im(Ey)']**2  + df['Re(Ez)']**2 + df['Im(Ez)']**2)
    #df['|H|'] = np.sqrt(df['Re(Hx)']**2  + df['Im(Hx)']**2 + df['Re(Hy)']**2  + df['Im(Hy)']**2  + df['Re(Hz)']**2 + df['Im(Hz)']**2)

    if S == 'S(E)':
        df['S'] = (df['|E|']**2)/(377)
    elif S == 'S(E2)':
        df['S'] = (df['|E|']**2)/(2*377)
    elif S == 'S(R)':
        df['S'] = (df['|E|']**2)/(377)
    elif S == 'S(ExH)':
        df['Ex'] = df['Re(Ex)'] + df['Im(Ex)']*1j
        df['Ey'] = df['Re(Ey)'] + df['Im(Ey)']*1j
        df['Ez'] = df['Re(Ez)'] + df['Im(Ez)']*1j
        df['Hx'] = df['Re(Hx)'] + df['Im(Hx)']*1j
        df['Hy'] = df['Re(Hy)'] + df['Im(Hy)']*1j
        df['Hz'] = df['Re(Hz)'] + df['Im(Hz)']*1j

        df['Re(Sx)'] = np.real(df['Ey']*df['Hz'] - df['Ez']*df['Hy'])
        df['Im(Sx)'] = np.imag(df['Ey']*df['Hz'] - df['Ez']*df['Hy'])

        df['Re(Sy)'] = np.real(df['Ez']*df['Hx'] - df['Ex']*df['Hz'])
        df['Im(Sy)'] = np.imag(df['Ez']*df['Hx'] - df['Ex']*df['Hz'])

        df['Re(Sz)'] = np.real(df['Ex']*df['Hy'] - df['Ey']*df['Hx'])
        df['Im(Sz)'] = np.imag(df['Ex']*df['Hy'] - df['Ey']*df['Hx'])

        df['S'] = np.sqrt(df['Re(Sx)']**2 + df['Im(Sx)']**2 + df['Re(Sy)']**2 + df['Im(Sy)']**2 + df['Re(Sz)']**2 + df['Im(Sz)']**2)

    if compress:
        df = df.drop(columns = ['Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','|E|'])


    #hdf = HDFStore('hdf_file.h5')
    #hdf.put('EMF', df, format='table', data_columns=True) #put data in hdf file
    #hdf.close()
    return Fekofield(source,frequency,coordSystem, xSamples, ySamples, zSamples,df)



#def CreateField(antenna,source='IEC-62232-panel-antenna',date=0,configuration='Created',frequency = f)
#   return fieldSolved(name,Fileformat,source,date,solverV,configuration,frequency,coordSystem, xSamples, ySamples, zSamples,df)


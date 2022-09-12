from fileinput import filename
from platform import freedesktop_os_release
from unittest import result
from kiwisolver import Solver
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import HDFStore
import math

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)


class fieldSolved:
    def __init__(self,name,Fileformat,source,date,solverV,configuration,frequency,coordSystem, xSamples, ySamples, zSamples,df):
        self.name = name
        self.format = Fileformat
        self.source = source
        self.date = date
        self.solverV = solverV
        self.configuration = configuration
        self.frequency = frequency
        self.coordSystem= coordSystem
        self.xSamples = xSamples
        self.ySamples = ySamples
        self.zSamples = zSamples
        self.df = df


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

    def PowerAtPoint(data):
        S = np.zeros(len(data))
        for j in range(len(data)):
            for i in range(3,9):
                S[j] += data[j][i]**2
        S = S/(2*377) 
        return(S)

    def plotPowerLine(line,axis):
        plt.plot(line[axis],line['S(E)'])
        plt.show() 

    def getS(self):
        dfS = self.df[['X','Y','Z','S(E)']].copy()
        dfS = dfS.astype(float)
        return dfS

    def GetE(self):
        dfE = self.df[['X','Y','Z','|E|']].copy()
        dfE =  dfE.astype(float)
        return dfE

    def GetH(self):
        dfH = self.df[['X','Y','Z','|H|']].copy()
        dfH =  dfH.astype(float)
        return dfH

    def plot2DZones(self,GPcolor = 'blue',Ocolor = 'yellow',RZcolor = 'red',xfig = 6,yfig = 4,show = False):
        colors = {'General Public':GPcolor,'Occupation':Ocolor,'Restricted Zone':RZcolor}
        #plt.scatter(x=self.df[X], y=self.df[Y],c= self.df['Restriction'].map(colors))

        groups = self.df.groupby('Restriction')
       
        fig, ax = plt.subplots(1, figsize=(xfig,yfig))
        for label, group in groups:
            ax.scatter(group[self.axis[0]], group[self.axis[1]], 
                    c=group['Restriction'].map(colors), label=label)

        ax.set(xlabel= self.axis[0], ylabel=self.axis[1])
        ax.set_title("Restricions with %d signal" % self.frequency)
        ax.legend(title='Restriction levels')
        if show:
            plt.show()

    def plot2D(self,field,color = 'Reds'):
        fig, ax = plt.subplots(1)
        ax1 =ax.scatter(x =self.df[self.axis[0]],y= self.df[self.axis[1]],c =self.df[field],cmap = color)
        plt.colorbar(ax1)
        ax.set_xlabel(self.axis[0])
        ax.set_ylabel(self.axis[1])
        ax.set_title("{} over {}{} plane".format(field,self.axis[0],self.axis[1]))
        ##plt.legend(title='Restriction levels')
        
        
def GetField(filenameE,filenameH):
    name = ''
    type = ''
    Fileformat= 0
    source= ''
    date= ''
    solverV= ''
    configuration= ''
    frequency= 0
    coordSystem= ''
    xSamples= 0
    ySamples= 0
    zSamples= 0
    global i
    i = 0


    with open(filenameE, 'r') as file:
        for line in file:
            if "##File Type: " in line:
                type = line[:-1].split("##File Type: ",1)[1]
            elif '##File Format: ' in line:
                Fileformat = int(line[:-1].split("##File Format: ",1)[1])
            elif '##Source: ' in line:
                source = line[:-1].split("##Source: ",1)[1]
            elif '##Date: ' in line:
                date = line[:-1].split("##Date: ",1)[1]
            elif "- Solver (seq)" in line:
                solverV = line[:-1].split("- Solver (seq) ",1)[1]
            elif "#Configuration Name: " in line:
                configuration = line[:-1].split("#Configuration Name: ",1)[1]
            elif "#Request Name:" in line:
                name = line[:-1].split("#Request Name:",1)[1] 
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
            elif "#Result Type: " in line:
                resultType = line[:-1].split("#Result Type: ",1)[1]
            elif "#No. of Header Lines: " in line:
                headerLines = line[:-1].split("#No. of Header Lines: ",1)[1]
            if line[0] != '#' and line[0] != '*' and line[0] != '\n':
                dataT[i] = line[4:-1].split('   ')
                i+=1
        df = pd.DataFrame(dataT,columns=['X','Y','Z','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)'])
        df = df.astype(float)
        df['R'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        df['phi'] = np.arccos(df['Z']/df['R'])
        df['theta'] = np.arccos(df['X']/(df['R']*np.sin(df['phi'])))
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
    df['|H|'] = np.sqrt(df['Re(Hx)']**2  + df['Im(Hx)']**2 + df['Re(Hy)']**2  + df['Im(Hy)']**2  + df['Re(Hz)']**2 + df['Im(Hz)']**2)
    df['S(E)'] = (df['|E|']**2)/(3770)

    frequency = frequency*(10**-6)
    frequency = 10
    df['Restriction'] ='Restricted Zone'
    if frequency < 0.3:
        df['Restriction'] ='General Public'
    elif frequency >= 0.3 and frequency < 3:
        df.loc[df['S(E)'] < 100,'Restriction'] = 'Occupation'
    elif frequency >= 3 and frequency < 30:
        df.loc[df['S(E)'] < 900/(frequency**2),'Restriction'] = 'Occupation'
    if frequency >= 0.3 and frequency < 1.34:
        df.loc[df['S(E)'] < 100,'Restriction'] = 'General Public'
    elif frequency >= 1.34 and frequency < 30:
        df.loc[df['S(E)'] < 180/(frequency**2),'Restriction'] = 'General Public'
    if frequency >= 30 and frequency < 300:
        df.loc[df['S(E)'] < 1,'Restriction'] = 'Occupation'
        df.loc[df['S(E)'] < 0.2,'Restriction'] = 'General Public'
    elif frequency >= 300 and frequency < 1500:
        df.loc[df['S(E)'] < frequency/300,'Restriction'] = 'Occupation'
        df.loc[df['S(E)'] < frequency/1500,'Restriction'] = 'General Public'
    elif frequency >= 1500 and frequency < 100000:
        df.loc[df['S(E)'] < 5,'Restriction'] = 'Occupation'
        df.loc[df['S(E)'] < 1,'Restriction'] = 'General Public'

       
    

    #hdf = HDFStore('hdf_file.h5')
    #hdf.put('EMF', df, format='table', data_columns=True) #put data in hdf file
    #hdf.close()
    return fieldSolved(name,Fileformat,source,date,solverV,configuration,frequency,coordSystem, xSamples, ySamples, zSamples,df)




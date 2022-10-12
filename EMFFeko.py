from cProfile import label
from cmath import cos, sin
from fileinput import filename
from platform import freedesktop_os_release
from unittest import result
from kiwisolver import Solver
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import HDFStore
import math
from Field import *
import mayavi.mlab as mlab
import time
from numpy import *


import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)



class Fekofield(Field):
    def __init__(self,power,source,frequency,coordSystem, xSamples, ySamples, zSamples,standard,df):
        super().__init__(df,frequency*(10**-6),type = 'Feko',standard=standard, restriction = False)
        self.source = source
        self.frequency = frequency
        self.coordSystem= coordSystem
        self.xSamples = xSamples
        self.ySamples = ySamples
        self.zSamples = zSamples
        self.power = power

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


def Validationtest1():
    ff = GetFarField('test1.ffe')

    #line1
    line1G = ff.loc[(ff['Phi'] == 0) & (ff['Theta'] == 90),'Directivity(Total)'].to_numpy()
    line1G = [line1G for i in range(80)]
    R = np.linspace(0.5,4,8)
    line1S = OET65Equation3_Dynamic(R,line1G)
    
    #line2
    line2 = ff.loc[(ff['Phi'] < 181) & (ff['Theta'] == 90)]
    x = 1
    Y = np.linspace(-40,40,401)
    R = [np.sqrt(y**2 + x**2) for y in Y]
    angle = [np.abs(np.round(np.arccos(x/r)*180/np.pi)) for r in R]
    line2G = []
    for a in angle:
        line2G.append(line2.loc[line2['Phi'] == a,'Directivity(Total)'].to_numpy()[0])
    line2S = OET65Equation3_Dynamic(R,line2G)


    line3 = ff.loc[ff['Phi'] == 0]
    Z = np.linspace(-40,40,401)
    x = 0.1
    R = [np.sqrt(z**2 + x**2) for z in Z]
    angle = [np.round(np.abs(np.arcsin(x/r)*180/np.pi)) for r in R]
    line3G = []
    for a in angle:
        line3G.append(line3.loc[line3['Theta'] == a,'Directivity(Total)'].to_numpy()[0])
    line3S = OET65Equation3_Dynamic(R,line3G)
    


    FCCoccupational = 30#getZone(900,'FCC')[1]
    l1 = [65.8, 39.4, 25, 13.4, 10.7, 10.4, 10.1, 9.44]
    l2 = [6.22, 9.16, 13.6, 19.6, 26.3, 32.8, 37.6, 39.4, 37.6, 32.8, 26.3, 19.6, 13.6, 9.16, 6.22]
    l3 = [0.173, 3.77, 129, 105, 111, 163, 272, 338, 272, 163, 111, 105, 129, 3.77, 0.173]

    IXUS1_persentage_occupation = [243.8, 138.7, 84.43, 43.62, 36.72, 36.46, 35.51, 33.07]
    IXUS2_persentage_occupation = [19.2, 29.18, 44.46, 66.19, 91.01, 114.7, 133, 138.7, 132.1, 114.1, 90.44, 65.52, 44.46, 29.18, 19.19]
    IXUS3_persentage_occupation = [3.623, 24.51, 595, 273.1, 508.7, 682.9, 869.8, 1279, 886.2, 683.6, 505.8, 267.4, 594.2, 24.24, 3.706]

    def Doline(*lines):
        
        for line in lines:
            plt.figure()
            #plt.plot(line['1D'], line['l'],label = 'Validation line')
            #plt.plot(line['1D'], line['IXUS'],label = 'IXUS')
            #plt.plot(line['1D'], line['Classical'], label = 'Classical')
            #plt.plot(line['1D'],line['OET65Equation3_Dynamic'],label = 'OET65Equation3_Dynamic')
           # plt.plot(line['1D'],line['OET65Equation3_Static'],label = 'OET65Equation3_Static')
            plt.plot(line['1D'], line['Full wave'], label = 'Full wave')
            plt.plot(line['1D'], line['OET651'], label = 'OET651')
            plt.plot(line['1D'], line['OET652'], label = 'OET652')
            plt.plot(line['1D'], line['ICNIRP Peak'], label = 'ICNIRP Peak')
            plt.plot(line['1D'], line['ICNIRP Average'], label = 'ICNIRP Average')
            #plt.plot(line['1D'], line['Peak Cylindrical'], label = 'Peak Cylindrical')
            #plt.plot(line['1D'], line['Average Cylindrical'], label = 'Average Cylindrical')
            #plt.plot(line['1D'], line['Adjusted Spherical'], label = 'Adjusted Spherical')
            #plt.plot(line['1D'], line['Simple Spherical'], label = 'Simple Spherical')
            plt.legend()


    line1 = GetField('IEC-62232-panel-antenna (4)_Line1.efe','IEC-62232-panel-antenna (4)_Line1.hfe',compress=False, power=80).df
    #line1 = line1.loc[line1['X'] < 4.1]
    #line1['IXUS'] = [x/100*FCCoccupational for x in IXUS1_persentage_occupation]
    #line1['l'] = l1

    line2 = GetField('IEC-62232-panel-antenna (4)_Line2.efe','IEC-62232-panel-antenna (4)_Line2.hfe',compress=False, power=80).df
   # line2 = line2.loc[(line2['Y'] < 5) & (line2['Y'] > -5)]
    #line2['IXUS'] = [x/100*FCCoccupational for x in IXUS2_persentage_occupation]
    #line2['l'] = l2

    line3 = GetField('IEC-62232-panel-antenna (4)_Line3.efe','IEC-62232-panel-antenna (4)_Line3.hfe',compress=False, power=80).df
    #line3 = line3.loc[(line3['Z'] < 5) & (line3['Z'] > -5)]
    #line3['IXUS'] = [x/100*FCCoccupational for x in IXUS3_persentage_occupation]
    #line3['l'] = l3

    Doline(line1.rename(columns = {'X': '1D'}), line2.rename(columns = {'Y': '1D'}),line3.rename(columns = {'Z': '1D'}))
    
    plt.show()



def GetField(filenameE,filenameH,S = 'S(E)',compress = False ,standard = 'FCC',power = 80):
    source= ''
    frequency= 900
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
        df['R'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
        df['phi'] = np.arctan(df['Y']/df['X'])
        df['theta'] = np.arctan(df['X']/df['Z'])
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

    df['Ex'] = (df['Re(Ex)'] + df['Im(Ex)']*1j)/np.sqrt(2)
    df['|Ex|'] = np.absolute(df['Ex'])
    df['Ey'] = (df['Re(Ey)'] + df['Im(Ey)']*1j)/np.sqrt(2)
    df['|Ey|'] = np.absolute(df['Ey'])
    df['Ez'] = (df['Re(Ez)'] + df['Im(Ez)']*1j)/np.sqrt(2)
    df['|Ez|'] = np.absolute(df['Ez'])
    df['Hx'] = (df['Re(Hx)'] + df['Im(Hx)']*1j)/np.sqrt(2)
    df['Hy'] = (df['Re(Hy)'] + df['Im(Hy)']*1j)/np.sqrt(2)
    df['Hz'] = (df['Re(Hz)'] + df['Im(Hz)']*1j)/np.sqrt(2)

    df['|E|'] = np.sqrt(np.absolute(df['Ex'])**2+ np.absolute(df['Ey'])**2 + np.absolute(df['Ez'])**2)

    #df['Hx'] = np.conj(df['Hx'])
    #df['Hy'] = np.conj(df['Hy'])
    #df['Hz'] = np.conj(df['Hz'])
    df['Sx'] = df['Ey']*df['Hz'] - df['Ez']*df['Hy']
    df['Sy'] = df['Ez']*df['Hx'] - df['Ex']*df['Hz']
    df['Sz'] = df['Ex']*df['Hy'] - df['Ey']*df['Hx']

    df['Full wave'] = np.sqrt(np.absolute(df['Sx'])**2 + np.absolute(df['Sy'])**2 + np.absolute(df['Sz'])**2)
    df['Classical'] = Classical(df['|E|'].to_numpy())
    df['OET65'] = OET65mesh1(df['R'])
    #df['OET651'] = OET65mesh2(df['R'])
    df['ICNIRP Peak'] = ICNIRPmeshPeak(df['R'], df['phi'], df['theta'])
    df['ICNIRP Average'] = ICNIRPmeshAverage(df['R'], df['phi'], df['theta'])

    df['S'] = df['Full wave']
    if compress:
        df = df.drop(columns = ['R','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','Ex','Ey','Ez','Hx','Hy','Hz','|E|','Sx','Sy','Sz','|Ex|','|Ey|','|Ez|','Full wave'])
    
    return Fekofield(source,power,frequency,coordSystem, xSamples, ySamples, zSamples,standard,df)


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

    FCCoccupational = 30#getZone(900,'FCC')[1]
    l1 = [65.8, 39.4, 25, 13.4, 10.7, 10.4, 10.1, 9.44]
    l2 = [6.22, 9.16, 13.6, 19.6, 26.3, 32.8, 37.6, 39.4, 37.6, 32.8, 26.3, 19.6, 13.6, 9.16, 6.22]
    l3 = [0.173, 3.77, 129, 105, 111, 163, 272, 338, 272, 163, 111, 105, 129, 3.77, 0.173]

    IXUS1_persentage_occupation = [243.8, 138.7, 84.43, 43.62, 36.72, 36.46, 35.51, 33.07]
    IXUS2_persentage_occupation = [19.2, 29.18, 44.46, 66.19, 91.01, 114.7, 133, 138.7, 132.1, 114.1, 90.44, 65.52, 44.46, 29.18, 19.19]
    IXUS3_persentage_occupation = [3.623, 24.51, 595, 273.1, 508.7, 682.9, 869.8, 1279, 886.2, 683.6, 505.8, 267.4, 594.2, 24.24, 3.706]

    def Doline(*lines):
        i = 0
        for line in lines:
            figure, axis = plt.subplots(2)
            axis[0].plot(line['1D'], line['l'],'k-',label = 'IEC Full wave refernce results')
            axis[0].plot(line['1D'], line['IXUS'],'k--',label = 'IXUS')
            axis[0].plot(line['1D'], line['Classical'],'k:', label = 'S=|E|^2/377')
            axis[0].plot(line['1D'], line['Full wave'],'k-.', label = 'S=ExH')
            axis[0].plot(line['1D'], line['OET65'],'k_', label = 'FCC OET 65')
            axis[0].plot(line['1D'], line['IEC Peak'],'ko', label = 'IEC Peak Estimation')
            axis[0].plot(line['1D'], line['EMSS Peak'],'k*', label = 'EMSS Peak Estimation')
            line = line.drop(columns = ['R','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','Ex','Ey','Ez','Hx','Hy','Hz','|E|','Sx','Sy','Sz','|Ex|','|Ey|','|Ez|'])
            #plt.plot(line['1D'], line['IEC Average'], label = 'IEC Average')
            axis[1].plot(line['1D'], line['IXUS']/line['l']*100,'k--',label = 'IXUS')
            axis[1].plot(line['1D'], line['Classical']/line['l']*100,'k:', label = 'S=|E|^2/377')
            axis[1].plot(line['1D'], line['Full wave']/line['l']*100,'k-.', label = 'S=ExH')
            axis[1].plot(line['1D'], line['OET65']/line['l']*100,'k_', label = 'FCC OET 65')
            axis[1].plot(line['1D'], line['IEC Peak']/line['l']*100,'ko', label = 'IEC Peak Estimation')
            axis[1].plot(line['1D'], line['EMSS Peak']/line['l']*100,'k*', label = 'EMSS Peak Estimation')

            if i ==0:
                axis[0].set_title('Validation test for line 1')
                axis[0].set_xlabel('x(m)') 
                axis[0].set_ylabel('S(W/m^2)')
                axis[1].set_title('Percentage of reference results for line 1')
                axis[1].set_xlabel('x(m)')
                axis[1].set_ylabel('Percentage of IEC reference results')
                #print(np.corrcoef(line['l'], line['Full wave'])[0,1])
                #print(np.corrcoef(line['l'], line['Classical'])[0,1])            
                #print(np.corrcoef(line['l'], line['IXUS'])[0,1])            
                #print(np.corrcoef(line['l'], line['OET65'])[0,1])            
                #print(np.corrcoef(line['l'], line['IEC Peak'])[0,1])   
                #print(np.corrcoef(line['l'], line['EMSS Peak'])[0,1])  

                print(np.max(np.abs(line['l'] - line['Full wave'])))
                print(np.max(np.abs(line['l'] - line['Classical'])))
                print(np.max(np.abs(line['l'] - line['IXUS'])))
                print(np.max(np.abs(line['l'] - line['OET65'])))
                print(np.max(np.abs(line['l'] - line['IEC Peak'])))
                print(np.max(np.abs(line['l'] - line['EMSS Peak'])))

            if i ==1:
                axis[0].set_title('Validation test for line 2')
                axis[0].set_xlabel('y(m)')
                axis[0].set_ylabel('S(W/m^2)')
                axis[1].set_title('Percentage of reference results for line 2')
                axis[1].set_xlabel('y(m)')
                axis[1].set_ylabel('Percentage of IEC reference results')
                
            if i ==2:
                axis[0].set_title('Validation test for line 3')
                axis[0].set_xlabel('z(m)')
                axis[0].set_ylabel('S(W/m^2)')
                axis[1].set_title('Percentage of reference results for line 3')
                axis[1].set_xlabel('z(m)')
                axis[1].set_ylabel('Percentage of IEC reference results')
            i+=1
            figure.legend(axis[0].get_legend_handles_labels()[0],axis[0].get_legend_handles_labels()[1])
            figure.tight_layout()
        plt.show()







    line1 = GetField('IEC-62232-panel-antenna (4)_Line1.efe','IEC-62232-panel-antenna (4)_Line1.hfe',compress=False, power=80).df
    line1['IXUS'] = [x/100*FCCoccupational for x in IXUS1_persentage_occupation]
    line1['l'] = l1

    line2 = GetField('IEC-62232-panel-antenna (4)_Line2.efe','IEC-62232-panel-antenna (4)_Line2.hfe',compress=False, power=80).df
    line2['IXUS'] = [x/100*FCCoccupational for x in IXUS2_persentage_occupation]
    line2['l'] = l2

    line3 = GetField('IEC-62232-panel-antenna (4)_Line3.efe','IEC-62232-panel-antenna (4)_Line3.hfe',compress=False, power=80).df
    line3['IXUS'] = [x/100*FCCoccupational for x in IXUS3_persentage_occupation]
    line3['l'] = l3

    Doline(line1.rename(columns = {'X': '1D'}), line2.rename(columns = {'Y': '1D'}),line3.rename(columns = {'Z': '1D'}))
    



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
        dfG = GetFarField('IEC-62232-panel-antenna (4)_FarField1.ffe')
        phi = df['phi']
        theta = df['theta']
        df['phi'] = np.abs(np.round(df['phi']*180/np.pi))
        df['theta'] = np.abs(np.round(df['theta']*180/np.pi))
        df = df.merge(dfG,how='left',on=['phi','theta'])
        df['phi'] = phi
        df['theta'] = theta
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
    df['OET65'] = OET65mesh(df['R'],df['Gain'])
    df['IEC Peak'] = IECmeshPeakSector(df['R'], df['phi'], df['theta'])
    df['IEC Average'] = IECmeshAverageSector(df['R'], df['phi'], df['theta'])
    df['EMSS Peak'] = EMSSmeshPeakSector(df['R'], df['phi'], df['theta'])
    df['EMSS Average'] = EMSSmeshAverageSector(df['R'], df['phi'], df['theta'])

    df['S'] = df['Full wave']
    if compress:
        df = df.drop(columns = ['R','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','Ex','Ey','Ez','Hx','Hy','Hz','|E|','Sx','Sy','Sz','|Ex|','|Ey|','|Ez|','Full wave'])
    
    return Fekofield(source,power,frequency,coordSystem, xSamples, ySamples, zSamples,standard,df)


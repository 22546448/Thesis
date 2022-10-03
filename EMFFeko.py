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
from Field import Field
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


def GetFarField(filename,compress = True,standard = 'FCC',power = 80):
    source= ''
    frequency= 900
    coordSystem= ''
    thetaSamples = 0
    phiSamples = 0
    global i
    i = 0
    filenameff = 'venv/Include/CADFeko/{}'.format(filename)

    with open(filenameff, 'r') as file:
        for line in file:
            if '##Source: ' in line:
                source = line[:-1].split("##Source: ",1)[1]
            elif "#Frequency: " in line:
                frequency = int(float(line[:-1].split("#Frequency:   ",1)[1]))  
            elif "#Coordinate System: " in line:
                coordSystem = line[:-1].split("#Coordinate System: ",1)[1]
            elif "#No. of Theta Samples: " in line:
                thetaSamples = int(line[:-1].split("#No. of Theta Samples: ",1)[1])
            elif "#No. of Phi Samples: " in line:
                phiSamples = int(line[:-1].split("#No. of Phi Samples: ",1)[1])
                global dataT
                dataT = np.zeros((thetaSamples*phiSamples,9)) 
            if line[0] != '#' and line[0] != '*' and line[0] != '\n':
                dataT[i] = line[4:-1].split('   ')
                i+=1
        df = pd.DataFrame(dataT,columns=['Theta','Phi','Re(Etheta)','Im(Etheta)','Re(Ephi)','Im(Ephi)','Directivity(Theta)','Directivity(Phi)','Directivity(Total)'])
        df = df.astype(float)
    file.close()

    df['Etheta'] = (df['Re(Etheta)'] + df['Im(Etheta)']*1j)/np.sqrt(2)
    df['Ephi'] = (df['Re(Ephi)'] + df['Im(Ephi)']*1j)/np.sqrt(2)
    df['|E|'] = np.sqrt(np.absolute(df['Etheta'])**2 + np.absolute(df['Ephi'])**2)
    df['S(E)'] = df['|E|']**2/(337*2)
    return df

def plotFarField(df):
    phi, theta  = mgrid[0:361:1,0:181:1]
    Gnum = 10**(df['Directivity(Total)'].to_numpy()/10)
    lamda = 1/3
    f = Gnum
    f = np.reshape(f,(361,181))
    x = f*np.sin(theta*np.pi/180)*np.cos(phi*np.pi/180)
    y = f*np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180)
    z = f*np.cos(theta*np.pi/180)
    mlab.mesh(x, y, z)
    mlab.show()

def GetField(filenameE,filenameH,S = 'S(E)',compress = False,standard = 'FCC',power = 80):
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
        df['theta'] = np.arcsin(df['X']/df['R'])
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
    df['Sx'] = df['Ey']*df['Hz'] - df['Ez']*df['Hy']
    df['Sy'] = df['Ez']*df['Hx'] - df['Ex']*df['Hz']
    df['Sz'] = df['Ex']*df['Hy'] - df['Ey']*df['Hx']

    df['Full wave'] = np.sqrt(np.absolute(df['Sx'])**2 + np.absolute(df['Sy'])**2 + np.absolute(df['Sz'])**2)
    df['Classical'] = Classical(df['|E|'].to_numpy())
    df['OET652'] = OET65mesh1(df['R'])
    df['OET651'] = OET65mesh2(df['R'])
    df['ICNIRP Peak'] = ICNIRPmeshPeak(df['R'], df['phi'], df['theta'])
    df['ICNIRP Average'] = ICNIRPmeshAverage(df['R'], df['phi'], df['theta'])

    df['S'] = df['Full wave']
    if compress:
        df = df.drop(columns = ['Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','Re(Sx)','Im(Sx)','Re(Sy)','Im(Sy)','Re(Sz)','Im(Sz)','Ex','Ey','Ez','Hx','Hy','Hz','|E|'])
    
    return Fekofield(source,power,frequency,coordSystem, xSamples, ySamples, zSamples,standard,df)

def plotSimulationMethod(df):
    Sarray = np.linspace(1, 100, 50)
    for S in Sarray:
        test = df.loc[df['S'] >= S]
        X = test.groupby(['Y','Z'])['X'].max()
        plt.plot(test['Y'].unique(),X)
    plt.show()
        #mlab.mesh(x, y, test['Z'])
        #mlab.show()

# all Sector coverage arrays
def AverageCylindricalSector(phi,R,P = 80,AHPBW = 85,L = 2.25,G= 17,y = 0,ry = None):
    G = 10**(G/10)
    AHPBW = np.pi*AHPBW/180
    ro = AHPBW*G*L*np.cos(y)**2/12
    ry = R/np.cos(y)
    return P*2**(-1*(2*phi/AHPBW)**2)/(AHPBW*ry*L*(np.cos(y)**2)*np.sqrt(1 + (ry/ro)**2))

def PeakCylindricalSector(phi,R,P = 80,AHPBW = 85,L = 2.25,G= 17,y = 0):
    G = 10**(G/10)
    AHPBW *= np.pi/180
    ro = AHPBW*G*L*np.cos(y)**2/12
    ry = R/np.cos(y)
    return 2*P*2**(-4*(phi/AHPBW)**2)/(AHPBW*ry*L*np.cos(y)**2*np.sqrt(1 + (2*ry/ro)**2))

def AdjustedSphericalSector(theta,phi,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = -3.6, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (theta - y - np.pi/2)/VHPBW
    b2 = 1.9*phi/AHPBW
    Gphitheta = 1.26*Globe + G*2**(-b1**2-b2**2)
    return 1.2*power*Gphitheta/(4*np.pi*R**2)

def SimpleSphericalSector(theta,phi,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = 0, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (2*(theta - y - np.pi/2)/VHPBW)**2
    b2 = (2*phi/AHPBW)**2
    Gphitheta = Globe + G*2**(-b1-b2)
    return power*Gphitheta/(4*np.pi*R**2)


def AverageCylindricalOmni(R, power = 80, VHPBW = 8.5, AHPBW = 85,G = 17, L =2.25, y = 0 ):
    G = 10**(G/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    ry = R/np.cos(y)
    ro = G*L*np.cos(y)**2/2
    return  power/(2*np.pi*ry*L*np.cos(y)**2*np.sqrt(1 + (ry/ro)**2))

def PeakCylindricalOmni(R, power = 80, L = 2.25, G = 17, y = 0):
    G = 10**(G/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    ry = R/np.cos(y)
    ro = G*L*np.cos(y)**2/2
    return  power/(np.pi*ry*L*np.cos(y)**2*np.sqrt(1 + (2*ry/ro)**2))

def SimpleSphericalOmni(theta,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = -3.6, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (2*(theta - y - np.pi/2)/VHPBW)**2
    Gphitheta = Globe + G*2**(-b1)
    return power*Gphitheta/(4*np.pi*R**2)

def AdjustedSphericalOmni(theta,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = -3.6, y = 0):
    G = 10**(G/10)
    Globe = 10**(Globe/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (theta - y - np.pi/2)/VHPBW
    Gphitheta = 1.26*Globe + G*2**(-b1**2)
    return power*Gphitheta/(4*np.pi*R**2)

def Classical(E):
    S = []
    for i in range(len(E)):
        S.append(E[i]**2/377)
    return np.array(S)

def OET65Equation3_Dynamic(R,G,power = 80):
    return [power*10**(g/10)/(4*np.pi*r**2) for g,r in zip(G,R)]

def CylindricalValidationTest():
    SectorCoverageSbar = [5.58, 3.54, 2.49, 1.86, 1.43, 1.02, 0.639]
    SectorCoverageS = [9.96, 5.74, 3.70, 2.56, 1.86, 1.25, 0.727]
    f = 925
    lamda = (3*10**8)/(f*10**6)
    power = 80
    L = 2.158
    AHPBW = 84
    y = 5
    Gs = 17 #dBi
    Go = 11 #dBi
    Globe = -9      #dBi
    Globe = -3.6    #dBi
    phi = np.pi/12
    Ry = [4, 6, 8, 10, 12, 15, 20]
    R = [ry*np.cos(y*np.pi/180) for ry in Ry]

    SectorAverage = AverageCylindricalSector(phi,R,power,AHPBW, L,Gs, y*np.pi/180,Ry)
    SectorPeak = PeakCylindricalSector(phi,R,power,AHPBW, L, Gs, y=y*np.pi/180)

    plt.figure()
    plt.plot(Ry,SectorAverage,label = 'SpacialPeakCylindrical')
    plt.plot(Ry,SectorCoverageS,label = 'Peak Cylindrical Validation line')
    plt.plot(Ry,SectorCoverageSbar,label = 'Average Cylindrical Validation line')
    plt.plot(Ry,SectorPeak,label = 'SpacialAverageCylindrical')
    plt.legend()
    plt.show()


def SphericalValidationTest():
    adjustedSectorS = [52, 353, 313, 210, 141, 98.6, 72, 54.5 ]
    adjustedSectorS = [a/1000 for a in adjustedSectorS]
    SectorCoverageS = [9.96, 5.74, 3.70, 2.56, 1.86, 1.25, 0.727]
    f = 925
    lamda = (3*10**8)/(f*10**6)
    power = 80
    L = 2.158
    Gs = 17 #dBi
    Go = 11 #dBi
    Globes = -3.6     #dBi
    Globe0 = -9   #dBi
    phi = np.pi/12
    Ry = np.linspace(10,80,8)
    R = [np.sqrt(ry**2 + 5**2) for ry in Ry]
    R = np.array(R)
    theta = [np.pi/2 + np.arctan(5/ry) for ry in Ry]
    theta = np.array(theta)
    Ry = np.array(Ry)

    adjustedSpherical = AdjustedSphericalSector(theta = theta, phi=phi, R = Ry,power = power, VHPBW=8, AHPBW=84, L = L, G=Gs, Globe=Globes, y =5*np.pi/180 )
    plt.figure()
    plt.plot(Ry,adjustedSpherical,label = 'SpacialPeakCylindrical')
    plt.plot(Ry,adjustedSectorS,label = 'Adjusted Spherical Validation line')
    #plt.plot(Ry,SectorCoverageSbar,label = 'Average Cylindrical Validation line')
    #plt.plot(Ry,SectorPeak,label = 'SpacialAverageCylindrical')
    plt.legend()
    plt.show()



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





##Near field
def getEfficiency(G = 17, f = 900,A = 2.25*0.3):
    lamda = 3*10**8/(f*10**6)
    #return (10**(G/10)*lamda**2)/(4*np.pi*A)

    top = 10**(G/10)*lamda**2/(4*np.pi)
    bottom =  np.pi*2.25**2/4#np.pi*2.25**2/4
    return top/bottom

def Ssurface(P = 80, A = 2.25*0.3):
    return 4*P/A

def Snf(G = 17, f = 900,w = 0.3,D = 2.25,power = 80):
    A = w*D
    n = getEfficiency(G, f, A)
    return 16*n*power/(np.pi*D**2)

def St(R):
    return Snf()*Rnf()/R


def Rnf(D = 2.25, f = 900):
    lamda = 3*10**8/(f*10**6)
    return D**2/(4*lamda)

def Rff(D = 2.25,f = 900):
    lamda = 3*10**8/(f*10**6)
    return 0.6*D**2/lamda

def Sff(R, power = 80, G = 17):
    return power*10**(G/10)/(4*np.pi*R**2) 

def OET65near(R, power = 80, D = 2.25, AHPBW = 85):
    return power*180/(R*D*AHPBW*np.pi)


#VSA pel average
#def OET65near(R, power = 80, D = 2.25, AHPBW = 85):
#    AHPBW = AHPBW*np.pi/180
#    return power/(R*D*AHPBW)


def OET65far(R,power = 80, G = 17):
    G = 10**(G/10)
    return power*G/(4*np.pi*R**2)

def OET65Modified(D = 2.25):
    Rtrans = D*1.5
    Rfar = Rff()
    return OET65near(Rtrans)*1/(Rfar/Rtrans)**2

def ICNIRPmeshPeak(R, phi, theta, f = 900, D = 2.25):
    lamda = 3*10**8/(f*10**6)
    Rreactive = 0.62*np.sqrt(D**3/lamda)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            S.append(PeakCylindricalSector(phi[i],R[i]))
        elif np.abs(R[i]) > Rnearfield:
            S.append(AdjustedSphericalSector(theta[i], phi[i], R[i]))
    return np.array(S)

def ICNIRPmeshAverage(R, phi, theta, f = 900, D = 2.25):
    lamda = 3*10**8/(f*10**6)
    Rreactive = 0.62*np.sqrt(D**3/lamda)
    Rnearfield = 2*D**2/lamda
    S = []
    for i in range(len(R)):
        if np.abs(R[i]) < Rnearfield:
            S.append(AverageCylindricalSector(phi[i],R[i]))
        elif np.abs(R[i]) > Rnearfield:
            S.append(SimpleSphericalSector(theta[i], phi[i], R[i]))
    return np.array(S)

def OET65mesh1(R, D = 2.25, f = 900):
    lamda = 3*10**8/(f*10**6)
    Rreactive = 0.62*np.sqrt(D**3/lamda)
    Rnearfield = 2*D**2/lamda
    S =[]
    for i in range(len(R)):
        if R[i] < Rnearfield:
            S.append(OET65near(R[i]))
        elif R[i] > Rnearfield:
            S.append(OET65far(R[i]))
    return np.array(S)
    

def OET65mesh2(R, f = 900,D = 2.25, a = True):
    lamda = 3*10**8/(f*10**6)
    Rreactive = 0.62*np.sqrt(D**3/lamda)
    Rnearfield = 2*D**2/lamda
    S = []
    if (a == True):
        for i in range(len(R)):
            if R[i] < 0.5:
                S.append(Ssurface())
            elif np.abs(R[i]) < Rnf():
                S.append(Snf())
            elif np.abs(R[i]) > Rff():
                S.append(Sff(R[i]))
            else:
                S.append(St(R[i]))
        return np.array(S)
    else:
        for i in range(len(R)):
            if np.abs(R[i]) < Rnearfield:
                S.append(Snf())
            elif np.abs(R[i]) > Rnearfield:
                S.append(Sff(R[i]))
            else:
                S.append(St(R[i]))
        return np.array(S)


from cmath import sin
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

def GetField(filenameE,filenameH,S = 'S(E)',compress = True,standard = 'FCC',power = 80):
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
    df['S(E)'] = df['|E|']**2/(377)
    #df['Snear'] = SpacialPeakCylindricalEstimation(df['phi'], df['R'])
    df['Sfar'] = AdjustedSphericalSector(theta=df['theta'] , phi=df['phi'] , R=df['R'])
    df['Snear'] = SpacialPeakCylindricalEstimation(df['phi'],df['R'])
    df['SfarSimple'] = SimpleSphericalSector(theta=df['theta'] , phi=df['phi'] , R=df['R'])


    df['Sx'] = df['Ey']*df['Hz'] - df['Ez']*df['Hy']
    df['Sy'] = df['Ez']*df['Hx'] - df['Ex']*df['Hz']
    df['Sz'] = df['Ex']*df['Hy'] - df['Ey']*df['Hx']
    df['S(ExH)'] = np.sqrt(np.absolute(df['Sx'])**2 + np.absolute(df['Sy'])**2 + np.absolute(df['Sz'])**2)


    #df['Sx*'] = np.real(df['Ey']*np.conj(df['Hz']) - df['Ez']*np.conj(df['Hy']))
    #df['Sy*'] = np.real(df['Ez']*np.conj(df['Hx']) - df['Ex']*np.conj(df['Hz']))
    #df['Sz*'] = np.real(df['Ex']*np.conj(df['Hy']) - df['Ey']*np.conj(df['Hx']))
    #df['S(ExH*)'] = np.sqrt(df['Sx*']**2 + df['Sy*']**2 + df['Sz*']**2)/2

    df['S'] = df['S(ExH)']
    if compress:
        df = df.drop(columns = ['Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','Re(Sx)','Im(Sx)','Re(Sy)','Im(Sy)','Re(Sz)','Im(Sz)','Ex','Ey','Ez','Hx','Hy','Hz','|E|'])
    
    return Fekofield(source,power,frequency,coordSystem, xSamples, ySamples, zSamples,standard,df)


# all Sector coverage arrays
def SpacialAverageCylindricalEstimation(phi,R,P = 80,AHPBW = 85,L = 2.25,G= 17,y = 0,ry = None):
    G = 10**(G/10)
    AHPBW = np.pi*AHPBW/180
    ro = AHPBW*G*L*np.cos(y)**2/12

    
    ry = R/np.cos(y)
    return P*2**(-1*(2*phi/AHPBW)**2)/(AHPBW*ry*L*(np.cos(y)**2)*np.sqrt(1 + (ry/ro)**2))

def SpacialPeakCylindricalEstimation(phi,R,P = 80,AHPBW = 85,L = 2.25,G= 17,y = 0):
    G = 10**(G/10)
    AHPBW *= np.pi/180
    ro = AHPBW*G*L*np.cos(y)**2/12
    ry = 0
    ry = R/np.cos(y)
    return 2*P*2**(-4*(phi/AHPBW)**2)/(AHPBW*ry*L*np.cos(y)**2*np.sqrt(1 + (2*ry/ro)**2))

def AdjustedSphericalSector(theta,phi,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = 3.6, y = 0):
    G = 10**(G/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (theta - y - np.pi/2)/VHPBW
    b2 = 1.9*phi/AHPBW
    Gphitheta = 1.26*Globe + G*2**(-b1**2-b2**2)
    return 1.2*power*Gphitheta/(4*np.pi*R**2)

def SimpleSphericalSector(theta,phi,R,power = 80, VHPBW = 8.5, AHPBW = 85, L = 2.25, G = 17,Globe = 0, y = 0):
    G = 10**(G/10)
    VHPBW *= np.pi/180
    AHPBW *= np.pi/180
    b1 = (2*(theta - y - np.pi/2)/VHPBW)**2
    b2 = (2*phi/AHPBW)**2
    Gphitheta = Globe + G*2**(-b1-b2)
    return power*Gphitheta/(4*np.pi*R**2)



def ClassicalSpherical(df,power = 80, L = 2.25):
    return df['|E|']**2/(377)

#def plotValidation1(*method):
#    for meth in method:

def nearSurface(ff,P = 80, h = 2.25, AHPBW = 8.5,dipoles = 9):
    x = 0.1
    linegain = ff.loc[ff['Phi'] == 0]
    Z = np.linspace(-1.4,1.4,15)
    R = [np.sqrt(z**2 + x**2) for z in Z]
    angle = [np.round(np.abs(np.arcsin(x/r)*180/np.pi)) for r in R]
    D = []
    for a in angle:
        D.append(linegain.loc[linegain['Theta'] == a,'Directivity(Total)'].to_numpy()[0])
    #S0 = [180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z+1.25)**2)*h) for z in Z]
    S1 = [180*P*(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z+1)**2)*h) for z in Z]
    S2 = [180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z+0.75)**2)*h) for z in Z]
    S3 = [180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z+0.5)**2)*h) for z in Z]
    S4 = [180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z+0.25)**2)*h) for z in Z]
    S5 = [180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z)**2)*h) for z in Z]
    S6 = [180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z-0.25)**2)*h) for z in Z]
    S7 = [180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z-0.5)**2)*h) for z in Z]
    S8 = [180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z-0.75)**2)*h) for z in Z]
    S9 = [180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z-1)**2)*h) for z in Z]
    #S10 =[180*P/(dipoles*AHPBW*2*np.pi*np.sqrt(x**2 + (z-1.25)**2)*h) for z in Z]
    temp = []
    for i in range(len(Z)):
        temp.append( (D[i]+2)*(S1[i] +  S2[i] + S3[i] + S4[i] + S5[i] + S6[i] + S7[i] + S8[i] + S9[i]))
    return D
    

#def CreateField(antenna,source='IEC-62232-panel-antenna',date=0,configuration='Created',frequency = f)
#   return fieldSolved(name,Fileformat,source,date,solverV,configuration,frequency,coordSystem, xSamples, ySamples, zSamples,df)

def test1():
    ff = GetFarField('test1.ffe')
    f = ff.loc[(ff['Directivity(Total)'] > 11.96) & (ff['Directivity(Total)'] < 12.1) & (ff['Phi'] < 181) & (ff['Theta'] == 90)]
    print(f.sort_values(by = ['Directivity(Total)']))

    line3near = nearSurface(ff = ff)
        #line1
    def ffline1():
        line1 = 10**(ff.loc[(ff['Phi'] == 0) & (ff['Theta'] == 90),'Directivity(Total)'].to_numpy()[0]/10)
        #line1 = (ff.loc[(ff['Phi'] == 0) & (ff['Theta'] == 90),'Directivity(Total)'].to_numpy()[0])
        R = np.linspace(0.5,4,8)
        return [80*line1/(4*np.pi*r**2) for r in R]
    
    #line2
    def ffline2(x = 1):
        line2 = ff.loc[(ff['Phi'] < 181) & (ff['Theta'] == 90)]
        Y = np.linspace(-1.4,1.4,15)
        R = [np.sqrt(y**2 + x**2) for y in Y]
        angle = [np.abs(np.round(np.arccos(x/r)*180/np.pi)) for r in R]
        D = []
        for a in angle:
            D.append(10**(line2.loc[line2['Phi'] == a,'Directivity(Total)'].to_numpy()[0]/10))

#            D.append(line2.loc[line2['Phi'] == a,'Directivity(Total)'].to_numpy()[0])

        #num = [10**(d/10) for d in D]
        #lamda = 1/3
        #D = [20*np.log10(9.73/(lamda*np.sqrt(n))) for n in num]

        return [80*d/(4*np.pi*r**2) for d,r in zip(D,R) ]



    def ffline3(x = 0.1):
        linegain = ff.loc[ff['Phi'] == 0]
        Z = np.linspace(-1.4,1.4,15)
        R = [np.sqrt(z**2 + x**2) for z in Z]
        angle = [np.round(np.abs(np.arcsin(x/r)*180/np.pi)) for r in R]
        D = []
        for a in angle:
            D.append(10**(linegain.loc[linegain['Theta'] == a,'Directivity(Total)'].to_numpy()[0]/10))
        #num = [10**(d/10) for d in D]
        #lamda = 1/3
        #D = [20*np.log10(9.73/(lamda*np.sqrt(n))) for n in num]
        return [80*d/(4*np.pi*r**2) for d,r in zip(D,R) ]
        


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
            plt.plot(line['1D'], line['l'],label = 'Validation line')
            plt.plot(line['1D'], line['IXUS'],label = 'IXUS')
            plt.plot(line['1D'], line['S(E)'], label = '|E|^2/377')
            plt.plot(line['1D'],line['Sff'],label = 'Sff')
            #plt.plot(line['1D'], line['S(ExH)'], label = 'S=ExH')
            plt.plot(line['1D'], line['Snear'], label = 'Snear')
            #plt.plot(line['1D'], line['Sfar'], label = 'Sfar')
            #plt.plot(np.linspace(-1.4,1.4,15),line3near,label = 'nearSurface')

            #axs[0, 0].plot(line1.df['X'], line1.df['S(R)far'], label = 'S(R)far')
            plt.legend()


    line1 = GetField('IEC-62232-panel-antenna (4)_Line1.efe','IEC-62232-panel-antenna (4)_Line1.hfe',compress=False, power=80).df
    line1['IXUS'] = [x/100*FCCoccupational for x in IXUS1_persentage_occupation]
    line1['Sff'] = ffline1()
    line1['l'] = l1
    line2 = GetField('IEC-62232-panel-antenna (4)_Line2.efe','IEC-62232-panel-antenna (4)_Line2.hfe',compress=False, power=80).df
    line2['IXUS'] = [x/100*FCCoccupational for x in IXUS2_persentage_occupation]
    line2['Sff'] = ffline2()
    line2['l'] = l2
    line3 = GetField('IEC-62232-panel-antenna (4)_Line3.efe','IEC-62232-panel-antenna (4)_Line3.hfe',compress=False, power=80).df
    line3['IXUS'] = [x/100*FCCoccupational for x in IXUS3_persentage_occupation]
    line3['Sff'] = ffline3()
    line3['l'] = l3

    Doline(line1.rename(columns = {'X': '1D'}), line2.rename(columns = {'Y': '1D'}),line3.rename(columns = {'Z': '1D'}))
    
    plt.show()


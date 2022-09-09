from fileinput import filename
from unittest import result
from kiwisolver import Solver
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math




class fieldSolved:
    def __init__(self,name,Fileformat,source,date,solverV,configuration,frequency,coordSystem, xSamples, ySamples, zSamples,dataframe):
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
        self.dataframe = dataframe


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
    resultType= ''
    headerLines= ''
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
                frequency = line[:-1].split("#Frequency: ",1)[1]  
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
        df['X'] = df['X'].astype(float)
        df['Y'] = df['Y'].astype(float)
        df['Z'] = df['Z'].astype(float)
        df['Re(Ex)'] = df['Re(Ex)'].astype(float)
        df['Im(Ex)'] = df['Im(Ex)'].astype(float)
        df['Re(Ey)'] = df['Re(Ey)'].astype(float)
        df['Im(Ey)'] = df['Im(Ey)'].astype(float)
        df['Re(Ez)'] = df['Re(Ez)'].astype(float)
        df['Im(Ez)'] = df['Im(Ez)'].astype(float)
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
    df['Re(Hx)'] = df['Re(Hx)'].astype(float)
    df['Im(Hx)'] = dataH[:,1]
    df['Im(Hx)'] = df['Im(Hx)'].astype(float)
    df['Re(Hy)'] = dataH[:,2]
    df['Re(Hy)'] = df['Re(Hy)'].astype(float)
    df['Im(Hy)'] = dataH[:,3]
    df['Im(Hy)'] = df['Im(Hy)'].astype(float)
    df['Re(Hz)'] = dataH[:,4]
    df['Re(Hz)'] = df['Re(Hz)'].astype(float)
    df['Im(Hz)'] = dataH[:,5]
    df['Im(Hz)'] = df['Im(Hz)'].astype(float)
    df['|E|'] = GetEPhasor(df)
    df['|H|'] = GetHPhasor(df)
    print(df)
    #print(GetEPhasor(df))
    return fieldSolved(name,Fileformat,source,date,solverV,configuration,frequency,coordSystem, xSamples, ySamples, zSamples,df)



def PowerAtPoint(data):
    S = np.zeros(len(data))
    for j in range(len(data)):
        for i in range(3,9):
            S[j] += data[j][i]**2
    S = S/(2*377) 
    return(S)

def getSdata(data):
    data['S(E)'] = data['Re(Ex)']**2  + data['Im(Ex)']**2 + data['Re(Ey)']**2  + data['Im(Ey)']*2  + data['Re(Ez)']**2 + data['Im(Ez)']**2 
    data ['S(E)']= data['S(E)']/(2*337)
    data['S(E)'] = data['S(E)'].astype(float)

    data['Restriction'] ='General Public'
    data.loc[data['S(E)'] > 20,'Restriction'] = 'Occupation'
    data.loc[data['S(E)'] > 100,'Restriction'] = 'Restricted Zone'

    return data

def plotPowerLine(line,axis):
    plt.plot(line[axis],line['S(E)'])
    plt.show()  


def GetEPhasor(data):
    dfE = data[['X','Y','Z']].copy()
    dfE['|E|'] = (data['Re(Ex)']  + data['Im(Ex)'])**2 + (data['Re(Ey)']  + data['Im(Ey)'])*2  + (data['Re(Ez)'] + data['Im(Ez)'])**2 
    dfE['|E|'] =  dfE['|E|'].astype(float)
    return dfE['|E|']

def GetHPhasor(data):
    dfH = data[['X','Y','Z']].copy()
    dfH['|H|'] = (data['Re(Hx)']  + data['Im(Hx)'])**2 + (data['Re(Hy)']  + data['Im(Hy)'])*2  + (data['Re(Hz)'] + data['Im(Hz)'])**2 
    dfH['|H|'] =  dfH['|H|'].astype(float)
    return dfH['|H|']


def plot2DColor(df,X,Y):
    #fig, ax = plt.subplots()
    colors = {'General Public':'blue','Occupation':'yellow','Restricted Zone':'red'}
    plt.scatter(x=df[X], y=df[Y],c= df['Restriction'].map(colors))
    plt.show()

def plot2D(df,X,Y,field):
    plt.scatter(x =df[X],y=df[Y],c =df[field],cmap = 'Reds')
    plt.show()

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import HDFStore
import math  
import mayavi.mlab as mlab
from Standard import getZone
import time


import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)
    
class Field:
    def __init__(self,df,f,type = 'Feko',standard = 'FCC'):
        self.standard = standard
        self.df = df
        self.f = f

        maxFreq = getZone(self.f,standard)[1]
        minFreq = getZone(self.f,standard)[0]  
        self.df['Restriction'] = 1
        self.df.loc[minFreq > self.df['S'],'Restriction'] = 0
        self.df.loc[maxFreq < self.df['S'],'Restriction'] = 2

        
    
    def PowerAtPoint(data):
        S = np.zeros(len(data))
        for j in range(len(data)):
            for i in range(3,9):
                S[j] += data[j][i]**2
        S = S/(2*377) 
        return(S)

    def plotPowerLine(line,axis):
        plt.plot(line[axis],line['S'])
        plt.show() 

    def getS(self):
        dfS = self.df[['X','Y','Z','S']].copy()
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

    def plot2DZones(self,Ncolor = 'blue',GPcolor = 'yellow',Ocolor = 'red',xfig = 6,yfig = 4,axis1 = 'X',axis2 = 'Y',show = True):
        colors = {0:Ncolor,1:GPcolor,2:Ocolor}
        #plt.scatter(x=self.df[X], y=self.df[Y],c= self.df['Restriction'].map(colors))

        groups = self.df.groupby('Restriction')
       
        fig, ax = plt.subplots(1, figsize=(xfig,yfig))
        for label, group in groups:
            ax.scatter(group[axis1], group[axis2], 
                    c=group['Restriction'].map(colors), label=label)

        ax.set(xlabel= axis1, ylabel=axis2)
        ax.set_title("Restricions with %d signal" % self.f)
        ax.legend(title='Restriction levels')
        if show:
            plt.show()



    def plot2D(self,c='S',color = 'Reds',method = 'matplotlib'):
        if method == 'matplotlib':
            fig, ax = plt.subplots(1)
            ax1 =ax.scatter(x =self.df[self.axis[0]],y= self.df[self.axis[1]],c =self.df[c],cmap = color)
            plt.colorbar(ax1)
            ax.set_xlabel(self.axis[0])
            ax.set_ylabel(self.axis[1])
            ax.set_title("{} over {}{} plane".format(c,self.axis[0],self.axis[1]))


    def compareToSelf(self,standard1,S2 = 'S',S1 = 'S',standard2 = None,Ncolor = 'blue',GPcolor = 'yellow',Ocolor = 'red',xfig = 6,yfig = 4,axis1 = 'X',axis2 = 'Y',show = True,c='Restriction'):
        if c == 'Restriction':
            colors = {0:Ncolor,1:GPcolor,2:Ocolor}

            if standard2 == None:
                standard1 = self.standard

            maxFreq1 = getZone(self.f,standard1)[1]
            minFreq1 = getZone(self.f,standard1)[0]  
            self.df['Restriction1'] = 1
            self.df.loc[minFreq1 > self.df[S1],'Restriction1'] = 0
            self.df.loc[maxFreq1 < self.df[S1],'Restriction1'] = 2


            maxFreq2 = getZone(self.f,standard2)[1]
            minFreq2 = getZone(self.f,standard2)[0]  
            self.df['Restriction2'] = 1
            self.df.loc[minFreq2 > self.df[S2],'Restriction2'] = 0
            self.df.loc[maxFreq2 < self.df[S2],'Restriction2'] = 2


            groups1 = self.df.groupby('Restriction1')
            groups2 = self.df.groupby('Restriction2')

        
            fig, (ax1,ax2) = plt.subplots(2,1)
            fig.set_size_inches(14.5,10.5)
            for label, group in groups1:
                ax1.scatter(group[axis1], group[axis2], 
                        c=group['Restriction1'].map(colors), label=label)

            for label, group in groups2:
                ax2.scatter(group[axis1], group[axis2], 
                        c=group['Restriction2'].map(colors), label=label)

            ax1.set(xlabel= axis1, ylabel=axis2)
            ax1.set_title("{}".format(standard1))
            ax1.legend(title='Restriction levels')
            ax2.set(xlabel= axis1, ylabel=axis2)
            ax2.set_title("{}".format(standard2))
            ax2.legend(title='Restriction levels')

            self.df = self.df.drop(columns = ['Restriction1','Restriction2'])
        if show:
            plt.show()


    def compareToSurface2D(self,field,Ncolor = 'blue',GPcolor = 'yellow',Ocolor = 'red',xfig = 6,yfig = 4,axis1 = 'X',axis2 = 'Y',show = True,c='Restriction'):
        if c == 'Restriction':
            colors = {0:Ncolor,1:GPcolor,2:Ocolor}
            #plt.scatter(x=self.df[X], y=self.df[Y],c= self.df['Restriction'].map(colors))
            groups1 = self.df.groupby('Restriction')
            groups2 = field.df.groupby('Restriction')

        
            fig, (ax1,ax2) = plt.subplots(2,1)
            for label, group in groups1:
                ax1.scatter(group[axis1], group[axis2], 
                        c=group['Restriction'].map(colors), label=label)

            ax1.set(xlabel= axis1, ylabel=axis2)
            ax1.set_title("Restricions with %d signal" % self.f)
            ax1.legend(title='Restriction levels')

            for label, group in groups2:
                ax2.scatter(group[axis1], group[axis2], 
                        c=group['Restriction'].map(colors), label=label)

            ax2.set(xlabel= axis1, ylabel=axis2)
            ax2.set_title("Restricions with %d signal" % self.f)
            ax2.legend(title='Restriction levels')
        else: 
            fig, (ax1,ax2) = plt.subplots(2,1)
            ax1.scatter(self.df[axis1], self.df[axis2], c=self.df[c])
            ax1.set(xlabel= axis1, ylabel=axis2)
            ax1.set_title("Restrictions with %d signal" % self.f)

            ax2.scatter(field.df[axis1], field.df[axis2], c=self.df[c])
            ax2.set(xlabel= axis1, ylabel=axis2)
            ax2.set_title("Restrictions with %d signal" % self.f)

        if show:
            plt.show()


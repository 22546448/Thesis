
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import HDFStore
import math  

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)
    
class Field:
    def __init__(self,df,f):
        self.df = df
        self.f = 10
        
        self.df['Restriction'] ='Restricted Zone'
        if self.f < 0.3:
            self.df['Restriction'] ='General Public'
        elif self.f >= 0.3 and self.f < 3:
            self.df.loc[self.df['S(E)'] < 100,'Restriction'] = 'Occupation'
        elif self.f >= 3 and self.f < 30:
            self.df.loc[self.df['S(E)'] < 900/(self.f**2),'Restriction'] = 'Occupation'
        if self.f >= 0.3 and self.f < 1.34:
            self.df.loc[self.df['S(E)'] < 100,'Restriction'] = 'General Public'
        elif self.f >= 1.34 and self.f < 30:
            self.df.loc[self.df['S(E)'] < 180/(self.f**2),'Restriction'] = 'General Public'
        if self.f >= 30 and self.f < 300:
            self.df.loc[self.df['S(E)'] < 1,'Restriction'] = 'Occupation'
            self.df.loc[self.df['S(E)'] < 0.2,'Restriction'] = 'General Public'
        elif self.f >= 300 and self.f < 1500:
            self.df.loc[self.df['S(E)'] < self.f/300,'Restriction'] = 'Occupation'
            self.df.loc[self.df['S(E)'] < self.f/1500,'Restriction'] = 'General Public'
        elif self.f >= 1500 and self.f < 100000:
            self.df.loc[self.df['S(E)'] < 5,'Restriction'] = 'Occupation'
            self.df.loc[self.df['S(E)'] < 1,'Restriction'] = 'General Public'

       
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
        ax.set_title("Restricions with %d signal" % self.f)
        ax.legend(title='Restriction levels')
        if show:
            plt.show()

    def plot2D(self,field='S(E)',color = 'Reds'):
        fig, ax = plt.subplots(1)
        ax1 =ax.scatter(x =self.df[self.axis[0]],y= self.df[self.axis[1]],c =self.df[field],cmap = color)
        plt.colorbar(ax1)
        ax.set_xlabel(self.axis[0])
        ax.set_ylabel(self.axis[1])
        ax.set_title("{} over {}{} plane".format(field,self.axis[0],self.axis[1]))


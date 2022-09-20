from fileinput import filename
from platform import freedesktop_os_release
from unittest import result
from kiwisolver import Solver
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import HDFStore
import math
import sys
from Field import Field
from Standard import getZone

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)
import errno
import os
import os.path


class IXUSField(Field):
    def __init__(self,csvfile,f):
        path = 'venv/Include/IXUS/{}'.format(csvfile)
        df = pd.read_csv(path)
        standard = ''
        column = ''
        if '% of ICNIRP Public' in df.columns:
            standard = 'ICNIRP'
            column = '% of ICNIRP Public'
        elif '% of ARPANSA Public' in df.columns:
            standard = 'ARPANSA'
            column = '% of ARPANSA Public'
        elif '% of FCC OET 65 Public' in df.columns:
            standard = 'FCC'
            column = '% of FCC OET 65 Public'
        elif '% of BGV B11 Public' in df.columns:
            standard = 'BGVB11'
            column = '% of BGV B11 Public'
        elif '% of IC Safety Code 6 Public' in df.columns:
            standard = 'Code6'
            column = '% of IC Safety Code 6 Public'

        df = df.rename(columns = {column:'percentage'})
        df['S'] = getZone(f,standard)[1]*df['percentage']

        super().__init__(df,f,type = 'IXUS',standard=standard)
        self.axis1  = 'X'
        self.axis2 = 'Y'

    def plot2DZones(self, Ncolor='blue', GPcolor='yellow', Ocolor='red', xfig=6, yfig=4, axis1='Y', axis2='X',show = True):
        return super().plot2DZones(Ncolor, GPcolor, Ocolor, xfig, yfig, axis1, axis2,show)

    def plot2D(self, field='% of ICNIRP Public', color='Reds'):
        return super().plot2D(field, color)

    def getS(self):
        raise AttributeError("IXUS Field has no attribute S")

    def GetE(self):
        raise AttributeError("IXUS Field has no attribute E")

    def getH(self):
        raise AttributeError("IXUS Field has no attribute H")

    def replaceStandard(self,newStandard):
        self.df['S'] = self.df['S']*getZone(self.f,newStandard)[1]/getZone(self.f,self.standard)[1]
        self.standard = newStandard

    def setf(self,newf):
        self.f = newf
        self.df['S'] = getZone(self.f,self.standard)[1]*self.df['percentage']
        

    def setRestrictions(self):
        maxFreq = getZone(self.f,self.standard)[1]
        minFreq = getZone(self.f,self.standard)[0]  
        self.df['Restriction'] = 1
        self.df.loc[minFreq > self.df['S'],'Restriction'] = 0
        self.df.loc[maxFreq < self.df['S'],'Restriction'] = 2



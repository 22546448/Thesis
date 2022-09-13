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

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)
import errno
import os
import os.path


class IXUSField(Field):
    def __init__(self,csvfile,f):
        path = 'venv/Include/IXUS/{}'.format(csvfile)
        if os.path.exists(path):
            df = pd.read_csv(path)
            super().__init__(df,f,type = 'IXUS')
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


    def plot2DZones(self, Ncolor='blue', GPcolor='yellow', Ocolor='red', xfig=6, yfig=4, axis1='X', axis2='Y',show = True):
        return super().plot2DZones(Ncolor, GPcolor, Ocolor, xfig, yfig, axis1, axis2,show)

    def plot2D(self, field='% of ICNIRP Public', color='Reds'):
        return super().plot2D(field, color)

    def getS(self):
        raise AttributeError("IXUS Field has no attribute S")

    def GetE(self):
        raise AttributeError("IXUS Field has no attribute E")

    def getS(self):
        raise AttributeError("IXUS Field has no attribute H")

    def getPersentage(self):
        dfP = self.df[['X','Y','Z','% of ICNIRP Public']].copy()
        dfP =  dfP.astype(float)
        return dfP
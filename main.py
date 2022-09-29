from audioop import lin2adpcm
from cProfile import label
from matplotlib.dates import SecondLocator
from sympy import GoldenRatio
from EMFFeko import GetField,GetFarField,nearSurface,ClassicalSpherical,SpacialAverageCylindricalEstimation,SpacialPeakCylindricalEstimation,AdjustedSphericalSector,SimpleSphericalSector
from Standard import getStandard,getZone
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from EMFIXUS import IXUSField
import time
import mayavi.mlab as mlab

def CylindricalValidationTest():
    SectorCoverageSbar = [5.58, 3.54, 2.49, 1.86, 1.43, 1.02, 0.639]
    SectorCoverageS = [9.96, 5.74, 3.70, 2.56, 1.86, 1.25, 0.727]
    f = 925
    lamda = (3*10**8)/(f*10**6)
    power = 80
    L = 2.158
    AHPBW = 84
    y = 0
    Gs = 17 #dBi
    Go = 11 #dBi
    Globe = -9      #dBi
    Globe = -3.6    #dBi
    phi = np.pi/12
    Ry = [4, 6, 8, 10, 12, 15, 20]
    R = [ry*np.cos(y*np.pi/180) for ry in Ry]

    SectorAverage = SpacialAverageCylindricalEstimation(phi,R,power,AHPBW, L,Gs, y*np.pi/180,Ry)
    SectorPeak = SpacialPeakCylindricalEstimation(phi,R,power,AHPBW, L, Gs, y=y*np.pi/180)

    plt.figure()
    plt.plot(Ry,SectorAverage,label = 'SpacialPeakCylindrical')
    plt.plot(Ry,SectorCoverageS,label = 'Peak Cylindrical Validation line')
    plt.plot(Ry,SectorCoverageSbar,label = 'Average Cylindrical Validation line')
    plt.plot(Ry,SectorPeak,label = 'SpacialAverageCylindrical')
    plt.legend()
    plt.show()















CylindricalValidationTest()


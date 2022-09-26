from EMFFeko import GetField
from Standard import getStandard,getZone
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from EMFIXUS import IXUSField
import time


FCCpublic = getZone(900,'FCC')[1]
print(FCCpublic)
l1 = [65.8, 39.4, 25, 13.4, 10.7, 10.4, 10.1, 9.44]
l2 = [6.22, 9.16, 13.6, 19.6, 26.3, 32.8, 37.6, 39.4, 37.6, 32.8, 26.3, 19.6, 13.6, 9.16, 6.22]
l3 = [0.173, 3.77, 129, 105, 111, 163, 272, 338, 272, 163, 111, 105, 129, 3.77, 0.173]




IXUS_E_l1 = [236.3, 151.5, 130.2, 90.72, 79.83, 79.23, 78.48, 76.06] 
IXUS_S_l1 = [x**2/(2*377) for x in IXUS_E_l1]

IXUS_E_l2 = [56.89, 70.7, 87.97, 106.1, 123.3, 137.8, 147.4, 151.5, 147.8, 138.5, 124, 106.6, 87.97, 70.55, 57]

#IXUS_E_l2 = [76.13, 94.48, 117.7, 142, 165.1, 184.4, 197.2, 202.7, 197.8, 185.3, 165.9, 142.6, 117.7, 94.41, 76.27]
IXUS_S_l2 = [x**2/(2*377) for x in IXUS_E_l2]

IXUS_E_l3 = [18.09, 38.67, 1.932*10**8, 127.7, 163.2, 190, 228.5, 1.413*10**8, 220.8, 189.4, 164.3, 132.8, 2.005*10**7, 39.58, 18.51]
IXUS_S_l3 = [x**2/(2*377) for x in IXUS_E_l3]


def Doline(line1,line2, line3):
    
    fig, axs = plt.subplots(2,2)

    axs[0, 0].plot(line1.df['X'], l1,label = 'S from Vitas')
    axs[0, 0].plot(line1.df['X'], IXUS_S_l1,label = 'IXUS')
    axs[0, 0].plot(line1.df['X'], line1.df['S(E)'], label = 'S(E)')
    axs[0, 0].plot(line1.df['X'], line1.df['S(ExH)'], label = 'S(ExH)')
    axs[0, 0].legend()
 
    axs[0, 1].plot(line2.df['Y'], l2,label = 'S from Vitas')
    axs[0, 1].plot(line2.df['Y'], IXUS_S_l2,label = 'IXUS')
    axs[0, 1].plot(line2.df['Y'], line2.df['S(E)'], label = 'S(E)')
    axs[0, 1].plot(line2.df['Y'], line2.df['S(ExH)'], label = 'S(ExH)')
    axs[0, 1].legend()

    axs[1, 0].plot(line3.df['Z'], l3,label = 'S from Vitas')
    axs[1, 0].plot(line3.df['Z'], IXUS_S_l3,label = 'IXUS')
    axs[1, 0].plot(line3.df['Z'], line3.df['S(E)'], label = 'S(E)')
    axs[1, 0].plot(line3.df['Z'], line3.df['S(ExH)'], label = 'S(ExH)')
    axs[1, 0].legend()
 
line1 = GetField('IEC-62232-panel-antenna (4)_Line1.efe','IEC-62232-panel-antenna (4)_Line1.hfe',compress=False)
line2 = GetField('IEC-62232-panel-antenna (4)_Line2.efe','IEC-62232-panel-antenna (4)_Line2.hfe',compress=False)
line3 = GetField('IEC-62232-panel-antenna (4)_Line3.efe','IEC-62232-panel-antenna (4)_Line3.hfe',compress=False)
Doline(line1, line2, line3)
plt.show()




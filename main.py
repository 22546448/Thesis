from EMFFeko import GetField
from Standard import getStandard
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from EMFIXUS import IXUSField
import time

l1 = [65.8, 39.4, 25, 13.4, 10.7, 10.4, 10.1, 9.44]
l2 = [6.22, 9.16, 13.6, 19.6, 26.3, 32.8, 37.6, 39.4, 37.6, 32.8, 26.3, 19.6, 13.6, 9.16, 6.22]
l3 = [0.173, 3.77, 129, 105, 111, 163, 272, 338, 272, 163, 111, 105, 129, 3.77, 0.173]

def Doline(line1,line2, line3):
    
    fig, axs = plt.subplots(2,2)
    line1.df['S from Vitas'] = l1
    axs[0, 0].plot(line1.df['X'], line1.df['S from Vitas'],label = 'S from Vitas')
    axs[0, 0].plot(line1.df['X'], line1.df['S(E)'], label = 'S(E)')
    axs[0, 0].plot(line1.df['X'], line1.df['S(ExH)'], label = 'S(ExH)')
    axs[0, 0].plot(line1.df['X'], line1.df['S(ExH*)'], label = 'S(ExH*)')
    axs[0, 0].legend()
    #line1.df = line1.df.drop(columns = ['|E|','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','S','Ex','Ey','Ez','Hx','Hy','Hz','Restriction'])

    line2.df['S from Vitas'] = l2
    axs[0, 1].plot(line2.df['Y'], line2.df['S from Vitas'],label = 'S from Vitas')
    axs[0, 1].plot(line2.df['Y'], line2.df['S(E)'], label = 'S(E)')
    axs[0, 1].plot(line2.df['Y'], line2.df['S(ExH)'], label = 'S(ExH)')
    axs[0, 1].plot(line2.df['Y'], line2.df['S(ExH*)'], label = 'S(ExH*)')
    axs[0, 1].legend()
    #line2.df = line2.df.drop(columns = ['|E|','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','S','Ex','Ey','Ez','Hx','Hy','Hz','Restriction'])

    line3.df['S from Vitas'] = l3
    axs[1, 0].plot(line3.df['Z'], line3.df['S from Vitas'],label = 'S from Vitas')
    axs[1, 0].plot(line3.df['Z'], line3.df['S(E)'], label = 'S(E)')
    axs[1, 0].plot(line3.df['Z'], line3.df['S(ExH)'], label = 'S(ExH)')
    axs[1, 0].plot(line3.df['Z'], line3.df['S(ExH*)'], label = 'S(ExH*)')
    axs[1, 0].legend()
    #line3.df = line3.df.drop(columns = ['|E|','Re(Ex)','Im(Ex)','Re(Ey)','Im(Ey)','Re(Ez)','Im(Ez)','Re(Hx)','Im(Hx)','Re(Hy)','Im(Hy)','Re(Hz)','Im(Hz)','S','Ex','Ey','Ez','Hx','Hy','Hz','Restriction'])

line1 = GetField('IEC-62232-panel-antenna (4)_Line1.efe','IEC-62232-panel-antenna (4)_Line1.hfe',compress=False)
line2 = GetField('IEC-62232-panel-antenna (4)_Line2.efe','IEC-62232-panel-antenna (4)_Line2.hfe',compress=False)
line3 = GetField('IEC-62232-panel-antenna (4)_Line3.efe','IEC-62232-panel-antenna (4)_Line3.hfe',compress=False)
Doline(line1, line2, line3)
plt.show()




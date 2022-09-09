from tkinter.ttk import Style
from turtle import colormode
from test import GetField,GetEPhasor, GetHPhasor, PowerAtPoint, getSdata, plot2D, plotPowerLine,plot2DColor
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns

sns.set(style='ticks')



surface1 = GetField("IEC-62232-panel-antenna_2Dxy.efe","IEC-62232-panel-antenna_2Dxy.hfe")
df = surface1.dataframe

E = GetEPhasor(df)
H = GetHPhasor(df)
S = getSdata(df)

print(E)



 

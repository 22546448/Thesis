from test import GetField, PowerAtPoint, getSdata, plot2D, plotPowerLine,plot2DColor
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd




surface1 = GetField("IEC-62232-panel-antenna_2Dxy.efe","IEC-62232-panel-antenna_2Dxy.hfe")
df = surface1.dataframe

S = getSdata(df)

#plot2DColor(S,'X','Y')

#plot2D(df,'X','Y','|E|')
#plot2D(df,'X','Y','|H|')
plot2D(df,'X','Y','S(E)')



 

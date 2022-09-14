from EMFFeko import GetField
import EMFIXUS
from CreateEMF import Surface,Antenna,Field
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from EMFIXUS import IXUSField

surface1 = GetField("IEC-62232-panel-antenna_2Dxy.efe","IEC-62232-panel-antenna_2Dxy.hfe")
#surface1.plot2DZones()
field = 'theta'
surface1.plot2D(field=field,method='mayavi')
#surface1.plot2D(field=field,method='cadfeko')
#surface1.plot2DZones()

#surface3 = IXUSField('EnvironmentalSlice2.csv',900)
#surface3.plot2DZones()


#plane1 = Surface(-10,500,10,-10,500,10)
#antenna1 = Antenna(f=50,P=100,G=30)
#surface2 = Field(antenna1,plane1,spaceMin = 2)
#surface2.plot2D(field='S(E)',method = 'cadfeko')
#surface2.plot2D(field='S(E)',method = 'mayavi')







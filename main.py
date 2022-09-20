from EMFFeko import GetField
import EMFIXUS
from CreateEMF import Surface,Antenna,Field
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from EMFIXUS import IXUSField

#surface1 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",standard='FCC')
surface2 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",standard='Code6')

surface2.plot2DZones()
plt.show()


#plt.scatter(x = surface1.df['X'],y = surface1.df['Y'],c = np.positive(surface1.df['|S|']-surface1.df['S']),cmap='Reds')
#surface1.df.to_csv('surface1.csv',sep=';')
#plt.show()
#surface1.plot2DZones()

#surface2 = GetField('IEC-62232-panel-antenna_2Dxy.efe','IEC-62232-panel-antenna_2Dxy.hfe')
#surface1.compare2D(surface2)


#surface3 = IXUSField('EnvironmentalSlice2.csv',900)
#surface3.plot2DZones()

#surface3.compare(surface1)

#plane1 = Surface(-10,500,10,-10,500,10)
#antenna1 = Antenna(f=50,P=100,G=30,x = -5)

#surface2 = Field(antenna1,plane1,spaceMin = 2)
#surface2.plot2D(field='S',method = 'cadfeko')
#surface2.plot2D(field='S',method = 'mayavi')






#default exposure standard
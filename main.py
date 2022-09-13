from EMFFeko import GetField
import EMFIXUS
from CreateEMF import Surface,Antenna,Field
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from EMFIXUS import IXUSField

#surface1 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe")
#df['R'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
#df.loc[df['% of ICNIRP Public'] > 4, '% of ICNIRP Public'] = 0.1




surface3 = IXUSField('EnvironmentalSlice2-1.csv',900)
surface3.plot2DZones()


#surface1.plot2DZones()

#plane1 = Surface(-50,100,50,-50,100,50)
#antenna1 = Antenna(f=900,P=50,G=30,y = 2,z=-5)
#surface2 = Field(antenna1,plane1)
#surface2.plot2DZones(show = True)


#ax = plt.scatter(x= df['X'],y=df['Y'],c=df['% of ICNIRP Public'],cmap='Reds')
#plt.colorbar(ax)

#plt.scatter(x=surface2['X'],y=surface2['Y'],c=surface2['S'],cmap='Reds')
#plt.show()

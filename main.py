from EMFFeko import GetField
from CreateEMF import Surface,Antenna,Field
from matplotlib import pyplot as plt
import pandas as pd

#surface1 = GetField("IEC-62232-panel-antenna_2Dxy.efe","IEC-62232-panel-antenna_2Dxy.hfe")



plane1 = Surface(-5,100,5,-5,100,5)
antenna1 = Antenna(f=900,P=50,G=30)
surface2 = Field(antenna1,plane1)
#surface2.plot2DZones(show = True)

df = pd.read_csv('IXUS/EnvironmentalSlice1.csv')
print(df)

plt.scatter(x= df['X'],y=df['Y'],c=df['% of ICNIRP Public'],cmap='Reds')


#plt.scatter(x=surface2['X'],y=surface2['Y'],c=surface2['S'],cmap='Reds')
plt.show()
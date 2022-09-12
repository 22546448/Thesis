from EMFFeko import GetField
from CreateEMF import Surface,Antenna,Field
from matplotlib import pyplot as plt

#surface1 = GetField("IEC-62232-panel-antenna_2Dxy.efe","IEC-62232-panel-antenna_2Dxy.hfe")



plane1 = Surface(-5,100,5,-5,100,5)
antenna1 = Antenna(f=900,P=50,G=30)
surface2 = Field(antenna1,plane1)
surface2.plot2DZones(show = True)



#plt.scatter(x=surface2['X'],y=surface2['Y'],c=surface2['S'],cmap='Reds')
#plt.show()
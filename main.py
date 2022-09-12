import imp
from EMFFeko import CreateSurface,CreateAntenna,CreatedField, antenna
from matplotlib import pyplot as plt

#surface1 = GetField("IEC-62232-panel-antenna_2Dxy.efe","IEC-62232-panel-antenna_2Dxy.hfe")

plane1 = CreateSurface(-5,100,5,-5,100,5)
antenna = antenna(f=900,P=50,G=90)



#plt.scatter(x=surface2['X'],y=surface2['Y'],c=surface2['S'],cmap='Reds')
#plt.show()
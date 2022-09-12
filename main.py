from EMF import GetField

from matplotlib import pyplot as plt 

surface1 = GetField("IEC-62232-panel-antenna_2Dxy.efe","IEC-62232-panel-antenna_2Dxy.hfe")
df = surface1.dataframe

#S = surface1.getSdata(df)

#plot2DColor(S,'X','Y')


#surface1.plot2D(df,'X','Y','S(E)')
#surface1.plot2D(df,'X','Y','Im(Ex)')
surface1.plot2DZones(df,'X','Y')
plt.show()
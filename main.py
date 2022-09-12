from EMF import GetField

from matplotlib import pyplot as plt 

surface1 = GetField("IEC-62232-panel-antenna_2Dxy.efe","IEC-62232-panel-antenna_2Dxy.hfe")

#S = surface1.getS()

surface1.plot2DZones(GPcolor = 'white',show = True)

#surface1.plot2D('R',color = 'Blues')

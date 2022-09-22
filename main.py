from EMFFeko import GetField
from Standard import getStandard
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from EMFIXUS import IXUSField
import time



surface1 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",compress=False,standard = 'FCC',S = 'S(E)')
surface2 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",compress=False,standard = 'FCC',S = 'S(ExH)')

surface1.compareToSurface2D(surface2)
#IXUSSurface = IXUSField("EnvironmentalSlice2-2.csv",900)
#IXUSSurface.compareToSelf('ARPANSA')


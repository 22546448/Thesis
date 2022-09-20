from EMFFeko import GetField
import EMFIXUS
from CreateEMF import Surface,Antenna,Field
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from EMFIXUS import IXUSField

surface1 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",standard='FCC')
surface2 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",standard='Code6')

surface1.compare2D(surface2)
plt.show()


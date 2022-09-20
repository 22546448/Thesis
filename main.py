from EMFFeko import GetField
import EMFIXUS
from Standard import getStandard
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from EMFIXUS import IXUSField
import time


FCC = getStandard('FCC')
Code6 = getStandard('Code6')
print(FCC.standard)
print(FCC.public)
print(FCC.occupational)
print(Code6.standard)
print(Code6.public)
print(Code6.occupational)

st = time.time()
surface1 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",compress=False,standard = 'FCC')
surface2 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",compress=False)

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


surface1.compareToSurface2D(surface2)




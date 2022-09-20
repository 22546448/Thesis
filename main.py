from EMFFeko import GetField
import EMFIXUS
from Standard import getStandard
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from EMFIXUS import IXUSField
import time

  

st = time.time()
surface1 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",compress=False)
surface2 = GetField("IEC-62232-panel-antenna (4)_NearField1.efe","IEC-62232-panel-antenna (4)_NearField1.hfe",compress=False)

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


surface1.compareStandards('FCC','Code6')




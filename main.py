from audioop import lin2adpcm
from cProfile import label
from matplotlib.dates import SecondLocator
from sympy import GoldenRatio
from EMFFeko import *
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from EMFIXUS import IXUSField
import time
import mayavi.mlab as mlab





df = GetField('IEC-62232-panel-antenna_2Dxy.efe','IEC-62232-panel-antenna_2Dxy.hfe').df
plotSimulationMethod(df)
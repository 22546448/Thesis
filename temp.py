import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def FCC(x):
    if x<30:
        return  0
    elif 30<x and x<300:
        return 2
    elif 300<x and x<1500:
        return x/150
    elif 1500<x and x<100000:
        return 10


def FCCs(f):
    if f<0.3:
        return  0
    elif 0.3<f and f<1.34:
        return 100
    elif 1.34<f and f<30:
        return  180/f**2  
    elif 30<f and f<300:
        return 2
    elif 300<f and f<1500:
        return f/150
    elif 1500<f and f<100000:
        return 10


y = []
x = np.arange(30,100000,2)

for i in range(len(x)):
   y.append(f(x[i]))


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)

ax.plot(x,y,c='red', ls='', ms=5, marker='.')
ax.set_xscale('log',base = 3)


plt.show()
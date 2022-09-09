import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd





# Program to draw scatter plot using Dataframe.plot
# Import libraries
import pandas as pd

# Prepare data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd

fig, ax = plt.subplots()

df = pd.DataFrame({'n1':[1,2,1,3], 'n2':[1,3,2,1], 'l':['a','b','c','d']})

colormap = cm.viridis
colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(df['l']))]
print(colorlist)
for i,c in enumerate(colorlist):

    x = df['n1'][i]
    y = df['n2'][i]
    l = df['l'][i]

    ax.scatter(x, y, label=l, s=50, linewidth=0.1, c=c)

ax.legend()

plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# generate data
nx,ny = (4,2)
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)


xv,yv= np.meshgrid(x,y)
print(xv,yv)

df = pd.DataFrame({
       'X':xv.ravel(),
       'Y':yv.ravel()
})

df['R'] = np.sqrt(df['X']**2 + df['Y']**2)
print(df)

plt.scatter(x=df['X'],y=df['Y'],c=df['R'],cmap='Reds')
plt.show()
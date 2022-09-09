import mayavi.mlab as mlab
import numpy as np

x, y, z, value = np.random.random((4, 40))
mlab.points3d(x, y, z, value)
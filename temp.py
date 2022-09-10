from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import curl
R = ReferenceFrame('R')

F = R[1]**2 * R[2] * R.x - R[0]*R[1] * R.y + R[2]**2 * R.z

G = curl(F, R) 


print(F)
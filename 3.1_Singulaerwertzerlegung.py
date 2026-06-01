import numpy as np

a = np.array([[5, 7, 1], [2, 6, 3], [2, 5, 4]])
print("A=")
print(a)
u,s,v  = np.linalg.svd(a)

# Matrix 
u = np.matrix(u[:,:2])
s = np.diag(s[:2])
v = np.matrix(v[:2,:])
print("U=")
print(u)
print("S=")
print(s)
print("V=")
print(v)
usv = u*s*v
print("U*S*V=")
print(usv)


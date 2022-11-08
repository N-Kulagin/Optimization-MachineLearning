import numpy as np
import matplotlib

a = np.array([1,2,3,4])
print(a)
print(type(a))
print(a.dtype)
print()
print(a[0])

b = a.reshape(2,2)
print(b)
print(b[0,0],b[0,1],b[1,0],b[1,1])

c = np.empty([3,2])
print(c)
c = np.eye(3,2)
print(c)
c = np.identity(3)
print(c)

print(matplotlib.get_backend())
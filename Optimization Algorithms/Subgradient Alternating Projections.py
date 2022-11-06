import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def distance_function(x):
    p = x - proj_plane(x)
    v = x - proj_ball(x)
    return max(np.dot(p,p)**0.5,np.dot(v,v)**0.5)

def proj_plane(x):
    #return x - a * (np.dot(a,x)-b)/(np.dot(a,a))
    return x - A.T @ np.linalg.inv(A @ A.T) @ (A @ x - b)

def proj_ball(x):
    return x / np.dot(x,x)**0.5

#a = np.array([2.0,3,-1])
#b = 10.0
A = np.array([[2.0,3,-1],[1.0,-1,-1]])
b = np.array([6.0,5]) # 10, 5

x = np.array([1.0,1,-8]) # 1,1,-8
x1_history, x2_history, x3_history = np.array([]),np.array([]),np.array([])
f_history = np.array([])

fig = plt.figure(figsize=(16,9))
ax1 = plt.subplot(projection='3d')

for i in range(50):
    x1_history = np.append(x1_history, x[0])
    x2_history = np.append(x2_history, x[1])
    x3_history = np.append(x3_history, x[2])

    f_history = np.append(f_history, distance_function(x))
    if i % 2 == 0:
        x = proj_plane(x)
    else:
        x = proj_ball(x)
    print(x)


g = np.arange(-0.4,2,0.1)
h = np.arange(-0.4,2,0.1)
xgrid, ygrid = np.meshgrid(g,h)
fgrid = A[0,0]*xgrid + A[0,1]*ygrid - b[0]
#fgrid = a[0]*xgrid + a[1]*ygrid - b

ax1.plot_wireframe(xgrid,ygrid,fgrid,color='blue')

g = np.arange(-1,3,0.1)
h = np.arange(-1,3,0.1)
xgrid, ygrid = np.meshgrid(g,h)
fgrid2 = A[1,0]*xgrid + A[1,1]*ygrid - b[1]

ax1.plot_wireframe(xgrid,ygrid,fgrid2,color='green')

# plot unit sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
g = np.cos(u)*np.sin(v)
h = np.sin(u)*np.sin(v)
z = np.cos(v)
ax1.plot_surface(g, h, z, color="r")

ax1.plot3D(x1_history,x2_history,x3_history,color='orange',marker='o',markerfacecolor='red',ms=10)

# set box aspect ratio
ax1.set_box_aspect([ub - lb for lb, ub in (getattr(ax1, f'get_{a}lim')() for a in 'xyz')])

plt.show()

fig2 = plt.figure(figsize=(16,9))
ax2 = plt.subplot()

ax2.plot(f_history,color='blue',marker='o',markerfacecolor='orange',ms=5,label='$dist(x_k,C_j)$')
ax2.set_xlabel('k')
ax2.set_ylabel('$dist(x_k,C_j)$')
ax2.legend(fontsize=10)

plt.show()
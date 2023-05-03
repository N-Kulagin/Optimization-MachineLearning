import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec



# minimze ||x||_1 subject to Ax=b using projection

def proj_plane(x):
    return x - A.T @ np.linalg.inv(A @ A.T) @ (A @ x - b)

def f(x):
    return np.dot(c, np.abs(x))

def subgradient(x):
    return c * np.sign(x)

A = np.array([[2.0,-1,-1],[1.0,1,-1]])
b = np.array([-1.0,-3])
c = np.array([1,1,0])

x = np.array([2.0,-1,-2]) # 2 -2 -2
eps = 0.01
step = 1.0
N = 1000

x1_history, x2_history = np.array([]),np.array([])
f_history = np.array([])
step_history = np.array([])

for i in range(N):
    f_history = np.append(f_history,f(x))

    grad = subgradient(x)
    step = eps / np.dot(grad,grad)
    step_history = np.append(step_history,step)

    x = x - step * grad
    x1_history = np.append(x1_history,x[0])
    x2_history = np.append(x2_history, x[1])

    x = proj_plane(x)
    x1_history = np.append(x1_history, x[0])
    x2_history = np.append(x2_history, x[1])
    print(np.round(x,3))

fig1 = plt.figure(figsize=(16,9))
gs1 = GridSpec(2,1,fig1)

ax1 = plt.subplot(gs1[0,0])
ax1.plot(f_history,label='$f(x_k)$',color='blue',marker='o',markerfacecolor='orange',markevery=50)
ax1.set_xlabel('Iteration number k')
ax1.set_ylabel('$f(x_k)$')
ax1.legend(fontsize=15)

ax2 = plt.subplot(gs1[1,0])
ax2.plot(step_history,label='$\\alpha_k$',color='blue',marker='o',markerfacecolor='orange',markevery=50)
ax2.set_xlabel('Iteration number k')
ax2.set_ylabel('$\\alpha_k$')
ax2.legend(fontsize=15)

plt.show()


g = np.arange(-3,4,0.1)
h = np.arange(-3,4,0.1)

xgrid,ygrid = np.meshgrid(g,h)
fgrid = np.zeros((len(g),len(h)))

for i in range(len(g)):
    for j in range(len(h)):
        fgrid[i,j] = f([g[i],h[j],0])

fig2 = plt.figure(figsize=(16,9))
ax3 = plt.subplot()

ax3.contourf(xgrid,ygrid,fgrid,np.arange(0,5,0.1),cmap='magma') # gist_stern
ax3.plot(x1_history,x2_history,color='orange',marker='o',markerfacecolor='red',ms=7,markevery=100)
ax3.plot([-3, 3.8],[-2.5,0.9],color='blue')

plt.show()
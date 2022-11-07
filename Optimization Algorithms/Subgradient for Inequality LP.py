import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


# Inequality constrained LP, Ax <= b, with Polyak Step size and subgradient method
# Stephen Boyd, Convex Optimization 2
# Polyak Step Size - https://youtu.be/B51GgGCHBRk?t=2068 - (f(x_k) - f*)/||g_k||^2
# Subgradient Descent for Constrained Optimization - https://youtu.be/kE3wtUaQzpA?t=2108

def f(x):
    return np.dot(c,x)

A = np.array([[1.0,1,-6,0.1],[2.0,-1,1,-1]]).T
b = np.array([10.0,10,-4,5])
c = np.array([1.0,1])

step = 1.0
x = np.array([8.0,7])
grad = np.zeros(len(x))
eps = 0.01

f_history, g_history = np.array([]),np.array([])
x1_history, x2_history = np.array([]),np.array([])

constr1_history, constr2_history = np.array([]),np.array([])
constr3_history, constr4_history = np.array([]),np.array([])

for i in range(5000):
    x1_history = np.append(x1_history,x[0])
    x2_history = np.append(x2_history,x[1])
    f_history = np.append(f_history,f(x))

    constr1_history = np.append(constr1_history,np.dot(A[0],x) - b[0])
    constr2_history = np.append(constr2_history, np.dot(A[1], x) - b[1])
    constr3_history = np.append(constr3_history, np.dot(A[2], x) - b[2])
    constr4_history = np.append(constr4_history, np.dot(A[3], x) - b[3])

    if False in(A @ x <= b):
        p = np.argmin(A @ x <= b)
        grad = A[p]
        # polyak step size (f_k - f*)/||g_k||^2, f* = 0 because inequality constraint <= 0
        step = (np.dot(A[p],x) - b[p])/np.dot(grad,grad)
    else:
        grad = c
        step = eps / np.dot(grad,grad)
    x = x - step * grad
    print(x)


fig1 = plt.figure(figsize=(16,9))
gs1 = GridSpec(2,1,fig1)

ax1 = plt.subplot(gs1[0,0])
ax1.plot(f_history,label='$f_0(x_k) = c^Tx_k$',color='blue',marker='o',markerfacecolor='orange',markevery=500)
ax1.set_xlabel('Iteration number k')
ax1.set_ylabel('Function value')
ax1.legend(fontsize=10)

ax2 = plt.subplot(gs1[1,0])
ax2.plot(constr1_history,label='$f_1(x_k) \\leq 0$',color='red',lw=2)
ax2.plot(constr2_history,label='$f_2(x_k) \\leq 0$',color='red',lw=2,ls=':')
ax2.plot(constr3_history,label='$f_3(x_k) \\leq 0$',color='blue',lw=2,ls='-.')
ax2.plot(constr4_history,label='$f_4(x_k) \\leq 0$',color='black',lw=2,ls='--')
ax2.set_xlabel('Iteration number k')
ax2.set_ylabel('Constraint feasiblity')
ax2.legend(fontsize=10)
plt.grid()

plt.show()

fig2 = plt.figure(figsize=(16,9))

ax3 = plt.subplot()

x_range = np.arange(-1,12,0.1)
y0 = (b[0] - A[0,0]*x_range)/A[0,1]
y1 = (b[1] - A[1,0]*x_range)/A[1,1]
y2 = (b[2] - A[2,0]*x_range)/A[2,1]
y3 = (b[3] - A[3,0]*x_range)/A[3,1]

ax3.plot(x_range,y0,label='$f_1(x)$')
ax3.plot(x_range,y1,label='$f_2(x)$')
ax3.plot(x_range,y2,label='$f_3(x)$')
ax3.plot(x_range,y3,label='$f_4(x)$')
ax3.plot(x1_history,x2_history,label='$x_k$',color='blue',marker='o',markerfacecolor='orange',markevery=500)
ax3.set_ylim([-10,10])
ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$x_2$')
ax3.legend(fontsize=10)


origin = np.array([[6,8,1,2],[2,-2,2,-4.8]])
V = np.array([-A[0]/np.dot(A[0],A[0])**0.5,-A[1],-A[2]/np.dot(A[2],A[2])**0.5,-A[3]])
plt.quiver(*origin,V[:,0],V[:,1],color=['blue','orange','green','red'],scale=14,width=0.002)
plt.grid()

plt.show()
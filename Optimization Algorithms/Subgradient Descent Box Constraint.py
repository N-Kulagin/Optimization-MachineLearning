import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# a <= x <= b
def f(x):
    return np.dot(c,np.abs(x))

def subgradient(x):
    return c * (np.sign(x) + 0.001)

def project_box(x):
    if x[0] < a[0]:
        x[0] = a[0]
    if x[0] > a[1]:
        x[0] = a[1]
    if x[1] < b[0]:
        x[1] = b[0]
    if x[1] > b[1]:
        x[1] = b[1]
    return x

c = np.array([0.5,1])
a = np.array([-1.0,1])
b = np.array([-2.0,-1.0])

x = np.array([2.0,0])
step = 1.0
eps = 0.01

f_history, step_history = np.array([]), np.array([])
x1_history, x2_history = np.array([]), np.array([])

for i in range(1000):
    x1_history = np.append(x1_history,x[0])
    x2_history = np.append(x2_history, x[1])
    f_history = np.append(f_history,f(x))

    grad = subgradient(x)
    step = eps / np.dot(grad,grad)
    x = project_box(x-step*grad)

    step_history = np.append(step_history,step)

    print(x)


fig1 = plt.figure(figsize=(16,9))
gs = GridSpec(2,1,fig1)

ax1 = plt.subplot(gs[0,0])
ax1.plot(f_history,label='$f(x_k)$',color='blue',marker='o',markerfacecolor='orange',markevery=50)
ax1.set_xlabel('k')
ax1.set_ylabel('$f(x_k)$')
ax1.legend(fontsize=15)
ax1.grid()

ax2 = plt.subplot(gs[1,0])
ax2.plot(step_history,label='$\\alpha_k$',color='black',marker='o',markerfacecolor='orange',markevery=50)
ax2.set_xlabel('k')
ax2.set_ylabel('$\\alpha_k$')
ax2.legend(fontsize=15)
ax2.grid()

plt.show()

fig2 = plt.figure(figsize=(16,9))

ax3 = plt.subplot()

g,h = np.arange(-3,3,0.1), np.arange(-3,3,0.1)

xgrid, ygrid = np.meshgrid(g,h)
fgrid = c[0] * np.abs(xgrid) + c[1] * np.abs(ygrid)

ax3.contourf(xgrid,ygrid,fgrid,np.arange(0,6,0.1),cmap='magma')
ax3.plot(x1_history,x2_history,color='red',marker='o',markerfacecolor='orange',markevery=20,ms=5,label='$x_k$')
ax3.plot([a[0],a[0]],[b[0],b[1]],color='black',ls='--',lw=1,label='Box constraint')
ax3.plot([a[1],a[1]],[b[0],b[1]],color='black',ls='--',lw=1)
ax3.plot([a[0],a[1]],[b[0],b[0]],color='black',ls='--',lw=1)
ax3.plot([a[0],a[1]],[b[1],b[1]],color='black',ls='--',lw=1)
ax3.legend(fontsize=15)

ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$x_2$')

plt.show()
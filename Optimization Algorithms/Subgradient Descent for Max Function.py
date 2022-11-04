import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def f1(x):
    return x[0]+x[1]

def f2(x):
    return x[0]**2 + 2*x[1]**2 - 2*x[0] + 5*x[1]

def f(x):
    return max(f1(x),f2(x))

def subgradient(x):
    p, v = f1(x), f2(x)

    if p > v:
        return np.array([1,1])
    else:
        return np.array([2*x[0]-2,4*x[1]+5])


x = np.array([5,5])
eps = 0.01
grad = subgradient(x)
step = 1

x1_history = np.array([])
x2_history = np.array([])
x1_history = np.append(x1_history,x[0])
x2_history = np.append(x2_history,x[1])
grad_norm_history = np.array([])
step_history = np.array([])
f_history = np.array([])
f_history = np.append(f_history,f(x))


for i in range(20000):
    grad = subgradient(x)
    t = np.dot(grad,grad)
    step = eps / t
    x = x - step * grad
    print(x)

    x1_history = np.append(x1_history, x[0])
    x2_history = np.append(x2_history, x[1])
    grad_norm_history = np.append(grad_norm_history,t**0.5)
    step_history = np.append(step_history, step)
    f_history = np.append(f_history,f(x))


fig1 = plt.figure(figsize=(16,9))
gs1 = GridSpec(2,1,fig1)
ax1 = plt.subplot(gs1[0,0])
ax1.plot(f_history - np.full(len(f_history),(2.0-(102.0)**0.5)/4.0),label='$f(x) - f(x^*) = \max (f_1(x),f_2(x)) - f(x^*)$',
         lw=1,ms=4,marker='o',markeredgecolor='black',markerfacecolor='orange',color='blue',markevery=300)
ax1.set_yscale('log')
ax1.set_xlabel('k')
ax1.set_ylabel('$f(x)-f(x^*)$')
ax1.legend(fontsize=15)

ax2 = plt.subplot(gs1[1,0])
ax2.plot(grad_norm_history,label='$||g_k||_2$',lw=1,ms=4,marker='o',markeredgecolor='black',
         markerfacecolor='orange',color='blue',markevery=300)
ax2.set_yscale('log')
ax2.set_xlabel('k')
ax2.set_ylabel('$||g_k||_2$')
ax2.legend(fontsize=15)

plt.show()

fig2 = plt.figure(figsize=(16,9))
gs2 = GridSpec(1,1,fig2)

ax3 = plt.subplot(gs2[0,0])
ax3.plot(step_history,label='$\\alpha_k$',lw=1,ms=4,marker='o',markeredgecolor='black',
         markerfacecolor='orange',color='blue',markevery=300)
ax3.set_yscale('log')
ax3.set_xlabel('k')
ax3.set_ylabel('$\\alpha_k$')
ax3.legend(fontsize=15)

plt.show()

a = np.arange(-5,5.1,0.1)
b = np.arange(-5,5.1,0.1)

xgrid,ygrid = np.meshgrid(a,b)

fgrid = np.zeros((len(a),len(b)))

for i in range(len(a)):
    for j in range(len(b)):
        fgrid[i,j] = max(a[i]+b[j],a[i]**2 - 2*a[i] + 2*b[j]**2 + 5*b[j])


fig3 = plt.figure(figsize=(16,9))
h1 = [np.log(t) for t in np.arange(0.1,10,0.1)]
h2 = np.arange(max(h1)+1,100,1)
H = np.concatenate((h1,h2)) # линии уровня



ax4 = fig3.add_subplot()
ax4.contourf(ygrid,xgrid,fgrid,H,cmap='gist_stern') # prism, flag, gist_stern
ax4.plot(x1_history,x2_history,label='$x_k$',color='yellow',marker='o',ms=5,markevery=250,markerfacecolor='orange',markeredgecolor='black',lw=2)
ax4.legend(facecolor=(0.2,0.2,0.2),labelcolor='white',fontsize=20)

ax4.set_xlabel('$x_1$')
ax4.set_ylabel('$x_2$')

plt.show()
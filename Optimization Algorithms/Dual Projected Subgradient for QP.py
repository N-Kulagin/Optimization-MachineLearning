import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


# Minimize 1/2 * x^T * A * x - b^T * x subject to x1^2 <= 1, x2^2 <= 1
# Stephen Boyd, Convex Optimization 2
# https://youtu.be/kE3wtUaQzpA

def primal_function(x):
    return 0.5 * np.dot(x,A @ x) - np.dot(b,x)

def dual_function(x,dual):
    inverse = np.linalg.inv(A + 2 * dual * np.identity(len(dual)))
    a = inverse @ b
    return 0.5 * np.dot(a, A @ a) - np.dot(b, a)

A = np.array([[2,0.5],[0.5,1]])
b = -np.array([1.0,-1])
number_of_constraints = 2

dual_variable = np.ones(number_of_constraints)
x = np.array([2.0,2.0])
step = 0.005 # do not use eps / ||g||^2, does not work

f_history, g_history = np.array([]),np.array([])
constr1_history, constr2_history = np.array([]),np.array([])
x1_history, x2_history = np.array([]),np.array([])

for i in range(500):
    x1_history = np.append(x1_history, x[0])
    x2_history = np.append(x2_history, x[1])

    x = np.linalg.inv((A + 2*np.diag(dual_variable))) @ b
    grad = 1 - x**2
    dual_variable = dual_variable - step * grad
    dual_variable = np.maximum(dual_variable,np.zeros(len(dual_variable))) # projection onto the non-negative orthant
    print(x, dual_variable)

    f_history = np.append(f_history, primal_function(x))
    g_history = np.append(g_history, dual_function(x, dual_variable))
    constr1_history = np.append(constr1_history,x[0]**2 - 1)
    constr2_history = np.append(constr2_history, x[1] ** 2 - 1)

fig1 = plt.figure(figsize=(16,9))
gs1 = GridSpec(2,1,fig1)
ax1 = plt.subplot(gs1[0,0])

ax1.plot(f_history,label='$f_0(x)$',color='blue',marker='o',markerfacecolor='orange',markevery=50)
ax1.plot(g_history,label='$g(\\lambda)$',color='black',marker='o',markerfacecolor='orange',markevery=50)
ax1.plot(constr1_history,label='$x_1^2 - 1 \\leq 0$',color='red')
ax1.plot(constr2_history,label='$x_2^2-1 \\leq 0$',color='red',ls=':')
ax1.set_xlabel('Iteration number k')
ax1.set_ylabel('Function value')
ax1.legend(fontsize=10)

ax2 = plt.subplot(gs1[1,0])
ax2.plot(f_history-g_history,label='$f_0(x)-g(\\lambda)$',color='blue',marker='o',markerfacecolor='orange',markevery=50)
ax2.set_xlabel('Iteration number k')
ax2.set_ylabel('Function value')
ax2.set_yscale('log')
ax2.legend(fontsize=15)

plt.show()

fig2 = plt.figure(figsize=(16,9))
ax3 = plt.subplot()

g = np.arange(-3,3.1,0.1)
h = np.arange(-3,3.1,0.1)
xgrid, ygrid = np.meshgrid(g,h)
fgrid = np.zeros((len(g),len(h)))

for i in range(len(g)):
    for j in range(len(h)):
        fgrid[i,j] = primal_function([g[i],h[j]])

ax3.contourf(ygrid,xgrid,fgrid,np.arange(-2,10,1))
ax3.plot(x1_history,x2_history,label='$x_k$',color='red',marker='o',markerfacecolor='orange',markeredgecolor='black',markevery=30,lw=1,ms=6)
ax3.plot([1,1],[1,-1],label='Box constraint',color='black',lw=3,ls=':')
ax3.plot([1,-1],[-1,-1],color='black',lw=3,ls=':')
ax3.plot([-1,-1],[-1,1],color='black',lw=3,ls=':')
ax3.plot([-1,1],[1,1],color='black',lw=3,ls=':')
ax3.legend(fontsize=20)
plt.show()

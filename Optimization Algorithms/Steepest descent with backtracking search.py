## On the algorithm - see S. Boyd Convex Optimization, page 464

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

def f(A,x,b):
    return 0.5*(np.dot(x,A @ x)) + np.dot(b,x)


def gradient(A,x,b):
    return A @ x + b

def get_step(x):
    t = 1
    while f(A,x-t*grad,b) > f(A,x,b) + alpha * t * np.dot(grad,-grad): # i. e. Armijo rule
        t = beta*t
    return t

A = np.array([[4,3],[3,4]])
x = np.ones(2)
b = np.array([5,-2])

eps = 0.01
grad_norm = 1
grad = np.zeros(2)
step_length = 0

alpha = 0.1
beta = 0.8

iter_counter = 0

f_history = np.zeros(1)
norm_history = np.zeros(1)
step_history = np.zeros(1)
iter_history = np.zeros(1)
x_history = x[0]
y_history = x[1]

f_history[0] = f(A,x,b)
norm_history[0] = np.dot(gradient(A,x,b),gradient(A,x,b))**0.5
iter_history[0] = iter_counter

print(f"Начальная точка: {x}\nЗначение функции: {f(A,x,b)}")

while grad_norm > eps:
    grad = gradient(A,x,b)
    step_length = get_step(x)
    x = x - step_length*grad
    grad_norm = np.dot(grad,grad)**0.5
    iter_counter += 1

    f_history = np.append(f_history,f(A,x,b))
    norm_history = np.append(norm_history,grad_norm)
    step_history = np.append(step_history,abs(step_length))
    iter_history = np.append(iter_history,iter_counter)
    x_history = np.append(x_history,x[0])
    y_history = np.append(y_history,x[1])


print()
print(f"Получено решение: {x}\nНорма градиента: {grad_norm}")
print(f"Прошло итераций: {iter_counter}.\nОбращений к оракулу: ?")

fig1 = plt.figure(figsize=(15,8))
gs1 = GridSpec(2,2,figure=fig1)

ax1 = plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[0,1])
ax3 = plt.subplot(gs1[1,:])

ax1.plot(f_history,marker='o',ms=4,markeredgecolor=(1,0.6,0),markerfacecolor=(1,0.6,0),label=r'$F(x_n)$')
ax2.plot(norm_history,marker='*',ms=6,markerfacecolor=(1,0,0),markeredgecolor=(1,0,0),label=r'$||\nabla F(x_n)||_2$')
ax3.plot(iter_history[1:],step_history[1:],label=r'$|\alpha_n|$')


ax1.grid()
ax1.set_facecolor((0.9,0.9,0.9))
ax1.legend()
ax1.xaxis.set_major_locator(MultipleLocator(base=2))
ax1.set_xlabel('n')
ax1.set_ylabel(r'$F(x_n)$')

ax2.grid()
ax2.set_facecolor((0.9,0.9,0.9))
ax2.legend()
ax2.xaxis.set_major_locator(MultipleLocator(base=2))
ax2.set_xlabel('n')
ax2.set_ylabel(r'$||\nabla F(x_n)||_2$')

ax3.grid(which='both')
ax3.set_facecolor((0.9,0.9,0.9))
ax3.legend(fontsize='x-large')
ax3.xaxis.set_major_locator(MultipleLocator(base=2))
ax3.set_xlabel('n')
ax3.set_ylabel(r'$|\alpha_n|$')
ax3.set_ylim([0,1])

plt.show()

a1 = np.arange(-8,5,0.1)
b1 = np.arange(-8,10,0.1)
agrid, bgrid = np.meshgrid(a1,b1)
zgrid = 0.5*(4*agrid**2+6*agrid*bgrid+4*bgrid**2)+5*agrid-2*bgrid

fig2 = plt.figure(figsize=(15,8))

ax4 = plt.contourf(agrid,bgrid,zgrid,cmap='plasma')
ax4 = plt.plot(x_history,y_history,marker='o',markerfacecolor='w',markeredgecolor='r',ls=':',ms=3,color='r')

plt.show()
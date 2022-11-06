import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


# ||Ax-b||_2^2 + t*||x||_1 -> min

def f(x):
    p = A @ x - b
    return np.dot(p,p)/2.0 + tau * np.sum(np.abs(x))

def subgradient(x):
    p = A @ x - b
    return A.T @ p + tau * np.sign(x)

A = np.array([[-1.0,-2,1,2,4,5,6,7,10],[1,1,1,1,1,1,1,1,1]]).T
b = np.array([0.0,-20,1,3,-15,6,6,9,25])
x = np.array([1.0,1])

tau = 40.0
eps = 0.01
step = 1.0
N = 10000

x1_history, x2_history = np.array([]),np.array([])
step_history = np.array([])
f_history = np.array([])
grad_norm_history = np.array([])

for i in range(N):
    x1_history = np.append(x1_history,x[0])
    x2_history = np.append(x2_history, x[1])
    f_history = np.append(f_history, f(x))

    grad = subgradient(x)
    step = eps / np.dot(grad,grad)
    x = x - step * grad
    print(x)

    step_history = np.append(step_history, step)
    grad_norm_history = np.append(grad_norm_history,np.dot(grad,grad)**0.5)


fig1 = plt.figure(figsize=(16,9))
gs1 = GridSpec(3,1,fig1)
ax1 = plt.subplot(gs1[0,0])
ax1.plot(f_history,color='blue',marker='o',markevery=500,
         label='$\\frac{1}{2}||Ax-b||_2^2 + \\tau ||x||_1, $' + f'$\\tau = {tau}$',markerfacecolor='orange')
ax1.set_xlabel('k')
ax1.set_ylabel('$f(x_k)$')
ax1.legend(fontsize=15)

ax2 = plt.subplot(gs1[1,0])
ax2.plot(step_history,color='black',marker='o',markevery=500,label='$\\alpha_k$',markerfacecolor='orange')
ax2.set_xlabel('k')
ax2.set_ylabel('$\\alpha_k$')
ax2.set_yscale('log')
ax2.legend(fontsize=15)

ax3 = plt.subplot(gs1[2,0])
ax3.plot(grad_norm_history,color='blue',marker='o',markevery=500,label='$||g_k||_2$',markerfacecolor='orange')
ax3.set_xlabel('k')
ax3.set_ylabel('$||g_k||_2$')
ax3.set_yscale('log')
ax3.legend(fontsize=15)

fig1.tight_layout(pad=2.0)
plt.show()


fig2 = plt.figure(figsize=(16,9))
ax4 = plt.subplot()

q = np.arange(-2,3.1,0.1)
w = np.arange(-2,3.1,0.1)
fgrid = np.zeros((len(q),len(w)))

xgrid, ygrid = np.meshgrid(q,w)

for i in range(len(q)):
    for j in range(len(w)):
        fgrid[i,j] = f(np.array([q[i],w[j]]))

ax4.contourf(ygrid,xgrid,fgrid,np.arange(400,700,5),cmap='gist_stern')
ax4.plot(x1_history,x2_history,color='purple',marker='o',markevery=1000,label='$x_k$',markerfacecolor='orange')
ax4.legend(fontsize=15)

plt.show()

fig3 = plt.figure(figsize=(16,9))
ax5 = plt.subplot()

x_star = np.linalg.inv(A.T @ A) @ (A.T @ b)
x_l2 = np.linalg.inv(A.T @ A + tau * np.identity(len(x))) @ (A.T @ b)

ax5.scatter(A.T[0],b)
ax5.plot([-12,12],[-12*x[0]+x[1],12*x[0]+x[1]],color='blue',label='L1 regularization')
ax5.plot([-12,12],[-12*x_star[0]+x_star[1],12*x_star[0]+x_star[1]],color='black',label='Least Squares',ls=':')
ax5.plot([-12,12],[-12*x_l2[0]+x_l2[1],12*x_l2[0]+x_l2[1]],color='orange',label='L2 regularization')
ax5.set_xlabel('$x_1$')
ax5.set_ylabel('$x_2$')
ax5.legend(fontsize=15)
ax5.grid()

plt.show()
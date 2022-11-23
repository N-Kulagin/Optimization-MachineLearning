import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def objective_hessian(x):
    H = np.zeros((len(A[0]),len(A[0])))
    for i in range(len(A)):
        H = H + np.tensordot(A[i],A[i],axes=0)/((b[i] - np.dot(A[i],x))**2)
    return H


def objective_gradient(x,mu,dual=True):
    grad = np.zeros(len(x))

    for i in range(len(A)):
        grad = grad + A[i]/(b[i] - np.dot(A[i],x))

    if dual:
        return t * c + grad + B * mu

    return t * c + grad


def objective_value(x):
    sum = np.sum(np.log(b-A@x))
    return t * np.dot(c,x) - sum


def backtracking(x,direction):
    alpha,beta = 0.1, 0.8

    function_value = objective_value(x)
    dot = np.dot(objective_gradient(x,mu,dual=False),direction)
    step = 1.0

    while objective_value(x+step*direction) > function_value + alpha * step * dot or np.isnan(objective_value(x+step*direction)):
        step = beta * step
    return step


A = np.array([[-1.0,-1],[2,-1],[-1,1],[1,5]])
b = np.array([2.0,10,4,40])
B = np.array([1.0,2])
d = 2.0
c = np.array([2.0,1])

m = len(A)
t = 1.0
multiplier = 1.2
duality_gap = m/t
eps = 0.01
decrement = 5.0

x = np.array([7.0,6])
mu = 1.0
step = 1.0

inner_iter_counter = 0
iter_counter = 0

x1_history, x2_history, t_history, step_history, grad_history, f_history, gap_history = \
    np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

while duality_gap >= eps:
    while decrement >= eps:
        x1_history = np.append(x1_history,x[0])
        x2_history = np.append(x2_history, x[1])
        t_history = np.append(t_history,t)
        step_history = np.append(step_history,step)
        grad_history = np.append(grad_history,np.dot(objective_gradient(x,mu,dual=False),objective_gradient(x,mu,dual=False))**0.5)
        f_history = np.append(f_history,objective_value(x))
        gap_history = np.append(gap_history,duality_gap)

        hessian = objective_hessian(x)
        KKT_Matrix = np.vstack((np.hstack((hessian,B[:,None])),np.concatenate((B,[0.0]))[:]))
        RHS = -np.concatenate((objective_gradient(x,mu) + B*mu,[np.dot(B,x)-d]))
        solution = np.linalg.solve(KKT_Matrix,RHS)

        step = backtracking(x,solution[:len(x)])

        x = x + step*solution[:len(x)]
        mu = mu + solution[len(x):]
        decrement = np.dot(solution[:len(x)],hessian @ solution[:len(x)])/2.0
        inner_iter_counter += 1
    iter_counter += 1
    print(inner_iter_counter,iter_counter)
    inner_iter_counter = 0
    decrement = 5.0
    t = multiplier * t
    duality_gap = m / t


print()
print(x)
print(t)
print(duality_gap)


fig1 = plt.figure(figsize=(16,9))
gs1 = GridSpec(3,1,fig1)

ax1 = plt.subplot(gs1[0,0])
ax1.plot(t_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$t_k$',markevery=1)
ax1.set_xlabel('Iteration number')
ax1.set_ylabel('$t_k$')
ax1.legend(fontsize=20)

ax2 = plt.subplot(gs1[1,0])
ax2.plot(step_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$\\gamma_k$',markevery=1)
ax2.set_xlabel('Iteration number')
ax2.set_ylabel('$\\gamma_k$')
ax2.legend(fontsize=20)

ax3 = plt.subplot(gs1[2,0])
ax3.plot(grad_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$||t_k\\nabla f_0(x_k) + \\nabla \phi(x_k)||_2$',markevery=1)
ax3.set_xlabel('Iteration number')
ax3.set_ylabel('$||t_k\\nabla f_0(x_k) + \\nabla \phi(x_k)||_2$')
ax3.legend(fontsize=20)

plt.tight_layout(pad=3)

plt.show()

fig2 = plt.figure(figsize=(16,9))
gs2 = GridSpec(2,1,fig2)

ax4 = plt.subplot(gs2[0,0])
ax4.plot(f_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$t_kf_0(x_k) + \phi (x_k)$',markevery=1)
ax4.set_xlabel('Iteration number')
ax4.set_ylabel('$t_kf_0(x_k) + \phi (x_k)$')
ax4.legend(fontsize=20)

ax5 = plt.subplot(gs2[1,0])
ax5.plot(gap_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$\\frac{m}{t_k}$',markevery=1)
ax5.set_xlabel('Iteration number')
ax5.set_ylabel('$\\frac{m}{t_k}$')
ax5.legend(fontsize=20)

plt.show()

fig2 = plt.figure(figsize=(16,9))
gs3 = GridSpec(1,1)
ax6 = plt.subplot(gs3[0,0])
for i in range(len(A)):
    ax6.plot([-10,10],[(b[i] - A[i,0]*(-10.0))/A[i,1],(b[i] - A[i,0]*(10.0))/A[i,1]],lw=3,color='black')
ax6.plot([-10,10],[(d-B[0]*(-10))/B[1],(d-B[0]*(10))/B[1]],lw=3,color='red',ls='--')

ax6.plot(x1_history,x2_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$x_k$',markevery=1,ms=6)
ax6.set_xlabel('$x_1$')
ax6.set_ylabel('$x_2$')
ax6.set_xlim([-10,10])
ax6.set_ylim([-10,10])
ax6.legend(fontsize=20)

g,h = np.arange(-10,10,0.1),np.arange(-10,10,0.1)
xgrid,ygrid = np.meshgrid(g,h)
fgrid = xgrid * c[0] + ygrid * c[1]

ax6.contourf(xgrid,ygrid,fgrid,np.arange(-52,52,1))

plt.show()
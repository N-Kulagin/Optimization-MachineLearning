import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def f(x,grad_coefficients=False):  # f(x) = sum of e^(ax+b)
    summation = 0
    f_grad = np.zeros(len(A))
    k = 0
    for i in range(len(A)):
        k = np.exp(np.dot(A[i],x)+b[i])
        f_grad[i] = k
        summation += k
    if grad_coefficients==False:
        return summation
    return f_grad


def gradient(x,f_grad):  # f'(x) = sum of a*e^(ax+b)
    g = np.zeros(len(x))
    for i in range(len(A)):
        g += A[i]*f_grad[i]
    return g


def hessian(x,f_grad):  # f''(x) = sum of a * a^T * e^(ax+b), rank 1 hessian (tensor product of two vectors)
    H = np.zeros((len(x),len(x)))

    for i in range(len(A)):
        H += np.tensordot(A[i],A[i],axes=0)*f_grad[i]
    return H


def get_step(x):  # find optimal step via Armijo rule (backtracking line search)
    t = 1  # initial step length
    while f(x+t*newton_step) > f(x) + alpha * t * np.dot(grad,newton_step):
        t = beta*t
    return t


A = np.array([[1,3],[1,-3],[-1,0]])  # Matrix of coefficients for affine functions which are exponentiated
x = np.array([-2,-4])  # starting point
b = np.array([-0.1,-0.1,-0.1])  # same as A, but a vector for affine functions
eps = 0.001  # desired accuracy

# Armijo rule (aka backtracking line search) parameters, 0 < alpha < 0.5; 0 < beta < 1;
alpha = 0.1
beta = 0.7  # higher beta produces more accurate search


grad_coeff = np.zeros(len(A))  # vector of coefficients for computing gradient and hessian, computational purpose only
grad = np.zeros(len(x))  # gradient vector
Hes = np.zeros((len(x),len(x)))  # Hessian matrix
newton_step = np.zeros(len(x))  # step direction
step_size = 0  # step length

stopping_criterion = 5  # stopping criterion
iter_counter = 0  # iteration counter

# keeps history of step size, function value, iterations, x points to plot it
step_history = np.zeros(0)
f_history = np.zeros(0)
iter_history = np.zeros(0)
x_history = np.array(x[0])
y_history = np.array(x[1])

# solution of the problem f(x) -> min and it's optimal value
x_optimal = np.array([-np.log(2)/2,0])
f_optimal = 2*(2**0.5)/np.exp(0.1)

while stopping_criterion > eps:
    grad_coeff = f(x,grad_coefficients=True)
    grad = gradient(x, grad_coeff)
    Hes = hessian(x, grad_coeff)
    newton_step = np.linalg.solve(Hes, -grad)
    stopping_criterion = np.dot(-grad, newton_step) * 0.5
    step_size = get_step(x)
    x = x + step_size * newton_step
    iter_counter += 1

    step_history = np.append(step_history,step_size)
    f_history = np.append(f_history,f(x)-f_optimal)
    iter_history = np.append(iter_history,iter_counter)
    x_history = np.append(x_history,x[0])
    y_history = np.append(y_history, x[1])

print(f"Found solution: {x} after {iter_counter} iterations")
print(f"Found optimal value: {f(x)}")
print(f"f(x_n) - f(x*) = {f(x) - f_optimal}")
print(f"Exact solution: {x_optimal}")

fig1 = plt.figure(figsize=(15,8))
gs1 = GridSpec(2,1,figure=fig1)

ax1 = plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[1,0])

ax1.plot(iter_history,step_history,
         marker='o',markerfacecolor=(1,0.5,0),markeredgecolor=(1,0.5,0),color='black',lw=3, label='$\\alpha_k$')
ax2.semilogy(iter_history,f_history,
             marker='o',markerfacecolor=(1,0.2,0),markeredgecolor=(1,0.2,0),color='black',lw=2,label='$F(x_n) - F(x^*)$')

ax1.set_facecolor((0.8,0.8,0.8))

ax2.minorticks_on()
ax1.grid()
ax2.grid()

ax1.legend(fontsize='x-large')
ax2.legend(fontsize='x-large')

ax1.xaxis.set_major_locator(MultipleLocator(base=2))
ax2.xaxis.set_major_locator(MultipleLocator(base=1))

ax1.set_xlabel('n')
ax1.set_ylabel('$\\alpha_k$')
ax2.set_ylabel('$F(x_n) - F(x^*)$')
ax2.set_xlabel('n')

ax1.set_title('Step length')
ax2.set_title('Function convergence')

fig1.tight_layout()

plt.show()

fig2 = plt.figure(figsize=(15,8))

a1 = np.arange(-6,6.1,0.1)
b1 = np.arange(-6,6.1,0.1)
zgrid = np.zeros((len(a1),len(b1)))

contour_levels = np.arange(0,500,10)

for i in range(len(a1)):
    for j in range(len(b1)):
        zgrid[i][j] = f([a1[i],b1[j]])

ax3 = plt.contourf(a1,b1,zgrid,contour_levels,cmap='plasma')
ax3 = plt.plot(x_history,y_history,marker='o',ls=':',lw=3,markerfacecolor='red',markeredgecolor='red')


plt.show()
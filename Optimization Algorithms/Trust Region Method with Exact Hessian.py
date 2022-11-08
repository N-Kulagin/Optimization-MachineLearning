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

def get_p(grad,Hes):  # Trust Region Dogleg approach
    p_u = -(np.dot(grad,grad)/np.dot(grad,Hes @ grad))*grad  # minimize model over a direction as in steepest descent
    p_b = np.linalg.solve(Hes,-grad)  # calculate global (unconstrained) minimum of the model
    if np.dot(p_b,p_b) <= delta_k**2:  # if the global minimum of the model lies within trusted region - go there
        return p_b
    # next algorithm implements the dogleg when the global minimum of the model lies outside the trusted region
    # here we need to solve the quadratic equation in terms of tau >= 0, which involves a bunch of dot products
    alpha = p_u  # for simplicity we'll use this renaming scheme
    beta = p_b - p_u  # consult formulas for Dogleg Trust Region Method for clearer notation
    aa = np.dot(alpha,alpha)  # here aa denotes alpha transpose times alpha, which is alpha dotted with itself
    ab = np.dot(alpha,beta)  # alpha dotted with beta
    bb = np.dot(beta,beta)  # beta dotted with itself
    #roots = polynomial.polyroots([aa+bb-2*ab-delta_k**2,2*(ab-bb),bb])
    #roots = numpy.polynomial.polynomial.polyroots([aa+bb-2*ab-delta_k**2,2*(ab-bb),bb])
    roots = np.roots([bb,2*(ab-bb),aa+bb-2*ab-delta_k**2])  # find roots of the polynomial in terms of tau
    tau = max(roots)  # pick tau as a maximum root of a quadratic polynomial
    print(f"Тау: {tau}")
    if tau <= 0 or tau > 2:  # tau can't be negative or greater than 2
        print(f"Error: tau roots are - {roots}")
        raise Exception
    if tau <= 1:  # consult the parametrized function p(tau) in Dogleg formulas
        return p_u
    else:
        return p_u + (tau-1)*beta


def model(p_k):  # quadratic model of the function at a point x, which depends on p as a variable vector
    return f(x)+np.dot(grad,p_k)+0.5*np.dot(p_k,Hes @ p_k)

A = np.array([[1,3],[1,-3],[-1,0]])  # Matrix of coefficients for affine functions which are exponentiated
x = np.array([-2,-4])  # starting point -2 -4
b = np.array([-0.1,-0.1,-0.1])  # same as A, but a vector for affine functions
eps = 0.001  # desired accuracy


grad = np.array([1,1])  # initialize arbitrary gradient to start while-loop
Hes = np.zeros((len(x),len(x)))  # initialize empty hessian

delta_max = 2  # maximum radius of a trusted sphere (region) in l2-norm
delta_k = 0.4  # arbitrary radius of a trusted region
eta = 0.2  # parameter used for controlling how much we trust the approximation, belongs tp [0,0.25) half-interval

iter_history = np.zeros(0)
x_history = np.zeros(0)
y_history = np.zeros(0)
grad_history = np.zeros(0)
f_history = np.zeros(0)
delta_history = np.zeros(0)
rho_history = np.zeros(0)
f_optimal = 2*(2**0.5)/np.exp(0.1)
iter_counter = 0

while np.dot(grad,grad)**0.5 > eps:  # stopping criterion unknown, using standard norm of a gradient
    grad_coeff = f(x,grad_coefficients=True)  # return coefficients for computing gradient and hessian
    grad = gradient(x,grad_coeff)  # compute gradient
    Hes = hessian(x, grad_coeff)  # compute hessian

    p_k = get_p(grad,Hes)  # x_next = x_last + p_k algorithm
    rho = (f(x)-f(x+p_k)) / (model([0,0]) - model(p_k))  # calculates how much f(x) changed vs predicted by our model

    if rho < 0.25:  # our quadratic model predicts the function behaviour poorly
        delta_k = 0.25*delta_k  # thus let's reduce the trusted region size to get more accurate steps
    elif rho > 0.75 and np.dot(p_k,p_k)**0.5 == delta_k:  # model prediction is good and is constrained by region size
        delta_k = min(2*delta_k,delta_max)  # thus let's double region size, but keep it no bigger than maximum size
    if rho > eta:  # if our prediction is good enough - change x, otherwise stay at the same point
        x = x + p_k
    iter_counter += 1
    iter_history = np.append(iter_history,iter_counter)
    x_history = np.append(x_history,x[0])
    y_history = np.append(y_history,x[1])
    grad_history = np.append(grad_history,np.dot(grad,grad)**0.5)
    f_history = np.append(f_history,f(x)-f_optimal)
    delta_history = np.append(delta_history,delta_k)
    rho_history = np.append(rho_history,rho)

print(f"Найдено решение: {x} за {iter_counter} итераций")

fig1 = plt.figure(figsize=(15,8))
gs1 = plt.GridSpec(4,1,fig1)

ax1 = plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[1,0])
ax3 = plt.subplot(gs1[2,0])
ax4 = plt.subplot(gs1[3,0])

ax1.semilogy(iter_history,grad_history,marker='o',markerfacecolor='orange',markeredgecolor='black',ls=':',color='black',lw=3,ms=10,label=r'$||\nabla F(x_k)||_2$')
ax2.semilogy(iter_history,f_history,marker='o',markerfacecolor='orange',markeredgecolor='black',ls=':',color='black',lw=3,ms=10,label=r'$F(x_k)-F(x^*)$')
ax3.plot(iter_history,delta_history,marker='o',markerfacecolor='orange',markeredgecolor='black',ls=':',color='black',lw=3,ms=10,label=r'$\Delta_k$')
ax4.plot(iter_history,rho_history,marker='o',markerfacecolor='orange',markeredgecolor='black',ls=':',color='black',lw=3,ms=10,label=r'$\rho_k$')

ax1.set_facecolor((0.9,0.9,0.9))
ax2.set_facecolor((0.9,0.9,0.9))
ax3.set_facecolor((0.9,0.9,0.9))
ax4.set_facecolor((0.9,0.9,0.9))

ax1.set_ylabel(r'$||\nabla F(x_K)||_2$')
ax1.set_xlabel(r'k')
ax2.set_ylabel(r'$F(x_k)-F(x^*)$')
ax2.set_xlabel(r'k')
ax3.set_ylabel(r'$\Delta_k$')
ax3.set_xlabel(r'k')
ax4.set_ylabel(r'$\rho_k$')
ax4.set_xlabel(r'k')

ax1.xaxis.set_major_locator(MultipleLocator(base=1))
ax2.xaxis.set_major_locator(MultipleLocator(base=1))
ax3.xaxis.set_major_locator(MultipleLocator(base=1))
ax4.xaxis.set_major_locator(MultipleLocator(base=1))

ax1.legend(fontsize='x-large')
ax2.legend(fontsize='x-large')
ax3.legend(fontsize='x-large')
ax4.legend(fontsize='x-large')

ax2.minorticks_on()
ax2.grid()

fig1.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(15,8))

a1 = np.arange(-8,8.1,0.1)
b1 = np.arange(-8,8.1,0.1)
zgrid = np.zeros((len(a1),len(b1)))

contour_levels = np.arange(0,500,10)

for i in range(len(a1)):
    for j in range(len(b1)):
        zgrid[i][j] = f([a1[i],b1[j]])


fig2 = plt.gcf()
ax5 = fig2.gca()

ax5.contourf(a1,b1,zgrid,contour_levels,cmap='plasma')
ax5.plot(x_history,y_history,marker='o',ls=':',lw=4,markerfacecolor='red',markeredgecolor='red')

for i in range(len(x_history)):
    if i<4 or i>len(x_history)-6:
        circle = plt.Circle((x_history[i], y_history[i]), delta_history[i], color='magenta', fill=False)
        ax5.add_patch(circle)

ax5.set_xlim(-6,6)
ax5.set_ylim(-6,6)


ax5.set_xlabel(r'X')
ax5.set_ylabel(r'Y')


plt.show()
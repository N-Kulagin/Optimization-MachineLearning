import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Stephen Boyd; Introduction to Applied Linear Algebra (page 413, exercise 18.4)

def model_function(x,theta):
    return theta[0]*np.exp(theta[1]*x)

def jacobian(x,theta):
    Jac = np.zeros((6,2))
    Jac[:,0] = np.exp(theta[1]*x)
    Jac[:,1] = theta[0] * x * np.exp(theta[1]*x)
    return Jac


x = np.array([0,1,2,3,4,5]).astype('float64')
y = np.array([5.2,4.5,2.7,2.5,2.1,1.9])

theta = np.array([-1.0,-2.0])
J = jacobian(x,theta)
f_val = model_function(x,theta)

eps = 0.01
lmbda = 1
iter_counter = 0

tmp = J.T @ f_val

iter_history = np.array([iter_counter])
lmbda_history = np.array([lmbda])
optimality_history = np.array([np.dot(tmp,tmp) ** 0.5])

while np.dot(tmp,tmp) ** 0.5 >= eps:
    J = jacobian(x, theta)
    f_val = model_function(x,theta) - y
    tmp = J.T @ f_val

    theta_test = theta - np.linalg.inv(J.T @ J + lmbda * np.identity(2)) @ J.T @ f_val

    tmp2 = (model_function(x,theta_test) - y)
    if np.dot(tmp2,tmp2) < np.dot(f_val,f_val):
        lmbda = 0.8 * lmbda
        theta = theta_test
    else:
        lmbda = 2.0 * lmbda
    tmp = jacobian(x,theta).T @ (model_function(x,theta) - y)
    iter_counter += 1
    print(theta)

    iter_history = np.append(iter_history,iter_counter)
    lmbda_history = np.append(lmbda_history,lmbda)
    optimality_history = np.append(optimality_history,np.dot(tmp,tmp)**0.5)

print(f"Найдено решение, параметры тета: {theta}")

fig = plt.figure(figsize=(15,8))

gs = GridSpec(3,1,fig)

ax1 = plt.subplot(gs[0,0])

ax1.plot(x,y,marker='o',lw=0,label='Data')
t = np.arange(-5,x[-1]+5,0.1)
ax1.plot(t,model_function(t,theta),label=r'$\theta_1 * e^ (\theta_2 x)$')
ax1.legend()
ax1.grid()

ax2 = plt.subplot(gs[1,0])
ax2.plot(iter_history,lmbda_history,label=r'$\lambda$',marker='o',color='black')

ax3 = plt.subplot(gs[2,0])
ax3.plot(iter_history,optimality_history,label=r'$||\ J^T \ f_k||_2$',marker='o',color='black',ls='--')

ax2.legend()
ax3.legend()

plt.show()
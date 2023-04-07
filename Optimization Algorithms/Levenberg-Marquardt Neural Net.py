import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math


def affine(x,theta):  # affine input for the sigmoid function
    return x[0]*theta[0]+x[1]*theta[1]+theta[2]


def sigmoid(x,theta):
    return 1.0/(1.0 + np.exp(-affine(x,theta)))


def sigmoid_derivative(x,theta):
    return sigmoid(x,theta)*(1-sigmoid(x,theta))


def model(x,theta):  # the model function which we try to find parameters theta for
    t1, t2, t3 = sigmoid(x, theta[1:4]), sigmoid(x, theta[5:8]), sigmoid(x, theta[9:12])
    return theta[0]*t1 + theta[4]*t2 + theta[8]*t3 + theta[12]


def objective_vector(X,theta,y):  # the vector of residuals of the model minus data (y), which we minimize
    return np.array([model(X[i],theta) for i in range(N)] - y)


def grad_f(x,theta):  # returns the gradient of the model function with respect to theta
    t1, t2, t3 = sigmoid(x,theta[1:4]), sigmoid(x,theta[5:8]), sigmoid(x,theta[9:12])
    s1, s2, s3 = sigmoid_derivative(x,theta[1:4]), sigmoid_derivative(x,theta[5:8]), sigmoid_derivative(x,theta[9:12])
    return np.array(
        [
            t1,
            theta[0]*x[0]*s1,
            theta[0]*x[1]*s1,
            theta[0]*s1,
            t2,
            theta[4]*x[0]*s2,
            theta[4]*x[1]*s2,
            theta[4]*s2,
            t3,
            theta[8]*x[0]*s3,
            theta[8]*x[1]*s3,
            theta[8]*s3,
            1.0
        ])


def jacobian(X,theta):  # returns jacobian of the function which is minimized
    J = np.zeros((N,p))

    for i in range(len(J)):
        J[i] = grad_f(X[i],theta)

    return J


def predicted_vector(X):  # returns the vector "y" of the observed values
    return X[:,0]*X[:,1]

# Stephen Boyd Introduction to Applied Linear Algebra, p. 413 Fitting a Neural Network

# (write this in LaTeX editor if it's not clear what's written below)
# Minimizing the function $F(\theta) = \frac{1}{2}||r(\theta)||_2^2 + \frac{\gamma}{2}||\theta||_2^2$
# where $r(\theta)_i = f(x_i|\theta) - y_i$ - the model function evaluated at the object $x_i$ minus $y_i$

# The goal is to fit the data X,y with a neural network


p = 13  # number of parameters in the model (do not change)
N = 200  # the number of observed objects (X)
K = 1000  # maximum number of iterations to optimize model for

theta = np.random.randn(p)  # initialize parameters with random values
np.random.seed(2)  # fix the seed to keep data (X,y) the same
X = np.random.randn(N, 2)  # observed objects
y = predicted_vector(X)  # observed value of the function at these objects


eps = 10**(-9.0)  # desired accuracy to solve the problem for (stop when norm(gradient) < epsilon)
gamma = 10**(-8.0)  # regularization parameter
lambd = 2.0  # levenberg-marquardt algorithm parameter

id = np.identity(p)  # initialize identity matrix to simplify calculations

print(f"THETA INITIAL = {theta}")
print()

J = jacobian(X,theta)  # jacobian of the objective function
grad = J.T @ objective_vector(X,theta,y) + gamma * theta  # evaluate gradient of the objective function
norm_grad = np.dot(grad, grad)  # square of the 2-norm of the gradient

# count iterations and record the values to build charts later
iter_counter = 0
f_history, grad_history = np.array([]), np.array([])
lambda_history, convergence_history = np.array([]), np.array([])

while norm_grad >= eps and lambd < 10**10:

    # levenberg-marquard step
    J = jacobian(X, theta)
    grad = J.T @ objective_vector(X,theta,y) + gamma * theta
    theta_test = theta - (np.linalg.inv(J.T @ J + (gamma + lambd) * id)) @ grad

    # check if the new value of theta decreases function value
    # if true - accept new theta and reduce lambda by 0.8, otherwise reject theta and increase lambda
    tmp = objective_vector(X,theta,y)
    tmp2 = objective_vector(X, theta_test, y)
    if np.dot(tmp2,tmp2) + gamma * np.dot(theta_test,theta_test) < np.dot(tmp,tmp) + gamma * np.dot(theta,theta):
        norm_grad = np.dot(grad,grad)
        lambd *= 0.8
        theta = theta_test
    else:
        lambd *= 2

    if iter_counter >= K:  # stop if the algorithm doesn't converge after K iterations
        break

    # record the values to build charts later on and print to show the progress of the algorithm LIVE
    print(norm_grad, iter_counter, lambd)
    k1 = objective_vector(X,theta,y)
    f_history = np.append(f_history, np.dot(k1,k1) + gamma * np.dot(theta,theta))
    grad_history = np.append(grad_history, norm_grad)
    lambda_history = np.append(lambda_history, lambd)
    convergence_history = np.append(convergence_history, np.dot(k1,k1))

    iter_counter += 1

# compute the linear least-squares solution of the problem
A = np.hstack((X, np.ones((N,1))))
beta = np.linalg.inv(A.T @ A) @ A.T @ y

# compute the RMS of the linear model and Neural Network
RMS_Linear = (np.dot(A @ beta - y, A @ beta - y) / len(y)) ** 0.5
RMS = (np.dot(objective_vector(X,theta,y), objective_vector(X,theta,y)) / len(y)) ** 0.5

print(f"Прошло {iter_counter} итераций")
print("Всё!")
print(f"theta = {theta}")
print()
print(f"RMS (Linear model): {RMS_Linear}")
print(f"RMS (Neural Network): {RMS}")

# following code creates charts and 3D plot

fig1 = plt.figure(figsize=(16,9))
gs1 = GridSpec(2,1,fig1)
fig1.suptitle('Convergence in function value and gradient norm')

ax1 = plt.subplot(gs1[0,0])
ax1.plot(f_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,
         label='$F(\\theta_k) = \\frac{1}{2}||r(\\theta_k)||_2^2+\\frac{\gamma}{2} ||\\theta_k||_2^2$',markevery=math.ceil(len(f_history)/10))
ax1.legend(fontsize=20,loc='upper right')
ax1.set_xlabel('$k$')
ax1.set_ylabel('$f(\\theta_k)$')
ax1.set_yscale('log')

ax2 = plt.subplot(gs1[1,0])
ax2.plot(grad_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,
         label='$||\\nabla F(\\theta_k)||_2^2$',markevery=math.ceil(len(grad_history)/10))
ax2.legend(fontsize=20,loc='upper right')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$||\\nabla f(\\theta_k)||_2^2$')
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(16,9))
gs2 = GridSpec(2,1,fig2)
fig2.suptitle('Value of the $\lambda_k$ in Levenberg–Marquardt algorithm and norm of the system $r(\\theta) = 0$')

ax3 = plt.subplot(gs2[0,0])
ax3.plot(lambda_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,
         label='$\lambda_k$',markevery=math.ceil(len(lambda_history)/10))
ax3.legend(fontsize=20,loc='upper right')
ax3.set_xlabel('$k$')
ax3.set_ylabel('$lambda_k$')
ax3.set_yscale('log')

ax4 = plt.subplot(gs2[1,0])
ax4.plot(convergence_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,
         label='$||r(\\theta)||_2^2$',markevery=math.ceil(len(convergence_history)/10))
ax4.legend(fontsize=20,loc='upper right')
ax4.set_xlabel('$k$')
ax4.set_ylabel('$||r(\\theta)||_2^2$')
ax4.set_yscale('log')

plt.tight_layout()
plt.show()

a = np.arange(-3, 3.5, 0.5)
xgrid, ygrid = np.meshgrid(a,a)
zgrid_model = np.identity(len(a))
zgrid_linear = beta[0]*xgrid + beta[1]*ygrid + beta[2]
zgrid_true = xgrid * ygrid

for i in range(len(a)):
    for j in range(len(a)):
        zgrid_model[i,j] = model(np.array([a[i],a[j]]),theta)

fig3 = plt.figure(figsize=(16,9))
ax_3d = fig3.add_subplot(projection="3d")
fig3.suptitle('Surfaces: True (green), Neural Network (blue), Linear Least Squares (orange) \n Red points are random samples (training set)')
ax_3d.set_zlim(-5,5)
ax_3d.plot_wireframe(xgrid,ygrid,zgrid_true,antialiased=False,color="green")
ax_3d.plot_surface(xgrid,ygrid,zgrid_model,antialiased=True)
ax_3d.plot_wireframe(xgrid,ygrid,zgrid_linear,color="orange")

ax_3d.scatter(X[:,0], X[:,1], y, color="red")

plt.show()
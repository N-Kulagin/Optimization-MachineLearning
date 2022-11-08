import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import openpyxl


def sigmoid(z):
    return 1/(1+np.exp(-z))


def logistic_loss(X,y,theta):
    sum = 0
    r,p = 0,0
    safety = 10 ** (-9)
    for i in range(len(X)):
        r = np.log(sigmoid(np.dot(theta,X[i])) + safety)
        p = np.log(1-sigmoid(np.dot(theta,X[i])) + safety)
        sum += y[i]*r + (1-y[i])*p
    return -sum


def logistic_gradient(X,y,theta):
    return X.T @ (sigmoid(X @ theta) - y)


def armijo(X,y,theta,gradient):
    t = 1
    alpha = 0.1  # 0.1
    beta = 0.8  # 0.8

    while logistic_loss(X,y,theta-t*gradient) > logistic_loss(X,y,theta) + alpha * t * np.dot(gradient,-gradient):
        t = beta * t
    return t

book = openpyxl.open('Logistic Regression Data.xlsx',read_only=True)
sheet = book.active
cells = sheet['A1':'D50']

X = np.zeros((50,3))
y = np.zeros(50)
theta = np.array([1,1,1])

k = 0
for x0,x1,x2,y_val in cells:
    X[k] = [x0.value,x1.value,x2.value]
    y[k] = y_val.value
    k += 1


eps = 0.01
step_length = 1
iter_counter = 0

grad_norm = np.zeros(52)
iter_history = np.arange(0,52,1)
loss_history = np.zeros(52)

while iter_counter <= 50:
    grad = logistic_gradient(X, y, theta)

    grad_norm[iter_counter] = np.dot(grad, grad) ** 0.5
    loss_history[iter_counter] = logistic_loss(X, y, theta)

    step_length = armijo(X,y,theta,grad)
    theta = theta - step_length * grad
    iter_counter += 1

loss_history[-1] = logistic_loss(X,y,theta)
grad_norm[-1] = np.dot(grad,grad) ** 0.5

print()
print("Метод сошёлся.")
print(f"Тета: {theta}")
print(f"Норма градиента: {np.dot(grad,grad) ** 0.5}")
print(f"LOSS: {logistic_loss(X,y,theta)}")

fig1 = plt.figure(figsize=(15,8))
gs = GridSpec(2,1,fig1)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])

ax1.plot(iter_history,grad_norm,color='blue',label=r'$||\nabla L(\theta_k) ||_2$')
ax1.grid()
ax1.set_xlabel('k')
ax1.set_ylabel(r'$||\nabla L(\theta_k) ||_2$')
ax1.set_yscale('log')
ax1.legend()

ax2.plot(iter_history,loss_history,color='blue',label=r'$L(\theta_k)$')
ax2.grid()
ax2.minorticks_on()
ax2.set_xlabel('k')
ax2.set_ylabel(r'$L(\theta_k)$')
ax2.set_yscale('log')
ax2.legend()

plt.show()

fig2 = plt.figure(figsize=(15,8))
gs = GridSpec(1,1,fig2)

def linear_fit(x,theta):
    return -1/theta[2] * (theta[0] + theta[1] * x)

ax3 = plt.subplot(gs[0,0])
ax3.scatter(X[0:25,1],X[0:25,2],label='group 0',color='orange')
ax3.scatter(X[25:,1],X[25:,2],label='group 1',color='blue')

x_points = np.arange(-15,15,0.5)
y_points = linear_fit(x_points,theta)

ax3.plot(x_points,y_points,label='Logistic Fit',color='black')

ax3.legend()
ax3.grid()
ax3.set_xlabel(r'$x_1$')
ax3.set_ylabel(r'$x_2$')

plt.show()
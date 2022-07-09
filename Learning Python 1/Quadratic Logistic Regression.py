import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
import openpyxl
import time


def sigmoid(z):
    return 1/(1+np.exp(-z))


def loss_function(X,y,theta):
    sum = 0
    safety = 10**(-7)
    r,p,sigma = 0,0,0

    for i in range(len(X)):
        sigma = sigmoid(np.dot(theta,X[i]))
        r = np.log(sigma + safety)
        p = np.log(1 - sigma + safety)
        sum += y[i]*r + (1-y[i])*p
    return -sum


def loss_gradient(X,y,theta):
    return X.T @ (sigmoid(X @ theta) - y)


def armijo(X,y,theta_param,gradient):
    t = 1
    alpha = 0.1
    beta = 0.6

    while loss_function(X,y,theta_param - t * gradient) > \
            loss_function(X,y,theta_param) + alpha * t * np.dot(gradient,-gradient):
        t = beta * t
    return t


book = openpyxl.open('Quadratic Logistic Regression Data.xlsx',read_only=True)
sheet = book.active
cells = sheet['A1':'G50']

X = np.zeros((50,6))
y = np.zeros(50)
theta = np.ones(len(X[0]))

k = 0
for x0,x1,x2,x3,x4,x5,y_val in cells:
    X[k] = np.array([x0.value,x1.value,x2.value,x3.value,x4.value,x5.value])
    y[k] = y_val.value
    k += 1

n = 1000
iter_counter = 0
grad = np.zeros(len(X))

iter_history = np.arange(1,n+1,1)
theta_history = np.zeros((n,len(theta)))
grad_norm = np.zeros(n)
loss_history = np.zeros(n)


while iter_counter <= n-1:
    grad = loss_gradient(X,y,theta)
    step_size = armijo(X,y,theta,grad)
    theta = theta - step_size * grad

    theta_history[iter_counter] = theta
    grad_norm[iter_counter] = np.dot(grad,grad) ** 0.5
    loss_history[iter_counter] = loss_function(X,y,theta)
    iter_counter += 1
    print(iter_counter)

print(f"Тета: {theta}")
print(f"Норма градиента: {np.dot(grad,grad) ** 0.5}")
print(f"LOSS: {loss_function(X,y,theta)}")

fig = plt.figure(figsize=(15,8))
gs = GridSpec(1,1,fig)
ax1 = plt.subplot(gs[0,0])
ax1.scatter(X[:25,1],X[:25,2],label='group 0')
ax1.scatter(X[25:,1],X[25:,2],label='group 1')


# implicit equation plot - https://www.tutorialspoint.com/is-it-possible-to-plot-implicit-equations-using-matplotlib
xrange = np.arange(-10,10,0.1)
yrange = np.arange(-20,10,0.1)

xp,yp = np.meshgrid(xrange,yrange)
equation = theta[0] + theta[1]*xp + theta[2]*yp + theta[3]*xp**2 + theta[4]*yp**2 + theta[5]*xp*yp
ax1.contour(xp,yp,equation,[0])
ax1.legend()

plt.show()

fig2 = plt.figure(figsize=(15,8))
gs2 = GridSpec(2,1)

ax2 = plt.subplot(gs2[0,0])
ax2.plot(iter_history,grad_norm,color='blue',ms=5,label=r'$||\nabla L(\theta_k)||_2$')
ax2.set_xlabel('k')
ax2.set_ylabel(r'$||\nabla L(\theta_k)||_2$')
ax2.set_yscale('log')
ax2.legend()

ax3 = plt.subplot(gs2[1,0])
ax3.plot(iter_history,loss_history,color='blue',ms=5,label=r'$L(\theta_k)$')
ax3.set_xlabel('k')
ax3.set_ylabel(r'$L(\theta_k)$')
ax3.set_yscale('log')
ax3.legend()

plt.show()


plt.ion()
matplotlib.rcParams['figure.figsize'] = (15, 8)
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
for i in range(len(theta_history)):
    plt.clf()

    plt.scatter(X[:25, 1], X[:25, 2], label='group 0')
    plt.scatter(X[25:, 1], X[25:, 2], label='group 1')
    plt.legend()
    equation = theta_history[i,0] + theta_history[i,1]*xp + theta_history[i,2]*yp + theta_history[i,3]*xp**2 + \
               theta_history[i,4]*yp**2 + theta_history[i,5] * xp * yp
    plt.contour(xp,yp,equation,[0])

    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.01)

plt.ioff()
plt.show()
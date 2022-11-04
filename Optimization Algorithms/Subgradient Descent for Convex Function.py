import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def f(c,x):
    return np.dot(c,np.abs(x))

def subgradient(c,x):
    return c * np.sign(x)


c = np.array([1,2])
x = np.array([6,10])
grad = subgradient(c,x)

x1_history = np.array([])
x2_history = np.array([])
function_history = np.array([])
min_function = 5.0
max_grad = 0.0
iter_counter = 0
N = 10000
eps = 0.01

R = np.dot(x,x)**0.5

for i in range(1,N+1):
    grad = subgradient(c,x)

    x1_history = np.append(x1_history,x[0])
    x2_history = np.append(x2_history,x[1])
    function_history = np.append(function_history,f(c,x))
    max_grad = max(max_grad,np.dot(grad,grad)**0.5)
    min_function = min(min_function, f(c, x))
    iter_counter = i

    if grad[0]==0 and grad[1]==0:
        break

    step1 = R/(max_grad * (N ** 0.5))
    step2 = eps/(max_grad**2)
    step3 = eps/np.dot(grad,grad)
    step4 = 1.0/i

    x = x - step2*grad
    print(x)

mean_values = np.array([])  # верна только для шага 1
x_mean = np.array([0,0])

for i in range(len(x1_history)):
    x_mean = x_mean + np.array([x1_history[i],x2_history[i]])
    mean_values = np.append(mean_values,f(c,x_mean/(i+1)))


fig1 = plt.figure(figsize=(16,9))
gs1 = GridSpec(2,1,fig1)

ax1 = plt.subplot(gs1[0,0])
ax1.plot(mean_values,label='$f(\\bar{x}_N)$')
ax1.plot([0,N+1],[max_grad * R / (N ** 0.5),max_grad * R / (N ** 0.5)],label='$\\frac{MR}{\sqrt{N}}' + f'\\approx {round((max_grad * R / (N ** 0.5)),3)}$')
ax1.set_xlabel('N')
plt.legend(fontsize=15)

ax2 = plt.subplot(gs1[1,0])
ax2.plot(function_history,label='$f(x_1,x_2)$')
ax2.set_xlabel('N')
ax2.set_ylabel('$f(x_1,x_2)$')
plt.legend(fontsize=15)

plt.show()

fig2 = plt.figure(figsize=(16,9))
ax3 = fig2.add_subplot()

a = np.arange(-11,12,0.1)
b = np.arange(-11,12,0.1)

x1grid,x2grid = np.meshgrid(a,b)
fgrid = c[0] * np.abs(x1grid) + c[1] * abs(x2grid)

ax3.contourf(x1grid,x2grid,fgrid,cmap='magma')
ax3.plot(x1_history,x2_history,markersize=2,color='yellow',marker='o',label='$x_k$')
ax3.legend(facecolor=(0.2,0.2,0.2),fontsize=20,labelcolor='white')
ax3.set_xlabel('$x_1$',fontsize=15)
ax3.set_ylabel('$x_2$',fontsize=15)

plt.show()
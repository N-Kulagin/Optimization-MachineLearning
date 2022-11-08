import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def f(x):
    return np.sin(x)/np.exp(x)


def df(x):
    return np.exp(-x)*(np.cos(x)-np.sin(x))

a = -4
b = abs(a)+1
x = np.zeros(1)
eps = 0.001
i = 0

if f(a)*f(b)<0:
    while abs(a-b)>eps:
        i += 1
        x = np.append(x,0.5*(a+b))
        if f(a)*f(x[i])<0:
            b = x[i]
        else:
            a = x[i]
print(f"Решение методом дихотомии: {x[i]}. Потребовалось итераций: {i}")

i = 0
x_0 = 1.2
x = np.array([x_0])
next = 0

while i < 100:
    if i>=2 and abs( (x[i]-x[i-1])/( 1 - (x[i]-x[i-1])/(x[i-1]-x[i-2])) ) < eps:
        break
    next = x[i] - f(x[i])/df(x[i])
    x = np.append(x,next)
    i+= 1

print(f"Решение методом Ньютона: {x[i]}. Потребовалось итераций: {i}")

print(f"|Pi - x*| = {abs(np.pi-x[i])}")

fig = plt.figure(figsize=(12,8))
k = np.arange(0,5.05,0.05)
plt.plot(k,f(k),k,[0]*len(k))
plt.scatter(x,f(x),color='r')


plt.show()
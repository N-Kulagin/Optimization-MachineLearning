import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Stephen Boyd; Introduction to Applied Linear Algebra (page 412, exercise 18.2)


def objective_function(r,c):
    sum = 0
    for i in range(len(c)):
        sum += c[i] / (1+r)**i
    return sum


def derivative(r,c):
    sum = 0
    t = 1+r
    for i in range(1,len(c)):
        sum += i*c[i]/(t**(i+1))
    return -sum



c = np.array([-1,-1,-1,0.3,0.3,0.3,0.3,0.3,0.6,0.6,0.6,0.6,0.6,0.6])
r = 0
eps = 0.00001
lmbda = 100
criterion = objective_function(r,c) ** 2
iter_counter = 0

r_history = np.array([r])
lmbda_history = np.array([lmbda])
iter_history = np.array([iter_counter])
function_history = np.array([criterion])

while criterion >= eps**2:
    r_next = r - derivative(r,c)*objective_function(r,c)/(lmbda + derivative(r,c)**2)
    tmp = objective_function(r_next,c) ** 2
    if tmp < criterion:
        criterion = tmp
        r = r_next
        lmbda = lmbda * 0.8

    else:
        lmbda = lmbda * 2.0
    iter_counter += 1
    r_history = np.append(r_history,r)
    lmbda_history = np.append(lmbda_history,lmbda)
    iter_history = np.append(iter_history,iter_counter)
    function_history = np.append(function_history,criterion)


print(f"Найдено решение: r = {r}")
print(f"Значение функции: {objective_function(r,c)}")
print(f"Значение производной: {derivative(r,c)}")
print(f"Прошло итераций: {iter_counter}")


fig = plt.figure(figsize=(15,8))

gs = GridSpec(2,1,fig)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])

ax1.plot(iter_history,function_history,label=r'$N(r)^2$',marker='o',color='black')
ax2.plot(iter_history,lmbda_history,label=r'$\lambda$',marker='o',color='black')

ax1.legend()
ax2.legend()

ax2.grid()

plt.show()

fig2 = plt.figure(figsize=(15,8))

gs = GridSpec(1,1,fig2)

ax3 = plt.subplot(gs[0,0])

ax3.plot(iter_history,r_history,label=r'$r$',marker='o',color='black',lw=0.5)

ax3.legend()

plt.show()
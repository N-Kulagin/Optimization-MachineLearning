import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D




def f(A,b,x):
    return 0.5*np.dot(x,A @ x)+np.dot(x,b)


A = np.array([[4,3],[3,4]])
b = np.array([5,-2])
#x = np.ones(len(A))
x = np.array([4,-2])

eps = 0.01
grad = np.ones(len(x))
iter_counter = 0

f_history = np.zeros(1)
grad_history = np.zeros(1)
beta_history = np.zeros(1)
iter_history = np.zeros(1)
x_history = np.array(x[0])
y_history = np.array(x[1])

print(f"Исходное значение функции: {f(A,b,x)}")

while np.dot(grad,grad)**0.5 > eps:

    # метод наискорейшего спуска
    iter_counter += 1
    grad = A @ x + b
    beta = np.dot(grad,grad)/np.dot(A @ grad, grad)
    x = x - beta*grad

    f_history = np.append(f_history, f(A, b, x))
    grad_history = np.append(grad_history, np.dot(grad, grad) ** 0.5)
    beta_history = np.append(beta_history, abs(beta))
    iter_history = np.append(iter_history, iter_counter)
    x_history = np.append(x_history,x[0])
    y_history = np.append(y_history, x[1])


print(f"Найден минимум: {x}.\nПрошло итераций: {iter_counter}\nНорма градиента: {np.dot(grad,grad)**0.5}")
print(f"Значение функции в точке минимума: {f(A,b,x)}")

fig = plt.figure(figsize=(15,8))
gs = GridSpec(2,2,fig)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[1,:])

ax1.plot(iter_history[1:],f_history[1:],
         marker='o',markerfacecolor='b',markeredgecolor='b',markersize=3,color=(1,0.4,0))
ax2.plot(iter_history[1:],grad_history[1:],
         marker='s',markerfacecolor='b',markeredgecolor='b',markersize=5,color=(0,0,1))
ax3.plot(iter_history[1:],beta_history[1:],
         marker='o',markerfacecolor=(0,0,0),markeredgecolor=(0,0,0),markersize=5,color=(1,0,0))

ax1.grid()
ax2.grid()
ax3.grid(which='major',axis='both',color=(0.4,0.4,0.4))
ax3.grid(which='minor',axis='y',ls=':')
ax3.minorticks_on()

ax1.xaxis.set_major_locator(MultipleLocator(base=2))
ax2.xaxis.set_major_locator(MultipleLocator(base=2))
ax3.xaxis.set_major_locator(MultipleLocator(base=1))

ax2.set_ylim([-1,np.max(grad_history)+1])
ax3.set_ylim([0,np.max(beta_history)+0.15])

ax1.set_xlabel('n',fontsize='x-large')
ax1.set_ylabel(r'$F(x_n)$')

ax2.set_xlabel('n',fontsize='x-large')
ax2.set_ylabel(r'$||\nabla F(x_n)||_2$')

ax3.set_xlabel('n',fontsize='x-large')
ax3.set_ylabel(r'$|\beta_n|$')

fig.suptitle('Заголовок')
ax1.set_title('Значения функции')
ax2.set_title('Норма градиента')
ax3.set_title('Шаг')

ax1.legend([r'$F(x_n)$'],edgecolor='black',fontsize='x-large')
ax2.legend([r'$||\nabla F(x_n)||_2$'],edgecolor='black',fontsize='x-large')
ax3.legend([r'$|\beta_n|$'],edgecolor='black',fontsize='x-large')

plt.show()

fig2 = plt.figure(figsize=(15,8))
gs2 = GridSpec(1,2,fig2)

ax4 = fig2.add_subplot(gs2[0,0],projection='3d')
ax5 = fig2.add_subplot(gs2[0,1])


ax4.set_xlabel(r'$x$')
ax4.set_ylabel(r'$y$')
ax4.set_zlabel(r'$\frac{1}{2}x^TAx + b^Tx$')

ax5.set_xlabel(r'$x$')
ax5.set_ylabel(r'$y$')


a1 = np.arange(-8,5,0.1)
b1 = np.arange(-8,10,0.1)

agrid, bgrid = np.meshgrid(a1,b1)
zgrid = 0.5*(4*agrid**2+6*agrid*bgrid+4*bgrid**2)+5*agrid-2*bgrid

ax4.plot_surface(agrid,bgrid,zgrid,cmap='plasma')
ax5.contourf(agrid,bgrid,zgrid,15,cmap='plasma')
ax5.plot(x_history,y_history,linewidth=1,color='r',marker='o',markerfacecolor='white',markersize=3)

#ax4.plot(x_history[:-1],y_history[:-1],(f_history+60)[1:])

ax4.set_title(r'3D поверхность $\frac{1}{2} x^TAx + b^Tx$')
ax5.set_title(r'Метод наискорейшего спуска')

plt.show()
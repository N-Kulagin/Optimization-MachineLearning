import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def gradient_step(x,step):
    return grad_matrix @ x + step * A.T @ b

def gradient(x,step):
    return A.T @ (A @ x - b)

def proximal_operator(x,step,regularization_parameter):
    prod = step * regularization_parameter
    out = [el-prod if el-prod >= 0 else el+prod for el in x]
    return np.array(out)

def f(x):
    p = A @ x - b
    return 0.5*np.dot(p,p) + tau * np.sum(np.abs(x))


# h(x) = f(x) + g(x) = \frac{1}{2}||Ax-b||_2^2 + \tau ||x||_1 \to \min_x
# y_k = x_k-\alpha\nabla f(x_k)
# x_{k+1} = prox_{\alpha g}(y_k) = \arg \min_u(||u||_1 + \frac{1}{2\alpha}||u-y_k||_2^2)

# https://youtu.be/h6-YUCiCx5w

A = np.array([[-1.0,-2,1,2,4,5,6,7,10],[1,1,1,1,1,1,1,1,1]]).T
b = np.array([0.0,-20,1,3,-15,6,6,9,25])
x = np.array([1.0,1])

grad = np.array([0.0,0.0])

tau = 40.0 # 40.0
step = 0.001 # 1.0
N=50
grad_matrix = np.identity(len(x))-step*A.T@A

counter = 0

f_history = np.array([])

# regular proximal gradient descent
while counter < N:
    f_history = np.append(f_history,f(x))
    grad = gradient_step(x,step)
    x = proximal_operator(grad, step, tau)
    print(x)
    counter+=1

# accelerated proximal gradient descent

# x_{k+1} = prox_{\alpha g}(y_k - \alpha \nabla f(y_k))
# y_{k+1} = x_{k+1} + \frac{k}{k+3}(x_{k+1}-x_k)

x = np.array([1.0,1])
x_last = np.copy(x)
y = np.array([0.0,0.0])
grad = np.array([0.0,0.0])
counter = 0
f_history2 = np.array([])

while counter < N:
    f_history2 = np.append(f_history2, f(x))
    if counter == 0:
        grad = gradient(x,step)
        x_last = np.copy(x)
        x = proximal_operator(x-step*grad,step,tau)
    else:
        y = x + counter/(counter + 3) * (x-x_last)
        grad = gradient(y,step)
        x_last = np.copy(x)
        x = proximal_operator(y-step*grad,step,tau)
    counter += 1
    print(x)

print(f"Решение задачи - {1.381, 6.006*10**(-4)}")


fig = plt.figure(figsize=(16,9))
gs = GridSpec(1,1,fig)

ax = plt.subplot(gs[0,0])
ax.plot(f_history, color='blue',marker='o',markevery=1,
         label='Regular proximal gradient descent',markerfacecolor='orange')
ax.plot(f_history2, color='black',marker='X',markevery=1,
         label='Accelerated proximal gradient descent',markerfacecolor='red')
ax.legend(fontsize=15)
plt.title('$\\frac{1}{2}||Ax-b||_2^2 + \\tau ||x||_1, $' + f'$\\tau = {tau}$')
ax.set_xlabel('k')
ax.set_ylabel('$f(x_k)$')
plt.show()
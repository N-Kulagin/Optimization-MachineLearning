import numpy as np

def sum_exp(A,x,b):
    sum = 0
    for i in range(len(A)):
        sum += np.exp(np.dot(A[i],x)+b[i])
    return sum


def grad_sum_exp(A,x,b):
    gradient = np.zeros(len(A[0]))

    for i in range(len(A)):
        gradient = gradient + A[i] * np.exp(np.dot(A[i],x)+b[i])
    return gradient


def backtracking(A,x,b,dir):
    alpha = 0.01
    beta = 0.1
    t = 1
    while (sum_exp(A,x+t*dir,b) > sum_exp(A,x,b) + t*alpha*np.dot(dir,-grad_sum_exp(A,x,b))):
        t = beta*t
    return t

A = np.array([[1,3],[1,-3],[-1,0]])
x = np.array([-1,-3])
b = np.array([-0.1,-0.1,-0.1])

eps = 0.001

rho = 0
step_size = 0

grad_last = grad_sum_exp(A,x,b)
grad_next = np.copy(grad_last)
direction = np.zeros(len(grad_last))
hessian = np.identity(len(x))

s_k = np.zeros(len(x))
y_k = np.zeros(len(grad_last))

I = np.identity(len(x))

iter_counter = 0

while np.dot(grad_last,grad_last)**0.5 > eps:
    direction = -hessian @ grad_last
    step_size = backtracking(A,x,b,direction)
    tmp = x + step_size * direction
    s_k = tmp - x
    x = np.copy(tmp)
    grad_next = grad_sum_exp(A,x,b)
    y_k = grad_next - grad_last
    grad_last = np.copy(grad_next)
    print(f"y_k * s_k = {np.dot(y_k,s_k)}")
    print(f"x = {x}")
    print(f"step = {step_size}")
    rho = 1/np.dot(y_k,s_k)

    # O(n^2) implementation below
    tmp2 = np.tensordot(s_k,y_k @ hessian,axes=0) + np.tensordot(hessian @ y_k,s_k,axes=0)
    tmp3 = np.dot(y_k,hessian @ y_k)
    tmp4 = np.tensordot(s_k,s_k,axes=0)
    hessian = hessian - rho*tmp2 + tmp4*(rho**2 * tmp3 + rho)

    # O(n^3) implementation below
    #hessian = np.matmul(I-rho*np.tensordot(s_k,y_k,axes=0),hessian) @ (I - rho*np.tensordot(y_k,s_k,axes=0)) + rho*np.tensordot(s_k,s_k,axes=0)
    iter_counter += 1

print(f"Найдено решение: {x} за {iter_counter} итераций")

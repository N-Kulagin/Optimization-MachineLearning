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
    beta = 0.6
    t = 1
    while sum_exp(A,x+t*dir,b) > sum_exp(A,x,b) + t*alpha*np.dot(dir,-grad_sum_exp(A,x,b)):
        t = beta*t
    return t


A = np.array([[1,3],[1,-3],[-1,0]])  # Matrix of coefficients for affine functions which are exponentiated
x = np.array([-1,-3])  # starting point -2 -4
b = np.array([-0.1,-0.1,-0.1])  # same as A, but a vector for affine functions

eps = 0.001
step_size = 0
beta_coefficient = 0

direction = -grad_sum_exp(A,x,b)
grad_last = np.copy(-direction)
grad_next = np.ones(len(grad_last))
iter_counter = 0

while np.dot(grad_next,grad_next)**0.5 > eps:
    if iter_counter == 0:
        step_size = backtracking(A,x,b,direction)
        x = x+step_size*direction
        grad_next = grad_sum_exp(A,x,b)
        iter_counter += 1
    else:
        beta_coefficient = np.dot(grad_next,grad_next)/np.dot(grad_last,grad_last) # выбор методом Флетчера-Ривса
        direction = -grad_sum_exp(A,x,b) + beta_coefficient * direction
        step_size = backtracking(A,x,b,direction)
        x = x + step_size * direction
        grad_last = np.copy(grad_next)
        grad_next = grad_sum_exp(A,x,b)
        iter_counter += 1

print(f"Найдено решение: {x} за {iter_counter} итераций")
print(f"Точное решение: {np.array([-np.log(2)/2,0])}")
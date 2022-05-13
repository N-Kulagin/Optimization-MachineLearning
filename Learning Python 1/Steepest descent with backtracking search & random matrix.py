## On the algorithm - see S. Boyd Convex Optimization, page 464

import numpy as np

def f(A,x,b):
    global oracle_f
    oracle_f += 1
    return 0.5*(np.dot(x,A @ x)) + np.dot(b,x)


def gradient(A,x,b):
    global oracle_grad
    oracle_grad += 1
    return A @ x + b

def get_step(x):
    t = 1
    while f(A,x-t*grad,b) > f(A,x,b) + alpha * t * np.dot(grad,-grad):
        t = beta*t
    return t

n = 10
m = 10

A_init = np.random.randint(low=-10,high=11,size=(n,m))
b_init = np.random.randint(low=-10,high=11,size=m)

A = A_init.T @ A_init
x = np.ones(m)
b = A_init.T @ b_init

eps = 0.001
grad_norm = 1
grad = np.zeros(2)
step_length = 0

alpha = 0.1
beta = 0.8

iter_counter = 0
oracle_f = 0
oracle_grad = 0

print(f"Начальная точка: {x}\nЗначение функции: {f(A,x,b)}")

while grad_norm > eps and iter_counter < 10000:
    grad = gradient(A,x,b)
    step_length = get_step(x)
    x = x - step_length*grad
    grad_norm = np.dot(grad,grad)**0.5
    iter_counter += 1


print()
print(f"Получено решение: {x}\nНорма градиента: {grad_norm}")
print(f"Новое значение функции: {f(A,x,b)}")
print(f"Прошло итераций: {iter_counter}.\nОбращения к оракулу: {oracle_f} вызовов функции, {oracle_grad} градиентов.")
print(f"Аналитическое решение: {-np.linalg.inv(A) @ b}")
print(f"Число обусловленности А в 2-норме: {np.linalg.cond(A)}")
print(f"Число обусловленности А_init в 2-норме: {np.linalg.cond(A_init)}")
#print(f"А_init:\n {A_init}")
#print(f"A:\n {A}")

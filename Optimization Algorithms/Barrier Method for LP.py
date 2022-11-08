from functools import reduce

import numpy as np
from operator import add


def f0(c,x):
    return np.dot(c,x)

def phi_grad(x):
    grad = np.zeros(len(A[0]))
    grad += t*c
    for i in range(len(A)):
        grad += 1/(b[i]-np.dot(A[i],x))*A[i]
    return grad

def phi_hessian(x):
    H = np.zeros((len(A[0]),len(A[0])))
    for i in range(len(A)):
        H += 1/(b[i]-np.dot(A[i],x))**2 * np.tensordot(A[i],A[i],axes=0)
    return H


def newton_backtracking():
    step = 1
    alpha = 0.01
    beta = 0.5
    g1 = np.array([F[0,0],F[0,1]])
    a1 = t*c + phi_grad(x+step*dx) + g1 * (nu+step*dnu)
    a2 = t*c + phi_grad(x) + g1 * nu

    while np.dot(a1,a1) ** 0.5 > (1-alpha*step) * np.dot(a2,a2) ** 0.5:
        step = step * beta
        a1 = t * c + phi_grad(x + step * dx) + g1 * (nu + step * dnu)
        a2 = t * c + phi_grad(x) + g1 * nu
    return step



x = np.array([2,5])

A = np.array([[1,1],[-1/3,1],[-1,-1],[-1/10,-1]])
b = np.array([10,5,-4,-1])
F = np.array([[1/2],[1]]).T
d = np.array([6])
c = np.array([-2,-1])
nu = 4.75 # 5

t = 1.1
mu = 50
eps = 0.01
m = len(A)
decrement = 100
step_size = 1
inner_iter_counter = 0

while m/t >= eps:
    while 0.5*decrement > eps:
        hessian = phi_hessian(x)
        KKT_Newton_Matrix = np.block([[hessian,F.T],[F,0]])
        KKT_Right_Side = np.append([t*c + phi_grad(x) + F * nu],0)
        sol = np.linalg.solve(KKT_Newton_Matrix,-KKT_Right_Side)

        dx = sol[:2]
        dnu = sol[2:]
        decrement = np.dot(dx,hessian @ dx)
        step_size = newton_backtracking()

        x = x + step_size*dx
        nu = nu + step_size*dnu
        inner_iter_counter += 1

    decrement = 100
    t = mu*t

print(f"Найдено решение: x = {x}")
print(f"Значение t = {t}")
print(f"Внутренних итераций: {inner_iter_counter}")
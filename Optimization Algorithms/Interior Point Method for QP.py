import numpy as np


def f(x):
    return np.exp(x[0]*x[1]) + x[0]**2 + x[1]**4


def f_grad(x):
    return np.array([x[1]*np.exp(x[0]*x[1]) + 2*x[0], x[0]*np.exp(x[0]*x[1]) + 4*x[1] ** 3])


def f_hessian(x):
    H = np.zeros((2,2))
    k = np.exp(x[0]*x[1])

    H[0,0] = x[1]**2 * k + 2
    H[0,1] = k * (x[0]*x[1] + 1)
    H[1,0] = H[0,1]
    H[1,1] = x[0]**2 * k + 12*x[1]**2
    return H


def c1(x):
    return x[0]**3 - x[1]**2 + 1


def c2(x):
    return - x[0] + x[1] ** 2 - 1


def c1_grad(x):
    return np.array([3*x[0]**2,-2*x[1]])


def c2_grad(x):
    return np.array([-1,2*x[0]])


def c1_hessian(x):
    return np.array([[6*x[0],0],[0,-2]])


def c2_hessian(x):
    return np.array([[0,0],[0,2]])


def lagrange_hessian(x,lamb):
    return f_hessian(x) + lamb[0]*c1_hessian(x) + lamb[1] * c2_hessian(x)


def quad_approx(p,x):
    return f(x) + np.dot(f_grad(x),p) + np.dot(p,lagrange_hessian(x) @ p)

def constr_jacobian(x): # not transposed
    A = np.zeros((2,2))
    A[0] = c1_grad(x)
    A[1] = c2_grad(x)
    return A


#x = np.ones(2)
#x = np.array([-2,0])
x = np.array([0,-2])
l = np.ones(2)

x_last = np.zeros(len(x))
l_last = np.zeros(len(l))

eps = 0.01

k = 2
g = 2
iter_counter = 0

while k > eps:
    L_hessian = lagrange_hessian(x, l)
    A = constr_jacobian(x)
    function_grad = f_grad(x)
    function_val = f(x)
    constr_val = np.array([c1(x), c2(x)])

    newton_matrix = np.block([[L_hessian, -A.T], [A, np.zeros((2, 2))]])
    sol = np.linalg.solve(newton_matrix,np.block([-function_grad + A.T @ l,-constr_val]))

    x_last = np.copy(x)
    x = x + sol[:2]

    l_last = np.copy(l)
    l = l + sol[2:4]

    k = np.dot(x - x_last, x - x_last) ** 0.5

    iter_counter += 1


print(f"Найдено решение: {x}")
print(f"Лямбда: {l}")
print(f"Прошло итераций: {iter_counter}")
print(g)
print(k)




import numpy as np

def quadratic(A,x,b):
    return 0.5*(np.dot(x,A @ x)) + np.dot(b,x)


def quadratic_grad(A,x,b):
    return A @ x + b

def quadratic_SD_step_size(A,grad):
    return np.dot(grad, grad) / np.dot(A @ grad, grad)


def quadratic_CG_step_size(res_last, dir, ad):
    return np.dot(res_last,res_last)/np.dot(dir,ad)

def armijo(A,x,b,direction,quad=True):
    alpha = 0.01
    beta = 0.6
    t = 1  # step length
    if quad==True:
        while quadratic(A,x+t*direction,b) > quadratic(A,x,b) + alpha*t*np.dot(quadratic_grad(A,x,b),direction):
            t = beta*t
        return t
    return 0

def test_quadratic_steepest_descent(A,b,armijo_rule=False):
    x = np.ones(len(A))
    grad = np.ones(len(x))
    eps = 0.001
    step_size = 0
    iter_counter = 0

    while np.dot(grad, grad) ** 0.5 > eps and iter_counter < 10000:
        grad = quadratic_grad(A, x, b)
        if armijo_rule == False:
            step_size = quadratic_SD_step_size(A, grad)
        else:
            step_size = armijo(A,x,b,-grad)
        x = x - step_size * grad
        iter_counter += 1

    print(f"SD: Найдено решение {x} за {iter_counter} итераций")


def test_quadratic_conjugate_gradient(A,b,armijo_rule=False):
    x = np.ones(len(A))
    direction = -quadratic_grad(A, x, b)
    residual_next = np.copy(direction)
    residual_last = np.zeros(len(residual_next))
    eps = 0.001
    step_size = 0
    beta_coefficient = 0
    iter_counter = 0

    while np.dot(direction, direction) ** 0.5 > eps and iter_counter < 10000:
        residual_last = np.copy(residual_next)
        Ad = A @ direction
        if armijo_rule == False:
            step_size = quadratic_CG_step_size(residual_last, direction,Ad)
        else:
            step_size = armijo(A,x,b,direction)
        x = x + step_size * direction
        residual_next = residual_last - step_size * Ad
        beta_coefficient = np.dot(residual_next, residual_next) / np.dot(residual_last, residual_last)
        direction = residual_next + beta_coefficient * direction
        iter_counter += 1

    print(f"CG: Найдено решение {x} за {iter_counter} итераций")


A = np.array([[4,3],[3,4]])
x = np.ones(len(A))
b = np.array([5,-2]) # квадратичная задача

test_quadratic_conjugate_gradient(A,b)
test_quadratic_steepest_descent(A,b)
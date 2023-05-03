import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def objective_function(x):
    return np.log(np.sum(np.exp(x)))-np.sum(np.log(x))

def model_function(f,gradient,hessian,p):
    return f + np.dot(gradient,p) + np.dot(p, hessian @ p)/2.0

def solve_quadratic(p_u,p_b,delta):
    u = p_u
    v = p_b - p_u

    arr = np.array([np.dot(v,v), np.dot(u,v), np.dot(u,u)])
    sol = np.roots([arr[0], 2 * (arr[1]-1), arr[0]-2*arr[1]+arr[2]-delta**2])
    for i in range(1,len(sol)+1):
        if 0 <= sol[-i] <= 1:
            return sol[-i]*p_u
        if 1 <= sol[-i] <= 2:
            return p_u * (sol[-i]-1) * v
    print(f"Ошибка поиска корней! =( Найденные корни: {sol}")

def gradient(x):
    z = np.exp(x)
    return z/np.sum(z)-1.0/x

def hessian(x):
    z = np.exp(x)
    sum = np.sum(z)
    return np.diag(z)/sum - np.tensordot(z,z,axes=0)/(sum)**2.0 + np.diag(1.0/x**2)


x = np.array([0.1,0.1])
next_step = np.zeros(len(x))
max_radius = 1.1
radius = 0.5
eta = 0.2
rho = 0.0
eps = 0.001

for i in range(100):
    H, g = hessian(x),gradient(x)
    next_step = np.linalg.solve(H,-g)
    obj = objective_function(x)

    if np.dot(next_step,next_step)**0.5 > radius+eps:
        next_step = solve_quadratic(-g * np.dot(g,g)/np.dot(g, H @ g),next_step,radius)

    rho = ( objective_function(x)-objective_function(x+next_step) ) / ( model_function(obj,g,H,np.zeros(len(x))) - model_function(obj,g,H,next_step) )

    if rho < 0.25:
        radius = radius/4.0
    else:
        if rho > 0.75 and radius - eps <= np.dot(next_step,next_step)**0.5 <= radius + eps:
            radius = min(2*radius,max_radius)

    if rho > eta:
        x = x + next_step
    print(x, radius)

print(np.dot(gradient(np.array([2.0,2.0])),gradient(np.array([2.0,2.0])))**0.5)
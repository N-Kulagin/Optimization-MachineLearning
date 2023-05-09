import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time


def f(X):  # objective function to be minimized, a quadratic
    if X.ndim == 2:
        F = np.array([0.5 * np.dot(y, A @ y) - np.dot(b,y) for y in X])
        return F
    else:
        return 0.5 * np.dot(X, A @ X) - np.dot(b,X)


def second_worst(Y,best_index,worst_index):  # find second worst objective value point in an array
    found = best_index
    for i in range(len(Y)):
        if Y[best_index] < Y[i] < Y[worst_index] and Y[i] >= Y[found]:
            found = i
    return found


def center_of_gravity(Y,worst_index):  # find center of gravity of the simplex
    sum = np.sum(Y,axis=0)
    avg = (sum - Y[worst_index])/(len(Y)-1)
    return avg


def convergence_criterion(center,f_values):  # stop iterations when convergence criterion < epsilon
    f_center = f(center)
    s = (np.sum((f_values - f_center)**2)/(n+1))**0.5
    return s


def reduction(Y,best_index):  # make simplex smaller, but keep the best point in place
    for i in range(len(Y)):
        Y[i] = Y[best_index] + 0.5 * (Y[i] - Y[best_index])
    return Y

# https://youtu.be/35pFgsZIoUw
# https://youtu.be/6Cisbqo-x1Q
# https://youtu.be/RvDxAtYlMHE


starting_point = bool(input("Welcome to Nelder-Mead! "
                            "\nPlease press Enter to select default starting simplex or "
                            "enter something to choose harder starting simplex: "))

n = 2  # dimension of the space
k = 1000  # number of planned iterations (used for plotting)

# objective function parameters
A = np.array([[5.0,3.0],[3.0,5.0]])
b = np.array([8.0,8.0])

# optimal value of the objective
f_optimal_value = -8.0

# two starting simplices to choose from
X1 = np.array([[-1.0,0,-2.0],[-2.0,0,0]]).T
X2 = np.array([[1.0,2,7.0],[-1.0,-2,2.0]]).T
X = X1 * (1.0 - starting_point) + X2 * starting_point

# default values of the algorithm constants
constants = np.array([1.0,0.5,2.0,0.001])
input_text = np.array(["Please enter α value, the reflection coefficient. \nTypical values are α = 1 (default). \n",
                       "Please enter β value, the squeeze coefficient. \nTypical values are 0.4 <= β <= 0.6. \nDefault is β = 0.5\n",
                       "Please enter γ value, the stretching coefficient. \nTypical values are 2.8 <= γ <= 3.0. \nDefault is γ = 2.0 \n",
                       "Please enter ε value, the desired accuracy. \nTypical values are 0.001 <= ε <= 0.1. \nDefault is ε = 0.001 \n"])
const_index = 0
while (const_index != 4):
    try:
        constants[const_index] = float(input(input_text[const_index]) or constants[const_index])
        const_index += 1
    except ValueError:
        continue

alpha = constants[0]  # α = 1, reflection coefficient
beta = constants[1]  # 0.4 <= β <= 0.6, squeeze coefficient
gamma = constants[2]  # 2.8 <= γ <= 3.0, stretching coefficient
eps = constants[3] * (constants[3] > 0) + 0.001 * (constants[3] <= 0)  # ε - desired accuracy

sigma = 1.0  # convergence criterion value
iter_counter = 0  # iteration counter

# store information for plots
triangles = np.zeros((k,3,2))
sigma_history = np.array([])
f_history = np.array([])

while True:  # repeat until convergence criterion is satisfied
    triangles[iter_counter] = X

    # evaluate function at vertices and find worst, best, second worst points
    F = f(X)
    i_worst, i_best = np.argmax(F), np.argmin(F)
    i_preworst = second_worst(F, i_best, i_worst)

    # compute center of the simplex, evaluate convergence criterion sigma
    center = center_of_gravity(X, i_worst)
    sigma = convergence_criterion(center, F)

    sigma_history = np.append(sigma_history,sigma)
    f_history = np.append(f_history,F[i_best]-f_optimal_value)

    iter_counter += 1
    print(fr"{iter_counter}) Convergence criterion σ: {round(sigma,5)}, best point - {X[i_best]}")

    if sigma < eps:
        print(f"\nFound solution: {X[i_best]} with the objective value {f(X[i_best])}")
        break

    # reflect the best point, try stretching the reflection, if succeeds pick stretch otherwise pick reflection
    reflection = center + alpha * (center - X[i_worst])
    tmp_reflection = f(reflection)
    if tmp_reflection <= F[i_best]:
        stretch = center + gamma * (reflection - center)
        if f(stretch) < F[i_best]:
            X[i_worst] = stretch
        else:
            X[i_worst] = reflection

    # reflection better than worst point, but worse than second worst
    # squeeze the simplex relative to the worst point, moves worst point closer to best ones
    elif F[i_preworst] < tmp_reflection <= F[i_worst]:
        squeeze = center + beta * (X[i_worst] - center)
        X[i_worst] = squeeze

    # reflection better than second worst
    elif tmp_reflection <= F[i_preworst]:
        X[i_worst] = reflection

    # reflection failed, reduce the simplex in size and keep best point
    elif tmp_reflection >= F[i_worst]:
        X = reduction(X,i_best)

print(f"Iterations have passed: {iter_counter}")
print(f"Coordinates of simplex vertices: \n{X}")

# plot code
u,v = np.arange(-3,5.1,0.1), np.arange(-3,5.1,0.1)
xgrid,ygrid = np.meshgrid(u,v)
fgrid = np.zeros((len(u),len(u)))
levels = np.arange(-10,20,0.1)

for i in range(len(u)):
    for j in range(len(u)):
        fgrid[i,j] = f(np.array([u[i],v[j]]))


plt.rcParams["figure.figsize"] = [16, 9]

fig = plt.figure()
gs = GridSpec(1,1,fig)

ax = plt.subplot(gs[0,0])
ax.plot(sigma_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$\sigma(x_k) = \sqrt{\sum_{i=1}^3(f(x_i)-f(x_c))^2}$')
ax.plot(f_history,marker='o',markerfacecolor='black',markeredgecolor='black',color=((0.9,0.4,0.0)),lw=2,label='$f(x_k) - f(x^*) = \\frac{1}{2}x^TAx - b^Tx - f(x^*)$')
ax.legend(fontsize=15,loc='upper right')
ax.set_xlabel('$k$')
ax.set_ylabel('$\sigma(x_k)$')
ax.set_yscale("log",base=10)
ax.set_title("Nelder-Mead convergence")

plt.show()


fig2, ax2 = plt.subplots()

# animation
while True:

    plt.ion()
    plt.show()
    for iter in range(iter_counter):
        plt.clf()
        plt.contourf(xgrid,ygrid,fgrid,levels)
        U,V = np.append(triangles[iter,:,0],triangles[iter,0,0]), np.append(triangles[iter,:,1],triangles[iter,0,1])
        plt.plot(U,V,marker='o',markersize=-np.log(iter+0.1)+6,markerfacecolor='orange',markeredgecolor='blue',color='blue')
        plt.plot(1,1,marker='*',markersize=-np.log(iter+0.1)+12,markerfacecolor='yellow',markeredgecolor='red',label=r'$x^*$ minimizer',linestyle="None")
        plt.legend(fontsize=20)
        plt.gcf().canvas.flush_events()
        fig2.canvas.draw()
        time.sleep(0.5)
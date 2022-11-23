import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def get_hessian(x):
    H = np.zeros((len(x),len(x)))
    inverse = np.linalg.inv(get_constraint(x))

    H[0,0] = np.trace(F1 @ inverse @ F1 @ inverse)
    H[0,1] = np.trace(F1 @ inverse @ F2 @ inverse)
    H[1,0] = np.trace(F2 @ inverse @ F1 @ inverse)
    H[0,1] = np.trace(F2 @ inverse @ F2 @ inverse)

    return H

def get_constraint(x):
    return -(x[0] * F1 + x[1] * F2 + G)

def get_gradient(x):
    grad = np.zeros(len(x))
    inverse = np.linalg.inv(get_constraint(x))

    grad[0] = np.trace(F1 @ inverse) + t * c[0]
    grad[0] = np.trace(F2 @ inverse) + t * c[1]

    return grad

# c^Tx - \log \det (\sum_{i=1}^n-x_iF_i - G) \to \min, \\\\ x_i \in \mathbb{R}, \ F_i,G \in \mathbb{S}^p
# (t\nabla f(x) + \nabla \phi(x))_k = tr (F_k \cdot (\sum_{i=1}^n-x_iF_i - G)^{-1}) = tr(F_k \cdot F(x)^{-1} )
# (t\nabla^2 f(x) + \nabla^2\phi(x))_{kp} = tr(F_k \cdot F(x)^{-1}F_p F(x)^{-1})
# Stephen Boyd Convex Optimization pdf page 616, book page 602

F1 = np.array([[-2.0,0],[0,-1]])
F2 = np.array([[-2.0,1],[1,-4]])
G = np.array([[-1.0,2],[2,-1]])
c = np.array([2.0,5])

x = np.array([1.0,2]) # 1 2, 3 4, 8 5 - feasible initial values
t = 1.0
multiplier = 1.2
duality_gap = len(F1)/t
decrement = 5.0
eps = 0.001
step = 0.01

eigval1_history, eigval2_history,  f_history, gap_history = np.array([]), np.array([]), np.array([]), np.array([])

while duality_gap >= eps:
    while decrement/2.0 >= eps:
        l = np.linalg.eigvals(-get_constraint(x))
        eigval1_history = np.append(eigval1_history,l[0])
        eigval2_history = np.append(eigval2_history,l[1])
        f_history = np.append(f_history,np.dot(c,x))
        gap_history = np.append(gap_history,duality_gap)


        hessian = get_hessian(x)
        solution = np.linalg.solve(hessian + 0.01*np.identity(2),-get_gradient(x))

        x = x + step * solution
        decrement = np.dot(solution,hessian @ solution)

        print(solution)
        print(x)
        print(decrement)
        print()
    decrement = 5.0
    t = t * multiplier
    duality_gap = len(F1)/t

print(f"Solution:{x}")
print(f"Duality gap: {duality_gap}")
print(f"t = {t}")
print(get_constraint(x))


fig = plt.figure(figsize=(16,9))
gs = GridSpec(2,1,fig)

ax1 = plt.subplot(gs[0,0])
ax1.plot(eigval1_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$\lambda_1$')
ax1.plot(eigval2_history,marker='o',markerfacecolor='orange',markeredgecolor='black',color='orange',lw=2,label='$\lambda_2$')
ax1.set_xlabel('Iteration number')
ax1.set_ylabel('$\lambda_k$')
ax1.legend(fontsize=20)

ax2 = plt.subplot(gs[1,0])
ax2.plot(f_history,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$c^Tx_k$')
ax2.plot(gap_history,marker='o',markerfacecolor='orange',markeredgecolor='black',color='orange',lw=2,label='$\\frac{p}{t_k}$')
ax2.set_xlabel('Iteration number')
ax2.set_ylabel('Function value and duality gap')
ax2.legend(fontsize=20)

plt.show()

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


# completing the matrix such that it's positive semi-definite by projecting onto PSD cone
# Stephen Boyd, Convex Optimization 2
# https://youtu.be/B51GgGCHBRk

def project_PSD(lamb):
    for i in range(len(lamb)):
        lamb[i] = max(0.0,lamb[i])
    return lamb

X_init = np.array([
              [1.0,2,1,0,0,0],
              [2.0,0,2,0,0,1],
              [1.0,2,2,3,0,0],
              [0.0,0,3,2,1,0],
              [0.0,0,0,1,0,2],
              [0.0,1,0,0,2,0]])
X = np.copy(X_init)

map = np.array([
              [1.0,1,1,0,0,0],
              [1.0,0,1,0,0,1],
              [1.0,1,1,1,0,0],
              [0.0,0,1,1,1,0],
              [0.0,0,0,1,0,1],
              [0.0,1,0,0,1,0]])

lamb, P = np.linalg.eig(X)
N = 80

frobenius_norm_history = np.array([])
lamb_history = np.zeros((N,len(lamb)))

fig1 = plt.figure(figsize=(16,9))
ax1 = plt.subplot()
colors=np.array(['blue','orange','black','green','red','purple'])


for i in range(N):
    lamb, P = np.linalg.eig(X)
    lamb_history[i] = lamb
    lamb = project_PSD(lamb)

    X_prev = np.copy(X)
    X = P @ np.diag(lamb) @ P.T

    frobenius_norm_history = np.append(frobenius_norm_history,np.linalg.norm(X-X_prev,ord='fro'))
    for j in range(len(map[0])):
        for k in range(len(map[0])):
            if map[j,k] == 1.0:
                X[j,k] = X_init[j,k]

for i in range(len(lamb)):
    ax1.plot(lamb_history.T[i],label=f'$\\lambda_{i+1}$',color=colors[i])
ax1.set_xlabel('Iteration number')
ax1.set_ylabel('Eigenvalue')
ax1.legend(fontsize=10)
ax1.grid()
plt.show()

fig2 = plt.figure(figsize=(16,9))
ax2 = plt.subplot()

ax2.plot(frobenius_norm_history,color='blue',marker='o',markerfacecolor='orange',label='$||X^{k+1}-X^{k}||_F$')
ax2.set_xlabel('k')
ax2.set_yscale('log')
ax2.legend(fontsize=15)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Stephen Boyd; Introduction to Applied Linear Algebra (page 379, exercise 17.6)

A = np.array([[0.99,0.03,-0.02,-0.32],
              [0.01,0.47,4.70,0],
              [0.02,-0.06,0.4,0],
              [0.01,-0.04,0.72,0.99]])

B = np.array([[0.01,0.99],
              [-3.44,1.66],
              [-0.83,0.44],
              [-0.47,0.25]])

x = np.array([0.0,0.0,0.0,1.0])
u = np.array([2,2])

x0_history = np.array([x[0]])
x1_history = np.array([x[1]])
x2_history = np.array([x[2]])
x3_history = np.array([x[3]])

t_history = np.array([0])


for t in range(1,301): # 121
    x = A @ x
    x0_history = np.append(x0_history,[x[0]])
    x1_history = np.append(x1_history,[x[1]])
    x2_history = np.append(x2_history,[x[2]])
    x3_history = np.append(x3_history,[x[3]])
    t_history = np.append(t_history,t)


fig = plt.figure(figsize=(15,8))
gs = GridSpec(2,2,fig)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[1,0])
ax4 = plt.subplot(gs[1,1])

ax1.plot(t_history,x0_history,label=r'$(x_t)_1$',color='black')
ax2.plot(t_history,x1_history,label=r'$(x_t)_2$',color='black')
ax3.plot(t_history,x2_history,label=r'$(x_t)_3$',color='black')
ax4.plot(t_history,x3_history,label=r'$(x_t)_4$',color='black')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$(x_t)_1$')
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$(x_t)_2$')
ax3.set_xlabel(r'$t$')
ax3.set_ylabel(r'$(x_t)_3$')
ax4.set_xlabel(r'$t$')
ax4.set_ylabel(r'$(x_t)_4$')

plt.show()
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import math


i = 27

freq = np.array([math.pi/16 * n for n in range(1,33)])
t = np.array([n for n in range(1,50)])
x = np.cos(t)
y = np.cos(freq[i]*t)

print(freq[i])


fig = plt.figure(figsize=(16,9))
gs = GridSpec(1,1,fig)

ax = plt.subplot(gs[0,0])

ax = plt.subplot(gs[0,0])
ax.plot(t,x,marker='o',markerfacecolor='orange',markeredgecolor='blue',color='blue',lw=2,label='$x(t)$',markevery=1)
ax.plot(t,y,marker='o',markerfacecolor='orange',markeredgecolor='black',color='black',lw=2,label=f'$x({round(freq[i],2)}t)$',markevery=1)
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
ax.set_ylim(ymin=-2,ymax=2)
ax.legend(fontsize=20)

plt.tight_layout(pad=3)
plt.grid(color='green',lw=1)

fig.suptitle('Figure Subtitle')

plt.show()
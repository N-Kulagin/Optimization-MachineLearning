import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullLocator

fig,ax = plt.subplots(2,2)

ax[0,0].plot(np.arange(0,20,1),linewidth=3,linestyle=':',marker='s',markerfacecolor='w',color='y')
ax[0,0].grid()
ax[0,0].set(xlim=(-3,15),ylim=(0,20))

ax[0,1].plot(np.arange(5,0,-0.5),linestyle='-.',marker='*')
ax[0,1].grid()
ax[0,1].set(xlim=(-10,10),ylim=(-5,15))

fig.set_size_inches(12,9)
fig.set_facecolor('#eee')

plt.show()

fig2 = plt.figure(figsize=(10,8))
gs = GridSpec(2,3,fig2)

ax1 = plt.subplot(gs[0,0])
ax1.plot(np.arange(0,20,1),linestyle='-.')

ax2 = plt.subplot(gs[1,0:2])
ax2.plot(np.random.random(10),linewidth=3)
ax2.set(xlim=(1,6),ylim=(0,0.8))
ax2.grid()

ax2.xaxis.set_major_locator(NullLocator())

ax3 = plt.subplot(gs[0:,2])
ax3.plot(np.random.random(5))

plt.show()
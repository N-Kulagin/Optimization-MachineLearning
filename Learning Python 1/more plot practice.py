import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

def f(x):
    return np.sinh(x)/(10*np.pi)

def g(x):
    return np.exp(np.cos(x))-1

x = np.arange(-5,5.05,0.05)

fig = plt.figure(figsize=(18,10))
gs = GridSpec(3,2,fig)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[1,:])
ax4 = plt.subplot(gs[2,0])
ax5 = plt.subplot(gs[2,1])

ax1.plot(x,f(x))
ax2.plot(x,g(x))
ax3.plot(x,f(x),label=r'$\frac{sinh(x)}{10\pi}$')
ax3.plot(x,g(x),label=r'$e^{cos(x)}$')
ax4.plot(x,f(x),linestyle=':',linewidth='3',color='blue')
ax4.plot([-1,0,2],f([-1,0,2]),marker='o',ms=10,linestyle='',markerfacecolor='yellow')
ax5.plot(x,g(x))
ax5.grid(which='both')
ax5.minorticks_on()

ax1.xaxis.set_major_locator(MultipleLocator(base=1))
ax2.xaxis.set_major_locator(MultipleLocator(base=1))
ax3.xaxis.set_major_locator(MultipleLocator(base=0.5))

ax3.legend(edgecolor='black',fontsize='x-large')

plt.show()
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import random


def ideal_function(x,a,w,p,b,t,c):
    return a * np.sin(w/10.0 * x + p) + b * np.cos(t/8.0 * x + c) + 10.0 * np.log10(x)


def noisy_function(x,a,w,p,b,t,c,mean,variance):
    noise = np.zeros(6)
    for i in range(len(noise)):
        noise[i] = random.gauss(mean,variance)
    print(noise)
    return (a + noise[0]) * np.sin((w/10.0 + noise[1]) * x + p + noise[2]) + \
           (b + noise[3]) * np.cos((t/8.0 + noise[4]) * x + c + noise[5]) + 10.0 * np.log10(x)

def convolve(data,kernel):
    conv_result = np.zeros(len(data)+len(kernel)-1)

    for n in range(len(conv_result)):
        for i in range(len(data)):
            for j in range(len(kernel)):
                if i+j==n:
                    conv_result[n] += data[i]*kernel[j]
    return conv_result


a,w,p,b,t,c = 20.0,2.0,8.0,2.0,20.0,5.0
mean, variance = 1.0, 6.0

x = np.arange(1,500,1)
y = ideal_function(x,a,w,p,b,t,c)
kernel_length = 11
convolution_kernel = np.full(kernel_length,1/kernel_length)


y_noisy = np.zeros(len(x))
for i in range(len(y_noisy)):
    y_noisy[i] = ideal_function(x[i],a,w,p,b,t,c) + random.gauss(mean,variance)

convolved_y = convolve(y_noisy,convolution_kernel)
margin = (len(convolution_kernel) - 1) // 2

print(len(y), len(convolution_kernel), len(x),len(convolved_y))
print(margin)
print(convolution_kernel)


fig = plt.figure(figsize=(18,9))
gs = GridSpec(4,1,fig)

ax1 = plt.subplot(gs[0,0])
ax1.plot(x,y,color='blue',lw=0.5)

ax2 = plt.subplot(gs[1,0])
ax2.plot(x,y_noisy,color='purple',lw=0.5)

ax3 = plt.subplot(gs[2,0])
ax3.plot(x,convolved_y[margin:len(convolved_y)-margin],color='orange',lw=1)

ax4 = plt.subplot(gs[3,0])
ax4.plot(x,y_noisy,color='purple',lw=0.5)
ax4.plot(x,convolved_y[margin:len(convolved_y)-margin],color='orange',lw=1)

plt.tight_layout(h_pad=3,w_pad=20)

plt.show()
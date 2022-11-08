import numpy as np
import matplotlib.pyplot as plt

m = 10 # number of equations
n = 2 # number of unknowns
eps = 0.001 # desired error
x_vec = np.array([0,0.1,0.7,0.8,1.1,1.6,1.9,2,3.4,4.5])
y_vec = np.array([3.2,3.4,4.6,4.4,5,5.5,6.3,6.8,10.6,12.6])

x = np.ones(n)
#A = np.array([[0,3.2],[0.1,3.4],[0.7,4.6],[0.8,4.4],[1.1,5],[1.6,5.5],[1.9,6.3],[2,6.8],[3.4,10.6],[4.5,12.6]])
A = np.array([[0,1],[0.1,1],[0.7,1],[0.8,1],[1.1,1],[1.6,1],[1.9,1],[2,1],[3.4,1],[4.5,1]])
b = np.array([3.2,3.4,4.6,4.4,5,5.5,6.3,6.8,10.6,12.6])

#print(A)
#print(b)

b_new = np.matmul(A.T,b) # A^T*b = ~b
A_new = np.matmul(A.T,A) # A^T*A = ~A
grad = np.ones(m) # градиент
iter_counter = 0

while np.dot(grad,grad)**0.5 > eps:
    iter_counter += 1
    grad = np.matmul(A_new, x) - b_new
    beta = np.dot(grad,grad)/np.dot(np.matmul(A_new,grad),grad) # шаг по методу наискорейшего спуска
    x = x - beta*grad

print(f"Найден минимум: {x}.\nПрошло итераций: {iter_counter}\nНорма градиента: {np.dot(grad,grad)**0.5}")

plt.plot(x_vec,y_vec,x_vec,np.matmul(A,x))
plt.show()
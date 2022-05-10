import numpy as np
import matplotlib.pyplot as plt
import time

def f(x,k=2,c=3, error=True):
    noise = np.zeros(len(x))
    if error == True:
        for i in range(len(x)):
            noise[i] = (-1) ** i * np.random.rand() * 30
    return k*x+c+noise

m = 100 # number of equations
n = 2 # number of unknowns
eps = 0.001 # desired error

x = np.zeros(2)

x1_vec = np.linspace(0,m,m)
x2_vec = np.ones(len(x1_vec))

A = np.column_stack((x1_vec,x2_vec)) # матрица из двух векторов столбцов
b = f(x1_vec) # правая часть Ax = b

A_new = A.T @ A
b_new = A.T @ b

grad = np.ones(m) # вектор градиента
beta = 0 # шаг градиентного спуска
iter_counter = 0 # счётчик итераций

function_value = np.zeros(1)
function_value[0] = np.dot(A @ x - b, A @ x - b)
iter_number = np.zeros(1)

plt.ion()
while np.dot(grad,grad)**0.5 > eps:
    # градиентный спуск
    iter_counter += 1
    grad = A_new @ x - b_new
    beta = np.dot(grad,grad)/np.dot(A_new @ grad,grad)
    x = x - beta*grad # следующее приближение

    # сбор данных об итерациях
    function_value = np.append(function_value,np.dot(A @ x - b, A @ x - b))
    iter_number = np.append(iter_number,iter_counter)

    # динамическое обновление графика
    plt.clf()
    plt.scatter(x1_vec, b, color="#FF8C00")
    plt.plot(x1_vec, f(x1_vec, x[0], x[1], False))
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.5)

plt.ioff()
plt.show()
print(f"Найден минимум: {x}\nПотребовалось итераций: {iter_counter}\nНорма градиента: {np.dot(grad,grad)**0.5}")

plt.plot(iter_number,function_value)
plt.show()
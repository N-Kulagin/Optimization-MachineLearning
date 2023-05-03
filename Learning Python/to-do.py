"""
Запрограммировать скалярное и тензорное умножение векторов, вычисление бесконечной, первой и второй норм вектора.
Запрограммировать метод Гаусса решения СЛАУ, градиентный метод решения СЛАУ (критерий остановки - норма градиента).
"""

"""
i = 5
j = 4

zero_matrix = [[0] * j] * i
print(zero_matrix)
"""


def dot(x, y):
    result = 0
    for i in range(0, len(x)):
        result += x[i] * y[i]
    return result

def tensor_product(x,y): #vector dimensions are equal
    result = [[0] * len(x)] * len(y)
    for i in range(0,len(y)):
        result[i] = [k * x[i] for k in y]
    return result


def infinity_norm(x):
    norm = 0
    for i in range(0,len(x)):
        if abs(x[i]) > norm:
            norm = abs(x[i])
    return norm


def l1_norm(x):
    norm = 0
    for i in range(0,len(x)):
        norm += abs(x[i])
    return norm

def l2_norm(x):
    norm = 0
    for i in range(0,len(x)):
        norm += x[i]**2
    return norm**(1/2)

def gauss(A,b):
    return 0

A = [[1,2,1],[2,1,2],[3,3,1]]
b = [8,10,12]
print(gauss(A,b))
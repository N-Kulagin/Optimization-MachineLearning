import numpy as np
import openpyxl as xl
from matplotlib import pyplot as plt
from random import *
import time


def k_means_function(x_array,z_array,assignment_array):
    summation = 0
    for i in range(len(x_array)):
        vec = x_array[i] - z_array[assignment_array[i]]
        summation += np.dot(vec,vec)
    return summation

book = xl.open('k-means input data.xlsx',read_only=True)
sheet = book.active

A = np.empty((30,2))

for row in range(1,30+1):
    b = np.array([sheet[row][1].value,sheet[row][2].value])
    A[row-1] = b

k = 3
eps = 1
seed(5)
repeats = set()  # защита от повторов
Z = np.empty((k,2))

for i in range(k):  # ининциализация векторов Z
    rnd = randint(0,len(A)-1)
    while rnd in repeats:
        rnd = randint(0,len(A)-1)
    repeats.add(rnd)
    Z[i] = A[rnd] + random()
print(f"Начальные векторы Z: {Z}")

c_vector_last = np.full(len(A),0)
c_vector = np.full(len(A),-1)

iter_counter = 0
f_val1 = 0.0
f_val2 = 1.0
f_val_array = np.array([])

# not np.array_equal(c_vector_last,c_vector) - один из возможных критериев останова
# abs(f_val2 - f_val1) >= eps - ещё один критерий останова
while True:
    if iter_counter >= 1:
        f_val1 = k_means_function(A,Z,c_vector)
    distances = np.zeros(k)
    c_vector_last = np.copy(c_vector)
    c_vector = np.full(len(A),-1)
    for i in range(len(A)):  # распределение векторов x по группам z
        for j in range(len(Z)):
                vector = A[i] - Z[j]
                distances[j] = np.dot(vector,vector)
        c_vector[i] = np.argmin(distances)

    if iter_counter >= 1:
        f_val2 = k_means_function(A,Z,c_vector)
        f_val_array = np.append(f_val_array,f_val2)

    if abs(f_val2 - f_val1) < eps:
        break

    group_list = np.array([set() for _ in range(k)])  # создание списка из групп

    # т. к. c_vector содержит ровно k различных значений от 0 до k-1 (k=4, c_vector = [0,1,2,3,1,0,2,1,3]),
    # то выбирая i-тое значение вектора - получаем номер множества (множества упорядочены) и в него присваем индекс i
    for i in range(len(c_vector)):  # выявление к какой группе принадлежит каждый вектор x
        group_list[c_vector[i]].add(i)

    sum_x = np.zeros(len(A[0]))
    for j in range(k):  # вычисление новых векторов z
        for el in group_list[j]:
            sum_x += A[el]
        Z[j] = 1/len(group_list[j]) * sum_x
        sum_x = np.zeros(len(A[0]))
    iter_counter += 1


print()
print(f"Прошло итераций: {iter_counter}")
print(f"История значений функции: {f_val_array}")
print(f"Итоговое разбиение: {c_vector}")
print(f"Итоговые векторы Z: {Z}")
print(f"Группы: {group_list}")

def detect(x_vectors,group):
    x = np.array([])
    y = np.array([])

    for i in range(len(group)):
        x = np.append(x,x_vectors[group[i],0])
        y = np.append(y,x_vectors[group[i],1])

    return x,y


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot()
colors = np.array(['red','blue','orange','green','black'])
plot_array = np.array([])
labels = np.array([])
for i in range(k):
    x,y = detect(A,list(group_list[i]))
    ax.scatter(x, y, c=colors[i], edgecolors='black', label=f'group-{i}')
    labels = np.append(labels,f'group-{i}')

ax.scatter((Z.T)[0],(Z.T)[1], c='yellow',edgecolors='black',s=150,marker='*', label='z')

leg1 = ax.legend(labels,loc='upper right')
#leg2 = ax.legend('z',loc='lower left')
#ax.add_artist(leg1)

ax.patch.set_facecolor((0.7,0.7,0.7))

plt.show()


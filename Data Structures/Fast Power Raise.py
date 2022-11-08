import math
import numpy as np

def pow(a,p):

    res = 1
    tmp = p

    if p == 0:
        return res
    mult = a
    iter_counter = 0

    while p != 0:
        iter_counter += 1
        if p % 2 == 1:
            res *= mult
        mult *= mult
        p //= 2
    print(f"Количество итераций: {iter_counter}")
    print(f"log_2({tmp}) = {math.log2(tmp)}")
    return res


def fast_matrix_power(A,n):  # assumes square matrices on the input

    if n == 0:
        return np.identity(len(A))

    if n % 2 == 1:
        return fast_matrix_power(A,n-1) @ A

    tmp = fast_matrix_power(A,n//2)
    return tmp @ tmp

x = 16
h = 9

print(pow(x,h))
print(math.pow(x,h))

A = np.array([[1,1,1],[1,1,1],[1,1,1]])

print(fast_matrix_power(A,5))
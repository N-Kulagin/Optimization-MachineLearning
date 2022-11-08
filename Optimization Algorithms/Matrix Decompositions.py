import numpy as np


def projection(vector_a,vector_b):  # project a onto b

    return np.dot(vector_a,vector_b)/np.dot(vector_b,vector_b)


def gram_schmidt(matrix):
    return_matrix = np.copy(matrix)

    for i in range(1,len(matrix)):
        for j in range(i):
            return_matrix[i] = return_matrix[i] - projection(matrix[i],return_matrix[j]) * return_matrix[j]

    return return_matrix


def normalize(matrix):
    for i in range(len(matrix)):
        matrix[i] = matrix[i]/(np.dot(matrix[i],matrix[i])**0.5)
    return matrix

def QR(matrix_A):
    Q = normalize(gram_schmidt(matrix_A)).T
    R = Q.T @ matrix_A
    return Q,R

'''
proj_matrix = np.array([[1,1,1],[1,1,2],[2,1,2]]).astype('float64')

A,B = QR(proj_matrix)
print(A)
print()
print(B)
print()
print(A @ B)
print()
print(A @ A.T)
'''

A = np.array([[1,1,1],[2,1,2],[2,3,4]]).astype('float64')

# QR-algorithm for finding eigenvalues
for i in range(100):
    Q,R = QR(A)
    A = Q.T @ A @ Q

print(A)
print(f"Собственные числа матрицы А: {A[0,0], A[1,1], A[2,2]}")

J_orig = np.array([[1,1,1,1,1],[1,1,1,1,2],[1,2,1,1,2],[1,3,2,3,2],[4,2,1,3,4]]).astype('float64')
J = gram_schmidt(J_orig)


print(np.dot(J[0],J[1]))
print(np.dot(J[0],J[2]))
print(np.dot(J[0],J[3]))
print(np.dot(J[0],J[4]))
print(np.dot(J[1],J[2]))
print(np.dot(J[1],J[3]))
print(np.dot(J[1],J[4]))
print(np.dot(J[2],J[3]))
print(np.dot(J[2],J[4]))
print(np.dot(J[3],J[4]))

B = np.array([[2,1,2],[1,3,1],[2,1,4]]).astype('float64')
Chol = np.linalg.cholesky(B)
print(Chol)
print(Chol @ Chol.T)
print()

C = np.array([[1,1,0],[1,2,0],[0,0,3]])
Chol2 = np.linalg.cholesky(C)
print(Chol2)
print()
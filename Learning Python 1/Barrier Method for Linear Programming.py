import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def lagrange_hessian(A,b,x):
    hessian = np.zeros((len(A[0]),len(A[0])))

    for index,row in enumerate(A):
        hessian += np.tensordot(row,row,axes=0) / (b[index] - np.dot(row,x))**2

    return hessian

def gradient(A,b,x):
    grad = np.zeros(len(A[0]))

    for index,row in enumerate(A):
        grad += row / (b[index] - np.dot(row,x))

    return grad


def find_feasible_point(A,x,b,B,d,mu,t):

    #F(x,s,\mu) = \begin{pmatrix} \underset{i}{\sum}\frac{a_i}{m_i}+B^T\mu \\ \textbf{1}t - \underset{i}{\sum}\frac{\textbf{1}}{m_i} - \underset{i}{\sum}\frac{\textbf{1}}{s_i} \\ Bx - d \\ \end{pmatrix} = \begin{pmatrix} \textbf{0}\\ \textbf{0}\\ \textbf{0}\\ \end{pmatrix}, m_i = s_i + b_i - a_i^Tx
    #J(F) = \begin{pmatrix} \underset{i}{\sum}\frac{a_i a_i^T}{m_i^2} \ \ -\underset{i}{\sum}\frac{\textbf{1}a_i^T}{m_i^2} \ \ \ \ \ \ \ \ \ \ \ \ \ \ B^T \\ -\underset{i}{\sum}\frac{\textbf{1}a_i^T}{m_i^2} \ \ \ \ \underset{i}{\sum}\frac{\textbf{11}^T}{m_i^2} + \underset{i}{\sum}\frac{\textbf{11}^T}{s_i^2}  \ \ \ \  \textbf{0} \\ \ \ \ B \ \ \ \ \ \ \ \ \ \ \textbf{0} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \textbf{0} \end{pmatrix}
    #t*\textbf{1}^Ts - \log (s+b-Ax) - \log(s) \rightarrow \underset{x,s}{\min} \\ Bx=d


    def eval_RHS(A,x,b,B,d,mu,t,s,tmp):
        first_component, second_component, third_component = np.zeros(len(A[0])), np.ones(len(s)), np.zeros(len(d))

        for index,row in enumerate(A):
            first_component = first_component + row / tmp[index]
        first_component = first_component + np.squeeze(np.asarray(B.T * mu))

        tmp2 = 0
        for el1,el2 in zip(tmp,s):
            tmp2 = tmp2 + 1/el1 + 1/el2
        second_component = second_component * (t-tmp2)

        third_component = B @ x - d

        return np.block([first_component,second_component,third_component])

    def eval_Block(A,x,b,B,d,mu,t,s,tmp):

        one, two, three = np.zeros((len(A[0]),len(A[0]))), np.zeros((len(A),len(A[0]))), np.zeros((len(A),len(A)))
        one_vector = np.ones(len(A))

        for index,row in enumerate(A):
            one = one + np.tensordot(row,row,axes=0) / (tmp[index]**2)

        for index,row in enumerate(A):
            two = two - np.tensordot(one_vector,row,axes=0) / (tmp[index] ** 2)

        tmp2 = 0.0
        for el1,el2 in zip(tmp,s):
            tmp2 += 1/(el1 ** 2) + 1/(el2 ** 2)
        three = tmp2 * np.tensordot(one_vector,one_vector,axes=0)

        lambd = 0.0001
        three = three + lambd * np.identity(len(three))

        return np.block([[one,two.T, B.T],[two, three, np.zeros((len(three),len(B.T[0])))],[B, np.zeros((len(B),len(three[0]))), np.zeros((len(B),len(B.T[0])))]])

    t = 1.0
    multiplier = 2.0
    max_over_s = 5
    dx,ds,dmu = np.zeros(len(A[0])), np.zeros(len(A)), np.zeros(len(B))

    s = np.ones(len(A))
    tmp = s + b - A @ x

    RHS = -eval_RHS(A,x,b,B,d,mu,t,s,tmp)
    Block = eval_Block(A,x,b,B,d,mu,t,s,tmp)

    solution = np.linalg.solve(Block,RHS)
    dx = solution[:len(x)]
    ds = solution[len(x):len(A)+len(x)]
    dmu = solution[len(A)+len(x):]

    loc_x, loc_s, loc_mu = x,s,mu
    loc_x = loc_x + dx
    loc_s = loc_s + ds
    loc_mu = loc_mu + dmu
    max_over_s = np.max(loc_s)

    if max_over_s == 0:
        return loc_x, loc_mu


    while max_over_s >= 0.1:
        t *= multiplier
        tmp = loc_s + b - A @ loc_x
        RHS = -eval_RHS(A, loc_x, b, B, d, loc_mu, t, loc_s, tmp)
        Block = eval_Block(A, loc_x, b, B, d, loc_mu, t, loc_s, tmp)

        solution = np.linalg.solve(Block, RHS)
        dx = solution[:len(x)]
        ds = solution[len(x):len(A) + len(x)]
        dmu = solution[len(A) + len(x):]

        loc_x = loc_x + dx
        loc_s = loc_s + ds
        loc_mu = loc_mu + dmu
        max_over_s = np.max(loc_s)

    return loc_x, loc_mu



# t * (c^T * x) - log(b-Ax) -> min, Ax <= b, Bx = d
# solve using newton's method for some initial value of t > 0
# then multiply t by multiplier > 1 and if m/t < eps terminate, where m is the number of inequalities
# m/t is duality gap or how close c^T * x to c^T * (x_optimal)
# mu is lagrange multiplier related to single equality constraint Bx = d


A = np.array([[-1,-2],[1,2],[1,-1],[-2,1],[-1,0],[0,-1]])
b = np.array([-2,10,6,0,0,0])
B = np.array([[1,1]])
d = np.array([5.0])
c = np.array([2,1])


x = np.array([-4,-4])
mu = 2.0

m = len(A)
t = 1.0
multiplier = 2.0
eps = 0.0001

x,mu = find_feasible_point(A,x,b,B,d,mu,t)

H = lagrange_hessian(A,b,x)
Block = np.block([[H,B.T],[B,np.zeros((len(B),len(B)))]])
RHS = np.block([-t*c-gradient(A,b,x) + np.squeeze(np.asarray(B.T * mu)),d - float(B @ x)])

sol = np.linalg.solve(Block,RHS)
dx, dmu = sol[0:2], sol[2:]

x = x + dx
mu = mu + dmu


iter_counter = 0
while m/t > eps:
    iter_counter += 1
    t *= multiplier
    H = lagrange_hessian(A, b, x)
    Block = np.block([[H, B.T], [B, np.zeros((len(B), len(B)))]])
    RHS = np.block([-t*c-gradient(A,b,x) - B.T @ mu,d - B @ x])

    sol = np.linalg.solve(Block,RHS)
    dx, dmu = sol[0:2], sol[2:]
    x = x + dx
    mu = mu + dmu

print("\nНайдено решение:")
print(x,mu)
print(f"Прошло {iter_counter} итераций")
print(f"Зазор двойственности: {m/t}")
print(f"Параметр t: {t}")
print(f"\nГессиан:\n {H}")


fig = plt.figure(figsize=(15,8))
gs = GridSpec(1,1,fig)
ax = plt.subplot(gs[0,0])
ax.set_ylim(-0.5,8)

a = np.arange(0,8,0.01)
for number,row in zip(b,A):
    k = np.array([])
    for el in a:
        if row[1] != 0:
            k = np.append(k,(number-row[0]*el)/row[1])
    if row[1] != 0:
        ax.plot(a,k)

a = np.arange(0.4,7.3333,0.1)
k = np.arange(0,4,0.1)

for el1 in a:
    for el2 in k:
        counter = 0
        for index,row in enumerate(A):
            if row[0]*el1 + row[1]*el2 <= b[index]:
                counter += 1
        if counter == len(A):
            ax.plot(el1, el2,'bo',ms=3)

ax.plot([a[0],a[-1]],[(d[0]-B[0,0]*a[0])/B[0,1],(d[0]-B[0,0]*a[-1])/B[0,1]],lw=3,color='black')


ax.plot(x[0],x[1],'r*',ms=20,)
plt.show()
import numpy as np


def find_initial_point(A,A_transp,c):
    tmp = np.linalg.inv(A @ A_transp)

    x_tilde = A_transp @ tmp @ b
    lambd_tilde = tmp @ A @ c
    s_tilde = c - A_transp @ lambd_tilde

    delta_x = np.max([0,-3/2*np.min(x_tilde)])
    delta_s = np.max([0,-3/2*np.min(s_tilde)])
    e = np.ones(len(x_tilde))

    x_hat = x_tilde + delta_x * e
    s_hat = s_tilde + delta_s * e

    delta_x_hat = 0.5 * (np.dot(x_hat,s_hat)/np.dot(e,s_hat))
    delta_s_hat = 0.5 * (np.dot(x_hat, s_hat) / np.dot(e, x_hat))

    return x_hat + delta_x_hat*e,lambd_tilde,s_hat+delta_s_hat*e

A = np.array([[2,1,1,0,0,0],[1,-1,0,1,0,0],[0,-1,0,0,1,0],[-2,1,0,0,0,1]])
A_transp = A.T
b = np.array([10,4,-0.5,4])

c = np.array([-1,-3,0,0,0,0])

x,lambd,s = find_initial_point(A,A_transp,c)
x_last = np.zeros(len(x))

e = np.ones(len(x))
eps = 0.01

X = np.diag(x)
S = np.diag(s)
I = np.identity(len(x))

m = len(A)
n = len(A[0])

iter_counter = 0

while np.dot(x-x_last,x-x_last)**0.5 > eps:
    X = np.diag(x)
    S = np.diag(s)
    KKT_Matrix = np.block(
        [[np.zeros((n, n)), A_transp, I], [A, np.zeros((m, m)), np.zeros((m, n))], [S, np.zeros((n, m)), X]])
    residual_c = A_transp @ lambd + s - c
    residual_b = A @ x - b
    XSe = X @ S @ e
    solution_affine = np.linalg.solve(KKT_Matrix,np.block([-residual_c,-residual_b,-XSe]))

    dx_affine = solution_affine[:len(x)]
    ds_affine = solution_affine[len(x)+len(lambd):len(x)+len(lambd)+len(s)]

    div_x_affine = -x/dx_affine
    div_s_affine = -s/ds_affine

    tmp1 = np.where(div_x_affine > 0,div_x_affine,1)
    tmp1 = np.where(tmp1 < 1,tmp1,1)
    tmp2 = np.where(div_s_affine > 0, div_s_affine, 1)
    tmp2 = np.where(tmp2 < 1, tmp2, 1)

    alpha_aff_primal = np.min(tmp1)
    alpha_aff_dual = np.min(tmp2)
    mu_affine = np.dot(x+alpha_aff_primal*dx_affine,s+alpha_aff_dual*ds_affine)/n
    mu = np.dot(x,s)/n
    sigma = (mu_affine/mu)**3

    DXDS_affine = np.diag(dx_affine) @ np.diag(ds_affine) @ e

    solution_corrector = np.linalg.solve(KKT_Matrix,np.block([-residual_c,-residual_b,-XSe-DXDS_affine+sigma*mu*e]))

    dx = solution_corrector[:len(x)]
    dl = solution_corrector[len(x):len(x)+m]
    ds = solution_corrector[len(x)+len(lambd):len(x)+len(lambd)+len(s)]

    tmp3 = np.where(-x/dx>0,-x/dx,100)
    tmp4 = np.where(-s / ds > 0, -s / ds, 100)

    alpha_primal_max = np.min(tmp3)
    alpha_dual_max = np.min(tmp4)

    eta = 0.99
    alpha_primal = np.min([1,eta*alpha_primal_max])
    alpha_dual = np.min([1,eta*alpha_dual_max])

    x_last = np.copy(x)
    x = x + alpha_primal * dx
    lambd = lambd + alpha_dual * dl
    s = s + alpha_dual * ds

    iter_counter += 1

print(f"Найдено решение за {iter_counter} итераций:")
print()
print(f"x = {x}")
print()
print(f"lambda = {lambd}")
print()
print(f"s = {s}")

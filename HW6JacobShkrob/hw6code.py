import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from tqdm import tqdm
import random
import emcee
from numba import jit, njit
from numba import int32, float32
from numba.experimental import jitclass
from scipy.stats import multivariate_normal

# Rosebrock function helper methods
@njit
def density_rosen(v):
    x = v[0]
    y = v[1]
    return np.exp(-1*(1/20)*(100*(y-x**2)**2 + (1-x)**2))
@njit
def grad_rosen(v):
    x = v[0]
    y = v[1]
    xx = (-1*1/10)*np.exp(-1*5*(y-x**2)**2 - (1/20)*(1-x)**2)*(200*x**3 - 200*x*y + x - 1)
    yy = -1*10*np.exp(-1*5*(x**2 - y)**2 - (1/20)*(x-1)**2)*(y-x**2)
    return(np.array([xx,yy]))
@njit 
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
    
@njit
def grad_log_rosen(v):
    x = v[0]
    y = v[1] 
    xx = -1*20*x**3 + x*(20*y - (1/10)) + (1/10)
    yy = -1*10*(y - x**2)
    return(np.array([xx,yy]))

@njit
def hessian_log_rosen(v):
    x = v[0]
    y = v[1]
    h_11 = -60*x**2 + 20*y - (1/10)
    h_12 = 20*x 
    h_22 = -1*10
    return(np.array([[h_11, h_12],[h_12, h_22]]))
    
@njit
def compute_R_mat(S:np.array):
    det_S = np.linalg.det(S)
    trace_S = np.trace(S) 
    R = (1/(np.sqrt(np.abs(trace_S) + 2*np.sqrt(np.abs(det_S)))))*(S + np.sqrt(np.abs(det_S)*np.eye(2)))
    return(R)

@njit 
def ensemble_proposal(z, alpha):
    if z >= 1/alpha and z <= alpha:
        return 1/np.sqrt(z)
    else:
        return 0

@njit
def ensemble_inv_cdf(z,alpha):
    if z == 0:
        return 0
    else:
        return (z**2/4) + (1/alpha)

def overdamped_scheme(metrop = True, h = 0.05, n = 1000, cov_mat = np.eye(2)):
    X = np.random.multivariate_normal(np.zeros(2), cov_mat)
    X_new = X
    M = np.linalg.inv(cov_mat)
    for i in range(n):
        xi = np.random.multivariate_normal(np.zeros(2), cov_mat)
        X_new = X + h* M @ grad_log_rosen(X) + np.sqrt(2*h)*xi
        if metrop:
            ratio_1 = (multivariate_normal.pdf(X_new, mean = X - h* M @ grad_log_rosen(X), cov = 2*h)) * density_rosen(X)
            ratio_2 = (multivariate_normal.pdf(X, mean = X_new - h*M @ grad_log_rosen(X_new), cov = 2*h)) * density_rosen(X_new)
            acc_p = min([1, ratio_2/ratio_1])
            t = np.random.binomial(1, acc_p, size = 1)
            if t == 1:
                X = X_new
            else:
                X = X
    return(X)

def stochastic_newton_scheme(metrop = True, h = 0.05, n = 1000, cov_mat = np.eye(2)):
    X = np.random.multivariate_normal(np.zeros(2), cov_mat)
    X_new = X
    for i in range(n):
        xi = np.random.multivariate_normal(np.zeros(2), cov_mat)
        S = hessian_log_rosen(X)
        S_ = compute_R_mat(S)
        S_inv = np.linalg.inv(S)
        if is_pos_def(S_inv):
            H = S_inv
            X_new = X + h* H @ grad_log_rosen(X) + np.sqrt(2*h)* (np.sqrt(H)) @ xi
        else:
            H = np.matmul(S_, S_)
            X_new = X + h* H @ grad_log_rosen(X) + np.sqrt(2*h)* S_ @ xi

        S_n = hessian_log_rosen(X_new)
        S_n_ = compute_R_mat(S_n)
        S_n_inv = np.linalg.inv(S_n)
        if is_pos_def(S_n_inv):
            H_n = S_n_inv
        else:
            H_n = np.matmul(S_n_,S_n_)

        if metrop:
            ratio_1 = (multivariate_normal.pdf(X_new, mean = X + h*H @ grad_log_rosen(X), cov = 2*h*H, allow_singular=True)) * density_rosen(X)
            ratio_2 = (multivariate_normal.pdf(X, mean = X_new + h*H_n @ grad_log_rosen(X_new), cov = 2*h*H_n, allow_singular = True)) * density_rosen(X_new)
            acc_p = min([1, ratio_2/ratio_1])
            t = np.random.binomial(1, acc_p, size = 1)
            if t == 1:
                X = X_new
            else:
                X = X
        else:
            X = X_new
            print(X)
            print(H)
            print(S_)
    return(X)

def ensemble_method(metrop = True, L = 10, n = 100, alpha = 0.5):
    X = np.random.multivariate_normal(np.zeros(2), 10*np.eye(2), size=L)
    for i in range(n):
        for j in range(L):
            k = np.random.choice(np.arange(L))
            while k == j:
                k = np.random.choice(np.arange(L))
            Z = np.random.uniform(low=0.0, high = 1.0)
            Z = ensemble_inv_cdf(Z, alpha)
            Y_new = X[k] + Z * (X[j] - X[k])
            try:
                p = (ensemble_proposal(1/Z, alpha)/ensemble_proposal(Z, alpha))*(density_rosen(Y_new)/density_rosen(X[j]))
            except:
                p = 1
            acc_p = min([1, p])
            t = np.random.binomial(1, acc_p, size = 1)
            if t == 1:
                X[j] = Y_new
    return (X)


### Outputs for Langevin schemes from exercise 71
list = [stochastic_newton_scheme(metrop = True, n = 10000, h = 0.1, cov_mat=np.array([[0.5,0],[0,0.5]])) for i in range(500)]
xx = [a[0] for a in list if (a[0] > -1*10 and a[0] < 10 and a[1] > -1*10 and a[1] < 10)]
yy = [a[1] for a in list if (a[0] > -1*10 and a[0] < 10 and a[1] > -1*10 and a[1] < 10)]
plt.scatter(x=  xx, y = yy, alpha = 0.4)

list = [stochastic_newton_scheme(metrop = True, n = 10000, h = 0.01, cov_mat=np.array([[0.5,0],[0,0.5]])) for i in range(500)]
xx = [a[0] for a in list if (a[0] > -1*10 and a[0] < 10 and a[1] > -1*10 and a[1] < 10)]
yy = [a[1] for a in list if (a[0] > -1*10 and a[0] < 10 and a[1] > -1*10 and a[1] < 10)]
plt.scatter(x=  xx, y = yy, alpha = 0.4)

list = [overdamped_scheme(metrop = True, n = 2000, h = 0.1, cov_mat=np.array([[1,0],[0,1]])) for i in range(1000)]
xx = [a[0] for a in list if (a[0] > -1*10 and a[0] < 10 and a[1] > -1*10 and a[1] < 10)]
yy = [a[1] for a in list if (a[0] > -1*10 and a[0] < 10 and a[1] > -1*10 and a[1] < 10)]
plt.scatter(x=  xx, y = yy, alpha = 0.4)

# Ensemble algorithm output
ensemble_list = ensemble_method(metrop = True, n = 10, L = 500, alpha = 30)
xx = [a[0] for a in ensemble_list if (a[0] > -1*10 and a[0] < 10 and a[1] > -1*10 and a[1] < 10)]
yy = [a[1] for a in ensemble_list if (a[0] > -1*10 and a[0] < 10 and a[1] > -1*10 and a[1] < 10)]
plt.scatter(x=  xx, y = yy, alpha = 0.4)
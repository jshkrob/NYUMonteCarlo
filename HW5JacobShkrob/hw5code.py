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

##### Question 64
@njit
def grad_XY(x, beta):
    y = np.empty(x.size)
    for i in range(x.size):
        if i == 0:
            y[i] = -1*beta*np.sin(x[i]-x[i+1])
        if i == (x.size-1):
            y[i] = beta*np.sin(x[i-1] - x[i])
        else:
            y[i] = beta*np.sin(x[i-1]-x[i]) - beta*np.sin(x[i]-x[i+1])
    return(y)
@njit
def unit_vec(x):
    return x / np.linalg.norm(x)

@njit
def XY_density(x, beta):
    y = np.zeros(x.size)
    for i in range(x.size-1):
        y[i] = beta*np.cos(x[i]-x[i+1])
    return(np.exp(np.sum(y)))

def cos_mg(x):
    si = np.zeros(2)
    # sum over all lattice the vectors
    for i in range(x.size):
        si += np.array([np.cos(x[i]), np.sin(x[i])])
    return(np.arccos(np.clip(np.dot(unit_vec(si), np.array([1,0])), a_min = -1.0, a_max = 1.0)))

### Old code for integrated autocorrelation time
@njit()
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr(y, c = 5.0):
    f = autocorr_func_1d(y)
    tau = 2*np.cumsum(f) - 1
    window = auto_window(tau, c)
    return tau[window]

# Model (5.8) w/ and w/ out Metropolization
def HMC(N = 10, beta = 1.0, iter_num = 1000, h = 0.05, prop_var = 1.0, metrop = False):
    mg = []
    # random initial angles
    theta = np.random.uniform(low=-1*np.pi, high=np.pi, size = N)
    for i in range(iter_num):
        # Unadjusted Langevin
        delt = grad_XY(theta, beta)
        if metrop:
            x = np.random.multivariate_normal(np.zeros(N), prop_var*np.eye(N))
            theta_new = theta + h * delt + np.sqrt(2*h)*x
            y1 = multivariate_normal.pdf(theta, theta_new + h*grad_XY(theta_new, beta), 2*h*np.eye(N))/multivariate_normal.pdf(theta_new, theta + h * delt, 2*h*np.eye(N))
            y2 = XY_density(theta_new,beta)/XY_density(theta, beta)
            acc_p = min(1, y2*y1)
           # print(acc_p)
            t = np.random.binomial(1, acc_p, size = 1)
            if t == 1:
                theta = theta_new
            else:
                theta = theta
        else:
            x = np.random.multivariate_normal(np.zeros(N), prop_var*np.eye(N))
            theta += h * delt + np.sqrt(2*h)*x
       # print(theta)
        mg.append(cos_mg(theta))
    return(mg)

def autocorr_func_results(n = 10):
    tau1 = [autocorr(np.asarray(HMC(N = 50, beta = 1.0, h = 0.1, iter_num=1000))) for i in range(n)]
    tau2 = [autocorr(np.asarray(HMC(N = 50, beta = 1.0, h = 0.1, iter_num=1000,metrop=True))) for i in range(n)]
    return((tau1, tau2))

tau1, tau2 = autocorr_func_results(n = 300)
bins = np.linspace(0, 120, 50)
plt.hist(tau1, bins=bins,alpha = 0.8, label = "no metropolis") # no metropolization
plt.hist(tau2, bins=bins, alpha = 0.8, label = "metropolis") # metropolization
plt.legend()
plt.xlabel("$IAT_f$ when using Langevin and metropolized Langevin, h = 0.1, N = 100")

############ Question 65
#### Hybrid Monte Carlo Method

'''
vel_ver_scheme: Velocity Verlet scheme: 

parameters:
h - time-discretization parameter
s - ending time of Hamilton ODE flow map
J - J matrix appearing in Hamilton's ODE discretization formulation
init_pt - initial point of the flow
beta - inverse temperature parameter
N - dimension (d_hat + d_tilde)
N_hat - dimension (d_hat) [same dimension as pi]
'''
@njit
def vel_ver_scheme(h, s, J, beta, init_pt, N, N_hat):
    y = init_pt
    J_hat = -1*J[0:N-N_hat, N-N_hat:N]
    for i in range(int(np.floor(s/h))):
        y_t = y[0:N-N_hat]
        y_h = y[N-N_hat:N]
        y_t_p = y_t + (h/2)*(J_hat.T @ grad_XY(y_h, beta).T)
        y_h = y_h + h*(J_hat @ y_t_p.T)
        y_t = y_t_p + (h/2)*(J_hat.T @ grad_XY(y_h, beta).T)
        y[0:N-N_hat] = y_t
        y[N-N_hat:N] = y_h
    return(y)

# K for us will be the gaussian in \R^d
def HybridMC(N = 10, 
             beta = 1.0, 
             iter_num = 1000, 
             h = 0.05, 
             n = 10,
             metrop = True
):  
    mg = []
    XX = np.zeros(2*N)
    J = np.block([
          [np.zeros((N,N)), -1*np.eye(N)],
          [np.eye(N), np.zeros((N,N))]
    ])
    X = np.random.uniform(low=-1*np.pi, high=np.pi, size = N)
    for i in range(iter_num):
        # d_tilde = N, K(x_tilde) = 1/2(x^T * x)
        Y = np.random.multivariate_normal(np.zeros(N), np.eye(N))
        XY = np.concatenate([X, Y])

        # h, n are used here
        Y_new = vel_ver_scheme(h=h, s=n, J=J, beta=beta, init_pt = XY, N=2*N, N_hat = N)
        if metrop:
            y1 = multivariate_normal.pdf(Y_new[N:2*N], np.zeros(N), np.eye(N))/multivariate_normal.pdf(Y, np.zeros(N), np.eye(N))
            y2 = XY_density(Y_new[0:N],beta)/XY_density(X, beta)
            acc_p = min(1, y2*y1)
            t = np.random.binomial(1, acc_p, size = 1)
            if t == 1:
                XX = Y_new
                X = Y_new[0:N]
            else:
                if i == 1:
                    XX = np.concatenate([X, Y])
                #otherwise, XX = XX i.e. XX stays the same
        else:
            XX = Y_new
            X = Y_new[0:N]

        mg.append(cos_mg(X))
    return(mg)

n = 300
tau_hybrid = [autocorr(np.asarray(HybridMC(N = 50, h = 0.1, beta = 1.0, n = 0.1, iter_num=1000, metrop=False))) for i in range(n)]
bins = np.linspace(0, 100, 40)
plt.hist(tau1, alpha = 0.8, bins=bins, label="no metrop") # no metropolization
plt.hist(tau2, alpha = 0.8, bins=bins, label = "metrop") # metropolization
plt.hist(tau_hybrid, alpha = 0.8, bins=bins, label="hybrid, no metrop") # hybrid mcmc w/ metropolization
plt.legend()
plt.xlabel("IAT estimates, $N = 50$, $h=0.1$, $n=0.1$")

############ Question 66
# Underdamped Langevin dynamics
def r_func(x,y, gamma, h, J, N, beta):
    xx = x[N:2*N]
    yy = y[N:2*N]
    x = x[0:N]
    y = y[0:N]
    grad_x = grad_XY(x, beta)
    grad_y = grad_XY(y, beta)
    t = np.linalg.norm(yy - np.exp(-1*gamma*h)*xx - (1/2)*h*J.T*(np.exp(-1*gamma*h)*grad_x.T + grad_y.T))**2
    return(np.exp(-1*(t/(2*(1-np.exp(-2*h*gamma))))))

def UnderdampedLangevin(N = 10, 
                beta = 1.0, 
                iter_num = 1000, 
                h = 0.05,
                gamma = 0.2,
                metrop = False):
    mg = []
    J = np.block([
          [np.zeros((N,N)), -1*np.eye(N)],
          [np.eye(N), np.zeros((N,N))]
    ])
    J_hat = -1*J[0:N, N:2*N]
    x = np.random.uniform(low=-1*np.pi, high=np.pi, size = N)
    y = np.random.multivariate_normal(np.zeros(N), np.eye(N))
    X = np.concatenate([x,y])
    for i in range(iter_num):
        Xpp = X[0:N]
        Ypp = X[N:2*N]
        XYpp = np.concatenate([Xpp, Ypp])
        X_t_p = X[0:N] + (h/2)*grad_XY(X[N:2*N], beta).T
        X_h_p = X[0:N] + (h/2)*X_t_p
        X_t_pp = np.exp(-1*gamma*h)*X_t_p + np.sqrt(1-np.exp(-2*gamma*h))*np.random.multivariate_normal(np.zeros(N), np.eye(N))
        X[N:2*N] = X_h_p + (h/2)*X_t_pp
        X[0:N] = X_t_pp + (h/2)*grad_XY(X[N:2*N], beta).T
        if metrop:
            arg1 = np.concatenate([X[0:N], -1*X[N:2*N]])
            arg2 = np.concatenate([Xpp, -1*Ypp])
            r1 = r_func(arg1, arg2, gamma, h, J_hat, N, beta)
            r2 = r_func(XYpp, X, gamma, h, J_hat, N, beta)

            y1 = multivariate_normal.pdf(X[N:2*N], np.zeros(N), np.eye(N))/multivariate_normal.pdf(Ypp, np.zeros(N), np.eye(N))
            y2 = XY_density(X[0:N],beta)/XY_density(Xpp, beta)
            #print((multivariate_normal.pdf(Y_new[N:2*N], np.zeros(N), np.eye(N)), multivariate_normal.pdf(Y, np.zeros(N), np.eye(N)), y1, y2))
            acc_p = min(1, y2*y1*(r1/r2))
            t = np.random.binomial(1, acc_p, size = 1)
            if t == 1:
                X[0:N] = X[0:N]
                X[N:2*N] = X[N:2*N]
            else:
                X[0:N] = Xpp
                X[N:2*N] = -1*Ypp
        mg.append(cos_mg(X))   

    return(mg)
tau_und_land = [autocorr(np.asarray(UnderdampedLangevin(N = 50, beta = 1.0, h = 0.1, iter_num=1000, gamma = 2.0))) for i in range(300)]
bins = np.linspace(0, 100, 40)
plt.hist(tau1, alpha = 0.8, bins=bins, label="no metrop") # no metropolization
plt.hist(tau2, alpha = 0.8, bins=bins, label = "metrop") # metropolization
plt.hist(tau_hybrid, alpha = 0.8, bins=bins, label="hybrid, no metrop") # hybrid mcmc w/ metropolization
plt.hist(tau_und_land, alpha = 0.8, bins = bins, label = "underdamped")
plt.legend()
plt.xlabel("IAT estimates, $N = 50$, $h=0.1$, $n=0.1$")

def f1(m,g):
    M = 0
    for i in range(m): 
        M+=autocorr(np.asarray(UnderdampedLangevin(iter_num=1000, gamma = g, h = 0.1)))
    return M/m 

def f2(m,h):
    M = 0
    for i in range(m):
        M+=autocorr(np.asarray(UnderdampedLangevin(iter_num = 1000, gamma = 1.0, h = h)))
    return M/m
tau_new_method_gamma = [f1(50, g=g) for g in np.arange(0.1, 3, 0.1)]
plt.plot(np.arange(0.1, 3, 0.1), tau_new_method_gamma, "o--")
plt.xlabel("IAT v.s. $\gamma$ value (Underdamped Langevin)")
tau_new_method_h = [f2(50, h) for h in np.arange(0.05, 1.0, 0.05)]
plt.plot(np.arange(0.05, 1.0, 0.05), tau_new_method_h, "o--")
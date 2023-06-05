import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py

# generates mean of exponential rvs
def mean_func(N):
    return np.average(np.random.exponential(scale = 1.0, size = N))

# outputs histogram 
def plot_hist(N, r_df = False, iterations = 1000):
    dat_hist = np.sqrt(N)*(np.asarray([mean_func(N) for i in range(iterations)]) - np.ones(iterations))
    n, bins, patches = plt.hist(dat_hist, bins = 25, density=True, facecolor='g', alpha = 0.6)
    plt.title("Histogram of $\sqrt{N}(\overline{X_N} - 1)$")
    plt.grid(True)
    plt.show()
    if r_df: return(dat_hist)

# running examples
plot_hist(10, iterations = 1000)
plot_hist(30, iterations = 1000)
plot_hist(50, iterations = 1000)
plot_hist(1000, iterations = 1000)

# qq plot
dat = plot_hist(100, r_df = True, iterations = 1000)
fig = sm.qqplot(dat, line = '45')
py.show

# for Q_N
def mean_func_p_N(N, M):
    return np.average((1/N) * np.random.gamma(shape = N, scale = 1, size = M) - np.ones(M) > 0.1)

def plot_hist_p_N(N, M, iterations = 1000):
    dat_hist = np.asarray([mean_func_p_N(N, M) for i in range(iterations)])
    n, bins, patches = plt.hist(dat_hist, bins = 25, density=True, facecolor='g', alpha = 0.6)
    plt.title("Histogram of $Q_N$")
    plt.grid(True)
    plt.show()
    
plot_hist_p_N(100, 100000)
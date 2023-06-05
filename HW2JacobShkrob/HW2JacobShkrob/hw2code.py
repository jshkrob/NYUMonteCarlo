import numpy as np
import scipy as scpy
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import time

########### Question 16 code:
vals = np.random.uniform(low = 0.0, high = 1.0, size = 1000)
dist_vals = vals**2
plt.xlim([0,1])
plt.ylim([0,10])
plt.hist(dist_vals, bins = 40,density=True)
plt.plot(np.sort(vals), 1/(np.sqrt(np.sort(vals))*2), 'k', linewidth = 2)

# Generates QQ-plot for our distribution
class sqrt_distr:
    def ppf(self, x):
        return x**2
#stats.probplot(
sqrt_fn = sqrt_distr()
stats.probplot(x = dist_vals, dist = sqrt_fn, plot = plt)


######## Question 18:
def transform_sampling(x_1, x_2, N = 100):
    # keeping track of running time
    t_0 = time.time()
    sample = np.empty([N,2])
    runif_coor1 = np.sqrt(x_1)*np.cos(2 * np.pi * x_2)
    runif_coor2 = np.sqrt(x_1)*np.sin(2 * np.pi * x_2)
    sample[:,0] = runif_coor1
    sample[:,1] = runif_coor2
    t_1 = time.time() - t_0
    return(sample, t_1)
(sample, t) = transform_sampling(x_1 = np.random.uniform(size = 100000),
                            x_2 = np.random.uniform(size = 100000),
                            N = 100000)
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

plt.hist2d(sample[:,0], sample[:,1], bins = (80,80))
plt.show

######### Question 19
def rejection_sampling(target, reference=scpy.stats.uniform(loc = 0, scale = 1), K = 1, N = 100):
    rej_samples = np.empty([N,2]) 
    for i in range(N):
        sample = reference.rvs(size = 2)
        test = (target(x = sample))/(K * reference.pdf(sample[0])*reference.pdf(sample[1]))
        #test = (target(x = sample))/(K)
        new_inst = reference.rvs(size = 1)
        while (new_inst > test):
            sample = reference.rvs(size = 2)
            test = (target(x = sample))/(K * reference.pdf(sample[0])*reference.pdf(sample[1]))
            # test = (target(x = sample))/(K)
            new_inst = reference.rvs(size = 1)
        # to create uniform, "spread" the sample randomly in each quadrant since the support only covers 1/4th of the distribution 
        new_inst = reference.rvs(size = 1)
        if 0 <= new_inst < 0.25:
            rej_samples[i, ] = np.array([sample[0], sample[1]])
        elif 0.25 <= new_inst < 0.5:
            rej_samples[i, ] = np.array([-1*sample[0], sample[1]])
        elif 0.5 <= new_inst < 0.75:
            rej_samples[i, ] = np.array([-1*sample[0], -1*sample[1]])
        else:
            rej_samples[i, ] = np.array([sample[0], -1*sample[1]])
    return(rej_samples)

def unif_sphere(x: np.array):
    x = (1/np.pi) * (1 if ((x[0]**2 + x[1]**2) <= 1) else 0)
    return(x)
fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

pts = rejection_sampling(target = unif_sphere, K = 1, N = 100000)
plt.scatter(pts[:,0], pts[:,1])
x_vals = np.arange(-1, 1.001, 0.001)
plt.plot(x_vals, np.sqrt(1 - np.sort(x_vals)**2), 'k', linewidth = 2)
plt.plot(x_vals, -1*np.sqrt(1 - np.sort(x_vals)**2), 'k', linewidth = 2)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_aspect('equal', adjustable='box')

plt.hist2d(pts[:,0], pts[:,1], bins = (80,80))

########## Question 20:
def importance_sampling_estimate(target, reference, f, N = 100):
    ref_sample = reference.rvs(N)
    vals = f(ref_sample)*target.pdf(ref_sample)/reference.pdf(ref_sample)
    return(np.average(vals))

# function I want to average over
def indic_funct(x):
    l = 1 if (x > 2) else 0
    return(l)

indic = np.vectorize(indic_funct)
# single random IS estimate
importance_sampling_estimate(target = scpy.stats.norm, 
                             reference = scpy.stats.norm(loc = 1, scale = 1),
                             f = indic,
                             N = 100)
def importance_sampling_sample(target, 
                               reference,
                               f,
                               N = 100,
                               M = 1000):
    vals = np.empty([M, 1])
    for i in range(M): 
        vals[i] = importance_sampling_estimate(target = target,
                                               reference = reference,
                                               f = f,
                                               N = N)
    return(vals)

vals = importance_sampling_sample(target = scpy.stats.norm(loc = 0, scale = 1), reference = scpy.stats.norm(loc = 0, scale = 2),
                           f = indic, N = 1000, M = 1000)
plt.hist(vals, bins = 100)

## Experiment 1 : changing various levels of s (standard dev) in the reference denesity for IS estimates: fix mean = 1
M = 1000 # number of experiments
vals_tbl_mean_lst = np.arange(0.1,5.0,0.01)
vals_tbl_mean = np.empty([M, len(vals_tbl_mean_lst)])
variance_list_std_dev = np.empty([len(vals_tbl_mean_lst), 1])
for (idx, i) in enumerate(vals_tbl_mean_lst):
    vals_tbl_mean[:,idx] = (importance_sampling_sample(target = scpy.stats.norm(loc = 0, scale = 1), reference = scpy.stats.norm(loc = 0, scale = i), f = indic, N = 1000, M = M)).reshape((M, ))
    variance_list_std_dev[idx] = np.var(vals_tbl_mean[:,idx])
plt.plot(vals_tbl_mean_lst, variance_list_std_dev)

## Experiment 2 : changing various levels of m (mean) in the reference denesity for IS estimates: fix scale = 1
M = 1000 # number of experiments
vals_tbl_mean_lst = np.arange(-1*5.0,5.0,0.1)
vals_tbl_mean = np.empty([M, len(vals_tbl_mean_lst)])
variance_list_mean = np.empty([len(vals_tbl_mean_lst), 1])
for (idx, i) in enumerate(vals_tbl_mean_lst):
    vals_tbl_mean[:,idx] = (importance_sampling_sample(target = scpy.stats.norm(loc = 0, scale = 1), reference = scpy.stats.norm(loc = i, scale = 1), f = indic, N = 1000, M = M)).reshape((M, ))
    variance_list_mean[idx] = np.var(vals_tbl_mean[:,idx])
plt.plot(vals_tbl_mean_lst[50:100], variance_list_mean[50:100])

########## Question 21:
def importance_sampling_estimate_nrml_const(target, reference, N = 100):
    ref_sample = reference.rvs(N)
    vals = target(ref_sample)/reference.pdf(ref_sample)
    return(np.average(vals))

def target_f(x):
    return (np.exp(-1*np.absolute(x)**3))

target = np.vectorize(target_f)
importance_sampling_estimate_nrml_const(target = target, 
                                        reference = scpy.stats.norm(loc = 0, scale = 1),
                                        N = 10000)
M = 1000 # number of IS estimates
vals = np.empty([M, 1])
for i in range(M): 
    vals[i] = importance_sampling_estimate_nrml_const(target = target, 
                                                      reference = scpy.stats.norm(loc = 0, scale = 1),
                                                      N = 10000)
plt.hist(vals, bins = 100)

########## Question 22:
# We consider the alternative importance sampling estimator instead.
def importance_sampling_estimate_alt(target, reference, f, N = 100):
    ref_sample = reference.rvs(N)
    vals = f(ref_sample)*target.pdf(ref_sample)/reference.pdf(ref_sample)
    l_vals = target.pdf(ref_sample)/reference.pdf(ref_sample)
    return(np.average(vals)/np.average(l_vals))

# function I want to average over
def indic_funct(x):
    l = 1 if (x > 2) else 0
    return(l)
# vectorized function
indic = np.vectorize(indic_funct)
def importance_sampling_alt(target, 
                               reference,
                               f,
                               N = 100,
                               M = 1000):
    vals = np.empty([M, 1])
    for i in range(M): 
        vals[i] = importance_sampling_estimate_alt(target = target,
                                               reference = reference,
                                               f = f,
                                               N = N)
    return(vals)

# single random IS estimate
importance_sampling_estimate_alt(target = scpy.stats.norm, 
                             reference = scpy.stats.norm(loc = 1, scale = 1),
                             f = indic,
                             N = 100)
vals = importance_sampling_alt(target = scpy.stats.norm,
                               reference = scpy.stats.norm(loc = -2, scale = 3),
                               f = indic,
                               N = 1000,
                               M = 3000)
vals_oth = importance_sampling_sample(target = scpy.stats.norm,
                                      reference =  scpy.stats.norm(loc = -2, scale = 3),
                                      f = indic,
                                      N = 1000,
                                      M = 3000)
plt.hist(vals, bins = 40, alpha = 0.6, label = 'alt')
plt.hist(vals_oth, bins = 40, alpha = 0.6, label = 'orig')
plt.legend(loc = 'upper right')
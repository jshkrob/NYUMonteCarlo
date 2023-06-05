import numpy as np
import scipy as scpy
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import time
import itertools
import random
import pylab

############################################ Exersize 28

## Outputs one importance sampling estiamte with resampling which compares the N(0,1) with N(0,\sigma^2)
def importance_sampling_resampling(type = 'multinomial', N = 100, sigma = 2.0):
    # sample from N(0,1):
    vals = np.random.normal(loc = 0.0, scale = 1.0, size = N)
    # weight samples w/ nomralized importance weights
    target_dens = (1/np.sqrt(2 * np.pi * sigma**2)*np.exp(-1 * ((vals)**2/(2*sigma**2))))
    approx_dens = (1/np.sqrt(2*np.pi))*np.exp(-1*((vals)**2/2))
    weights_inter = target_dens/approx_dens
    weights_IS = weights_inter/np.sum(weights_inter)
        
    if type == 'multinomial':
        # resample using multinomial
        N_n = np.random.multinomial(n = N, pvals = weights_IS)
        var_N_n = np.var(N_n)
        ll = [[vals[i]]*N_n[i] for i in range(len(N_n))]
        resample_vals = list(itertools.chain(*ll))
    
    elif type == 'Bernoulli':
        # resample using Bernoulli
        floor_weights_IS = np.floor(N * weights_IS)
        unif = np.random.uniform(low = 0.0, high = 1.0, size = N)
        indic_bool = unif < N*weights_IS - floor_weights_IS
        indic = indic_bool.astype(int)
        N_n = floor_weights_IS + indic
        N_n = N_n.astype(int)
        var_N_n = np.var(N_n)
        ll = [[vals[i]]*N_n[i] for i in range(len(N_n))]
        resample_vals = list(itertools.chain(*ll))
        
    elif type == 'systematic':
        unif = np.random.uniform(low = 0.0, high = 1.0, size = 1)
        alt_unif = (1/N) * (np.arange(1, N+1) - unif)
        cum_weights_IS = np.append(0, np.cumsum(weights_IS))
        N_n = np.zeros((N,), dtype=int)
        for i in range(N):
            N_n[i] = np.sum([1 for u in alt_unif if (u >= cum_weights_IS[i] and u < cum_weights_IS[i+1])])
        var_N_n = np.var(N_n)
        ll = [[vals[i]]*N_n[i] for i in range(len(N_n))] 
        resample_vals = list(itertools.chain(*ll))

    return({"orig_sample": vals, "resample_sample": resample_vals, "var": var_N_n, "is_weights":weights_IS})


# Output of resampled sample and original sample
test = importance_sampling_resampling(type = "systematic", N = 2000, sigma=5)
plt.hist(test["orig_sample"], bins = 35,density=True, alpha = 0.5)
plt.hist(test["resample_sample"], bins = 35,density=True, alpha = 0.5)

# Scatter plot of importance weights
plt.scatter(test["orig_sample"], test["is_weights"])
plt.title("Importance Sampling Weights for $N = 2000$ (Multinomial)")

####
# Observation of variance for N_n's
sigma_array = np.arange(1, 10, step = 0.2)
s_N = len(sigma_array)
M = 1000
N = 100
variance_mat_multinomial = np.zeros(shape = (s_N, M))
variance_mat_Bernoulli = np.zeros(shape = (s_N, M))
variance_mat_systematic = np.zeros(shape = (s_N, M))
for i, si in enumerate(sigma_array):
    for j in range(M):
        test_multinomial = importance_sampling_resampling(type = "multinomial", N = N, sigma = si)
        test_Bernoulli = importance_sampling_resampling(type = "Bernoulli", N = N, sigma = si)
        test_systematic = importance_sampling_resampling(type = "systematic", N = N, sigma = si)
        variance_mat_multinomial[i, j] = test_multinomial["var"]
        variance_mat_Bernoulli[i, j] = test_Bernoulli["var"]
        variance_mat_systematic[i, j] = test_systematic["var"]

var_avg_multinomial = np.mean(variance_mat_multinomial, axis = 1)
var_avg_Bernoulli = np.mean(variance_mat_Bernoulli, axis = 1)
var_avg_systematic = np.mean(variance_mat_systematic, axis = 1)

fig,ax = plt.subplots(nrows = 3, ncols = 1, figsize = (10,8))
fig.tight_layout(pad = 2.0)
ax[0].set_title("Multinomial resample averages of $N_n$ v.s. variances $\sigma^2$")
ax[0].plot(sigma_array, var_avg_multinomial)
ax[0].tick_params('x', labelbottom=False)

# share x only
ax[1] = plt.subplot(312, sharex=ax[0], sharey=ax[0])
ax[1].set_title("Bernoulli resample averages of $N_n$ v.s. variances $\sigma^2$")
ax[1].plot(sigma_array, var_avg_Bernoulli)
# make these tick labels invisible
ax[1].tick_params('x', labelbottom=False)

# share x and y
ax[2] = plt.subplot(313, sharex=ax[0], sharey=ax[0])
ax[2].set_title("Systematic resample averages of $N_n$ v.s. variances $\sigma^2$")
ax[2].plot(sigma_array, var_avg_systematic)
#plt.xlim(0.01, 5.0)
plt.show()


######################################################### Exercise 29

# defining the number of steps
n = 200
 
#creating two array for containing x and y coordinate
#of size equals to the number of size and filled up with 0's
x = np.zeros(n)
y = np.zeros(n)
final_n = 0
d = 100    
taken_positions = [(0,0)]

# Force random walk to Z^2_d lattice and be SAW! 
for i in range(1, n):
    val = np.random.randint(1, 5)
    good_direc = []
    
    # check feasibility of (1,0):
    if ((x[i-1]+1, y[i-1]) not in taken_positions and x[i-1]+1 <= d-1):
        good_direc.append((1,0))
        
    # check feasibility of (-1,0):
    if ((x[i-1]-1, y[i-1]) not in taken_positions and x[i-1]-1 >= 0):
        good_direc.append((-1,0))
        
    # check feasibility of (0,1):
    if ((x[i-1], y[i-1]+1) not in taken_positions and y[i-1]+1 <= d-1):
        good_direc.append((0,1))
    
    # check feasibility of (0,-1):
    if ((x[i-1], y[i-1]-1) not in taken_positions and y[i-1]-1 >= 0):
        good_direc.append((0,-1))

    if len(good_direc) == 0:
        direc = (0,0) 
    else: 
        direc = random.choice(good_direc)
        
    #print(direc)
    x[i] = x[i-1] + direc[0]
    y[i] = y[i-1] + direc[1]
    taken_positions.append((x[i], y[i]))

# plotting stuff:
pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
pylab.plot(x, y)
pylab.savefig("rand_walk"+str(n)+".png",bbox_inches="tight",dpi=600)
pylab.show()

# Sample a single SAW in Z^2_d lattice using the observation of nearby neighbors approach
def sample_myopic_saw(n, dimension = d):
    # Force random walk to Z^2_d lattice and be SAW! 
    x = np.zeros(n) # x positions
    y = np.zeros(n) # y positions
    num_nghs = np.zeros(n) # number of neighbors to visit (legal!)
    num_nghs[0] = 2
    taken_positions = [(0,0)] # positions that are already occupied
    
    for i in range(1, n):
        good_direc = []
        # check feasibility of (1,0):
         # check feasibility of (1,0):
        if ((x[i-1]+1, y[i-1]) not in taken_positions and x[i-1]+1 <= d-1):
            good_direc.append((1,0))

        # check feasibility of (-1,0):
        if ((x[i-1]-1, y[i-1]) not in taken_positions and x[i-1]-1 >= 0):
            good_direc.append((-1,0))

        # check feasibility of (0,1):
        if ((x[i-1], y[i-1]+1) not in taken_positions and y[i-1]+1 <= d-1):
            good_direc.append((0,1))

        # check feasibility of (0,-1):
        if ((x[i-1], y[i-1]-1) not in taken_positions and y[i-1]-1 >= 0):
            good_direc.append((0,-1))

        if len(good_direc) == 0:
            direc = (0,0) 
        else: 
            num_nghs[i] = len(good_direc)
            direc = random.choice(good_direc)

        #print(direc)
        x[i] = x[i-1] + direc[0]
        y[i] = y[i-1] + direc[1]
        taken_positions.append((x[i], y[i]))
    return((x,y,num_nghs))

def sample_one_step_SAW(x,y,d,taken_positions):
    num_nghs = 0
    good_direc = []
    # check feasibility of (1,0):
    if ((x+1, y) not in taken_positions and x + 1 <= d-1):
        good_direc.append((1,0))

    # check feasibility of (-1,0):
    if ((x-1, y) not in taken_positions and x - 1 >= 0):
        good_direc.append((-1,0))

    # check feasibility of (0,1):
    if ((x, y+1) not in taken_positions and y + 1 <= d-1):
        good_direc.append((0,1))

    # check feasibility of (0,-1):
    if ((x, y-1) not in taken_positions and y - 1 >= 0):
        good_direc.append((0,-1))

    if len(good_direc) == 0:
        direc = (0,0) 
    else: 
        num_nghs = len(good_direc)
        direc = random.choice(good_direc)
    #print(direc)
    x_new = x + direc[0]
    y_new = y + direc[1]
    taken_positions.append((x,y))
    return((x_new, y_new, num_nghs, taken_positions))
    
# Function which checks the number of times a specific lattice site is visitied (as detailed in assignment)
def visit_func(x, y, num_nghs, lattice_site = (5,5)):
    x_0 = lattice_site[0]; y_0 = lattice_site[1] 
    num = 0
    for i in range(len(x)):
        if x[i] == x_0 and y[i] == y_0:
            num = num+1
    return(num)

# Function which returns 1 if the random walk eventually crashes into itself
def crash_func(x, y, num_nghs):
    if 0 in num_nghs:
        return(1)
    else:
        return(0)
        
# Function which returns the distance from the origin at the final point:
def mean_square_dist(x, y, num_nghs):
    return(x[len(x)-1]**2 + y[len(y)-1]**2)

################################################################## Sequential Importance Sampling for SAW(d):
# f - function averaged over w.r.t. uniform measure on SAW(d)
# d - dimension of the random walk (and length!)
# N - number of samples averaged over in the SIS procedure
def sequential_importance_sampling_SAW(f, d, norm_cnst = False, N = 20):
    # N samples from myopic SAW
    X = np.zeros((N, d)); Y = np.zeros((N, d)); NGHS = np.zeros((N,d))
    for i in range(N):
        x, y, nghs = sample_myopic_saw(n = d, dimension = d)
        X[i,:] = x; Y[i,:] = y; NGHS[i,:] = nghs
    # to compute normalized importance sampling weights:
    weights_mat = np.zeros((N, d))
    for j in range(d):
        if j == 0:
            weights = NGHS[:,j]
            weights_mat[:,j] = weights/np.sum(weights)
        else:
            temp_wght = np.multiply(weights_mat[:,j-1],NGHS[:,j])
            weights_mat[:,j] = temp_wght/np.sum(temp_wght)
    is_weights = weights_mat[:, d-1]
    f_vals = [f(X[i,:], Y[i,:], NGHS[i,:]) for i in range(N)]
    if norm_cnst == False:
        return(np.sum(np.multiply(is_weights, f_vals)))
    else:
        norm_const_avg = np.average(np.prod(NGHS, axis = 1))
        return(np.sum(np.multiply(is_weights, f_vals)), norm_const_avg)

################################################################## Resampling Sequential Importance Sampling for SAW(d):
def resample_sequential_importance_sampling_SAW(f, d, N = 20):
    # compute up to the dimensionality of the target measure
    # X = history of x-coordinate
    # Y = history of y-coordinate
    # NGHS = history of # of neighbors
    # X, Y, NGHS are updated during the next sampling step and copied/shuffled during resampling step
    X = np.zeros((N, d)); Y = np.zeros((N, d)); NGHS = np.zeros((N,d))
    
    # Canonically, at (0,0), there are always 2 available neighbors,
    NGHS[:,0] = 2*np.ones(shape = (N,))
    
    # Taken position is origin at initial step
    taken_positions = [[(0,0)] for i in range(N)]
    
    # Importance weights
    weights_mat = np.zeros((N, d-1))

    # size of walks
    for j in range(d-1):
        # samples
        for i in range(N):
            (X[i,j+1], Y[i,j+1], NGHS[i,j+1], ts_i)  = sample_one_step_SAW(X[i,j],Y[i,j],d,taken_positions = taken_positions[i])
            taken_positions[i] = ts_i
        
        if j == 0:
                weights = NGHS[:,j+1]
                weights_mat[:,j] = weights/np.sum(weights)
                multinomial_N = np.random.multinomial(n = N, pvals = weights_mat[:,j])
                copy_X = np.zeros((1,d)); copy_Y = np.zeros((1,d)); copy_NGHS = np.zeros((1,d)); copy_taken_positions = []
                #multinomial_N = np.ones(N, dtype=int)
                ### Resampling
                # Do multinomial resampling
                for idx, val in enumerate(multinomial_N):
                    curr_trajectory_x = X[idx,:] # idx'th object
                    curr_trajectory_y = Y[idx, :]
                    curr_nghs = NGHS[idx, :]
                    curr_taken_positions = taken_positions[idx]
                    for k in range(val):
                        copy_X = np.concatenate((copy_X, np.reshape(curr_trajectory_x, (1,d))), axis = 0)
                        copy_Y = np.concatenate((copy_Y, np.reshape(curr_trajectory_y, (1,d))), axis = 0)
                        copy_NGHS = np.concatenate((copy_NGHS, np.reshape(curr_nghs, (1,d))), axis = 0)
                        copy_taken_positions.append(curr_taken_positions)
                X = np.delete(copy_X, (0), axis=0)
                Y = np.delete(copy_Y, (0), axis=0)
                NGHS = np.delete(copy_NGHS, (0), axis=0)
                taken_positions = copy_taken_positions
        else:
                #weights_mat[:,j] = NGHS[:, j+1]/np.sum(NGHS[:, j+1])
                temp_wght = np.multiply(weights_mat[:,j-1],NGHS[:,j+1])
                weights_mat[:,j] = temp_wght/np.sum(temp_wght)
                #weights_mat[:,j] = np.array([temp_wght[ll]/np.sum(temp_wght) if NGHS[ll,j+1] != 0 else 0 for ll in range(len(temp_wght))])
                multinomial_N = np.random.multinomial(n = N, pvals = weights_mat[:,j])
                multinomial_N = np.ones(N, dtype=int)

                copy_X = np.zeros((1,d)); copy_Y = np.zeros((1,d)); copy_NGHS = np.zeros((1,d)); copy_taken_positions = []
                ### Resampling
                # Do multinomial resampling
                for idx, val in enumerate(multinomial_N):
                    curr_trajectory_x = X[idx,:] # idx'th object
                    curr_trajectory_y = Y[idx, :]
                    curr_nghs = NGHS[idx, :]
                    curr_taken_positions = taken_positions[idx]
                    # Copy according to 
                    for k in range(val):
                        copy_X = np.concatenate((copy_X, np.reshape(curr_trajectory_x, (1,d))), axis = 0)
                        copy_Y = np.concatenate((copy_Y, np.reshape(curr_trajectory_y, (1,d))), axis = 0)
                        copy_NGHS = np.concatenate((copy_NGHS, np.reshape(curr_nghs, (1,d))), axis = 0)
                        copy_taken_positions.append(curr_taken_positions)
                X = np.delete(copy_X, (0), axis=0)
                Y = np.delete(copy_Y, (0), axis=0)
                NGHS = np.delete(copy_NGHS, (0), axis=0)
                taken_positions = copy_taken_positions
    # final weights
    is_weights = weights_mat[:, d-2]
    # final sample average
    f_vals = [f(X[i,:], Y[i,:], NGHS[i,:]) for i in range(N)]

    return(np.sum(np.multiply(is_weights, f_vals)))


############# Testing of methods
#### Different functions to test go in f argument
vals = [resample_sequential_importance_sampling_SAW(f = mean_square_dist, d = 15, N = 100) for i in range(500)]
vals_2 = [sequential_importance_sampling_SAW(f = mean_square_dist, norm_cnst=False, d = 15, N = 100) for i in range(500)]
plt.hist(vals, bins = 45, alpha = 0.6)
plt.hist(vals_2, bins = 45, alpha = 0.6)

# Checking normalizing constant growth
norm_const_vec = []
for i in range(1, 20):
    nc_vals_2 = [sequential_importance_sampling_SAW(f = visit_func, norm_cnst = True,d = i+1, N = 500)[1] for j in range(500)]
    norm_const_vec.append(np.average(nc_vals_2))
plt.plot(np.arange(1,20), (norm_const_vec), 'o-')
plt.title("IS Estimates of Normalizing Constant $\mathcal{Z}_d$")
plt.xlabel("Dimension $d$ of SAW(d)")

plt.plot(np.arange(1,20), np.log(norm_const_vec), 'o-')
plt.title("IS Estimates of Normalizing Constant $\mathcal{Z}_d$ on $\log_{10}$ scale")
plt.xlabel("Dimension $d$ of SAW(d)")
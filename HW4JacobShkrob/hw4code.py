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


###
# README: run by writing python hw4code.py, it should just run everything I used to generate the project
#  
class IsingModel:
    def __init__(self, N, beta) -> None:
        self.N = N
        self.beta = beta
        self.config = np.random.choice([-1,1], size = (self.N, self.N))
        self.mg = []
    '''
    Checks to see if given site is within the grid size N x N
    '''
    @jit
    def density_Ising(self):
        d = 0
        for i in range(self.N):
            for j in range(self.N):
                if j+1 < self.N:
                    d += self.config[i,j]*self.config[i, j+1] 
                if i+1 < self.N:
                    d += self.config[i,j]*self.config[i+1, j]
        return(np.exp(self.beta * d))

    def magnetization(self):
        return(np.sum(self.config)/self.N**2)
    
    @staticmethod
    @jit(nopython = True)
    def bc_check(N, site):
        i = site[0]
        j = site[1]
        if (0 <= i < N) and (0 <= j < N):
            return True
        else:
            return False

    @staticmethod
    def compute_nn(config, beta, N, i, j):
        E = 0 
        if IsingModel.bc_check(N, (i-1, j)):
            E = (beta * config[i-1, j]) + E
        if IsingModel.bc_check(N, (i,j-1)): 
            E = (beta * config[i, j-1]) + E
        if IsingModel.bc_check(N, (i+1,j)): 
            E = (beta * config[i+1, j]) + E 
        if IsingModel.bc_check(N, (i,j+1)):  
            E = (beta * config[i, j+1]) + E
        return(E)

def GibbsSampler(Lattice: IsingModel, iter_num = 100, type = "random"):
    if type == "random":
        for n in range(iter_num):
            ## Random index selection
            i = random.randrange(Lattice.N)
            j = random.randrange(Lattice.N)
            E = IsingModel.compute_nn(Lattice.config, Lattice.beta, Lattice.N, i, j)
            p = np.exp(E) / (np.exp(E) + np.exp(-1*E))
            t = np.random.binomial(1, p, size = 1)
            if t == 1:
                Lattice.config[i,j] = 1
            elif t == 0:
                Lattice.config[i,j] = -1
            # Updates to magnetization
            Lattice.mg.append(Lattice.magnetization())
    elif type == "deterministic":
        ## gibbs conditioning step sampling
        # Loop floor(iter_num / N^2) times
        r = iter_num % Lattice.N**2
        lp = (iter_num - r) / Lattice.N**2
        for w in range(int(lp)):
            for i in range(Lattice.N):
                for j in range(Lattice.N):
                    # check boundary conditions:
                    E = IsingModel.compute_nn(Lattice.config, Lattice.beta, Lattice.N, i, j)
                    p = np.exp(E) / (np.exp(E) + np.exp(-1*E))
                    t = np.random.binomial(1, p, size = 1)
                    if t == 1:
                        Lattice.config[i,j] = 1
                    elif t == 0:
                        Lattice.config[i,j] = -1
                    Lattice.mg.append(Lattice.magnetization())
        # remainder looping:
        it = 0
        while(it <= r):
            for i in range(Lattice.N):
                for j in range(Lattice.N):
                    it+=1 
                    # check boundary conditions:
                    E = IsingModel.compute_nn(Lattice.config, Lattice.beta, Lattice.N, i, j)
                    p = np.exp(E) / (np.exp(E) + np.exp(-1*E))
                    t = np.random.binomial(1, p, size = 1)
                    if t == 1:
                        Lattice.config[i,j] = 1
                    elif t == 0:
                        Lattice.config[i,j] = -1
                    Lattice.mg.append(Lattice.magnetization())

### Shows sampled element of lattice
lattice = IsingModel(N = 100, beta = 0.05)
plt.imshow(lattice.config, interpolation='nearest', cmap='jet')
# after 10000 iterations
GibbsSampler(lattice, iter_num = 10000, type='random')
plt.imshow(lattice.config, interpolation='nearest', cmap='jet')

# Plot magnetization:
def GibbsSamplerMagnetization(n = 100, N = 10, beta=1.0, iter_num=1000, type="deterministic"):
    mg_hist = []
    for i in range(n):
        lattice = IsingModel(N = N, beta = beta)
        GibbsSampler(lattice, iter_num = iter_num, type=type)
        mg_hist.append(lattice.mg[iter_num-1])
    return(mg_hist)
mg_hist = GibbsSamplerMagnetization(n = 1000, beta = 1.0, N = 50, iter_num = 3000, type="random")
plt.hist(mg_hist, bins = 50)
plt.xlabel("Histogram of (normalized) magnetization f($\sigma$)")
GibbsSampler(lattice, iter_num = 10000, type='random')
plt.imshow(lattice.config, interpolation='nearest', cmap='jet')

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
'''
IAT_Gibbs:
Intergrated autocorrelation time computation
for magnetization under certain number of iterations

Parameters:
N - lattice number
beta - inverse temperature
iter_num - number of iterations 
type - 'random' or 'deterministic' (deterministic shuffling)
'''
def IAT_Gibbs(N, beta, iter_num, type):
    lattice = IsingModel(N = N, beta = beta)
    GibbsSampler(lattice, iter_num = iter_num, type=type)
    return(autocorr(np.asarray(lattice.mg)))

def autocorr_func_results(n = 10):
    tau1 = [IAT_Gibbs(N = 50, beta = 1.0, iter_num=1000, type = "deterministic") for i in range(n)]
    tau2 = [IAT_Gibbs(N = 50, beta = 1.0, iter_num = 1000, type = "random") for i in range(n)]
    return((tau1, tau2))

def autocorr_func_results_temp(n = 100):
    betas = np.arange(0.1, 5, 0.1)
    tau = [(1/n)*sum([IAT_Gibbs(N = 50, beta = b, iter_num=2000, type = "random") for i in range(n)]) for b in betas]
    return(tau)

'''
Fixed temperature at beta = 1.0
Fixed scheme as 'deterministic'
Fixed number of mcmc steps as 10000
'''
def autocorr_func_results_size(n = 100):
    sizes = np.arange(10,50,2)
    tau = [(1/n)*sum([IAT_Gibbs(N = ss, beta = 1.0, iter_num=3000, type = "random") for i in range(n)]) for ss in sizes]
    return(tau)

(tau_deter, tau_rand) = autocorr_func_results(n = 300)
bins = np.linspace(20, 380, 70)
plt.hist(tau_deter, alpha = 0.8, bins=bins, label = "deterministic")
plt.hist(tau_rand, alpha = 0.8, bins=bins, label = "random")
plt.xlabel("Histogram of $IAT_f$")
plt.legend(fontsize = 10)

TEMP_RESULTS = autocorr_func_results_temp(n = 300)
SIZE_RESULTS = autocorr_func_results_size(n = 300)
# size (lattice number) plot
plt.plot(np.arange(10, 50, 2), SIZE_RESULTS, "o--", label = "")
plt.xlabel("$IAT_f$ plotted against lattice number $L$ ($\mathcal{Z}^2_L$)")
# temp (beta) plot
plt.plot(np.arange(0.1, 5, 0.1), TEMP_RESULTS, "o--")
plt.xlabel("$IAT_f$ plotted against temperature parameter")
# temperature (betas) plot
plt.plot(np.arange(0.1, 5, 0.1), TEMP_RESULTS, "o-")
plt.ylabel("$IAT_f$ plotted against temperature $\beta$")

def MetropolisMCMC(Lattice: IsingModel, iter_num = 100):
    for n in range(iter_num):
        ## Random index selection
        i = random.randrange(Lattice.N)
        j = random.randrange(Lattice.N)
        old_val = Lattice.config[i,j]
        Lattice.config[i,j] *= -1
        E = IsingModel.compute_nn(Lattice.config, Lattice.beta, Lattice.N, i, j)
        R = np.exp(-1*4*old_val*E)
        acc_p = min(1, R)
        t = np.random.binomial(1, acc_p, size = 1)
        if t == 0:
            Lattice.config[i,j] = old_val        
        else:
            Lattice.config[i,j] = Lattice.config[i,j]    
        # Updates
        Lattice.mg.append(Lattice.magnetization())
def MetropolisSamplerMagnetization(n = 100, N = 10, beta=1.0, iter_num=1000):
    mg_hist = []
    for i in range(n):
        lattice = IsingModel(N = N, beta = beta)
        MetropolisMCMC(lattice, iter_num = iter_num, type=type)
        mg_hist.append(lattice.mg[iter_num-1])
    return(mg_hist)
def IAT_Metropolis(N, beta, iter_num):
    lattice = IsingModel(N = N, beta = beta)
    MetropolisMCMC(lattice, iter_num = iter_num)
    return(autocorr(np.asarray(lattice.mg)))

def autocorr_func_results_metr(n = 10):
    tau1 = [IAT_Metropolis(N = 50, beta = 1.0, iter_num=1000) for i in range(n)]
    return(tau1)

tau_metr = autocorr_func_results_metr(n = 300)
plt.hist(tau_metr, alpha = 0.6, bins=35, label = "metropolis")
plt.hist(tau_rand, alpha = 0.6, bins=35, label = "random")
plt.legend()
plt.xlabel("Histogram of $IAT_f$")
print(np.average(tau_rand))
print(np.average(tau_metr))


############# atempt at Jarzynski's method, kind of a failure
def JarzynskiMethod(Lattice_List, jm_N = 1000, type = "Metropolis"):
    # assume lenght of Lattice_List and num_chains are the same\
    N = Lattice_List[0].N
    num_chains = len(Lattice_List)
    mg = np.empty((num_chains, jm_N))
    # weights initialization
    weights = np.ones(num_chains)
    w_k = np.ones(num_chains)
    # Initial spins are already iid
    if type == "Metropolis":
        for k in range(jm_N):
            for m in range(num_chains):
                Lattice = Lattice_List[m]
                # Metropolis but with a varying π 
                i = random.randrange(Lattice.N)
                j = random.randrange(Lattice.N)
                old_val = Lattice.config[i,j]
                Lattice.config[i,j] *= -1
                E = IsingModel.compute_nn(Lattice.config, Lattice.beta, Lattice.N, i, j)
                # Change π^{k/N} here
                R = np.exp(-1*4*Lattice.beta*((k+1)/jm_N)*old_val*E)
                acc_p = min(1, R)
                t = np.random.binomial(1, acc_p, size = 1)
                if t == 0:
                    Lattice.config[i,j] = old_val        
                else:
                    Lattice.config[i,j] = Lattice.config[i,j]    
                # Updates
                w_k[m] = (Lattice.density_Ising())
                mg[m, k] = Lattice.magnetization()
            W = np.multiply(weights, w_k)
            weights = W/np.sum(W)
            mg[:,k] = np.multiply(mg[:,k], weights)
    elif type == "Gibbs":
        for k in range(jm_N):
            for m in range(num_chains):
                Lattice = Lattice_List[m]            
                ## Random index selection
                i = random.randrange(Lattice.N)
                j = random.randrange(Lattice.N)
                E = IsingModel.compute_nn(Lattice.config, Lattice.beta*((k+1)/N), Lattice.N, i, j)
                p = np.exp(E) / (np.exp(E) + np.exp(-1*E))
                t = np.random.binomial(1, p, size = 1)
                if t == 1:
                    Lattice.config[i,j] = 1
                elif t == 0:
                    Lattice.config[i,j] = -1
                # Updates to magnetization
                w_k[m] = Lattice.density_Ising()
                mg[m,k] = (Lattice.magnetization())
            W = np.multiply(weights, w_k)
            weights = W/np.sum(W)
            mg[:,k] = np.multiply(mg[:,k], weights)

    return(mg)
lattices = [IsingModel(50, 1.0) for i in range(30)]
out = JarzynskiMethod(lattices, jm_N=1000)
tau_jm = [autocorr(out[i]) for i in range(len(out))]
(np.average(tau_jm))
bins = np.linspace(0, 150, 40)
plt.hist(tau_jm, alpha = 0.8, bins=bins, color="green")
plt.hist(tau_metr, alpha = 0.8, bins = bins)
plt.hist(tau_rand, alpha = 0.8, bins = bins)

mag_vals_weighted = np.average(out, axis = 1)
plt.hist(mag_vals_weighted, alpha = 0.6, bins = 30, range = (-0.5,0.5))
mag_vals_weighted
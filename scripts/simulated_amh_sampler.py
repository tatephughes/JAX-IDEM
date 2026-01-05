print("Script started. Importing modules...", flush=True)

import jax
# Some filters, particularily the sqrt filters, may work in 32-bit, but expect instabilities.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jax.scipy.linalg import solve
from jax.scipy.linalg import solve_triangular as st
from jax.scipy.stats.multivariate_normal import logpdf as lmvn
from jax.scipy.stats.norm import logpdf as lpnorm

# Module imports
import jaxidem.utils as utils
from jaxidem.utils import add_variance
import jaxidem.idem as idem
import jaxidem.filters as filts

import jax.lax as jl
import jax.random as jr
from jax.random import multivariate_normal as smvn
from jax.random import normal as snorm

from functools import partial

from tqdm.auto import tqdm

import csv
from datetime import datetime
import os
import pickle

print("This is adaptive metropolis with a prior on the simulated data example.")

print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Creating model...", flush=True)

seed = 2
key = jr.PRNGKey(seed)
keys = jr.split(key, 10)


amh_n = 100000


T = 20
nobs = 100
process_basis = utils.place_cosine_basis(N=10) 

# This puts a point of mass at 0.1,0.9 and a point of 0 at 0.9, 0.1.
# Using the least squares estimator, this creates a point at the corner.
inp_data = jnp.array([[0.1, 0.9, 100],
#                      [0.9, 0.1, 0],
                      [0.3, 0.9, 100],
                      [0.1, 0.7, 100],
                      [0.3, 0.7, 100]])


# create a 'ball' at the top left using least squares
PHI = process_basis.mfun(inp_data[:,0:2])
alpha_0 = jnp.linalg.pinv(PHI.T@PHI)@PHI.T @ inp_data[:,2]


# this function creates models like in this example. for practical uses, use `init_model`.

# Fudged numbers to look at least a little interesting.
K_basis = (
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
)
# These values create reasonable drift and diffusion at
s = 0.0001
k = (
    jnp.array([1/(2*jnp.pi*s)]), # Scale parameter    (θ_1)
    jnp.array([0.5*s]), # Shape paramter  (θ_2)
    jnp.array([-0.025]), # X-axis drift    (θ_3) 
    jnp.array([0.025]), # Y-axis drift     (θ_4)
)
kernel = idem.param_exp_kernel(K_basis, k)

model = idem.gen_example_idem(keys[0],
                              process_basis = process_basis,
                              beta = jnp.array([1.0]),
                              kernel=kernel)




coords = jr.uniform(
                keys[0],
                shape=(nobs, 2),
                minval=0,
                maxval=1,
            )
times = jnp.repeat(jnp.arange(1, T + 1), coords.shape[0])
rep_coords = jnp.tile(coords, (T, 1))
x = rep_coords[:,0]
y = rep_coords[:,1]


process_data, obs_data = model.simulate(keys[1], x, y, times, alpha_0 = alpha_0)

# For normalising axes
zmin = float(jnp.nanmin(obs_data.z))
zmax = float(jnp.nanmax(obs_data.z))
zbounds = [zmin, zmax]






# Model to fit to the data

K_basis = (
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
)
s = 0.002
k = (
    jnp.array([1/(2*jnp.pi*s)]), # Scale parameter    (θ_1)
    jnp.array([2*s]), # Shape paramter  (θ_2)
    jnp.array([-0.00]), # X-axis drift    (θ_3) 
    jnp.array([0.00]), # Y-axis drift     (θ_4)
)
kernel = idem.param_exp_kernel(K_basis, k)

model_0 = idem.gen_example_idem(keys[0],
                                process_basis = process_basis,
                                beta = jnp.array([0.0]),
                                kernel=kernel,
                                S2_eta = 0.01**2,
                                S2_eps = 0.01**2)



print("Done!", flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Creating log marginal...", flush=True)


log_marginal = model_0.get_log_like(obs_data, method="sqinf", likelihood='partial', P_0 = 1000*jnp.eye(process_basis.nbasis))



print('Log marginal made!', flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Creating priors and posterior...", flush=True)


alp_eta_0 = 2
lam_eta_0 = model_0.S2_eta
V = jnp.eye(model_0.nbasis)
# variance of process noise is S2_eta * V

# S2_eps prior
alp_eps_0 = 2
lam_eps_0 = model_0.S2_eps

# beta prior
p = model_0.beta.size
s2_beta_0 = 1
mu_beta_0 = model_0.beta

p = model.beta.size

init_tker = model_0.params["trans_kernel_params"]
# Kernel prior and general posterior pdf
@jax.jit
def kern_prior(tker_params):

    return (lpnorm(tker_params[0], init_tker[0], 1) 
            + lpnorm(tker_params[1], init_tker[1], 10)
            + lpnorm(tker_params[2], init_tker[2], 0.1)
            + lpnorm(tker_params[3], init_tker[3], 0.1))[0]


def inv_gamma_logpdf(x, alpha, beta):

    return (-alpha-1)*jnp.log(x) - beta/x

def log_post(params):
    
    log_beta_prior = lmvn(params['beta'], mu_beta_0, s2_beta_0 * jnp.eye(p))

    log_eta_prior = inv_gamma_logpdf(jnp.exp(params['log_S2_eta']), alp_eta_0, lam_eta_0)
    log_eps_prior = inv_gamma_logpdf(jnp.exp(params['log_S2_eps']), alp_eps_0, lam_eps_0)

    log_kern_prior = kern_prior(params['trans_kernel_params'])

    return log_marginal(params) + log_beta_prior + log_eta_prior + log_eps_prior + log_kern_prior

    
print('Log posterior made!', flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)




#directory = os.path.dirname(os.path.abspath(__file__))
#directory = "~/Projects/JAX-IDEM/scripts/"

script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)



current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
csv_file = os.path.join(directory, f'results/{current_datetime}_results_adaptive_prior.csv')
pickle_file = os.path.join(directory, f'results/{current_datetime}_results_adaptive.pickle')




back_key, sample_key = jr.split(key, 2)





init = model_0.params
init_flat, unflat = utils.flatten_and_unflatten(init)


init_mean = init_flat
x = init_mean
x_mean = init_mean
accept_count = 0
d = x.shape[0]
prop_cov = 0.1*jnp.eye(d)

mix = True

acc_count = 0




for j in range(1, amh_n):
    
    amh_key = jr.fold_in(sample_key, j)

    mix_key, prop_key, acc_key = jr.split(amh_key, 3)
    
    #keys = jr.split(prop_key,3)
    eps = 0.05

    
    prop = jl.cond((j <= 5*d) | (mix & (jr.uniform(mix_key) < eps)),
                   lambda key: jr.normal(key, shape=(d,))/(jnp.sqrt(1000*d)) + x, # 'Safe' sampler
                   lambda key: jr.multivariate_normal(key, x, prop_cov), # 'Adaptive' sampler
                   prop_key)
    
    # Compute the log Hastings ratio

    val = log_post(unflat(x))
    alpha = log_post(unflat(prop)) - val
    
    log_prob = jnp.minimum(0.0, alpha)
  
    u = jr.uniform(acc_key)

    x_new, is_accepted = jl.cond((jnp.log(u) < log_prob),
                                 0, lambda _: (prop, 1),
                                 0, lambda _: (x, 0))

    acc_count = acc_count + is_accepted
    
    # update empirical mean
    x_mean_new = (x_mean*j + x_new)/(j+1)
  
    # update proposal covariance
    # update proposal covariance
    prop_cov_new = jnp.select(condlist   = [mix | (j<2*d), (not mix) & (j>=2*d)],
                              choicelist = [
                                  prop_cov*((j-1)/j) +
                                  (j*jnp.outer(x_mean-x_mean_new, x_mean-x_mean_new) +
                                   jnp.outer(x_new - x_mean_new, x_new - x_mean_new)
                                   )*5.6644/(j*d),
                                  prop_cov*((j-1)/j) +
                                  (j*jnp.outer(x_mean-x_mean_new, x_mean-x_mean_new) +
                                   jnp.outer(x_new - x_mean_new, x_new - x_mean_new) +
                                   0.01*jnp.identity(d)
                                   )*5.6644/(j*d)],
                              default = 1)

    x= x_new
    x_mean = x_mean_new
    prop_cov = prop_cov_new

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(jnp.concatenate([jnp.array([is_accepted]), x]))


    
    if j%10 == 0:
        print(f"Acceptance rate is {acc_count/j}", flush=True)
        print(f"Current value is {x}", flush=True)
        # Save the PyTree to a file using pickle
        with open(pickle_file, 'wb') as file:
            pickle.dump((j, x, x_mean, prop_cov), file)

        print(f"Current density is {val}", flush=True)
        


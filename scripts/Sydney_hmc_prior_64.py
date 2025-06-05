import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as rand
import optax
import pandas as pd
import pickle

from tqdm.auto import tqdm

import jaxidem.idem as idem
import jaxidem.utils as utils

import importlib
import os
dir = os.path.dirname(os.path.abspath(__file__))
from datetime import datetime

import sys
hmc_n = 1_000


print("This is HMC with a strong prior on shape and scale. (64-bit precision)")

print("Done!", flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Reading and censoring data...", flush=True)


radar_df = pd.read_csv(os.path.join(dir, 'radar_df.csv'))
# Censor the data!
radar_df_censored = radar_df
# remove the final time measurements (for forecast testing)
radar_df_censored = radar_df_censored[radar_df_censored['time'] != "2000-11-03 08:45:00"]
# remove the a specific time (for intracast testing)
radar_df_censored = radar_df_censored[radar_df_censored['time'] != "2000-11-03 10:15:00"]
# three randomly chose indices ('dead pixels')
import numpy as np
np.random.seed(42) # reproducibility (jax.random is used elsewhere)
random_indices = np.random.choice(radar_df_censored.index, size=300, replace=False)
radar_df_censored = radar_df_censored.drop(random_indices)


# no covariates (besides intercept)
radar_data = utils.pd_to_st(radar_df_censored, 's2', 's1', 'time', 'z')

print("Done!", flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Creating model...", flush=True)

sigma2_eta = jnp.var(radar_data.z)/2
sigma2_eps = jnp.var(radar_data.z)/2
beta = jnp.array([0.]) # only intercept, no covariates

# stations are stationary and there is no missing data, so there is the same number of observations per time period
#nobs = radar_data.as_wide()['x'].size 

process_basis = utils.place_basis(data = radar_data.coords,
                                  nres = 2,
                                  min_knot_num = 3,) # defaults to bisquare basis functions

process_grid = utils.create_grid(radar_data.bounds, jnp.array([41, 41]))
int_grid = utils.create_grid(radar_data.bounds, jnp.array([100, 100]))

const_basis = utils.constant_basis
K_basis = (const_basis,const_basis,const_basis,const_basis)
k = (jnp.array([150]) / (process_grid.area*process_grid.ngrid), # kernel scale
    jnp.array([0.002]) * (process_grid.area*process_grid.ngrid), # kernel shape
    jnp.array([0.]), # x drift
    jnp.array([0.])) # y drift
kernel = idem.param_exp_kernel(K_basis, k)

model = idem.Model(process_basis=process_basis,
                   kernel=kernel,
                   process_grid=process_grid,
                   sigma2_eta=sigma2_eta,
                   sigma2_eps=sigma2_eps,
                   beta=beta,
                   int_grid=int_grid)

print("Done!", flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Creating log marginal...", flush=True)

log_marginal = model.get_log_like(radar_data, method="sqinf", likelihood='partial', P_0 = 1000*jnp.eye(process_basis.nbasis))

print('Log marginal made!', flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)



lmvn = jax.scipy.stats.multivariate_normal.logpdf

import pickle
with open(os.path.join(dir,'./pickles/Hamilton/24_4_25/mle_params.pkl'), 'rb') as file: 
    mle_params = pickle.load(file)

params = mle_params

def log_prior(params):
    return (-0.5*(params.trans_kernel_params[0] - mle_params.trans_kernel_params[0])**2/0.1**2
            -0.5*(params.trans_kernel_params[1] - mle_params.trans_kernel_params[1])**2/0.1**2)[0]

def log_post(params):
    return log_prior(params) + log_marginal(params)

fparams, unflat = utils.flatten_and_unflatten(params)

init_mean = fparams.astype(jnp.float64)

# no more reproducibility!
rng_key = jax.random.PRNGKey(np.random.choice(range(1000000)))

parshape = init_mean.shape
npars = parshape[0]

#with open(os.path.join(dir, 'pickles/Hamilton/24_4_25/rmh_sample.pkl'), 'rb') as file:
#    rmh_sample, acc_ratio = pickle.load(file)

with open(os.path.join(dir,'./pickles/adaptive_params_prior.pkl'), 'rb') as file: 
    j, x, x_mean, prop_cov = pickle.load(file)

back_key, sample_key = jax.random.split(rng_key, 2)

# Build the kernel
step_size = 5e-3
imm = prop_cov.astype(jnp.float64)
# / (5.6644/7)

import blackjax

hmc = blackjax.hmc(lambda flatpars: log_post(unflat(flatpars)), step_size, imm, num_integration_steps=10)

# Initialize the state
state = hmc.init(init_mean)



hmc_sample = []

# Iterate
step = jax.jit(hmc.step)

import csv
from datetime import datetime

# Example data
current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

csv_file = os.path.join(dir, f'results/{current_datetime}_results_hmc_prior_64.csv')



for i in tqdm(range(hmc_n), desc="Sampling... "):

    hmc_key = jax.random.fold_in(sample_key, i)
    state, info = step(hmc_key, state)

    accepted = jnp.asarray(info.is_accepted, dtype=int).reshape((1,))
    hmc_sample.append(jnp.concatenate([accepted, state.position]))
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(jnp.concatenate([accepted, state.position]))


#hmc_sample_arr = jnp.array(hmc_sample)

acc_ratio = info.acceptance_rate
print(f"Acceptance rate: {acc_ratio}")
#post_mean = jnp.mean(hmc_sample_arr[int(hmc_n/3):,:], axis=0)
#print(post_mean)
#post_params_mean = unflat(post_mean)
#idem.print_params(post_params_mean)


print("Pickling...")
with open(os.path.join(dir, 'pickles/hmc_sample.pkl'), 'wb') as file:
    pickle.dump((jnp.array(hmc_sample), acc_ratio), file)
print("Done!")

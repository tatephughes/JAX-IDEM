import jax
import jax.numpy as jnp
import jax.random as rand
import optax
import pandas as pd
import pickle

from tqdm.auto import tqdm

import jaxidem.idem as idem
import jaxidem.utils as utils

import importlib

importlib.reload(idem)
importlib.reload(utils)

import os
dir = os.path.dirname(os.path.abspath(__file__))


rmh_n = 100

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

radar_data = utils.pd_to_st(radar_df_censored, 's2', 's1', 'time', 'z')

sigma2_eta = jnp.var(radar_data.z)/2
sigma2_eps = jnp.var(radar_data.z)/2
beta = jnp.array([0.]) # only intercept, no covariates

# stations are stationary and there is no missing data, so there is the same number of observations per time period
nobs = radar_data.wide['x'].size 

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
log_marginal = model.get_log_like(radar_data, method="sqinf", likelihood='partial', P_0 = 1000*jnp.eye(process_basis.nbasis))


import pickle
with open('./pickles/Hamilton/24_4_25/mle_params.pkl', 'rb') as file: 
    mle_params = pickle.load(file)

params = mle_params


fparams, unflat = utils.flatten_and_unflatten(params)

init_mean = fparams
rng_key = jax.random.PRNGKey(1)
parshape = init_mean.shape
npars = parshape[0]

#with open(os.path.join(dir, 'pickles/Hamilton/24_4_25/rmh_sample.pkl'), 'rb') as file:
#    rmh_sample, acc_ratio = pickle.load(file)



# Load the CSV file using NumPy
csv_data = np.loadtxt(os.path.join(dir, 'results/2025-04-30_09:46:25_results.csv'), delimiter=',')
# Convert the NumPy array to a JAX array
rmh_sample_arr = jnp.array(csv_data)
    
prop_var = (2.38**2/7) *jnp.cov(rmh_sample_arr[:,1:].T)
prop_sd = jnp.linalg.cholesky(prop_var, upper=True)

back_key, sample_key = jax.random.split(rng_key, 2)

sample_keys = jax.random.split(sample_key, rmh_n)


init_state = jnp.concatenate([jnp.array([1.0]), init_mean])


rmh_sample = [init_state]
accepted = 0

import csv
from datetime import datetime

# Example data
current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

csv_file = os.path.join(dir, f'results/{current_datetime}_results.csv')


for i in tqdm(range(rmh_n), desc="Sampling... "):
    current_state = rmh_sample[-1][1:]
    prop_key, acc_key = jax.random.split(sample_keys[i], 2)
    
    proposal = current_state + prop_sd @ jax.random.normal(prop_key, shape=parshape)
    r = log_marginal(unflat(proposal)) - log_marginal(unflat(current_state))

    log_acc_prob = min((jnp.array(0.0), r))
    if jnp.log(jax.random.uniform(acc_key)) > log_acc_prob:
        new_state = jnp.concatenate([jnp.array([0.0]), current_state])
    else:
        accepted = accepted + 1
        new_state = jnp.concatenate([jnp.array([1.0]), proposal])

    rmh_sample.append(new_state)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_state)

acc_ratio = accepted/rmh_n
rmh_sample_arr = jnp.array(rmh_sample)

print(f"Acceptance rate: {acc_ratio}")
post_mean = jnp.mean(rmh_sample_arr[int(rmh_n/3):,1:], axis=0)
print(post_mean)
post_params_mean = unflat(post_mean)
idem.print_params(post_params_mean)

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

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


hmc_n = 1000

radar_df = pd.read_csv(os.path.join(dir, 'radar_df.csv'))
# Censor the data!
radar_df_censored = radar_df

# remove the final time measurements (for forecast testing)
radar_df_censored = radar_df_censored[radar_df_censored['time'] != "2000-11-03 08:45:00"]

# remove the a specific time (for intracast testing)
radar_df_censored = radar_df_censored[radar_df_censored['time'] != "2000-11-03 10:15:00"]

# three randomly chose indices ('dead pixels')
import numpy as np
#np.random.seed(42) # reproducibility (jax.random is used elsewhere)
random_indices = np.random.choice(radar_df_censored.index, size=300, replace=False)
radar_df_censored = radar_df_censored.drop(random_indices)

radar_data = utils.pd_to_st(radar_df_censored, 's2', 's1', 'time', 'z')

print("Done!")

print("Creating model...")

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

print("Done!")
print("Creating log marginal...")

log_marginal = model.get_log_like(radar_data, method="sqinf", likelihood='partial', P_0 = 1000*jnp.eye(process_basis.nbasis))

print('Log marginal made!')

lmvn = jax.scipy.stats.multivariate_normal.logpdf

import pickle
with open(os.path.join(dir,'./pickles/Hamilton/24_4_25/mle_params.pkl'), 'rb') as file: 
    mle_params = pickle.load(file)

params = mle_params


fparams, unflat = utils.flatten_and_unflatten(params)

init_mean = fparams

# no more reproducibility!
rng_key = jax.random.PRNGKey(np.random.choice(range(1000000)))

parshape = init_mean.shape
npars = parshape[0]

#with open(os.path.join(dir, 'pickles/Hamilton/24_4_25/rmh_sample.pkl'), 'rb') as file:
#    rmh_sample, acc_ratio = pickle.load(file)



# Load the CSV file using NumPy
csv_data = np.loadtxt(os.path.join(dir, './results/bigruns/2025-05-08_09:17:44_results_rmh-32core.csv'), delimiter=',')
# Convert the NumPy array to a JAX array

hmc_sample_prev = jnp.array(csv_data)

# with open(os.path.join(dir, 'pickles/hmc_sample.pkl'), 'rb') as file: 
#     hmc_sample_prev, _ = pickle.load(file)
#     hmc_sample_prev = jnp.array(hmc_sample_prev)

prop_var = jnp.cov(hmc_sample_prev[300000:,1:].T)
prop_sd = jnp.linalg.cholesky(prop_var, upper=True)

back_key, sample_key = jax.random.split(rng_key, 2)

# Build the kernel
step_size = 1e-2
inverse_mass_matrix = prop_var

import blackjax

hmc = blackjax.hmc(lambda flatpars: log_marginal(unflat(flatpars)), step_size, inverse_mass_matrix, num_integration_steps=25)

# Initialize the state
state = hmc.init(init_mean)

hmc_sample = []

# Iterate
step = jax.jit(hmc.step)

import csv
from datetime import datetime

# Example data
current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

csv_file = os.path.join(dir, f'results/{current_datetime}_results_hmc.csv')



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

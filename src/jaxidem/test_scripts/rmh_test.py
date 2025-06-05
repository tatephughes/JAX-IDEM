import sys
import os
sys.path.append(os.path.abspath('../src/jaxidem'))

import jax
import jax.numpy as jnp
import jax.random as rand
import idem
import utilities

from tqdm import tqdm

import pandas as pd
radar_df = pd.read_csv('../../data/radar_df.csv')

radar_data = utilities.st_data(x = jnp.array(radar_df.s1),
                               y = jnp.array(radar_df.s2),
                               t = jnp.array(radar_df['t'].astype('category').cat.codes + 1.),
                               z = jnp.array(radar_df.z))

unique_times = jnp.unique(radar_data.t)
T = unique_times.size


radar_data_wide = radar_data.as_wide()

#initial variances
sigma2_eta = jnp.var(radar_data.z)/2
sigma2_eps = jnp.var(radar_data.z)/2
beta = jnp.array([0.]) # only intercept, no covariates

# stations are stationary and there is no missing data, so there is the same number of observations per time period
nobs = radar_data_wide['x'].size 
X_obs = jnp.ones((nobs,1)) # intercept

station_locs = jnp.column_stack([radar_data_wide["x"], radar_data_wide["y"]])

process_basis = utilities.place_basis(data = station_locs,
                                      nres = 2,
                                      min_knot_num = 3,) # defaults to bisquare basis functions

xmin = jnp.min(station_locs[:, 0])
xmax = jnp.max(station_locs[:, 0])
ymin = jnp.min(station_locs[:, 1])
ymax = jnp.max(station_locs[:, 1])
bounds = jnp.array([[xmin, xmax], [ymin, ymax]])

process_grid = utilities.create_grid(bounds, jnp.array([41, 41]))
int_grid = utilities.create_grid(bounds, jnp.array([100, 100]))

K_basis = (
    utilities.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1), # contant basis
    utilities.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utilities.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utilities.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
)
k = (
    jnp.array([150])/(process_grid.area*process_grid.ngrid) ,
    jnp.array([0.002])/(process_grid.area*process_grid.ngrid),
    jnp.array([0.]),
    jnp.array([0.])
)
kernel = idem.param_exp_kernel(K_basis, k)


model0 = idem.IDEM(process_basis=process_basis,
                   kernel=kernel,
                   process_grid=process_grid,
                   sigma2_eta=sigma2_eta,
                   sigma2_eps=sigma2_eps,
                   beta=beta,
                   int_grid=int_grid)

log_marginal = model0.get_log_like(radar_data, [X_obs for _ in range(T)], method="sqinf", likelihood='full')

init_mean = model0.params
prop_var = idem.IdemParams(log_sigma2_eps=jnp.array(1.),
                           log_sigma2_eta=jnp.array(1.),
                           trans_kernel_params=(jnp.array([1.]),
                                                jnp.array([1.]),
                                                jnp.array([1.]),
                                                jnp.array([1.])),
                           beta=jnp.array([1.]))

def sample_gaussian(key):
    return mean+jax.random.normal(key, shape=prop_var.shape) * jnp.sqrt(variance)

nparams = sum(arr.size for arr in jax.tree.leaves(init_mean))

def sample_gaussian_noise(key):
    keys = jax.random.split(key, nparams)
    keys_tree = idem.IdemParams(log_sigma2_eps=jnp.array(keys[0]),
                                log_sigma2_eta=jnp.array(keys[1]),
                                trans_kernel_params=(jnp.array([keys[2]]),
                                                     jnp.array([keys[3]]),
                                                     jnp.array([keys[4]]),
                                                     jnp.array([keys[5]])),
                                beta=jnp.array([keys[6]]))
    tup = (keys_tree, means, variances)
    return jax.tree.map(sample_gaussian, tup)
    

rng_key = jax.random.PRNGKey(3)




# Function to recursively split keys to match the PyTree structure
def split_keys_tree(key, pytree):
    leaves, treedef = jax.tree.flatten(pytree)
    keys = jax.random.split(key, num=len(leaves))
    return jax.tree.unflatten(treedef, keys)

# Function to sample from Gaussian
def sample_gaussian(variance, subkey):
    return jax.random.normal(subkey, shape=variance.shape) * jnp.sqrt(variance)

def sample_gaussian_tree(key):
    keys_tree = split_keys_tree(key, prop_var)
    sampled_prop_var = jax.tree.map(
        lambda var, subkey: sample_gaussian(var, subkey),
        prop_var,
        keys_tree
    )
    return sampled_prop_var

n=100
keys = jax.random.split(rng_key, n)

sample = [init_mean]

accepted = 0

for i in tqdm(range(n), desc="Sampling... "):

    current_state = sample[-1]

    sample_key, acc_key = jax.random.split(keys[i], 2)
    
    proposal = jax.tree.map(jnp.add, current_state, sample_gaussian_tree(sample_key))

    r = log_marginal(proposal) - log_marginal(current_state)
    log_acc_prob = min((jnp.array(0.0), r))

    if jnp.log(jax.random.uniform(acc_key)) > log_acc_prob:
        sample.append(current_state)
    else:
        sample.append(proposal)
        accepted = accepted + 1

acc_ratio = accepted/n

sample_stack = jax.tree.map(lambda *arrays: jnp.stack(arrays), *sample)
new_mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), sample_stack)
new_var = jax.tree.map(lambda x: jnp.var(x, axis=0), sample_stack)

prop_var= jax.tree_map(lambda x: x * (2.38**2)/nparams, new_var)

# (side note, why not use the adaptive metropois from last year? might be easy)


# masybe use 2.38^2/d?

def sample_gaussian_tree(key):
    keys_tree = split_keys_tree(key, prop_var)
    sampled_prop_var = jax.tree.map(
        lambda var, subkey: sample_gaussian(var, subkey),
        new_var,
        keys_tree
    )
    return sampled_prop_var


# round 2 with updated variance

n=1000
keys = jax.random.split(rng_key, n)

sample = [new_mean]

accepted = 0

for i in tqdm(range(n), desc="Sampling... "):

    current_state = sample[-1]

    sample_key, acc_key = jax.random.split(keys[i], 2)
    
    proposal = jax.tree.map(jnp.add, current_state, sample_gaussian_tree(sample_key))

    r = log_marginal(proposal) - log_marginal(current_state)
    log_acc_prob = min((jnp.array(0.0), r))

    if jnp.log(jax.random.uniform(acc_key)) > log_acc_prob:
        sample.append(current_state)
    else:
        sample.append(proposal)
        accepted = accepted + 1

acc_ratio = accepted/n

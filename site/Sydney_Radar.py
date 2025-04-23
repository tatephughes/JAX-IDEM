import jax
import jax.numpy as jnp
import jax.random as rand
import numpy as np
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


mle_n = int(input("How many maximum likelihood iterations?: (int)"))
rmh_n = int(input("How many RMH samples?"))
mala_n = int(input("How many MALA samples?"))
hmc_n = int(input("How many HMC samples?"))


radar_df = pd.read_csv(os.path.join(dir, '../data/radar_df.csv'))


# Censor the data!
radar_df_censored = radar_df

# remove the final time measurements (for forecast testing)
#radar_df_censored = radar_df_censored[radar_df_censored['time'] != "2000-11-03 08:45:00"]

# remove the a specific time (for intracast testing)
#radar_df_censored = radar_df_censored[radar_df_censored['time'] != "2000-11-03 10:15:00"]

# three randomly chose indices ('dead pixels')
#np.random.seed(42) # reproducibility (jax.random is used elsewhere)
#random_indices = np.random.choice(radar_df_censored.index, size=300, replace=False)
#radar_df_censored = radar_df_censored.drop(random_indices)

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


import optax

# create a function to find the value and gradient of the negative log likelihood
nll_val_grad = jax.value_and_grad(lambda par: -log_marginal(par))

init_nll, _ = nll_val_grad(model.params)

optimizer = optax.adam(1e-1)
opt_state = optimizer.init(model.params)

params = model.params

if mle_n > 0:
    for i in tqdm(range(mle_n), desc="Optimising"):
        nll, grad = nll_val_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        #idem.print_params(params)

        # Save the PyTree to a file using pickle
        with open(os.path.join(dir, 'pickles/mle_params.pkl'), 'wb') as file:
            pickle.dump(params, file)


# results from R-IxDE
#ride_k = (
#    jnp.log(jnp.array([0.1353081])),
#    jnp.log(jnp.array([2.49728])),
#    jnp.array([-5.488385]),
#    jnp.array([-1.860784]),
#)
#ride_params = idem.IdemParams(log_sigma2_eps=jnp.log(28.38365),
#                                 log_sigma2_eta=jnp.log(7.270833),
#                                 trans_kernel_params = ride_k,
#                                 beta = jnp.array([0.5816754]))




# rmh sampling

fparams, unflat = utils.flatten_and_unflatten(model.params)

init_mean = fparams

# initial run gave the following for estimated optimal tuning
#prop_var = jnp.array([0.16133152, 0.00453646, 0.01214727, 0.392362, 0.789936, 0.41011548, 0.14044523])
#prop_var = jnp.array([0.0051443721, 0.00011324495, 0.0092018154, 0.42602387, 0.14025699, 0.045133684, 0.10179958])
#prop_var = jnp.array([0.24664642, 0.00706895, 0.01684521, 0.01354943, 0.5177155, 0.20242365, 0.75817305])

# hand-tuned
prop_var = jnp.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01])
prop_sd = jnp.diag(jnp.sqrt(prop_var))
prop_var = jnp.diag(prop_var)

# or we cheat
#with open(os.path.join(dir, 'pickles/rmh_sample_14_04.pkl'), 'rb') as file:
#    rmh_sample, acc_ratio = pickle.load(file)
#prop_var = (2.38**2/7) *jnp.cov(jnp.array(rmh_sample).T)
#prop_sd = jnp.linalg.cholesky(prop_var, upper=True)
#init_mean = rmh_sample[-1]



rng_key = jax.random.PRNGKey(1)
parshape = init_mean.shape
npars = parshape[0]



back_key, sample_key = jax.random.split(rng_key, 2)

sample_keys = jax.random.split(sample_key, rmh_n)

rmh_sample = [init_mean]
accepted = 0

if rmh_n > 0:
    for i in tqdm(range(rmh_n), desc="Sampling... "):
        current_state = rmh_sample[-1]
        prop_key, acc_key = jax.random.split(sample_keys[i], 2)

        proposal = current_state + prop_sd @ jax.random.normal(prop_key, shape=parshape)
        r = log_marginal(unflat(proposal)) - log_marginal(unflat(current_state))
        #print(jnp.exp(r))
        log_acc_prob = min((jnp.array(0.0), r))
        if jnp.log(jax.random.uniform(acc_key)) > log_acc_prob:
            rmh_sample.append(current_state)
        else:
            accepted = accepted + 1
            rmh_sample.append(proposal)

    acc_ratio = accepted/rmh_n
    # Save the PyTree to a file using pickle
    with open(os.path.join(dir, 'pickles/rmh_sample.pkl'), 'wb') as file:
        pickle.dump((rmh_sample, acc_ratio), file)

    print(f"Acceptance rate: {acc_ratio}")
    post_mean = jnp.mean(jnp.array(rmh_sample[int(len(rmh_sample)/3):]), axis=0)
    print(post_mean)
    post_params_mean = unflat(post_mean)
    idem.print_params(post_params_mean)

# hand-tuned
prop_var = 0.2*jnp.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01])
prop_sd = jnp.diag(jnp.sqrt(prop_var))
prop_var = jnp.diag(prop_var)

back_key, sample_key = jax.random.split(back_key, 2)

sample_keys = jax.random.split(sample_key, mala_n)


ll_val_grad = jax.value_and_grad(lambda par: log_marginal(par))

# start from the end of the last chain
mala_sample = [init_mean]
# use estimated theoretically optimalish proposal variance
#with open(os.path.join(dir, 'pickles/rmh_sample.pkl'), 'rb') as file:
#    rmh_sample, acc_ratio = pickle.load(file)
#prop_var = (2.38**2/7**3) *jnp.cov(jnp.array(rmh_sample[int(len(rmh_sample)/3):]).T)
#prop_sd = jnp.linalg.cholesky(prop_var) 

accepted = 0
lmvn = jax.scipy.stats.multivariate_normal.logpdf

if mala_n > 0:
    for i in tqdm(range(mala_n), desc="Sampling... "):
        current_state = mala_sample[-1]
        prop_key, acc_key = jax.random.split(sample_keys[i], 2)

        val, grad = ll_val_grad(unflat(current_state))
        grad, _ = utils.flatten_and_unflatten(grad)

        mean = 0.5 * prop_var @ grad + current_state

        proposal = (mean + prop_sd @ jax.random.normal(prop_key, shape=parshape))

        r = (log_marginal(unflat(proposal)) - val
             + lmvn(current_state, mean, prop_var) - lmvn(proposal, mean, prop_var))
        log_acc_prob = min((jnp.array(0.0), r))
        
        if jnp.log(jax.random.uniform(acc_key)) > log_acc_prob:
            mala_sample.append(current_state)
        else:
            accepted = accepted + 1
            mala_sample.append(proposal)
        #print(r, i, current_state[4:6])

    acc_ratio = accepted/mala_n
    # Save the PyTree to a file using pickle
    with open(os.path.join(dir, 'pickles/mala_sample.pkl'), 'wb') as file:
        pickle.dump((mala_sample, acc_ratio), file)

    print(f"Acceptance rate: {acc_ratio}")
    post_mean = jnp.mean(jnp.array(mala_sample[int(len(mala_sample)/3):]), axis=0)
    print(post_mean)
    post_params_mean = unflat(post_mean)
    idem.print_params(post_params_mean)


from jax_tqdm import scan_tqdm
import blackjax

if hmc_n > 0:
    
    imm = prop_sd
    num_int = 3
    samp = blackjax.hmc(log_marginal, 1e-3, imm, num_int)
    kernel=samp.step
    init = samp.init(model.params)

#    @scan_tqdm(hmc_n, desc='Sampling...')
    def step(carry, i):
        nuts_key = jax.random.fold_in(rng_key, i)
        new_state, info = kernel(nuts_key, carry)
        return new_state, (new_state, info)

    _, (hmc_sample, info) = jax.lax.scan(step, init, jnp.arange(hmc_n))

    post_mean = jax.tree.map(lambda arr: jnp.mean(arr, axis=0), hmc_sample.position)
    idem.print_params(post_mean)

    with open(os.path.join(dir, 'pickles/hmc_sample.pkl'), 'wb') as file:
        pickle.dump((hmc_sample, info), file)



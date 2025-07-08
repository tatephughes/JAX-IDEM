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
importlib.reload(idem)
importlib.reload(utils)
import os
dir = os.path.dirname(os.path.abspath(__file__))

import csv
from datetime import datetime
rmh_n = 10_000

print("This is RMH (manual version) with a strong prior on shape and scale. (64-bit precision)")

print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Reading and censoring data...", flush=True)

radar_df = pd.read_csv(os.path.join(dir, 'data/radar_df.csv'))
radar_df_censored = radar_df
radar_df_censored = radar_df_censored[radar_df_censored['time'] != "2000-11-03 08:45:00"]
radar_df_censored = radar_df_censored[radar_df_censored['time'] != "2000-11-03 10:15:00"]
import numpy as np
np.random.seed(42) # reproducibility (jax.random is used elsewhere)
random_indices = np.random.choice(radar_df_censored.index, size=300, replace=False)
radar_df_censored = radar_df_censored.drop(random_indices)

radar_data = utils.pd_to_st(radar_df_censored, 's2', 's1', 'time', 'z')


print("Done!", flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Creating model...", flush=True)

model = idem.init_model(radar_data)

print("Done!", flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Creating log marginal...", flush=True)

log_marginal = model.get_log_like(radar_data, method="inf", likelihood='partial', P_0 = 1000*jnp.eye(model.process_basis.nbasis))

print('Log marginal made!', flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)


def log_prior(params):
    return (-0.5*(params.trans_kernel_params[0] - mle_params.trans_kernel_params[0])**2/0.1**2
            -0.5*(params.trans_kernel_params[1] - mle_params.trans_kernel_params[1])**2/0.1**2)[0]

def log_post(params):
    return log_prior(params) + log_marginal(params)



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


with open(os.path.join(dir,'./pickles/adaptive_params_prior.pkl'), 'rb') as file: 
    j, x, x_mean, prop_cov = pickle.load(file)


back_key, sample_key = jax.random.split(rng_key, 2)

sample_keys = jax.random.split(sample_key, rmh_n)


init_state = jnp.concatenate([jnp.array([1.0]), init_mean])


rmh_sample = [init_state]
state = init_state
accepted = 0


# Example data
current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

csv_file = os.path.join(dir, f'results/{current_datetime}_results_rmh_maual_64.csv')


for i in tqdm(range(rmh_n), desc="Sampling... "):
    current_state = state[1:]
    prop_key, acc_key = jax.random.split(sample_keys[i], 2)
    
    proposal = rand.multivariate_normal(prop_key, current_state, prop_cov)
    r = log_post(unflat(proposal)) - log_post(unflat(current_state))

    log_acc_prob = min((jnp.array(0.0), r))
    if jnp.log(jax.random.uniform(acc_key)) > log_acc_prob:
        state = jnp.concatenate([jnp.array([0.0]), current_state])
    else:
        accepted = accepted + 1
        state = jnp.concatenate([jnp.array([1.0]), proposal])

#    rmh_sample.append(new_state)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(state)

acc_ratio = accepted/rmh_n
#rmh_sample_arr = jnp.array(rmh_sample)

print(f"Acceptance rate: {acc_ratio}")
#post_mean = jnp.mean(rmh_sample_arr[int(rmh_n/3):,1:], axis=0)
#print(post_mean)
#post_params_mean = unflat(post_mean)
#idem.print_params(post_params_mean)

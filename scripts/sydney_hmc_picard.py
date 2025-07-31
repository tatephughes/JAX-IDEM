import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import pandas as pd
import pickle

from jax_tqdm import scan_tqdm

import jaxidem.idem as idem
import jaxidem.utils as utils

import importlib
import os
dir = os.path.dirname(os.path.abspath(__file__))
#dir = "/home/tate/Projects/JAX-IDEM/scripts"
from datetime import datetime

import sys

key = jax.random.PRNGKey(2)


print(f"output of jax.devices: {jax.devices()}")

print("This is RMH testing using the picard map, on the Sydney Radar data with a strong prior on the shape and scale. (no censoring)")

print("Done!", flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Reading data...", flush=True)

radar_df = pd.read_csv(os.path.join(dir, 'data/radar_df.csv'))
radar_data = utils.pd_to_st(radar_df, 's2', 's1', 'time', 'z')

print("Done!", flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Creating model...", flush=True)

sigma2_eta = jnp.var(radar_data.z)/2
sigma2_eps = jnp.var(radar_data.z)/2
beta = jnp.array([0.]) # only intercept, no covariates

process_basis = utils.place_cosine_basis(data = radar_data.coords, N=10)

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
print("Creating log marginal (inf)...", flush=True)

log_marginal = model.get_log_like(radar_data, method="inf", likelihood='partial', P_0 = 1000*jnp.eye(process_basis.nbasis))

print('Log marginal made!', flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Constructing posterior...", flush=True)

lmvn = jax.scipy.stats.multivariate_normal.logpdf

with open(os.path.join(dir,'./pickles/mle_params.pkl'), 'rb') as file: 
    mle_params = pickle.load(file)

    # promote to 64 bit
    mle_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), mle_params)

params = mle_params

def log_prior(params):
    return (-0.5*(params.trans_kernel_params[0] - mle_params.trans_kernel_params[0])**2/0.1**2
            -0.5*(params.trans_kernel_params[1] - mle_params.trans_kernel_params[1])**2/0.1**2)[0]

def log_post(params):
    return log_prior(params) + log_marginal(params)

print('Posterior done!', flush=True)
print("Current Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
print("Setting up RMH...", flush=True)




def flatten(tree):
    flat_leaves, treedef = jax.tree.flatten(tree)
    flat_array = jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves])
    return flat_array



fparams, unflat_params = utils.flatten_and_unflatten(params)

init_mean = fparams

with open(os.path.join(dir,'./pickles/adaptive_params_prior.pkl'), 'rb') as file: 
    j, x, x_mean, prop_cov = pickle.load(file)

rng_key = jax.random.PRNGKey(1)

# Build the kernel
imm = prop_cov.astype(jnp.float64)

import blackjax

rmh = blackjax.additive_step_random_walk(lambda x: log_post(unflat_params(x)), blackjax.mcmc.random_walk.normal(imm))

# Initialize the state
init_state = rmh.init(init_mean)


from jax_tqdm import scan_tqdm


n = 1000

@scan_tqdm(n)
def rmh_step(state, i):
    key = keys[i]
    new_state, _ = rmh.step(key, state)
    return new_state, new_state

keys = jax.random.split(rng_key, 10000000)

import time



start = time.time()
_, seq = jax.lax.scan(rmh_step, init_state, jnp.arange(n))
chain = jax.tree.map(lambda x,y: jnp.concatenate([x[None, ...],y]), init_state, seq)
end = time.time()

X_true = jnp.concatenate([jnp.reshape(x, (x.shape[0], -1)) for x in jax.tree.leaves(chain)], axis=-1)
accepted = jnp.any(X_true[1:] != X_true[:-1], axis=1)
acceptance_rate = jnp.mean(accepted)
print(f"Acceptance rate is approximately: {100*acceptance_rate:6f}%.")

seq_time = end - start

print(f"Sequential RMH took {seq_time:.6f} seconds")





@jax.jit
def rmh_step(state, i):
    key = keys[i]
    new_state, _ = rmh.step(key, state)
    return new_state, new_state

def get_picard_map(step, init_state, n):
    finit_state, unflat = utils.flatten_and_unflatten(init_state)

    X_0 = jnp.stack([finit_state for _ in range(n+1)])

    vf = jax.vmap(lambda s, i: flatten(step(unflat(s), i)[0]) - s)

    @jax.jit
    def picard_map(fstates, indices):
        finit_state = fstates[0]
        #fs = vf((fstates[0:-1], indices))
        fs = vf(fstates[0:-1], indices)
        cumsum = jnp.cumsum(fs, axis=0) + finit_state
        return jnp.vstack([finit_state, cumsum])
    
    return X_0, picard_map

X_0, pic_map = get_picard_map(rmh_step, init_state, n=n)


# TRYING TO APPLY THE MAP TO THE WHOLE CHAIN WILL GIVE A MEMORY ERROR
# X_n = pic_map(X_0, jnp.arange(n))

# will work on smaller subsets though (up to K=~40)
K = 10
print(f"Doing {K-1} computations of the log-density in parallel. Reduce if you get a memory error.")
#print(pic_map(X_0[0:K], jnp.arange(0, K-1)))


# Instead, we define the online step, which only applies the map to a subset of the chain

#def get_online_step(pic_map, K):
#
#    def op_step(states, L):
#        U = L+K
#        
#        states_to_update = states[L:U,:]
#        inds_to_update_with = jnp.arange(L, U-1)
#        
#        updated_states = pic_map(states_to_update, inds_to_update_with)
#        #states_new = jax.lax.dynamic_update_slice(states, updated_states, (L,0))
#
#        states_new = states.at[L:U].set(updated_states)
#        
#        L_new = L + jnp.max(jnp.where(jnp.all(states_to_update == updated_states, axis=1))[0])+1
#        
#        return states_new, L_new
#    
#    return op_step


def jit_luc(pic_map, K):

    @jax.jit
    def online_step(tup):

        states, L = tup
        
        d = states.shape[1]

        states_to_update = jax.lax.dynamic_slice(states, (L,0), (K, d))
        #inds_to_update_with = jax.lax.dynamic_slice(keys, (L,), (K-1,))
        inds_to_update_with = jnp.arange(K-1) + L
        
        updated_states = pic_map(states_to_update, inds_to_update_with)

        states_new = jax.lax.dynamic_update_slice(states, updated_states, (L,0))

        states_incorrect = jnp.all(states_to_update != updated_states, axis=1)

        L_new = L + jnp.where(states_incorrect, size=1, fill_value=K-1)[0][0]
        
        return states_new, L_new
    
    return online_step



#op_step = get_online_step(pic_map, K)
jit_step = jit_luc(pic_map, K)


# X_0 now needs a 'buffer' to avoid index out of bounds problems
# Remember to cut the last K samples out of the chain
finit_state, unflat = utils.flatten_and_unflatten(init_state)
X_0 = jnp.stack([finit_state for _ in range(n+1+K)])


X_n = X_0


L = 0
#i=0
init_val = (X_0, jnp.array(L))

start = time.time()
X_n, L = jax.lax.while_loop(lambda val: val[1]<n, jit_step, init_val)
#while L < n:
#    i = i+1
#    new_states, L = op_step(X_n, L)
#    L = L.item()
#    X_n = new_states
#    if i % 5 ==0:
#        print(f"\rL = {L}, iteration {i}", end="\n", flush=True)
end = time.time()

par_time = end - start

#print(f"Convergence complete in {i} iterations.")
X_n = X_n[:n+1]
print(f"Converge Success: {jnp.allclose(X_n, X_true)}")
print(f"\nOnline-Picard accelerated RMH took {par_time:.6f} seconds")
print(f"Compared to Sequential RMH {seq_time:.6f} seconds")








print("\n Quickly testing the speed of the log-prior computations, single eval vs parallel eval")

rng_key = jax.random.PRNGKey(2)


f = jax.jit(lambda fpar: log_post(unflat_params(fpar)))
vf = jax.vmap(f)

f_tup = lambda tup: f(tup[0])
vf_tup = lambda tup: vf(tup[0])

sing_param = (fparams, )
array_params = (X_true[:, :-1][:K-1],)

sing_time = utils.time_jit(rng_key, f_tup, sing_param, n=10)
mult_time = utils.time_jit(rng_key, vf_tup, array_params, n=10)

print(f"\ncomputation of log-likelihood for a single parameter took {sing_time.average_time:.3f} seconds on average.")

print(f"computation of log-likelihood for {K-1} parameters took {mult_time.average_time:.3f} seconds, taking {mult_time.average_time/(K-1):.3f} per function evaluation.")








print("\n Quickly testing the speed of a much simpler computation, single eval vs parallel eval")

rng_key = jax.random.PRNGKey(2)


f = jax.jit(lambda fpar: jnp.sum(jnp.exp(fpar)))
vf = jax.vmap(f)

f_tup = lambda tup: f(tup[0])
vf_tup = lambda tup: vf(tup[0])

sing_param = (fparams, )
array_params = (X_true[:, :-1][:K-1],)

sing_time = utils.time_jit(rng_key, f_tup, sing_param, n=1000)
mult_time = utils.time_jit(rng_key, vf_tup, array_params, n=1000)

print(f"\ncomputation of sum-exp for a single parameter took {1000*sing_time.average_time:.3f} ms on average.")

print(f"computation of sum-exp for {K-1} parameters took {1000*mult_time.average_time:.3f} ms, taking {1000*mult_time.average_time/(K-1):.3f} ms per function evaluation.")

import sys
import os
sys.path.append(os.path.abspath('src/jaxidem'))
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.random as jr
import jax.numpy as jnp
import idem
import utilities
import filter_smoother_functions as fsf
import warnings
import matplotlib.pyplot as plt

import importlib
# unnecessary unless running interactively
importlib.reload(idem)
importlib.reload(utilities)
importlib.reload(fsf)

seed = 4
key = jr.PRNGKey(seed)
keys = jr.split(key, 3)
process_basis = utilities.place_basis(nres=1, min_knot_num=5)
model = idem.gen_example_idem(keys[0], k_spat_inv=True,
                              ngrid=jnp.array([40, 40]),
                              process_basis=process_basis)

nbasis = model.process_basis.nbasis

# Simulation
T = 9

nobs = jr.randint(keys[2], shape=(T,), minval=5, maxval=20)
obs_keys = jr.split(keys[3], T)

obs_locs = jnp.vstack([jnp.column_stack([jnp.repeat(i+1,n), jr.uniform(
                            obs_keys[i],
                            shape=(n, 2),
                            minval=0.0,
                            maxval=1.0,
                        )]) for i,n in enumerate(nobs)])

process_data, obs_data = model.simulate(key, T=T, obs_locs = obs_locs)

unique_times = jnp.unique(obs_data.t)
obs_locs = jnp.column_stack(jnp.column_stack((obs_data.x, obs_data.y))).T
obs_locs_tuple = [obs_locs[obs_data.t == t][:, 0:] for t in unique_times]

X_obs_tuple = jax.tree.map(
    lambda locs: jnp.column_stack((jnp.ones(len(locs)), locs)), obs_locs_tuple
)

zs_tree = [
            obs_data.z[obs_data.t == t] - X_obs_tuple[i] @ model.beta
            for i, t in enumerate(unique_times)
        ]
PHI_tree = jax.tree.map(model.process_basis.mfun, obs_locs_tuple)

max_len = max(z.shape[0] for z in zs_tree)
zs = jnp.array([jnp.pad(z, (0, max_len - z.shape[0]), constant_values=jnp.nan) for z in zs_tree])
PHIs = jnp.array([jnp.pad(PHI, ((0, max_len - PHI.shape[0]), (0, 0)), constant_values=jnp.nan) for PHI in PHI_tree])

def rm_nan(matrix):
    valid_rows = jnp.any(~jnp.isnan(matrix), axis=1)
    valid_cols = jnp.any(~jnp.isnan(matrix), axis=0)
    return matrix[valid_rows][:, valid_cols]

m_0 = jnp.zeros(nbasis)
P_0 = 100*jnp.eye(nbasis)

P_pred = jnp.eye(nbasis)
e = z - PHI_pad@m_0
Sigma = PHI_pad @ P_pred @ PHI_pad.T + 0.01*jnp.eye(PHI_pad.shape[0])
K_t = (jnp.linalg.solve(remove_nan_rows_and_columns(Sigma), remove_nan_rows_and_columns(PHI_pad)) @ remove_nan_rows_and_columns(P_pred).T).T



ll, nus, Qs, _,_ = model.information_filter(obs_data)
ms = jnp.linalg.solve(Qs, nus[..., None]).squeeze(-1)

filt_data = idem.basis_params_to_st_data(ms, model.process_basis, model.process_grid)


import jax.lax as jl

ll, ms, Ps, _ ,_ = kalman_filter_indep_vd(m_0, P_0, model.M, PHI_tree, model.sigma2_eta, model.sigma2_eps, zs_tree)



from jax import lax
def scan_fn(carry, z):
    result = jnp.dot(z,z)
    return result, result  # Carry is unchanged

# Example data
data = jnp.array([10, 20, 30, 40])

# Generate indices
indices = jnp.arange(len(data))

# Use lax.scan with indices and data together
_, output = lax.scan(scan_fn, init=None, xs=(indices, data))

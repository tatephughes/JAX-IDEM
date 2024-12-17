import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("../src/jax_idem"))

import jax
import utilities
import IDEM
from functools import partial

from utilities import *
from IDEM import *
import warnings

from tqdm import tqdm

import optax

key = jax.random.PRNGKey(12)
keys = rand.split(key, 5)


df = pd.read_csv("z_obs.csv")

date_mapping = {date: i-1 for i, date in enumerate(df['time'].unique(), 1)}
df['time'] = df['time'].map(date_mapping)

obs_data = ST_Data_Long(x = jnp.array(df['s1']), y=jnp.array(df['s2']), t=jnp.array(df['time']), z = jnp.array(df['z']))
obs_data_wide = ST_towide(obs_data)
obs_locs = jnp.array([obs_data_wide.x,obs_data_wide.y]).T
X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

K_basis = (
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
        )
# scale and shape of the kernel will be the same, but the offsets will be estimated
k = (
    jnp.array([150.]),
    jnp.array([0.002]),
    jnp.array([-0.]),
    jnp.array([0.]),
)
# This is the kind of kernel used by ```gen_example_idem```
kernel = param_exp_kernel(K_basis, k)

process_basis = place_basis()

#PHI_obs = process_basis.mfun(obs_locs)
#m_0 = PHI_obs.T @ (obs_data_wide.z[:,0] - X_obs@jnp.array([0.2, 0.2, 0.2]))
#print(m_0)

m_0 = jnp.zeros(process_basis.nbasis).at[64].set(1.)

model0 = IDEM(
        process_basis = process_basis,
        kernel=kernel,
        process_grid = create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([41, 41])),
        sigma2_eta = 0.01**2,
        sigma2_eps = 0.01**2,
        beta = jnp.array([0.2, 0.2, 0.2]),
        m_0 = m_0,
        sigma2_0 = 1000
)


ll0= model0.filter(obs_data_wide, X_obs)[0]

print(ll0)

quick_fit_model = model0.data_mle_fit(
    obs_data,
    X_obs,
    fixed_ind=["m_0","sigma2_0", "ks1", "ks2"],
    optimizer=optax.adam(1e-1),
    nits=10,
)

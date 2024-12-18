import sys
import os

sys.path.append(os.path.abspath("./src/jax_idem"))

import jax
import utilities
import IDEM

from jax.numpy.linalg import vector_norm, solve

from filter_smoother_functions import *
import warnings

import importlib

importlib.reload(utilities)
importlib.reload(IDEM)

from utilities import *
from IDEM import *

key = jax.random.PRNGKey(12)
keys = rand.split(key, 3)

process_basis = place_basis(nres=2, min_knot_num=5)
nbasis = process_basis.nbasis

m_0 = jnp.zeros(nbasis).at[16].set(1)
sigma2_0 = 0.001

model = gen_example_idem(
    keys[0], k_spat_inv=True, process_basis=process_basis, m_0=m_0, sigma2_0=sigma2_0
)


# Simulation
T = 9

nobs = jax.random.randint(keys[1], (T,), 10, 51)

locs_keys = jax.random.split(keys[2], T)

obs_locs = jnp.vstack(
    [
        jnp.column_stack(
            [
                jnp.repeat(t + 1, n),
                rand.uniform(
                    locs_keys[t],
                    shape=(n, 2),
                    minval=0,
                    maxval=1,
                ),
            ]
        )
        for t, n in enumerate(nobs)
    ]
)

# Simulation
process_data, obs_data = model.simulate(
    nobs=50, T=T, key=keys[1], obs_locs=obs_locs
)

obs_data_2 = st_data(obs_data.x, obs_data.y, obs_data.t, obs_data.z)

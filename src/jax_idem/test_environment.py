import sys
import os

sys.path.append(os.path.abspath("../src/jax_idem"))

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
T = 10
process_data, obs_data = model.simulate(nobs=50, T=T, key=keys[1])
obs_data_wide = ST_towide(obs_data)

print(T)

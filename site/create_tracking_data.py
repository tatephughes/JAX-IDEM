import jax
import jax.numpy as jnp

import jaxidem.filters as filt

import pandas as pd


key = jax.random.PRNGKey(1)

alpha_0 = jnp.ones(2)  # 2D, easily plottable
M = jnp.array([[jnp.cos(0.3), -jnp.sin(0.3)],
              [jnp.sin(0.3), jnp.cos(0.3)]])  # spinny

alphas = [alpha_0]
zs = []

T = 50
keys = jax.random.split(key, T*2)

sigma2_eta = jnp.array(0.001)
sigma2_eps = jnp.array(0.01)

PHI = jnp.array([[1, 0], [0.6, 0.4], [0.4, 0.6]])

for i in range(T):
    alphas.append(M @ alphas[i] + jnp.sqrt(sigma2_eta)*jax.random.normal(keys[2*i], shape=(2,)))
    zs.append(PHI @ alphas[i+1] + jnp.sqrt(sigma2_eps)*jax.random.normal(keys[2*i+1], shape=(3,)))

alphas = jnp.array(alphas)
zs_tree = zs
    
alphas_df = pd.DataFrame(alphas, columns = ["x", "y"])
zs_df = pd.DataFrame(zs, columns = ["x", "y", "z"])

import pickle

with open('./pickles/alphas.pkl', 'wb') as file:
    pickle.dump((alphas, zs_tree, alphas_df, zs_df), file)

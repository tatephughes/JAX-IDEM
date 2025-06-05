import jax
is_64_bit = True
jax.config.update("jax_enable_x64", False)
import os
import jax.numpy as jnp
import jaxidem.utils as utils
import jaxidem.idem as idem
import jaxidem.filters as filters

seed = 4
key = jax.random.PRNGKey(seed)
keys = jax.random.split(key, 10)
nreps = 100
T = 12

#process_basis = utils.place_cosine_basis(N = 5)
process_basis = idem.place_basis()

#sigma2_eta = (0.01*jnp.ones(process_basis.nbasis)).at[1].set(40.0).at[30].set(80.0).at[31].set(60.0)

sigma2_eta = 0.01

model = idem.gen_example_idem(keys[0], k_spat_inv=False, ngrid=jnp.array([40, 40]),
                              process_basis = process_basis, sigma2_eta = sigma2_eta,
                              covariate_labels = ['Intercept', 'x', 'y'])


coords = jax.random.uniform(
        keys[0],
        shape=(100, 2),
        minval=0,
        maxval=1,
    )

times = jnp.repeat(jnp.arange(1, T + 1), coords.shape[0])
rep_coords = jnp.tile(coords, (T, 1))
x = rep_coords[:,0]
y = rep_coords[:,1]
    
process_data, obs_data = model.simulate(key, x, y, times,
                                        covariates = jnp.column_stack([x,y]),)


nbasis = model.nbasis
zs_tree = obs_data.zs_tree
obs_locs = obs_data.coords_tree[0]
PHI_obs = model.process_basis.mfun(obs_locs)
m_0 = jnp.zeros(nbasis)
P_0 = 100 * jnp.eye(nbasis)

(
    log_sigma2_eps,
    log_sigma2_eta,
    ks,
    beta,
) = model.params

ztildes_tree = obs_data.tildify(beta)
logks1, logks2, ks3, ks4 = ks
ks1 = jnp.exp(logks1)
ks2 = jnp.exp(logks2)
sigma2_eta = jnp.exp(log_sigma2_eta)
sigma2_eps = jnp.exp(log_sigma2_eps)
M = model.con_M((ks1, ks2, ks3, ks4))


filt_results = filters.kalman_filter(
    m_0,
    P_0,
    M,
    PHI_obs,
    sigma2_eta,
    sigma2_eps,
    ztildes_tree,
    likelihood="full",
    sigma2_eta_dim = model.sigma2_eta_dim,
    sigma2_eps_dim = model.sigma2_eps_dim,
)

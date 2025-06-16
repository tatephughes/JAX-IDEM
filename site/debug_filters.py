import jax
is_64_bit = False
jax.config.update("jax_enable_x64", is_64_bit)
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

process_basis = utils.place_cosine_basis(N = 5)
#process_basis = idem.place_basis()

#sigma2_eta = (0.01*jnp.ones(process_basis.nbasis)).at[1].set(40.0).at[30].set(80.0).at[31].set(60.0)

sigma2_eta = 0.01

model = idem.gen_example_idem(keys[0], k_spat_inv=False, ngrid=jnp.array([40, 40]),
                              process_basis = process_basis, sigma2_eta = sigma2_eta,
                              covariate_labels = ['Intercept', 'x', 'y'])


coords = jax.random.uniform(
        keys[0],
        shape=(1000, 2),
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
nu_0 = jnp.zeros(nbasis)
P_0 = 100 * jnp.eye(nbasis)
U_0 = 10 * jnp.eye(nbasis)
Q_0 = 0.01 * jnp.eye(nbasis)
R_0 = 0.1 * jnp.eye(nbasis)


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

sigma2_eps_tree = [sigma2_eps for _ in range(T)]
PHI_obs_tree = [PHI_obs for _ in range(T)]

filt_results_1 = filters.kalman_filter(
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

filt_results_2 = filters.sqrt_filter(
    m_0,
    U_0,
    M,
    PHI_obs,
    sigma2_eta,
    sigma2_eps,
    ztildes_tree,
    likelihood="full",
    sigma2_eta_dim = model.sigma2_eta_dim,
    sigma2_eps_dim = model.sigma2_eps_dim,
)

filt_results_3 = filters.information_filter(
    nu_0,
    Q_0,
    M,
    PHI_obs_tree,
    sigma2_eta,
    sigma2_eps_tree,
    ztildes_tree,
    likelihood="full",
    sigma2_eta_dim = model.sigma2_eta_dim,
    sigma2_eps_dim = model.sigma2_eps_dim,
)

filt_results_4 = filters.sqrt_information_filter(
    nu_0,
    R_0,
    M,
    PHI_obs_tree,
    sigma2_eta,
    sigma2_eps_tree,
    ztildes_tree,
    likelihood="full",
    sigma2_eta_dim = model.sigma2_eta_dim,
    sigma2_eps_dim = model.sigma2_eps_dim,
)

log_marginal_A = model.get_log_like(obs_data, method="kalman", likelihood='full', P_0=P_0)
log_marginal_B = model.get_log_like(obs_data, method="sqrt", likelihood='full', P_0=P_0)
log_marginal_C = model.get_log_like(obs_data, method="inf", likelihood='full', P_0=P_0)
log_marginal_D = model.get_log_like(obs_data, method="sqinf", likelihood='full', P_0=P_0)

print(log_marginal_A(model.params))
print(log_marginal_B(model.params))
print(log_marginal_C(model.params))
print(log_marginal_D(model.params))

r = m_0.shape[0]
nobs = ztildes_tree[0].size


sigma_eta = jnp.sqrt(sigma2_eta) * jnp.eye(r)
sigma_eps = jnp.sqrt(sigma2_eps) * jnp.eye(nobs)

carry_a = (m_0, P_0, m_0, P_0, 0, jnp.zeros((r, nobs)))
carry_b = (m_0, U_0, m_0, U_0, 0, jnp.zeros((r, nobs)))


from jaxidem.filters import qr_R
from jax.scipy.linalg import solve_triangular as st

z_t = ztildes_tree[0]
PHI = PHI_obs

# Kalman First step

m_tt, P_tt, _, _, ll, _ = carry_a
m_pred = M @ m_tt
P_prop = M @ P_tt @ M.T
P_pred = jnp.fill_diagonal(P_prop, sigma2_eta + jnp.diag(P_prop), inplace=False)
e_t = z_t - PHI @ m_pred
P_oprop = PHI @ P_pred @ PHI.T
Sigma_t = jnp.fill_diagonal(P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False)
K_t = (jnp.linalg.solve(Sigma_t, PHI) @ P_pred.T).T
m_up_A = m_pred + K_t @ e_t
P_up = (jnp.eye(r) - K_t @ PHI) @ P_pred

# sqrt first step


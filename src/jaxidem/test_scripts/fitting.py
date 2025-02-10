import numpy as np
import timeit
import idem
import utilities
import filter_smoother_functions as fsf
import importlib
import jax
import jax.random as rand
import jax.numpy as jnp

import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('../src/jaxidem'))

importlib.reload(idem)
importlib.reload(utilities)
importlib.reload(fsf)
from utilities import mat_hug, mat_sq

jax.config.update('jax_enable_x64',True)

key = jax.random.PRNGKey(1)
keys = rand.split(key, 3)

process_basis = utilities.place_basis(nres=1, min_knot_num=5)
nbasis = process_basis.nbasis

truemodel = idem.gen_example_idem(
    keys[0], k_spat_inv=True,
    process_basis=process_basis,
    sigma2_eta=0.02**2,
    sigma2_eps=0.05**2
)

process_data, obs_data = truemodel.simulate(nobs=100, T=3, key=keys[1])

process_data.show_plot()

K_basis = truemodel.kernel.basis
k = (
    jnp.array([200.0]),
    jnp.array([0.001]),
    jnp.array([0.01]),
    jnp.array([0.01]),
)
# This is the kind of kernel used by ```gen_example_idem```
kernel = idem.param_exp_kernel(K_basis, k)

model0 = idem.IDEM_Model(
    process_basis=process_basis,
    kernel=kernel,
    process_grid=utilities.create_grid(jnp.array([[0, 1], [0, 1]]),
                                       jnp.array([41, 41])),
    sigma2_eta=0.01**2,
    sigma2_eps=0.01**2,
    beta=jnp.array([0.0, 0.0, 0.0]),
    m_0=jnp.zeros(nbasis),
    sigma2_0=10)

# | output: false
obs_data_wide = obs_data.as_wide()
X_obs = jnp.column_stack(
    [jnp.ones(obs_data_wide['x'].shape[0]), obs_data_wide['x'], obs_data_wide['y']])

obs_locs = jnp.column_stack([obs_data_wide['x'], obs_data_wide['y']])
PHI_obs = model0.process_basis.mfun(obs_locs)

ks0 = (
    jnp.log(model0.kernel.params[0]),
    jnp.log(model0.kernel.params[1]),
    model0.kernel.params[2],
    model0.kernel.params[3],
)

params0 = (
    jnp.log(model0.sigma2_eta),
    jnp.log(model0.sigma2_eps),
    ks0,
    model0.beta,
)


def objective(params):
    (log_sigma2_eta,
     log_sigma2_eps,
     ks,
     beta, ) = params

    m_0 = model0.m_0
    P_0 = model0.sigma2_0 * jnp.eye(nbasis)

    logks1, logks2, ks3, ks4 = ks

    ks1 = jnp.exp(logks1)
    ks2 = jnp.exp(logks2)
    M = model0.con_M((ks1, ks2, ks3, ks4))

    sigma2_eta = jnp.exp(log_sigma2_eta)
    sigma2_eps = jnp.exp(log_sigma2_eps)

    ztildes = obs_data_wide['z'] - (X_obs @ beta)[:, None]

    ll, _, _, _, _, _ = fsf.kalman_filter_indep(
        m_0,
        P_0,
        M,
        PHI_obs,
        sigma2_eta,
        sigma2_eps,
        ztildes,
        likelihood='partial'
    )
    return -ll


obj_grad = jax.grad(objective)

params = params0

grad = obj_grad(params)
res = jax.tree.map(lambda x, y: x-y, params, grad)


ztildes = obs_data_wide['z'] - (X_obs @ model0.beta)[:, None]

P_0 = model0.sigma2_0 * jnp.eye(nbasis)

ll, _, _, _, l_, _ = fsf.kalman_filter_indep(
    model0.m_0,
    P_0,
    model0.M,
    PHI_obs,
    model0.sigma2_eta,
    model0.sigma2_eps,
    ztildes,
    likelihood='partial'
)


# the kalman filter_isolated


def step(carry, z_t):
    m_tt, P_tt, _, _, ll, _ = carry

    # predict
    m_pred = M @ m_tt

    # Add sigma2_eps to the diagonal intelligently
    P_prop = mat_hug(M, P_tt)
    P_pred = jnp.fill_diagonal(
        P_prop, sigma2_eta + jnp.diag(P_prop), inplace=False)

    # Update

    # Prediction error
    e_t = z_t - PHI_obs @ m_pred

    # Prediction Variance
    P_oprop = mat_hug(PHI_obs, P_pred)
    Sigma_t = jnp.fill_diagonal(
        P_oprop, sigma2_eps+jnp.diag(P_oprop), inplace=False)

    # Kalman Gain
    K_t = (
        jnp.linalg.solve(Sigma_t, PHI_obs)
        @ P_pred.T
    ).T

    m_up = m_pred + K_t @ e_t

    P_up = (jnp.eye(nbasis) - K_t @ PHI_obs) @ P_pred

    # likelihood of epsilon, using cholesky decomposition
    chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
    z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e_t, lower=True)
    if likelihood == 'full':
        ll_new = ll - jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - \
            0.5 * jnp.dot(z, z) - 0.5 * nobs * jnp.log(2*jnp.pi)
    elif likelihood == 'partial':
        ll_new = ll - \
            jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)
    elif likelihood == 'none':
        ll_new = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial').")

    print(ll)

    return (m_up, P_up, m_pred, P_pred, ll_new, K_t), (
        m_up,
        P_up,
        m_pred,
        P_pred,
        ll_new,
        K_t,
    )


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, ys


m_0 = model0.m_0
P_0 = model0.sigma2_0*jnp.eye(nbasis)
nobs = ztildes.shape[0]
M = model0.M
sigma2_eta = model0.sigma2_eta
sigma2_eps = model0.sigma2_eps
beta = model0.beta
likelihood = 'partial'

carry, seq = scan(
    step,
    (m_0, P_0, m_0, P_0, 0.0, jnp.zeros((nbasis, nobs))),
    ztildes.T,
)

ll = carry[4]
#print(ll)


PHI = model0.process_basis.mfun(obs_locs)

A = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#print(mat_sq(A))
#print(A@A.T)


#efficient_time = timeit.timeit(lambda: mat_sq(PHI), number=1000)
#print(f"mat_sq: {efficient_time:.6f} seconds")

# Time the simple matrix multiplication
#simple_time = timeit.timeit(lambda: PHI@PHI.T, number=1000)
#print(f"just multiplying: {simple_time:.6f} seconds")


Sigma = jnp.array([[2, 1], [1, 3]])
A = jnp.array([[1, 2], [4, 5], [7, 8]])


# step-by-step


carry = (m_0, P_0, m_0, P_0, 0.0, jnp.zeros((nbasis, nobs)))
i=0


z_t = ztildes.T[i]
m_tt, P_tt, _, _, ll, _ = carry
m_pred = M @ m_tt
P_prop = M @ P_tt @ M.T
P_pred = jnp.fill_diagonal(
    P_prop, sigma2_eta + jnp.diag(P_prop), inplace=False)
e_t = z_t - PHI_obs @ m_pred
P_oprop = PHI_obs @ P_pred @ PHI_obs.T
Sigma_t = jnp.fill_diagonal(
    P_oprop, sigma2_eps+jnp.diag(P_oprop), inplace=False)
K_t = (
    jnp.linalg.solve(Sigma_t, PHI_obs)
    @ P_pred.T
).T
m_up = m_pred + K_t @ e_t
P_up = (jnp.eye(nbasis) - K_t @ PHI_obs) @ P_pred
chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e_t, lower=True)
ll_new = ll - \
    jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)

jnp.linalg.cholesky(P_up)[0:3,0:3]

carry = (m_up, P_up, m_pred, P_pred, ll_new, K_t)
i=i+1

def ensure_pos(A):
   return jnp.linalg.pinv(jnp.linalg.pinv(A, hermitian=True), hermitian=True)

def ensure_sym(A):
    return 0.5*(A+A.T)

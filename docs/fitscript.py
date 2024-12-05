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

process_basis = place_basis(nres=2, min_knot_num=5)
nbasis = process_basis.nbasis

m_0 = jnp.zeros(nbasis).at[16].set(1)
sigma2_0 = 0.001

truemodel = gen_example_idem(
    keys[0], k_spat_inv=True, process_basis=process_basis, m_0=m_0, sigma2_0=sigma2_0
)

# Simulation
T = 10

process_data, obs_data = truemodel.simulate(nobs=50, T=T + 1, key=keys[1])

# Plotting
gif_st_grid(process_data, output_file="target_process.gif")
gif_st_pts(obs_data, output_file="synthetic_observations.gif")
plot_kernel(truemodel.kernel, output_file="target_kernel.png")

K_basis = truemodel.kernel.basis
k = (
    jnp.array([100.0]),
    jnp.array([0.001]),
    jnp.array([0.0]),
    jnp.array([0.0]),
)
# This is the kind of kernel used by ```gen_example_idem```
kernel = param_exp_kernel(K_basis, k)


process_basis2 = place_basis(
    nres=1, min_knot_num=5
)  # courser process basis with 25 total basis functions
nbasis0 = process_basis2.nbasis

model1 = IDEM(
    process_basis=process_basis2,
    kernel=kernel,
    process_grid=create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([41, 41])),
    sigma2_eta=0.01,
    sigma2_eps=0.01,
    beta=jnp.array([0.0, 0.0, 0.0]),
    m_0=jnp.zeros(nbasis0),
    sigma2_0=0.01,
)
# a model with inaccurate 'guesses'

# Currently, the Kalman filter requires the data to be in wide format.
obs_data_wide = ST_towide(obs_data)

# although it is irrelevent for this particular model, we need to put in the covariate matrix into filter
obs_locs = jnp.column_stack((obs_data_wide.x, obs_data_wide.y))
nobs = obs_locs.shape[0]
X_obs = jnp.column_stack([jnp.ones(nobs), obs_locs])
nobs = obs_locs.shape[0]
PHI_obs = model1.process_basis.mfun(obs_locs)
PHI = model1.process_basis.mfun(model1.process_grid.coords)
GRAM = (PHI.T @ PHI) * model1.process_grid.area

v_unfit_process_data, v_unfit_obs_data = model1.simulate(nobs=1, T=T + 1, key=keys[2])
# Plotting
gif_st_grid(v_unfit_process_data, output_file="very_unfit_process.gif")

ll, _, _, _, _, _ = model1.filter(obs_data_wide, X_obs)


@jax.jit
def objective(params):
    m_0, sigma2_0, sigma2_eta, sigma2_eps, ks = params

    M = model1.con_M(ks)

    Sigma_eta = sigma2_eta * jnp.eye(nbasis0)
    Sigma_eps = sigma2_eps * jnp.eye(nobs)
    P_0 = sigma2_0 * jnp.eye(nbasis0)

    carry, seq = kalman_filter(
        m_0,
        P_0,
        M,
        PHI_obs,
        Sigma_eta,
        Sigma_eps,
        model1.beta,
        obs_data_wide,
        X_obs,
    )
    return -carry[4]


obj_grad = jax.grad(objective)

params0 = (
    model1.m_0,
    model1.sigma2_0,
    model1.sigma2_eta,
    model1.sigma2_eps,
    model1.kernel.params,
)


lower = (
    jnp.full(nbasis0, -jnp.inf),
    jnp.array(0.0),
    jnp.array(0.0),
    jnp.array(0.0),
    (jnp.array(0.0), jnp.array(0.0), jnp.array(-jnp.inf), jnp.array(-jnp.inf)),
)

upper = (
    jnp.full(nbasis0, jnp.inf),
    jnp.array(0.1),
    jnp.array(0.1),
    jnp.array(0.1),
    (jnp.array(500.0), jnp.array(0.01), jnp.array(jnp.inf), jnp.array(jnp.inf)),
)

# with this many parameters, must use a lower starting learning rate
start_learning_rate = 1e-3
optimizer = optax.adam(start_learning_rate)


params_ad = params0
opt_state = optimizer.init(params_ad)

# A simple update loop.
for i in tqdm(range(5), desc="Adam Optimiser..."):
    grad = obj_grad(params_ad)
    updates, opt_state = optimizer.update(grad, opt_state)
    params_ad = optax.apply_updates(params_ad, updates)
    params_ad = optax.projections.projection_box(params_ad, lower, upper)
    nll = objective(params_ad)

new_fitted_model = IDEM(
    process_basis=process_basis2,
    kernel=param_exp_kernel(K_basis, params_ad[4]),
    process_grid=create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([41, 41])),
    sigma2_eta=params_ad[2],
    sigma2_eps=params_ad[3],
    beta=jnp.array([0, 0, 0]),
    m_0=params_ad[0],
    sigma2_0=params_ad[1],
)

fit_process_data, fit_obs_data = new_fitted_model.simulate(
    nobs=50, T=T + 1, key=keys[3]
)
#gif_st_grid(fit_process_data, output_file="new_fitted_process.gif")
#plot_kernel(new_fitted_model.kernel, output_file="new_fitted_kernel.png")

true_ll, _, _, _, _, _ = truemodel.filter(obs_data_wide, X_obs)

print(f"\nthe final parameters are {params_ad}\n\n")

print(f"\nThe log likelihood (up to a constant) of the true model is {true_ll}")
print(
    f"The final log likelihood (up to a constant) of the fit model is {-objective(params_ad)}\n"
)


print("and now for the easier way!")

kernel = param_exp_kernel(
    K_basis,
    (
        jnp.array([100.0]),
        truemodel.kernel.params[1],
        jnp.array([0.0]),
        jnp.array([0.0]),
    ),
)


model2 = IDEM(
    process_basis=process_basis2,
    kernel=kernel,
    process_grid=create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([41, 41])),
    sigma2_eta=truemodel.sigma2_eta,
    sigma2_eps=truemodel.sigma2_eps,
    beta=jnp.array([0.0, 0.0, 0.0]),
    m_0=jnp.zeros(nbasis0),
    sigma2_0=truemodel.sigma2_0,
)


quick_fit_model = model2.data_mle_fit(
    obs_data,
    X_obs,
    fixed_ind=["sigma2_0", "beta"],
    optimizer=optax.adam(1e-3),
    nits=10,
)

new_fit_process_data, new_fit_obs_data = quick_fit_model.simulate(
    nobs=50, T=T + 1, key=keys[4]
)
gif_st_grid(new_fit_process_data, output_file="quickfit_new_fitted_process.gif")
plot_kernel(quick_fit_model.kernel, output_file="quickfit_new_fitted_kernel.png")



import pandas as pd

df = pd.read_csv("z_obs.csv")

date_mapping = {date: i-1 for i, date in enumerate(df['time'].unique(), 1)}
df['time'] = df['time'].map(date_mapping)

data = ST_Data_Long(x= df['s1'], y=df['s2'], t=df['time'], z=df['z'])

obs_locs = jnp.array([data.x,data.y]).T
X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

print(X_obs.shape)
print(model1.m_0.shpae)

quick_fit_model = model1.data_mle_fit(
    data,
    X_obs,
    fixed_ind=["sigma2_0"],
    optimizer=optax.adam(1e-3),
    nits=100,
)

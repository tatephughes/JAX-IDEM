import time
import optax
import pandas as pd
import csv
import os
import sys
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as rand
import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import idem
import utilities
import filter_smoother_functions as fsf
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)
df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../../../data/obs_data_r-ide.csv")
)
df["time"] = pd.to_datetime(df["time"])
reference_date = pd.to_datetime("2017-12-01")
df["t"] = (df["time"] - reference_date).dt.days + 1

obs_data = utilities.st_data(
    x=jnp.array(df["s1"]),
    y=jnp.array(df["s2"]),
    t=jnp.array(df["t"]),
    z=jnp.array(df["z"]),
)

obs_locs = jnp.column_stack((obs_data.x, obs_data.y))
X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

betahat = jnp.linalg.solve(X_obs.T @ X_obs, X_obs.T) @ obs_data.z

ztilde = obs_data.z - X_obs @ betahat

const_basis = utilities.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1)

K_basis = (
    const_basis,
    const_basis,
    const_basis,
    const_basis,
)
k = (
    jnp.array([150.0]),
    jnp.array([0.002]),
    jnp.array([0.0]),
    jnp.array([0.0]),
)

kernel = idem.param_exp_kernel(K_basis, k)
process_basis = utilities.place_basis(nres=2, min_knot_num=3)
model0 = idem.IDEM(
    process_basis=process_basis,
    kernel=kernel,
    process_grid=utilities.create_grid(
        jnp.array([[0, 1], [0, 1]]), jnp.array([41, 41])
    ),
    sigma2_eta=jnp.var(ztilde),
    sigma2_eps=jnp.var(ztilde),
    beta=betahat,
)

truek = (
    jnp.array([150.0]),
    jnp.array([0.002]),
    jnp.array([-0.1]),
    jnp.array([0.1]),
)

truekernel = idem.param_exp_kernel(K_basis, truek)
process_basis = utilities.place_basis(nres=2, min_knot_num=3)

truemodel = idem.IDEM(
    process_basis=process_basis,
    kernel=truekernel,
    process_grid=utilities.create_grid(
        jnp.array([[0, 1], [0, 1]]), jnp.array([41, 41])
    ),
    sigma2_eta=0.01**2,
    sigma2_eps=0.01**2,
    beta=jnp.array([0.2, 0.2, 0.2]),
)

truektrans = (
    jnp.log(truemodel.kernel.params[0]),
    jnp.log(truemodel.kernel.params[1]),
    truemodel.kernel.params[2],
    truemodel.kernel.params[3],
)

trueparams = (
    jnp.log(truemodel.sigma2_eps),
    jnp.log(truemodel.sigma2_eta),
    truektrans,
    truemodel.beta,
)


obs_data_wide = obs_data.as_wide()
X_obs = jnp.column_stack(
    [jnp.ones(obs_data_wide["x"].shape[0]), obs_data_wide["x"], obs_data_wide["y"]]
)
start_time = time.time()
model1, params = model0.fit_sqrt_filter(
    obs_data=obs_data,
    X_obs=X_obs,
    optimizer=optax.adamax(1e-2),
    max_its=1000,
    # target_ll=jnp.array(3217.945),
    # fixed_ind = ['ks1', 'ks2'],
    likelihood="partial",
    eps=None,
    debug=False,
)
end_time = time.time()
print(f"Time Elapsed is {end_time - start_time}")

print(f"\nFitted parameters are: \n{idem.format_params(params)}")
print(
    f"with likelihood {model1.filter(obs_data, X_obs=X_obs, likelihood='partial')[0].tolist()}"
)
print(f"\nTrue parameters are: \n{idem.format_params(trueparams)}")
print(
    f"with likelihood {truemodel.filter(obs_data, X_obs=X_obs, likelihood='partial')[0].tolist()}"
)

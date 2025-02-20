import os
import sys
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as rand
import time
import optax
import numpy as np
import timeit
import importlib
import jax
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_platform_name', 'gpu')
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import filter_smoother_functions as fsf
import utilities
import idem

print(f"Current working directory: {os.getcwd()}")


# unnecessary unless running interactively
importlib.reload(idem)
importlib.reload(utilities)
importlib.reload(fsf)


key = jax.random.PRNGKey(3)
keys = rand.split(key, 3)

# True model

process_basis = utilities.place_basis(nres=2, min_knot_num=3)
nbasis = process_basis.nbasis

truemodel = idem.gen_example_idem(
    keys[0], k_spat_inv=True,
    process_basis=process_basis,
    sigma2_eta=0.01**2,
    sigma2_eps=0.01**2,
    beta=jnp.array([.2, .2, .2]),
)

alpha_0 = jnp.zeros(nbasis).at[81].set(10)

truek = (jnp.log(truemodel.kernel.params[0]),
         jnp.log(truemodel.kernel.params[1]),
         truemodel.kernel.params[2],
         truemodel.kernel.params[3])

trueparams = (jnp.log(truemodel.sigma2_eps), jnp.log(
    truemodel.sigma2_eta), truek, truemodel.beta)

process_data, obs_data = truemodel.simulate(
    nobs=100, T=9, key=keys[1], alpha_0=alpha_0)


# Shell model

K_basis = truemodel.kernel.basis

k = (
    jnp.array([100]),
    jnp.array([0.001]),
    jnp.zeros(truemodel.kernel.basis[2].nbasis),
    jnp.zeros(truemodel.kernel.basis[2].nbasis),
)
kernel = idem.param_exp_kernel(K_basis, k)

model0 = idem.IDEM(process_basis=process_basis,
                   kernel=kernel,
                   process_grid=utilities.create_grid(jnp.array([[0, 1], [0, 1]]),
                                                      jnp.array([41, 41])),
                   sigma2_eta=0.01**2,
                   sigma2_eps=0.01**2,
                   beta=jnp.array([0.0, 0.0, 0.0]),)

obs_data_wide = obs_data.as_wide()
X_obs = jnp.column_stack(
    [jnp.ones(obs_data_wide['x'].shape[0]), obs_data_wide['x'], obs_data_wide['y']])
X_obs_tuple = [X_obs for _ in range(len(obs_data.z))]

start_time = time.time()
model1, params = model0.fit_sqrt_filter(obs_data=obs_data,
                                        X_obs=X_obs,
                                        optimizer=optax.adamax(1e-2),
                                        debug=False,
                                        max_its=100,
                                        likelihood='full',
                                        loading_bar=True,
                                        eps=None)
end_time = time.time()
print(f"Time Elapsed is {end_time - start_time}")

ll, ms, Ps, _, _ = truemodel.filter(obs_data, X_obs=X_obs)
# ms = jnp.linalg.solve(Qs, nus[..., None]).squeeze(-1)

print("\nFitted parameters are: \n", idem.format_params(params))
print(f"with likelihood {model1.filter(obs_data, X_obs=X_obs)[0].tolist()}")
print("\nTrue parameters are: \n", idem.format_params(trueparams))
print(f"with likelihood {truemodel.filter(obs_data, X_obs=X_obs)[0].tolist()}")

truemodel.kernel.save_plot('truekernel.png')
model1.kernel.save_plot('fitkernel.png')

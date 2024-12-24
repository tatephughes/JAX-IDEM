import pathmagic
import jax.random as rand
import jax.numpy as jnp
import jax.lax as jl
import jax
import importlib
import sys

with pathmagic.context():
    if 'IDEM' in sys.modules:
        import IDEM
        import utilities
        import filter_smoother_functions
        importlib.reload(IDEM)
        importlib.reload(utilities)
        importlib.reload(filter_smoother_functions)
    import IDEM
    import utilities
    import filter_smoother_functions


key = jax.random.PRNGKey(12)
keys = rand.split(key, 3)

process_basis = utilities.place_basis(nres=2, min_knot_num=3)
nbasis = process_basis.nbasis

m_0 = jnp.zeros(nbasis).at[73].set(10)
sigma2_0 = 0.001

model = IDEM.gen_example_idem(
    keys[0], k_spat_inv=True, process_basis=process_basis, m_0=m_0,
    sigma2_0=sigma2_0
)

init_ob = IDEM.basis_params_to_st_data(jnp.array([m_0]),
                                       model.process_basis, model.process_grid)

# Simulation
T = 9

nobs = jax.random.randint(keys[1], (T,), 100, 151)

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

process_data = utilities.st_data(t=process_data.t, x=process_data.x,
                                 y=process_data.y, z=process_data.z)

obs_data = utilities.st_data(obs_data.x, obs_data.y, obs_data.t, obs_data.z)

unique_times = jnp.unique(obs_data.t)

obs_locs_tuple = [obs_locs[obs_locs[:, 0] == t][:, 1:] for t in unique_times]
PHI_obs_tuple = jax.tree.map(model.process_basis.mfun, obs_locs_tuple)

X_obs_tuple = jax.tree.map(lambda locs:
                           jnp.column_stack((jnp.ones(len(locs)),
                                             locs)), obs_locs_tuple)

nu_0 = jnp.zeros(nbasis)
Q_0 = jnp.zeros((nbasis, nbasis))

z = [obs_data.z[obs_data.t == t] for t in unique_times]

carry, seq = filter_smoother_functions.information_filter(
    nu_0,
    Q_0,
    model.M,
    PHI_obs_tuple,
    model.sigma2_eta,
    model.sigma2_eps,
    model.beta,
    z,
    X_obs_tuple,)

nus, Qs = seq

ms = jnp.linalg.solve(Qs, nus)

m_0_lse = jnp.linalg.solve(PHI_obs_tuple[0].T@PHI_obs_tuple[0],
                           PHI_obs_tuple[0].T) @ z[0]


PHI_obs_new = model.process_basis.mfun(obs_locs_tuple[0])
print(jnp.linalg.det(PHI_obs_new.T @ PHI_obs_new))

filt_data = IDEM.basis_params_to_st_data(ms,
                                         model.process_basis,
                                         model.process_grid)

import os
import sys
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import jax.random as rand
import time
import optax
import importlib
import timeit
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import filter_smoother_functions as fsf
import utilities
import idem
jax.config.update('jax_enable_x64', True)

print(f"Current working directory: {os.getcwd()}")


# unnecessary unless running interactively
importlib.reload(idem)
importlib.reload(utilities)
importlib.reload(fsf)


key = jax.random.PRNGKey(1)
keys = rand.split(key, 3)

# True model

process_basis = utilities.place_basis(nres=1, min_knot_num=5)
nbasis = process_basis.nbasis

model = idem.gen_example_idem(
    keys[0], k_spat_inv=True,
    process_basis=process_basis,
    sigma2_eta=0.01**2,
    sigma2_eps=0.01**2,
    beta=jnp.array([.2, .2, .2]),
)

alpha_0 = jnp.zeros(nbasis).at[81].set(10)

process_data, obs_data = model.simulate(
    nobs=100, T=9, key=keys[1], alpha_0=alpha_0)

unique_times = jnp.unique(obs_data.t)

obs_data_wide = obs_data.as_wide()
X_obs = jnp.column_stack(
    [jnp.ones(obs_data_wide['x'].shape[0]), obs_data_wide['x'], obs_data_wide['y']])


obs_locs = jnp.column_stack([obs_data_wide["x"], obs_data_wide["y"]])
obs_locs_long = jnp.column_stack(jnp.column_stack((obs_data.x, obs_data.y))).T

obs_locs_tuple = [obs_locs_long[obs_data.t == t][:, 0:] for t in unique_times]



PHI_obs = model.process_basis.mfun(obs_locs)
PHI_obs_tuple = jax.tree.map(model.process_basis.mfun, obs_locs_tuple)

X_obs_tuple = jax.tree.map(
            lambda locs: jnp.column_stack((jnp.ones(len(locs)), locs)), obs_locs_tuple)

nbasis = model.process_basis.nbasis

m_0 = jnp.zeros(nbasis)
P_0 = 100 * jnp.eye(nbasis)
U_0 = 10 * jnp.eye(nbasis)

nu_0 = jnp.zeros(nbasis)
Q_0 = 0.01 * jnp.eye(nbasis)


beta= model.beta
M = model.M
sigma2_eps = model.sigma2_eps
sigma2_eta = model.sigma2_eta


zs_tuple = [obs_data.z[obs_data.t == t] for t in unique_times]

ztildes = obs_data_wide["z"] - (X_obs @ model.beta)[:, None]
def is_leaf(node):
    return jax.tree.structure(node).num_leaves == 2

mapping_elts = tuple(
            [[zs_tuple[i], X_obs_tuple[i]] for i in range(len(zs_tuple))])

def tildify(z, X_obs_i, beta):
            return z - X_obs_i @ beta
        
ztildes_tuple = jax.tree.map(
                lambda tup: tildify(tup[0], tup[1], beta), mapping_elts, is_leaf=is_leaf)

def kalman_filter():
        ll_kf, _, _, _, _, _ = fsf.kalman_filter_indep(
                        m_0,
                        P_0,
                        model.M,
                        PHI_obs,
                        model.sigma2_eta,
                        model.sigma2_eps,
                        ztildes,
                        likelihood="full",
                    )
        return ll_kf

def sqrt_filter():
        ll_sq, _, _, _, _, _ = fsf.sqrt_filter_indep(
                        m_0,
                        U_0,
                        model.M,
                        PHI_obs,
                        model.sigma2_eta,
                        model.sigma2_eps,
                        ztildes,
                        likelihood="full",
                    )
        return ll_sq

def if_filter():
    ll_if, _, _, _, _ = fsf.information_filter_indep(
                    nu_0,
                    Q_0,
                    M,
                    PHI_obs_tuple,
                    sigma2_eta,
                    sigma2_eps,
                    ztildes_tuple,
                    likelihood="full",
                )
    return ll_if

def sqif_filter():
    ll_if, _, _, _, _ = fsf.sqrt_information_filter_indep(
                    nu_0,
                    U_0,
                    M,
                    PHI_obs_tuple,
                    sigma2_eta,
                    sigma2_eps,
                    ztildes_tuple,
                    likelihood="full",
                )
    return ll_if


cpu_kf = jax.jit(kalman_filter, backend="cpu")
cpu_sq = jax.jit(sqrt_filter, backend="cpu")
cpu_if = jax.jit(if_filter, backend="cpu")
cpu_sqif = jax.jit(sqif_filter, backend="cpu")
gpu_kf = jax.jit(kalman_filter, backend="gpu")
gpu_sq = jax.jit(sqrt_filter, backend="gpu")
gpu_if = jax.jit(if_filter, backend="gpu")
gpu_sqif = jax.jit(sqif_filter, backend="gpu")

print(f"\nKalman Filter (CPU), ll: {cpu_kf()}")
print(f"Sqrt Filter (CPU), ll: {cpu_sq()}")
print(f"Inf Filter (CPU), ll: {cpu_if()}")
print(f"SqInf Filter (CPU), ll: {cpu_sqif()}")
print(f"Kalman Filter (GPU), ll: {gpu_kf()}")
print(f"Sqrt Filter (GPU), ll: {gpu_sq()}")
print(f"Inf Filter (GPU), ll: {gpu_if()}")
print(f"SqInf Filter (GPU), ll: {gpu_sqif()}\n")

print(f"Kalman filter (CPU) took {timeit.timeit(cpu_kf, number=1000)}ms.")
print(f"Sqrt filter (CPU) took {timeit.timeit(cpu_sq, number=1000)}ms.")
print(f"Inf filter (CPU) took {timeit.timeit(cpu_if, number=1000)}ms.")
print(f"SqInf filter (CPU) took {timeit.timeit(cpu_sqif, number=1000)}ms.\n")
print(f"Kalman filter (GPU) took {timeit.timeit(gpu_kf, number=1000)}ms.")
print(f"Sqrt filter (GPU) took {timeit.timeit(gpu_sq, number=1000)}ms.")
print(f"Inf filter (GPU) took {timeit.timeit(gpu_if, number=1000)}ms.")
print(f"SqInf filter (GPU) took {timeit.timeit(gpu_sqif, number=1000)}ms.")

# jaxpr = jax.make_jaxpr(kalman_filter)(P_0)
# print(jaxpr)


# with jax.profiler.trace("/tmp/jax_trace"):
#     result = kalman_filter(P_0)
# jaxpr = jax.make_jaxpr(P_0)
#ll_kf, ms_kf, Ps_kf, _,_ = model.kalman_filter(obs_data, X_obs)
#ll_sq, ms_sq, Ps_sq, _,_ = model.sqrt_filter(obs_data, X_obs)
#ll_if, nus_if, Qs_if, _,_ = model.information_filter(obs_data)
#ms_if = jnp.linalg.solve(Qs_if, nus_if[..., None]).squeeze(-1)

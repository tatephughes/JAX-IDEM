import os
import sys
import jax.numpy as jnp
import numpy as np
import jax.random as rand
import time
import optax
import importlib
import jax
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import filter_smoother_functions as fsf
import utilities
import idem
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_platform_name', 'cpu')
import blackjax
from tqdm.auto import tqdm
from jax_tqdm import scan_tqdm


print(f"Current working directory: {os.getcwd()}")

# unnecessary unless running interactively
importlib.reload(idem)
importlib.reload(utilities)
importlib.reload(fsf)

import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(3)
keys = rand.split(key, 3)

# True model

process_basis = utilities.place_basis(nres=2, min_knot_num=3)
nbasis = process_basis.nbasis

truemodel = idem.gen_example_idem(
    keys[0], k_spat_inv=False,
    process_basis=process_basis,
    Sigma_eta=0.01**2*jnp.eye(nbasis),
)

alpha_0 = jnp.zeros(nbasis).at[81].set(10)

truek = (jnp.log(truemodel.kernel.params[0]),
         jnp.log(truemodel.kernel.params[1]),
         truemodel.kernel.params[2],
         truemodel.kernel.params[3])

trueparams = (jnp.log(truemodel.sigma2_eps), jnp.log(
    truemodel.Sigma_eta[0,0]), truek, truemodel.beta)

process_data, obs_data = truemodel.simulate(
    nobs=100, T=9, key=keys[1], alpha_0=alpha_0)


# Shell model

K_basis = truemodel.kernel.basis

obs_data_wide = obs_data.as_wide()
obs_locs = jnp.column_stack([obs_data_wide["x"], obs_data_wide["y"]])
unique_times = jnp.unique(obs_data.t)
obs_locs_long = jnp.column_stack(
    jnp.column_stack((obs_data.x, obs_data.y))).T
obs_locs_tuple = [obs_locs_long[obs_data.t == t][:, 0:]
                  for t in unique_times]
X_obs = jnp.column_stack(
    [jnp.ones(obs_data_wide['x'].shape[0]), obs_data_wide['x'], obs_data_wide['y']])
obs_locs = jnp.column_stack([obs_data_wide["x"], obs_data_wide["y"]])
PHI_obs = truemodel.process_basis.mfun(obs_locs)
nbasis = truemodel.process_basis.nbasis
m_0 = jnp.zeros(nbasis)
nu_0 = jnp.zeros(nbasis)
U_0 = 10 * jnp.eye(nbasis)
P_0 = U_0@U_0
Q_0 = jnp.linalg.inv(P_0)
R_0 = jnp.linalg.cholesky(Q_0)
PHI_obs_tuple = jax.tree.map(truemodel.process_basis.mfun, obs_locs_tuple)
X_obs_tuple = jax.tree.map(
    lambda locs: jnp.column_stack((jnp.ones(len(locs
                                                )), locs)), obs_locs_tuple)
zs_tuple = [obs_data.z[obs_data.t == t] for t in unique_times]
def is_leaf(node):
    return jax.tree.structure(node).num_leaves == 2
def tildify(z, X_obs_i, beta):
    return z - X_obs_i @ beta
mapping_elts = tuple(
    [[zs_tuple[i], X_obs_tuple[i]] for i in range(len(zs_tuple))])


@jax.jit
def log_marginal(params):
    (
        log_sigma2_eta,
        log_sigma2_eps,
        ks,
        beta,
    ) = params

    logks1, logks2, ks3, ks4 = ks

    ks1 = jnp.exp(logks1)
    ks2 = jnp.exp(logks2)

    sigma2_eta = jnp.exp(log_sigma2_eta)
    sigma2_eps = jnp.exp(log_sigma2_eps)
    M = truemodel.con_M((ks1, ks2, ks3, ks4))

    ztildes = obs_data_wide["z"] - (X_obs @ beta)[:, None]

    #ztildes_tuple = jax.tree.map(
    #    lambda tup: tildify(tup[0], tup[1], beta), mapping_elts, is_leaf=is_leaf)

    ll, _, _, _, _, _ = fsf.sqrt_filter_indep(
        nu_0,
        U_0,
        M,
        PHI_obs,
        sigma2_eta,
        sigma2_eps,
        ztildes,
        likelihood="full",
    )
    return ll

priorvar = jnp.var(obs_data_wide["z"])

ks0 = (
    jnp.array([jnp.log(100.0)]),
    jnp.array([jnp.log(0.002)]),
    jnp.zeros(truemodel.kernel.basis[2].nbasis),
    jnp.zeros(truemodel.kernel.basis[3].nbasis),
)
params0 = (jnp.log(jnp.var(obs_data_wide["z"])),
           jnp.log(jnp.var(obs_data_wide["z"])),
           ks0,
           jnp.array([0.0,0.0,0.0]))

def log_prior_density(param):

    (
        log_sigma2_eta,
        log_sigma2_eps,
        ks,
        beta,
    ) = param

    logdens_log_sigma2_eta = jax.scipy.stats.norm.logpdf(log_sigma2_eta, loc = priorvar, scale=3.0)
    logdens_log_sigma2_eps = jax.scipy.stats.norm.logpdf(log_sigma2_eps, loc = priorvar, scale=3.0)

    logdens_ks1 = jax.scipy.stats.multivariate_normal.logpdf(ks[0], ks0[0], 20*jnp.eye(ks[0].shape[0]))
    logdens_ks2 = jax.scipy.stats.multivariate_normal.logpdf(ks[1], ks0[1], 20*jnp.eye(ks[1].shape[0]))
    logdens_ks3 = jax.scipy.stats.multivariate_normal.logpdf(ks[2], ks0[2], 20*jnp.eye(ks[2].shape[0]))
    logdens_ks4 = jax.scipy.stats.multivariate_normal.logpdf(ks[3], ks0[3], 20*jnp.eye(ks[3].shape[0]))

    logdens_beta = jax.scipy.stats.multivariate_normal.logpdf(beta, params0[3], 20*jnp.eye(beta.shape[0]))

    return logdens_log_sigma2_eta+logdens_log_sigma2_eps+logdens_ks1+logdens_ks2+logdens_ks3+logdens_ks4+logdens_beta
    




# Build the kernel
step_size = 1e-3
nparams = sum(leaf.size for leaf in jax.tree.leaves(params0))

# imm should be an estimate of the posterior variance matric, but do this for now
inverse_mass_matrix = 1.0*jnp.eye(nparams)

@jax.jit
def log_post_dens(param):
    return log_prior_density(param) + log_marginal(param)

nuts = blackjax.nuts(log_post_dens, step_size, inverse_mass_matrix)

init_state = nuts.init(params0)

# Iterate

step = jax.jit(nuts.step)

n=3
rng_key = jax.random.key(2)
burn_key, it_key = rand.split(rng_key,2)

@scan_tqdm(n, desc='Sampling...')
def body_fn_sample(carry, i):
    nuts_key = jax.random.fold_in(it_key, i)
    new_state, info = step(nuts_key, carry)
    return new_state, (new_state, info)



_, (param_post_sample,_) = jax.lax.scan(body_fn_sample, init_state, jnp.arange(n))

post_mean = jax.tree.map(lambda arr: jnp.mean(arr, axis=0), param_post_sample.position)

def plot_kernel_from_params(par, filename):
    logks1, logks2, ks3, ks4 = par[2]
    ks1 = jnp.exp(logks1)
    ks2 = jnp.exp(logks2)
    new_kernel_params = (ks1, ks2, ks3, ks4)
    model = idem.IDEM(
        process_basis=truemodel.process_basis,
        kernel=idem.param_exp_kernel(truemodel.kernel.basis, new_kernel_params),
        process_grid=truemodel.process_grid,
        Sigma_eta=jnp.exp(par[0]) * jnp.eye(truemodel.process_basis.nbasis),
        sigma2_eps=jnp.exp(par[1]),
        beta=par[3],
    )
    model.kernel.save_plot(filename)

for i in range(n):
    par = jax.tree.map(lambda arr: arr[i], param_post_sample.position)
    plot_kernel_from_params(par, f"./kernelplots/kernel2_{i}.png")

truemodel.kernel.save_plot("./kernelplots/kernel_true.png")

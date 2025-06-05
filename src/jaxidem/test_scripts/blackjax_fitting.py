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
jax.config.update('jax_enable_x64', True)
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

process_basis = utilities.place_basis(nres=1, min_knot_num=4)
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
prior_mean = (jnp.log(jnp.var(obs_data_wide["z"])),
           jnp.log(jnp.var(obs_data_wide["z"])),
           ks0,
           jnp.array([0.0,0.0,0.0]))

prior_var = (jnp.log(priorvar*100),
           jnp.log(priorvar*100),
           (
               jnp.array([jnp.log(100.0)]),
               jnp.array([jnp.log(10)]),
               10*jnp.eye(truemodel.kernel.basis[2].nbasis),
               10*jnp.eye(truemodel.kernel.basis[3].nbasis),
           ),
           jnp.diag(jnp.array([5.0,5.0,5.0])))


prior_var = post_var

def log_prior_density(param):

    (
        log_sigma2_eta,
        log_sigma2_eps,
        ks,
        beta,
    ) = param

    logdens_log_sigma2_eta = jax.scipy.stats.norm.logpdf(log_sigma2_eta, loc = prior_mean[0], scale=prior_var[0])
    logdens_log_sigma2_eps = jax.scipy.stats.norm.logpdf(log_sigma2_eps, loc = prior_mean[1], scale=prior_var[1])

    logdens_ks1 = jax.scipy.stats.norm.logpdf(ks[0], prior_mean[2][0], prior_var[2][0])
    logdens_ks2 = jax.scipy.stats.norm.logpdf(ks[1], prior_mean[2][1], prior_var[2][1])
    logdens_ks3 = jax.scipy.stats.multivariate_normal.logpdf(ks[2], prior_mean[2][2], jnp.diag(prior_var[2][2]))
    logdens_ks4 = jax.scipy.stats.multivariate_normal.logpdf(ks[3], prior_mean[2][3], jnp.diag(prior_var[2][3]))

    logdens_beta = jax.scipy.stats.multivariate_normal.logpdf(beta, prior_mean[3], jnp.diag(prior_var[3]))
    return logdens_log_sigma2_eta+logdens_log_sigma2_eps+logdens_ks1+logdens_ks2+logdens_ks3+logdens_ks4+logdens_beta
    




# Build the kernel
step_size = 1e-3
nparams = sum(leaf.size for leaf in jax.tree.leaves(prior_mean))

# imm should be an estimate of the posterior variance matric, but do this for now
inverse_mass_matrix = jnp.array([0.1928944,
                                 0.1928944,
                                 4.6051702,
                                 2.3025851,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 10.0,
                                 5.0,
                                 5.0,
                                 5.0,])

@jax.jit
def log_post_dens(param):
    return (log_prior_density(param) + log_marginal(param))[0]

step_size = 1e-3

nuts = blackjax.nuts(log_post_dens, step_size, inverse_mass_matrix)
hmc = blackjax.hmc(log_post_dens, step_size, inverse_mass_matrix, num_integration_steps = 5)


init_state = hmc.init(prior_mean)

# Iterate

step = jax.jit(hmc.step)

rng_key = jax.random.key(2)
burn_key, it_key = rand.split(rng_key,2)

n=100
burnin=1000
@scan_tqdm(n, desc='Sampling...')
def body_fn_sample(carry, i):
    nuts_key = jax.random.fold_in(it_key, i)
    new_state, info = step(nuts_key, carry)
    return new_state, (new_state, info)

@scan_tqdm(burnin, desc='Burning in...')
def body_fn_burnin(carry, i):
    nuts_key = jax.random.fold_in(burn_key, i)
    new_state, info = step(nuts_key, carry)
    return new_state, (new_state, info)


burnin_state, _ = jax.lax.scan(body_fn_burnin, init_state, jnp.arange(burnin))
_, (param_post_sample,info) = jax.lax.scan(body_fn_sample, burnin_state, jnp.arange(n))


post_mean = jax.tree.map(lambda arr: jnp.mean(arr, axis=0), param_post_sample.position)
post_var = jax.tree.map(lambda arr: jnp.var(arr, axis=0), param_post_sample.position)

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


## ARVIS?
## condor
## get hamilton to work again
## apptainer
## ncc


sigma2_eta_vec = jnp.repeat(sigma2_eta, 2)
sigma2_eps_vec = jnp.repeat(sigma2_eps, 3)

sigma2_eta_mat = jnp.diag(jnp.repeat(sigma2_eta, 2))
sigma2_eps_mat = jnp.diag(jnp.repeat(sigma2_eps, 3))


ll, ms, Ps, _, _, _ = kalman_filter_new(m_0,
                                          P_0,
                                          M,
                                          PHI,
                                          sigma2_eta_mat,
                                          sigma2_eps_mat,
                                          zs.T,
                                          likelihood='full',
                                          sigma2_eps_dim=2,
                                          sigma2_eta_dim=2
    )



Q_0 = 0.01 * jnp.eye(2) 
nu_0 = jnp.zeros(2)

PHI_tuple = tuple([PHI for _ in range(T)])
zs_tuple = tuple(zs)

ll1, nus1, Qs1, nupreds ,Qpreds1  = information_filter_new(nu_0,
                                            Q_0,
                                            M,
                                            PHI_tuple,
                                            sigma2_eta,
                                            [sigma2_eps for _ in range(T)],
                                            zs_tuple,
                                            likelihood='full',
                                            sigma2_eps_dim=0,
                                            sigma2_eta_dim=0
    )


ll2, nus2, Qs2,nupreds2 ,Qpreds2  = information_filter_new(nu_0,
                                            Q_0,
                                            M,
                                            PHI_tuple,
                                            sigma2_eta_vec,
                                            [sigma2_eps_vec for _ in range(T)],
                                            zs_tuple,
                                            likelihood='full',
                                            sigma2_eta_dim=1,
                                            sigma2_eps_dim=0,
    )

ll3, nus3, Qs3, nupreds3 ,Qpreds3  = information_filter_new(nu_0,
                                            Q_0,
                                            M,
                                            PHI_tuple,
                                            sigma2_eta_mat,
                                            [sigma2_eps_mat for _ in range(T)],
                                            zs_tuple,
                                            likelihood='full',
                                            sigma2_eps_dim=2,
                                            sigma2_eta_dim=2
    )



Q_pred =  Qpreds2[4]
PHI = PHI_tuple[4]
z =zs_tuple[4]

P_oprop = PHI @ jnp.linalg.solve(Q_pred, PHI.T)
Sigma_t = jnp.fill_diagonal(
    P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
)
chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)

P_oprop2 = PHI @ jnp.linalg.solve(Q_pred, PHI.T)
Sigma_t2 = jnp.fill_diagonal(
    P_oprop, jnp.diag(sigma2_eps_vec) + jnp.diag(P_oprop), inplace=False
)
chol_Sigma_t2 = jnp.linalg.cholesky(Sigma_t)



m_0 = jnp.zeros(2)
U_0 = 10*jnp.eye(2)

# Since we have independant errors, we can use the faster sqrt_filter_indep.
    
ll4, ms4, Us, _, _, _ = sqrt_filter_new(m_0,
                                              U_0,
                                              M,
                                              PHI,
                                              sigma2_eta_mat,
                                              sigma2_eps_mat,
                                              zs.T,  
                                              likelihood='full',
                                              sigma2_eps_dim=2,
                                              sigma2_eta_dim=2
                                              )



R_0 = 0.1*jnp.eye(2)
    
ll5, nus2, Rs2,_ ,Rpreds2  = sqrt_information_filter_new(nu_0,
                                                 R_0,
                                                 M,
                                                 PHI_tuple,
                                                 sigma2_eta_mat,
                                                 [sigma2_eps_mat for _ in range(T)],
                                                 zs_tuple,
                                                 likelihood='full',
                                                 sigma2_eps_dim=2,
                                                 sigma2_eta_dim=2
    )


ll6, nus3, Rs3,_ ,Rpreds3  = fsf.sqrt_information_filter(nu_0,
                                                 R_0,
                                                 M,
                                                 PHI_tuple,
                                                 sigma2_eta_mat,
                                                 [sigma2_eps_mat for _ in range(T)],
                                                 zs_tuple,
                                                 likelihood='full',
    )






sigma2_eps_tree = [sigma2_eps_mat for _ in range(T)]

match sigma2_eps:
        case jnp.ndarray():  # JAX array
            print("Variable is a JAX array.")
        case _ if isinstance(jax.tree.flatten(sigma2_eps_tree)[0][0], jnp.ndarray):  # Pytree
            print("Variable is a pytree of JAX arrays.")
        case _:
            print("Variable is neither a JAX array nor a pytree of JAX arrays.")

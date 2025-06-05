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
import jax.lax as jl
import jaxidem.idem as idem
import jaxidem.utils as utils
import jaxidem.filters as filts
#jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_platform_name', 'cpu')
import blackjax
from tqdm.auto import tqdm
from jax_tqdm import scan_tqdm

from functools import partial

#!/usr/bin/env python3

# JAX imports
import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
# Typing imports
from jaxtyping import ArrayLike, PyTree
from typing import Callable, NamedTuple  # , Union
from functools import partial

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from PIL import Image
import io

# nice loading bars
from tqdm.auto import tqdm
import time

import pandas as pd
from pandas.api.types import is_string_dtype, is_float_dtype, is_integer_dtype

st = jax.scipy.linalg.solve_triangular 

@jax.jit
def qr_R(A, B):
    """Wrapper for the stacked-QR decompositon"""
    return jnp.linalg.qr(jnp.vstack([A, B]), mode="r")


@jax.jit
def ql_L(A, B):
    """Wrapper for the stacked-QL decompositon"""
    A_flipped = jnp.flip(jnp.vstack([A, B]), axis=1)
    R = jnp.linalg.qr(A_flipped, mode='r')
    L = jnp.flip(R, axis=(0, 1))

    return L






print(f"Current working directory: {os.getcwd()}")

# unnecessary unless running interactively
importlib.reload(idem)
importlib.reload(utils)
importlib.reload(filts)

import jax
import jax.numpy as jnp


# Choose a reference time (first entry)
start_time = radar_df['time'].iloc[0]

radar_df = pd.read_csv('../data/radar_df.csv')

# Compute time differences in seconds
radar_df['time_float'] = (radar_df['time'] - start_time).dt.total_seconds()


def flatten_and_unflatten(pytree):
    # Flatten the PyTree
    flat_leaves, treedef = jax.tree.flatten(pytree)

    # Convert to a 1D array
    flat_array = jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves])

    # **Ensure sizes and split indices are concrete**
    sizes = [leaf.size for leaf in flat_leaves]  # Extract sizes concretely
    split_indices = list(jnp.cumsum(jnp.array(sizes[:-1])))  # Precompute split indices outside JIT

    # JIT-compatible unflattening
    @jax.jit
    def unflatten(flat_array):
        # Dynamically reshape leaves
        splits = [flat_array[start:end] for start, end in zip([0] + split_indices, split_indices + [len(flat_array)])]
        reshaped_leaves = [split.reshape(leaf.shape) for split, leaf in zip(splits, flat_leaves)]
        return jax.tree.unflatten(treedef, reshaped_leaves)

    return flat_array, unflatten









def safe_cholesky(matrix):
    # Define a function for the zero case
    def zero_case(_):
        return jnp.zeros_like(matrix)

    # Define a function for the non-zero case
    def nonzero_case(matrix):
        return jnp.linalg.cholesky(matrix, upper=True)

    # Use lax.cond to handle both cases
    return jax.lax.cond(
        jnp.all(matrix == 0),  # Condition
        zero_case,            # If condition is True (zero matrix)
        nonzero_case,         # If condition is False (non-zero matrix)
        operand=matrix         # Argument passed to the functions
        )

@partial(jax.jit, static_argnames=["sigma2_eta_dim", "sigma2_eps_dim", "forecast", "likelihood"])
def sqrt_information_filter(
        nu_0: ArrayLike,
        R_0: ArrayLike,
        M: ArrayLike,
        PHI_tree: tuple,
        sigma2_eta: ArrayLike,
        sigma2_eps_tree: tuple,
        zs_tree: tuple,
        sigma2_eta_dim: int,
        sigma2_eps_dim: int,
        forecast:int = 0,
        likelihood: str = "partial",
) -> tuple:

    r = nu_0.shape[0]
    
    mapping_elts = jax.tree.map(
        lambda t: (
            zs_tree[t],
            PHI_tree[t],
            sigma2_eps_tree[t]
        ),
        tuple(range(len(zs_tree))),
    )

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_k = tup[1]
        sigma2_eps_k = tup[2]


        # THIS WILL FAIL if sigma2_eps_dim!=0 and there is a time point with no data.
        match sigma2_eps_dim:
            case 0:
                i_k = PHI_k.T @ z_k / sigma2_eps_k
                #R_k = jnp.linalg.qr((PHI_k / jnp.sqrt(sigma2_eps_k)), mode="r")
                R_k = safe_cholesky(PHI_k.T @ PHI_k / sigma2_eps_k)
            case 1:
                sigma_eps = jnp.diag(jnp.sqrt(sigma2_eps_k))
                i_k = PHI_k.T @ st(sigma_eps, st(sigma_eps.T, z_k, lower=False), lower=True)
                R_k = jnp.linalg.qr(st(sigma_eps.T, PHI_k, lower=False), mode="r")
            case 2:
                sigma_eps = jnp.linalg.cholesky(sigma2_eps_k)
                i_k = PHI_k.T @ st(sigma_eps, st(sigma_eps.T, z_k, lower=False), lower=True)
                R_k = jnp.linalg.qr(st(sigma_eps.T, PHI_k, lower=False), mode="r")
        
        return jnp.vstack((i_k, R_k))

    def is_leaf(node):
        return jax.tree.structure(node).num_leaves == 3

    scan_elts = jnp.array(jax.tree.map(
        informationify, mapping_elts, is_leaf=is_leaf))

    # Minv = jnp.linalg.solve(M, jnp.eye(r))
    match sigma2_eta_dim:
        case 0:
            sigma_eta = jnp.sqrt(sigma2_eta) * jnp.eye(r)
        case 1:
            sigma_eta = jnp.diag(jnp.sqrt(sigma2_eta))
        case 2:
            sigma_eta = jnp.linalg.cholesky(sigma2_eta)

    r = nu_0.shape[0]

    def step(carry, scan_elt):
        nu_tt, R_tt, _, _ = carry

        i_tp = scan_elt[0, :]
        R_tp = scan_elt[1:, :]

        U_pred = ql_L(st(R_tt.T, M.T, lower=True), sigma_eta)
        R_pred = st(U_pred, jnp.eye(r), lower=True).T
        nu_pred = R_pred.T @ R_pred @ M @ st(R_tt,
                                             st(R_tt.T,
                                                nu_tt,
                                                lower=True), lower=False)

        nu_up = nu_pred + i_tp
        R_up = qr_R(R_pred, R_tp)

        return (nu_up, R_up, nu_pred, R_pred), (nu_up, R_up, nu_pred, R_pred)

    carry, seq = jl.scan(
        step,
        (nu_0, R_0, jnp.zeros(r), jnp.eye(r)),
        scan_elts,
    )

    if likelihood in ("full", "partial"):
        mapping_elts = jax.tree.map(
            lambda t: (zs_tree[t], PHI_tree[t], seq[2]
                       [t], seq[3][t], sigma2_eps_tree[t]),
            tuple(range(len(zs_tree))),
        )

        def likelihood_func(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            nu_pred = tree[2]
            R_pred = tree[3]
            sigma2_eps = tree[4]

            Q_pred = R_pred.T@R_pred
            e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)
            
            P_oprop = PHI @ jnp.linalg.solve(Q_pred, PHI.T)
            match sigma2_eps_dim:
                case 0:
                    Sigma_t = jnp.fill_diagonal(
                        P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
                    )
                    chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
                case 1:
                    Sigma_t = jnp.fill_diagonal(
                        P_oprop, jnp.diag(sigma2_eps) + jnp.diag(P_oprop), inplace=False
                    )
                    chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
                case 2:
                    Sigma_t = P_oprop + sigma2_eps
                    chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)

            z = st(chol_Sigma_t, e, lower=True)

            match likelihood:
                case 'full':
                    ll = (
                        -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                        - 0.5 * jnp.dot(z, z)
                        - 0.5 * nobs * jnp.log(2 * jnp.pi)
                    )
                case 'partial':
                    ll = (
                        -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                        - 0.5 * jnp.dot(z, z)
                    )

            return ll

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 5

        lls = jnp.array(
            jax.tree.map(likelihood_func,
                         mapping_elts, is_leaf=is_leaf)
        )
        ll = jnp.sum(lls)
    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

    nus, Rs, nupreds, Rpreds = (seq[0], seq[1], seq[2], seq[3])

    fc_scan_elts = jnp.repeat(jnp.expand_dims(jnp.zeros((r+1, r)), axis=0), forecast, axis=0)
    
    carry_pred, seq_pred = jl.scan(
        step,
        (nus[-1], Rs[-1], jnp.zeros(r), jnp.eye(r)),
        fc_scan_elts,)


    filt_results = {"ll": ll,
                    "nus": nus,
                    "Rs": Rs,
                    "nupreds": nupreds,
                    "Rpreds": Rpreds,
                    "nuforecast": seq_pred[0],
                    "Rforecast": seq_pred[1]}
    
    return filt_results

obs_data = radar_data
obs_locs_tree = obs_data.coords_tree
PHI_obs_tree = jax.tree.map(model.process_basis.mfun, obs_locs_tree)

nbasis = model.nbasis

m_0 = jnp.zeros(nbasis)
P_0 = 100 * jnp.eye(nbasis)

nu_0 = jnp.linalg.solve(P_0, m_0)

init_mat = jnp.linalg.cholesky(jnp.linalg.inv(P_0))
likelihood='full'

@jax.jit
def objective_A(params):
                (
                    log_sigma2_eta,
                    log_sigma2_eps,
                    ks,
                    beta,
                ) = params
                ztildes_tree = obs_data.tildify(beta)
                logks1, logks2, ks3, ks4 = ks
                ks1 = jnp.exp(logks1)
                ks2 = jnp.exp(logks2)
                sigma2_eta = jnp.exp(log_sigma2_eta)
                sigma2_eps = jnp.exp(log_sigma2_eps)
                M = model.con_M((ks1, ks2, ks3, ks4))
                filt_results = sqrt_information_filter(
                    nu_0,
                    R_0=init_mat,
                    M=M,
                    PHI_tree=PHI_obs_tree,
                    sigma2_eta=sigma2_eta,
                    sigma2_eps_tree = [sigma2_eps for _ in range(obs_data.T)],
                    zs_tree = ztildes_tree,
                    likelihood=likelihood,
                    sigma2_eta_dim = model.sigma2_eta_dim,
                    sigma2_eps_dim = 0
                )
                return -filt_results['ll']

@partial(jax.jit, static_argnames=["sigma2_eta_dim", "sigma2_eps_dim", "forecast", "likelihood"])
def sqrt_information_filter_B(
        nu_0: ArrayLike,
        R_0: ArrayLike,
        M: ArrayLike,
        PHI_tree: tuple,
        sigma2_eta: ArrayLike,
        sigma2_eps_tree: tuple,
        zs_tree: tuple,
        sigma2_eta_dim: int,
        sigma2_eps_dim: int,
        forecast:int = 0,
        likelihood: str = "partial",
) -> tuple:

    r = nu_0.shape[0]
    
    mapping_elts = jax.tree.map(
        lambda t: (
            zs_tree[t],
            PHI_tree[t],
            sigma2_eps_tree[t]
        ),
        tuple(range(len(zs_tree))),
    )

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_k = tup[1]
        sigma2_eps_k = tup[2]

        match sigma2_eps_dim:
            case 0:
                i_k = PHI_k.T @ z_k / sigma2_eps_k
                R_k = jnp.linalg.qr(PHI_k/jnp.sqrt(sigma2_eps_k), mode="r")
            case 1:
                sigma_eps = jnp.diag(jnp.sqrt(sigma2_eps_k))
                i_k = PHI_k.T @ st(sigma_eps, st(sigma_eps.T, z_k, lower=False), lower=True)
                R_k = jnp.linalg.qr(st(sigma_eps.T, PHI_k, lower=False), mode="r")
            case 2:
                sigma_eps = jnp.linalg.cholesky(sigma2_eps_k)
                i_k = PHI_k.T @ st(sigma_eps, st(sigma_eps.T, z_k, lower=False), lower=True)
                R_k = jnp.linalg.qr(st(sigma_eps.T, PHI_k, lower=False), mode="r")
        
        return jnp.vstack((i_k, R_k))

    def is_leaf(node):
        return jax.tree.structure(node).num_leaves == 3

    scan_elts = jnp.array(jax.tree.map(
        informationify, mapping_elts, is_leaf=is_leaf))

    # Minv = jnp.linalg.solve(M, jnp.eye(r))
    match sigma2_eta_dim:
        case 0:
            sigma_eta = jnp.sqrt(sigma2_eta) * jnp.eye(r)
        case 1:
            sigma_eta = jnp.diag(jnp.sqrt(sigma2_eta))
        case 2:
            sigma_eta = jnp.linalg.cholesky(sigma2_eta)

    r = nu_0.shape[0]

    def step(carry, scan_elt):
        nu_tt, R_tt, _, _ = carry

        i_tp = scan_elt[0, :]
        R_tp = scan_elt[1:, :]

        U_pred = ql_L(st(R_tt.T, M.T, lower=True), sigma_eta)
        R_pred = st(U_pred, jnp.eye(r), lower=True).T
        nu_pred = R_pred.T @ R_pred @ M @ st(R_tt,
                                             st(R_tt.T,
                                                nu_tt,
                                                lower=True), lower=False)

        nu_up = nu_pred + i_tp
        R_up = qr_R(R_pred, R_tp)

        return (nu_up, R_up, nu_pred, R_pred), (nu_up, R_up, nu_pred, R_pred)

    carry, seq = jl.scan(
        step,
        (nu_0, R_0, jnp.zeros(r), jnp.eye(r)),
        scan_elts,
    )

    if likelihood in ("full", "partial"):
        mapping_elts = jax.tree.map(
            lambda t: (zs_tree[t], PHI_tree[t], seq[2]
                       [t], seq[3][t], sigma2_eps_tree[t]),
            tuple(range(len(zs_tree))),
        )

        def likelihood_func(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            nu_pred = tree[2]
            R_pred = tree[3]
            sigma2_eps = tree[4]

            Q_pred = R_pred.T@R_pred
            e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)
            
            P_oprop = PHI @ jnp.linalg.solve(Q_pred, PHI.T)
            match sigma2_eps_dim:
                case 0:
                    Sigma_t = jnp.fill_diagonal(
                        P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
                    )
                    chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
                case 1:
                    Sigma_t = jnp.fill_diagonal(
                        P_oprop, jnp.diag(sigma2_eps) + jnp.diag(P_oprop), inplace=False
                    )
                    chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
                case 2:
                    Sigma_t = P_oprop + sigma2_eps
                    chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)

            z = st(chol_Sigma_t, e, lower=True)

            match likelihood:
                case 'full':
                    ll = (
                        -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                        - 0.5 * jnp.dot(z, z)
                        - 0.5 * nobs * jnp.log(2 * jnp.pi)
                    )
                case 'partial':
                    ll = (
                        -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                        - 0.5 * jnp.dot(z, z)
                    )

            return ll

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 5

        lls = jnp.array(
            jax.tree.map(likelihood_func,
                         mapping_elts, is_leaf=is_leaf)
        )
        ll = jnp.sum(lls)
    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

    nus, Rs, nupreds, Rpreds = (seq[0], seq[1], seq[2], seq[3])

    fc_scan_elts = jnp.repeat(jnp.expand_dims(jnp.zeros((r+1, r)), axis=0), forecast, axis=0)
    
    carry_pred, seq_pred = jl.scan(
        step,
        (nus[-1], Rs[-1], jnp.zeros(r), jnp.eye(r)),
        fc_scan_elts,)


    filt_results = {"ll": ll,
                    "nus": nus,
                    "Rs": Rs,
                    "nupreds": nupreds,
                    "Rpreds": Rpreds,
                    "nuforecast": seq_pred[0],
                    "Rforecast": seq_pred[1]}
    
    return filt_results


def objective_B(params):
                (
                    log_sigma2_eta,
                    log_sigma2_eps,
                    ks,
                    beta,
                ) = params
                ztildes_tree = obs_data.tildify(beta)
                logks1, logks2, ks3, ks4 = ks
                ks1 = jnp.exp(logks1)
                ks2 = jnp.exp(logks2)
                sigma2_eta = jnp.exp(log_sigma2_eta)
                sigma2_eps = jnp.exp(log_sigma2_eps)
                M = model.con_M((ks1, ks2, ks3, ks4))
                filt_results = filts.sqrt_information_filter(
                    nu_0,
                    init_mat,
                    M,
                    PHI_obs_tree,
                    sigma2_eta,
                    sigma2_eps_tree = [sigma2_eps for _ in range(obs_data.T)],
                    zs_tree = ztildes_tree,
                    likelihood=likelihood,
                    sigma2_eta_dim = model.sigma2_eta_dim,
                    sigma2_eps_dim = 0
                )

                return -filt_results['ll']


rng_key = jax.random.PRNGKey(1)
                

times_A = utils.time_jit(rng_key, jax.jit(objective_A), model.params, n=100, noise_scale=1e-5, desc = "Objective_A...")
print(times_A.average_time)
times_B = utils.time_jit(rng_key, jax.jit(objective_B), model.params, n=100, noise_scale=1e-5, desc = "Objective_B...")
print(times_B.average_time)






def cholesky_method(PHI):
    return filts.safe_cholesky(PHI.T @ PHI)

def qr_method(PHI):
    return jnp.linalg.qr(PHI, mode="r")

times_A = utils.time_jit(rng_key, jax.jit(cholesky_method), PHI, n=1000, noise_scale=1e-5, desc = "Cholesky...")
print(times_A.average_time)
times_B = utils.time_jit(rng_key, jax.jit(qr_method), PHI, n=1000, noise_scale=1e-5, desc = "QR...")
print(times_B.average_time)









import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file using NumPy
csv_data = np.loadtxt('results/Hamilton/bigruns/2025-05-08_09:17:44_results_rmh-32core.csv', delimiter=',')
csv_data = np.loadtxt('results/bigruns/2025-05-12_22:37:34_results_hmc.csv', delimiter=',')
csv_data = np.loadtxt('results/bigruns/2025-05-14_11:01:18_results_hmc.csv', delimiter=',')
csv_data = np.loadtxt('results/2025-05-15_09:35:09_results_hmc', delimiter=',')

# Convert the NumPy array to a JAX array
rmh_sample = jnp.array(csv_data)[:,1:]

print(f"Acceptance Ratio: {jnp.mean(csv_data[:,0])}")
post_mean = jnp.mean(jnp.array(rmh_sample[int(len(rmh_sample)/3):, :]), axis=0)
print(f"Posterior Mean of (transformed) parameters: \n{post_mean}")


samples = rmh_sample

# Number of parameters (columns in the array)
num_params = samples.shape[1]

# Create a figure and axes for the stacked trace plots
fig, axes = plt.subplots(num_params, 1, figsize=(10, 2 * num_params), sharex=True)

# Plot each parameter's trace
for i in range(num_params):
    axes[i].plot(samples[:, i], lw=0.8, color='b')
    axes[i].set_ylabel(f'Parameter {i+1}')
    axes[i].grid(True)

# Label the x-axis for the last subplot
axes[-1].set_xlabel('Iteration')

# Add a title to the entire figure
fig.suptitle('Trace Plots', fontsize=16, y=0.95)

# Adjust spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figure/hmc_ncc_small.png")










def basis_params_to_st_data(alphas, process_basis, process_grid, times=None):

    PHI_proc = process_basis.mfun(process_grid.coords)

    T = alphas.shape[0]
    if times is None:
        times = jnp.arange(T)

    # Hmmm... make a proper error message?
    assert T == len(times)

    #vals = (PHI_proc@alphas.T).T  # process values
    vals = alphas @ PHI_proc.T
    grids = jnp.tile(process_grid.coords, (T, 1, 1))
    t_locs = jnp.vstack(
        jl.map(
            lambda i: jnp.column_stack(
                [jnp.tile(times[i], grids[i].shape[0]), grids[i]]
            ),
            jnp.arange(T),
        )
    )
    pdata = jnp.column_stack([t_locs, jnp.concatenate(vals)])
    data = utils.st_data(x=pdata[:, 1], y=pdata[:, 2], times=pdata[:, 0], z=pdata[:, 3])
    return data

process_data = basis_params_to_st_data(alphas, model.process_basis, model.process_grid)

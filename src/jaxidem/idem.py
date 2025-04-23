#!/usr/bin/env .venv/bin/python
#JAX imports
import jax.random as rand
import jax
import jax.numpy as jnp
import jax.lax as jl
from jax.numpy.linalg import solve
from jax_tqdm import scan_tqdm
from tqdm.auto import tqdm
import optax
import blackjax

# Plotting imports
import matplotlib.pyplot as plt

# typing imports
from jaxtyping import ArrayLike, PyTree
from jaxtyping import PyTree, Float, Array
from typing import Callable, Union, NamedTuple
from functools import partial

import warnings


# In-Module imports
from jaxidem.utils import create_grid, place_basis, outer_op, Basis, Grid, st_data

import jaxidem.filters as filt

ngrids = jnp.array([41, 41])
bounds = jnp.array([[0, 1], [0, 1]])


class Kernel:
    """
    Generic class defining a kernel, or a basis expansion of a kernel with
    its parameters.
    """

    def __init__(
        self,
        function: Callable,
        basis: tuple = None,
        params: tuple = None,
        form: str = "expansion",
    ):
        self.basis = basis
        self.params = params
        self.function = function
        self.form = form

    def update(self, params):
        return Kernel(self.function, self.basis, params, self.form)
        
    def show_plot(self, width=5, height=4):
        """Shows a plot of the direction of the kernel."""

        if self.form != "expansion":
            raise Exception(
                """Kernel graphs only available for kernels formed
                              with knot-based basis functions"""
            )
        else:
            with plt.style.context("seaborn-v0_8-dark-palette"):
                fig, axes = plt.subplots(figsize=(width, height))
                bounds = jnp.array([[0, 1], [0, 1]])
                grid = create_grid(bounds, jnp.array([10, 10])).coords

                def offset(s):
                    return -jnp.array(
                        [
                            self.params[2] @ self.basis[2].vfun(s),
                            self.params[3] @ self.basis[3].vfun(s),
                        ]
                    )

                vecoffset = jax.vmap(offset)

                offsets = vecoffset(grid)

                axes.quiver(grid[:, 0], grid[:, 1], offsets[:, 0], offsets[:, 1])
                # ax.quiverkey(q, X=0.3, Y=1.1, U=10)

                axes.set_xticks([])
                axes.set_yticks([])

                axes.set_title("Kernel Direction")

                fig.show()

    def save_plot(self, filename, width=6, height=4, dpi=300, title=None):
        """Saves a plot of the direction of the kernel."""

        with plt.style.context("seaborn-v0_8-dark-palette"):
            fig, axes = plt.subplots(figsize=(width, height))
            bounds = jnp.array([[0, 1], [0, 1]])
            grid = create_grid(bounds, jnp.array([10, 10])).coords

            def offset(s):
                return -jnp.array(
                    [
                        self.params[2] @ self.basis[2].vfun(s),
                        self.params[3] @ self.basis[3].vfun(s),
                    ]
                )

            vecoffset = jax.vmap(offset)

            offsets = vecoffset(grid)

            axes.quiver(grid[:, 0], grid[:, 1], offsets[:, 0], offsets[:, 1])
            # ax.quiverkey(q, X=0.3, Y=1.1, U=10)

            axes.set_xticks([])
            axes.set_yticks([])

            if title is not None:
                axes.set_title(title)

            fig.savefig(filename, dpi=dpi)


def param_exp_kernel(K_basis: tuple, k: tuple):
    """Creates a kernel in the style of AZM's R-IDE package"""

    @jax.jit
    def kernel(s, r):
        """
        Generates the kernel function from the kernel basis and basis
        coefficients
        """
        theta = (
            k[0] @ K_basis[0].vfun(s),
            k[1] @ K_basis[1].vfun(s),
            jnp.array(
                [
                    k[2] @ K_basis[2].vfun(s),
                    k[3] @ K_basis[3].vfun(s),
                ]
            ),
        )

        return theta[0] * jnp.exp(-(jnp.sum((r - s - theta[2]) ** 2)) / theta[1])

    return Kernel(basis=K_basis, params=k, function=kernel)


class IdemParams(NamedTuple):
    log_sigma2_eps: Union[Float[Array, "()"],
                          PyTree[Float[Array, "(nobs[i],)"]],
                          PyTree[Float[Array, "(nobs[i], nobs[i])"]]]
    log_sigma2_eta: Union[Float[Array, "()"],
                          Float[Array, "(r,)"],
                          Float[Array, "(r, r)"]]
    trans_kernel_params: PyTree[Array]
    beta: ArrayLike


class Model:
    """The Integro-differential Equation Model.
    Unlike R-IDE, this does not take in data as part of the model, so the
    process grid and all involved bases must be manually made to be the domain
    of interest.
    """

    def __init__(
        self,
        process_basis,
        kernel,
        process_grid,
        sigma2_eta,
        sigma2_eps,
        beta,
        int_grid=create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([100, 100])),
    ):
        self.process_basis = process_basis
        self.kernel = kernel
        self.process_grid = process_grid
        self.sigma2_eta = jnp.array(sigma2_eta)
        self.int_grid = int_grid
        self.PHI_proc = process_basis.mfun(process_grid.coords)
        self.GRAM = (self.PHI_proc.T @ self.PHI_proc) * process_grid.area
        self.M = self.con_M(kernel.params)
        self.beta = beta
        self.nbasis = process_basis.nbasis

        trans_kernel_params = (jnp.log(self.kernel.params[0]),
                               jnp.log(self.kernel.params[1]),
                               self.kernel.params[2],
                               self.kernel.params[3])
        
        self.params = IdemParams(log_sigma2_eps=jnp.log(sigma2_eps),
                                 log_sigma2_eta=jnp.log(sigma2_eta),
                                 trans_kernel_params = trans_kernel_params,
                                 beta = self.beta)

        self.nparams = sum(arr.size for arr in jax.tree.leaves(self.params))
        
        self.sigma2_eta_dim = len(self.sigma2_eta.shape)

        match sigma2_eps:
            case jnp.ndarray():
                self.sigma2_eps = sigma2_eps
                self.sigma2_eps_dim = len(self.sigma2_eps.shape)
                self.eps_type = "array"
            case _ if isinstance(sigma2_eps, float):
                self.sigma2_eps = jnp.array(sigma2_eps)
                self.sigma2_eps_dim = 0
                self.eps_type = "array"
            case _ if isinstance(jax.tree.flatten(sigma2_eps_tree)[0][0], jnp.ndarray):
                self.sigma2_eps = sigma2_eps
                self.sigma2_eps_dim = len(jax.tree.flatten(sigma2_eps_tree)[0][0].shape)
                self.eps_type = "pytree"

    def simulate(
        self,
        key,
        obs_locs=None,
        fixed_data=True,
        nobs=100,
        T=9,
        int_grid: Grid = create_grid(bounds, ngrids),
        alpha_0=None,
    ):
        """
        Simulates from the model, using the jit-able function sim_idem.

        Parameters
        ----------
        key: ArrayLike
            PRNG key
        obs_locs: ArrayLike (3, n)
            the observation locations in long format. This should be a
            (3, n) array where the first column corresponds to time, and
            the last two to spatial coordinates. If this is not
            provided, 50 random points per time are chosen in the domain
            of interest.d
        fixed_data: Bool
            Whether the simulated data are randomly place or static across
            time (unless obs_locs is manually given)
        T: int
            The number of time points to simulate
        int_grid: ArrayLike (3, nint)
            The grid over which to compute the Riemann integral.
        alpha_0: ArrayLike (nbasis,)
            Initial value of the process coefficients

        Returns
        ----------
        tuple
            A tuple containing the Process data and the Observed data, both
            in long format in the st_data type (see
            [utilities](/.env.example))
        """

        M = self.M
        PHI_proc = self.PHI_proc
        beta = self.beta
                
        process_grid = self.process_grid

        # Check that M is not explosive
        if jnp.max(jnp.absolute(jnp.linalg.eig(M)[0])) > 1.0:
            warnings.warn(
                """Eigenvalue above the absolute value of 1. Result
                will be explosive."""
            )

        bounds = jnp.array(
            [
                [
                    jnp.min(process_grid.coords[:, 0]),
                    jnp.max(process_grid.coords[:, 0]),
                ],
                [
                    jnp.min(process_grid.coords[:, 1]),
                    jnp.max(process_grid.coords[:, 1]),
                ],
            ]
        )

        keys = rand.split(key, 3)

        if obs_locs is None:
            if fixed_data:
                obs_locs = jnp.column_stack(
                    [
                        jnp.repeat(jnp.arange(T), nobs) + 1,
                        jnp.tile(
                            rand.uniform(
                                keys[0],
                                shape=(nobs, 2),
                                minval=bounds[:, 0],
                                maxval=bounds[:, 1],
                            ),
                            (T, 1),
                        ),
                    ]
                )
            else:
                obs_locs = jnp.column_stack(
                    [
                        jnp.repeat(jnp.arange(T), nobs) + 1,
                        rand.uniform(
                            keys[0],
                            shape=(T * nobs, 2),
                            minval=bounds[:, 0],
                            maxval=bounds[:, 1],
                        ),
                    ]
                )

            times = jnp.unique(obs_locs[:, 0])
        else:
            times = jnp.unique(obs_locs[:, 0])

        nobs = obs_locs.shape[0]
            
        match len(self.sigma2_eta.shape):
            case 0:
                Sigma_eta = self.sigma2_eta * jnp.eye(self.nbasis)
            case 1:
                Sigma_eta = jnp.diag(self.sigma2_eta)
            case 2:
                Sigma_eta = self.sigma2_eta

        match len(self.sigma2_eps.shape):
            case 0:
                Sigma_eps = self.sigma2_eps * jnp.eye(nobs)
            case 1:
                Sigma_eps = jnp.diag(self.sigma2_eps)
            case 2:
                Sigma_eps = self.sigma2_eps

            
        if T != len(times):
            raise ValueError("The times in obs_locs does not match inputted T")

        obs_locs_tree = jax.tree.map(
            lambda t: obs_locs[jnp.where(obs_locs[:, 0] == t)][:, 1:], list(times)
        )
        PHI_tree = jax.tree.map(self.process_basis.mfun, obs_locs_tree)
        # really should consider exploring a sparse matrix solution!
        PHI_obs = jax.scipy.linalg.block_diag(*PHI_tree)

        nbasis = self.process_basis.nbasis

        if alpha_0 is None:
            alpha_0 = jax.random.multivariate_normal(
                keys[1], jnp.zeros(nbasis), 0.1 * jnp.eye(nbasis)
            )

        process_vals, obs_vals = sim_idem(
            key=keys[2],
            T=T,
            M=M,
            PHI_proc=PHI_proc,
            PHI_obs=PHI_obs,
            beta=beta,
            alpha_0=alpha_0,
            obs_locs=obs_locs,
            process_grid=process_grid,
            Sigma_eta=Sigma_eta,
            Sigma_eps=Sigma_eps,
        )

        # Create st_data object
        process_grids = jnp.tile(process_grid.coords, (T, 1, 1))

        t_process_locs = jnp.vstack(
            jl.map(
                lambda i: jnp.column_stack(
                    [jnp.tile(i, process_grids[i].shape[0]), process_grids[i]]
                ),
                jnp.arange(T) + 1,
            )
        )

        pdata = jnp.column_stack([t_process_locs, jnp.concatenate(process_vals)])

        process_data = st_data(
            x=pdata[:, 1], y=pdata[:, 2], t=pdata[:, 0], z=pdata[:, 3]
        )

        obs_data = st_data(
            x=obs_locs[:, 1], y=obs_locs[:, 2], t=obs_locs[:, 0], z=obs_vals
        )

        return (process_data, obs_data)

    def get_log_like(self,
            obs_data,
            method="sqrt",
            m_0=None,
            P_0=None,
            likelihood='partial',
            negative=False):

        nbasis = self.nbasis
        
        if method in ("sqrt", "kalman"):
                
            obs_data_wide = obs_data.as_wide()

            if jnp.isnan(obs_data_wide["z"]).any():
                raise ValueError("Missing data detected. This is not supported for method='kalman' or 'sqrt'. Please use method='inf' or 'sqinf'. Note that errors must be iid for those methods.")
            # if not isinstance(X_obs, jax.numpy.ndarray):
            #     raise ValueError("X_obs must be an ndarray for Kalman/square-root filtering. If it is a PyTree, consider method='inf' or 'sqinf', where the spatial stations are allowed to vary with time (hence X_obs is a tree with T elements corresponding to the covariance matrices at each time).")
                         
            obs_locs = jnp.column_stack([obs_data_wide["x"], obs_data_wide["y"]])
            PHI_obs = self.process_basis.mfun(obs_locs)
            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            match method:
                case "sqrt":
                    init_mat = jnp.linalg.cholesky(P_0)
                    filterer = filt.sqrt_filter
                case "kalman":
                    init_mat = P_0
                    filterer = filt.kalman_filter
                
            @jax.jit
            def objective(params):
                (
                          log_sigma2_eta,
                          log_sigma2_eps,
                          ks,
                          beta,
                ) = params
                ztildes_mat = jnp.column_stack(obs_data.tildify(beta))
                logks1, logks2, ks3, ks4 = ks
                ks1 = jnp.exp(logks1)
                ks2 = jnp.exp(logks2)
                sigma2_eta = jnp.exp(log_sigma2_eta)
                sigma2_eps = jnp.exp(log_sigma2_eps)
                M = self.con_M((ks1, ks2, ks3, ks4))
                filt_results = filterer(
                    m_0,
                    init_mat,
                    M,
                    PHI_obs,
                    sigma2_eta,
                    sigma2_eps,
                    ztildes_mat,
                    likelihood=likelihood,
                    sigma2_eta_dim = self.sigma2_eta_dim,
                    sigma2_eps_dim = self.sigma2_eps_dim,
                )
                if negative:
                    return -filt_results['ll']
                else:
                    return filt_results['ll']
            return objective
                
        elif method in ("inf", "sqinf"):

            # if isinstance(X_obs, jax.numpy.ndarray):
            #     raise ValueError(f"X_obs is a JAX array, but for method={method} it must be a PyTree of length T, with each element beingt the covariate matrix for that time. Assuming the spatial stations are stationary across time and no missing data , try [X_obs for _ in range(T)], or consider method'kalman' or 'sqrt'.")
                
            #unique_times = jnp.unique(obs_data.t)
            #T = len(unique_times)
            #zs_tuple = [obs_data.z[obs_data.t == t] for t in unique_times]
            zs_tree = obs_data.zs_tree

            obs_locs_tree = obs_data.coords_tree
                
            PHI_obs_tree = jax.tree.map(self.process_basis.mfun, obs_locs_tree)
            
            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            if self.sigma2_eps_dim != 0:
                raise ValueError("Non-iid measurement errors are not supported for method='inf' or 'sqinf'. Please use methof='kalman' or 'sqrt'.")
            nu_0 = jnp.linalg.solve(P_0, m_0)

            match method:
                case "sqinf":
                    init_mat = jnp.linalg.cholesky(jnp.linalg.inv(P_0))
                    filterer = filt.sqrt_information_filter
                case "inf":
                    init_mat = jnp.linalg.inv(P_0)
                    filterer = filt.information_filter

            @jax.jit
            def objective(params):
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
                M = self.con_M((ks1, ks2, ks3, ks4))
                filt_results = filterer(
                    nu_0,
                    init_mat,
                    M,
                    PHI_obs_tree,
                    sigma2_eta,
                    [sigma2_eps for _ in range(obs_data.T)],
                    ztildes_tree,
                    likelihood=likelihood,
                    sigma2_eta_dim = self.sigma2_eta_dim,
                    sigma2_eps_dim = 0
                )
                if negative:
                    return -filt_results['ll']
                else:
                    return filt_results['ll']
            return objective
        else:
            raise ValueError(f"Invalid method, {method}, Please select one of ['kalman', 'sqrt', 'inf', 'sqinf'].")

    def filter(self, obs_data, X_obs, method="sqrt", params = None, m_0=None, P_0=None, likelihood='partial'):

        nbasis = self.nbasis
        if params is None:
            params = self.params
        
        if method in ("sqrt", "kalman"):
                
            obs_data_wide = obs_data.as_wide()

            if jnp.isnan(obs_data_wide["z"]).any():
                raise ValueError("Missing data detected. This is not supported for method='kalman' or 'sqrt'. Please use method='inf' or 'sqinf'. Note that errors must be iid for those methods.")
            if not isinstance(X_obs, jax.numpy.ndarray):
                raise ValueError("X_obs must be an ndarray for Kalman/square-root filtering. If it is a PyTree, consider method='inf' or 'sqinf', where the spatial stations are allowed to vary with time (hence X_obs is a tree with T elements corresponding to the covariance matrices at each time).")
                         
            obs_locs = jnp.column_stack([obs_data_wide["x"], obs_data_wide["y"]])
            PHI_obs = self.process_basis.mfun(obs_locs)
            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            match method:
                case "sqrt":
                    init_mat = jnp.linalg.cholesky(P_0)
                    filterer = filt.sqrt_filter
                case "kalman":
                    init_mat = P_0
                    filterer = filt.kalman_filter
                

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
            M = self.con_M((ks1, ks2, ks3, ks4))
            ztildes = obs_data_wide["z"] - (X_obs @ beta)[:, None]
            filt_results = filterer(
                m_0,
                init_mat,
                M,
                PHI_obs,
                sigma2_eta,
                sigma2_eps,
                ztildes,
                likelihood=likelihood,
                sigma2_eta_dim = self.sigma2_eta_dim,
                sigma2_eps_dim = self.sigma2_eps_dim
            )

            ms = filt_results[1]
            filt_data = basis_params_to_st_data(ms, self.process_basis, self.process_grid)

            return (filt_data, filt_results)
                
        elif method in ("inf", "sqinf"):

            if isinstance(X_obs, jax.numpy.ndarray):
                raise ValueError(f"X_obs is a JAX array, but for method={method} it must be a PyTree of length T, with each element beingt the covariate matrix for that time. Assuming the spatial stations are stationary across time and no missing data , try [X_obs for _ in range(T)], or consider method'kalman' or 'sqrt'.")
                
            unique_times = jnp.unique(obs_data.t)
            T = len(unique_times)
            zs_tuple = [obs_data.z[obs_data.t == t] for t in unique_times]

            obs_locs = jnp.column_stack(jnp.column_stack((obs_data.x, obs_data.y))).T
            obs_locs_tuple = tuple([obs_locs[obs_data.t == t][:, 0:] for t in unique_times])
                
            PHI_obs_tuple = jax.tree.map(self.process_basis.mfun, obs_locs_tuple)
            
            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            if self.sigma2_eps_dim != 0:
                raise ValueError("Non-iid measurement errors are not supported for method='inf' or 'sqinf'. Please use methof='kalman' or 'sqrt'.")
            nu_0 = jnp.linalg.solve(P_0, m_0)

            match method:
                case "sqinf":
                    init_mat = jnp.linalg.cholesky(jnp.linalg.inv(P_0))
                    filterer = filt.sqrt_information_filter
                case "inf":
                    init_mat = jnp.linalg.inv(P_0)
                    filterer = filt.information_filter
            @jax.jit
            def tildify(z, X_obs_ind, beta):
                return z - X_obs_ind @ beta
            mapping_elts = tuple(
                [[zs_tuple[i], X_obs[i]] for i in range(len(zs_tuple))]
            )
            def is_leaf(node):
                return jax.tree.structure(node).num_leaves == 2

            (
                log_sigma2_eta,
                log_sigma2_eps,
                ks,
                beta,
            ) = params
            ztildes = jax.tree.map(
                lambda tup: tildify(tup[0], tup[1], beta), mapping_elts, is_leaf=is_leaf
            )
            logks1, logks2, ks3, ks4 = ks
            ks1 = jnp.exp(logks1)
            ks2 = jnp.exp(logks2)
            sigma2_eta = jnp.exp(log_sigma2_eta)
            sigma2_eps = jnp.exp(log_sigma2_eps)
            M = self.con_M((ks1, ks2, ks3, ks4))
            filt_results = filterer(
                nu_0,
                init_mat,
                M,
                PHI_obs_tuple,
                sigma2_eta,
                [sigma2_eps for _ in range(T)],
                ztildes,
                likelihood=likelihood,
                sigma2_eta_dim = self.sigma2_eta_dim,
                sigma2_eps_dim = 0
            )

            nus = filt_results[1]
            
            match method:
                case "sqinf":
                    Rs = (filt_results[2], False)
                    ms = jax.scipy.cho_solve(Rs, nus[..., None]).squeeze(-1)
                case "inf":
                    Qs = filt_results[2]
                    ms = jnp.linalg.solve(Qs, nus[..., None]).squeeze(-1)

            filt_data = basis_params_to_st_data(ms, self.process_basis, self.process_grid)

            return (filt_data, filt_results)
        
        else:
            raise ValueError(f"Invalid method, {method}, Please select one of ['kalman', 'sqrt', 'inf', 'sqinf'].")


    def update(self, params):

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

        ker_params = (ks1,ks2,ks3,ks4)

        new_kernel = self.kernel.update(ker_params)
        
        newmodel = Model(self.process_basis,
                         new_kernel,
                         self.process_grid,
                         sigma2_eta=sigma2_eta,
                         sigma2_eps=sigma2_eps,
                         beta=beta,
                         int_grid=self.int_grid)

        return newmodel

            
    def smooth(self, ms, Ps, mpreds, Ppreds):
        """NOT FULLY IMPLEMENTED """
        M = self.M
        nbasis = ms[-1].shape[0]

        carry, seq = kalman_smoother(ms, Ps, mpreds, Ppreds, M)

        return (
            jnp.vstack([jnp.flip(seq[0], axis=1), ms[-1]]),
            jnp.concatenate(
                [jnp.flip(seq[1], axis=1), jnp.reshape(Ps[-1], (1, nbasis, nbasis))]
            ),
            jnp.flip(seq[2], axis=1),
        )

    def lag1smooth(self, Ps, Js, K_T, PHI_obs):
        """NOT FULLY IMPLEMENTED OR TESTED"""
        M = self.M
        nbasis = Ps[0].shape[0]

        carry, seq = filt.lag1_smoother(Ps, Js, K_T, PHI_obs, M)

        P_TTmT = (jnp.eye(nbasis) - K_T @ PHI_obs) @ M @ Ps[-2]

        return jnp.concatenate(
            [jnp.flip(seq, axis=1), jnp.reshape(P_TTmT, (1, nbasis, nbasis))]
        )

    @partial(jax.jit, static_argnames=["self"])
    def con_M(self, ks):
        """
        Creates the propegation matrix, M, with a given set of kernel parameters.

        Params
        ----------
        ks: PyTree(ArrayLike)
            The kernel parameters used to construct the matrix (must match the
            structure of self.kernel.params).
        Returns
        ----------
        M: ArrayLike (r, r)
            The propegation matrix M.
        """

        def kernel(s, r):
            theta = (
                ks[0] @ self.kernel.basis[0].vfun(s),
                ks[1] @ self.kernel.basis[1].vfun(s),
                jnp.array(
                    [
                        ks[2] @ self.kernel.basis[2].vfun(s),
                        ks[3] @ self.kernel.basis[3].vfun(s),
                    ]
                ),
            )
            return theta[0] * jnp.exp(-(jnp.sum((r - s - theta[2]) ** 2)) / theta[1])

        K = outer_op(self.process_grid.coords, self.process_grid.coords, kernel)
        # TODO: Investigate better, faster, more accurate ways to ocmpute this?
        return (
            solve(self.GRAM, self.PHI_proc.T @ K @ self.PHI_proc)
            * self.process_grid.area**2
        )

    def fit_mle(
        self,
        obs_data: st_data,
        X_obs: ArrayLike,
        fixed_ind: list = [],
        optimizer=optax.adam(1e-3),
        debug=False,
        max_its: int = 100,
        target_ll: ArrayLike = jnp.inf,
        likelihood: str = "partial",
        eps=None,
        loading_bar=True,
        method = 'sqrt'
    ):
        """
        Fits a new model by maximum likelihood estimation, maximizing the
        data likelihood, computed by the standard Kalman filter, using a given
        OPTAX optimiser.

        Params
        ----------
        obs_data: st_data
          The observed data, as an st_data object containing the data to be fit
          to.
        X_obs: ArrayLike (nobs, p)
          Matrix of covariate data, where p is the number of covariates
          (including a column of 1s)
        fixed_ind: list = []
          List of strings representing the variables to keep fixed at the value
          in ```self```. Possible values; "sigma2_eps", "sigma2_eta", "ks1",
          "ks2", "ks3", "ks4", "beta".
        lower: tuple = None
          Lower bounds on the parameters
        upper:tuple = None
          Upper bounds on the parameters
        optimizer: Callable = optax.adam(1e-3)
          Optimiser to use
          (see [here](https://optax.readthedocs.io/en/latest/api/optimizers.html)
          for available options)
        m_0: ArrayLike = None (r,)
          Initial mean vector for Kalman filter
        P_0: ArrayLike = None (r,r)
          Initial Variance matrix for Kalman filter
        debug: bool = False
          Whether to print diagnostics during the fitting
        max_its: int = 100
          Maximum number of iterations to perform (if other stopping rules
          don't stop the loop early)
        target_ll: ArrayLike = jnp.inf
          Target log likelihood which, once reached, the main loop will stop
          early
        likelihood: str = 'partial'
          Type of likelihood for computation ('full' or 'partial').
        eps: float = None
          How close two loops should be before the loop is stopped early (None
          removes this stopping rule
        loading_bar:bool = True
          Displays a tqdm bar during the main loop.

        Returns: tuple
        ----------
        A tuple containing a new, fitted idem.IDEM object and the corresponding
        parameters.
        """

        obs_data_wide = obs_data.as_wide()
        obs_locs = jnp.column_stack([obs_data_wide["x"], obs_data_wide["y"]])
        PHI_obs = self.process_basis.mfun(obs_locs)
        nbasis = self.process_basis.nbasis

        if m_0 is None:
            m_0 = jnp.zeros(nbasis)
        if P_0 is None:
            P_0 = 100 * jnp.eye(nbasis)

        bound_di = jnp.max(self.process_grid.ngrids * self.process_grid.deltas)

        ks0 = (
            jnp.log(self.kernel.params[0]),
            jnp.log(self.kernel.params[1]),
            self.kernel.params[2],
            self.kernel.params[3],
        )

        params0 = (
            jnp.log(self.Sigma_eta[0, 0]),
            jnp.log(self.sigma2_eps),
            ks0,
            self.beta,
        )

        print(f"Initial Parameters:\n\n{format_params(params0)}\n")

        objective = get_log_like(self, obs_data, X_obs, method=method, likelihood='partial', negative=True)

        obj_grad = jax.grad(objective)

        ll = -objective(params0)
        params = params0
        opt_state = optimizer.init(params)

        if loading_bar:
            progress = tqdm(range(max_its), desc="Optimising")
        else:
            progress = range(max_its)

        for i in progress:
            grad = obj_grad(params)
            updates, opt_state = optimizer.update(grad, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            # params = optax.projections.projection_box(params, lower, upper)
            llprev = ll
            ll = -objective(params)
            if eps is not None and (jnp.isclose(ll, llprev, atol=eps)):
                print("Likelihood stopped improving. Stopping early...")
                break
            if ll > target_ll:
                print("Achieved target likelihood. Stopping early...")
                break
            if loading_bar:
                progress.set_postfix_str(
                    f"ll: {round(ll)}, offsets: {[round(params[2][2].tolist()[0], 4), round(params[2][3].tolist()[0], 4)]}"
                )
            if debug & loading_bar:
                progress.write(f"\nIteration: {i}")
                progress.write(format_params(params))
                progress.write(f"Current log-likelihood {ll.tolist()}")
            elif debug:
                print(f"\nIteration: {i}")
                print(format_params(params))
                print(f"Current log-likelihood {ll.tolist()}")

        logks1, logks2, ks3, ks4 = params[2]

        ks1 = jnp.exp(logks1)
        ks2 = jnp.exp(logks2)

        new_kernel_params = (ks1, ks2, ks3, ks4)

        new_fitted_model = IDEM(
            process_basis=self.process_basis,
            kernel=param_exp_kernel(self.kernel.basis, new_kernel_params),
            process_grid=self.process_grid,
            Sigma_eta=jnp.exp(params[0]) * jnp.eye(self.process_basis.nbasis),
            sigma2_eps=jnp.exp(params[1]),
            beta=params[3],
        )

        init_ll = -objective(params0)

        print(
            f"""The log likelihood (up to a constant) of the initial model is
               {init_ll}"""
        )
        print(
            f"""The final log likelihood (up to a constant) of the fit model is
               {-objective(params)}"""
        )

        print(
            f"""\nthe final offset parameters are {params[2][2]} and
               {params[2][3]}\n\n"""
        )

        return (new_fitted_model, params)
    
    def sample_posterior(self,
                         key,
                         n,
                         burnin,
                         obs_data,
                         X_obs,
                         init_params=None,
                         log_prior_density=None,
                         inverse_mass_matrix=None,
                         sampling_method=blackjax.hmc,
                         num_integration_steps = 5,
                         step_size = 1e-3,
                         likelihood_method="kalman"):
        
        if init_params is None:
            init_params = self.params
        nparams = sum(arr.size for arr in jax.tree.leaves(init_params))
        
        log_marginal = self.get_log_like(obs_data,
               X_obs,
               method=likelihood_method, likelihood='partial')

        if log_prior_density is None:
            # flat prior by default
            @jax.jit # really not sure if this is necessary (but it is weirdly funny)
            def log_prior_density(params): return jnp.array(1)

        if inverse_mass_matrix is None:
            inverse_mass_matrix = 1*jnp.ones(nparams)

        @jax.jit
        def log_post_dens(params):
            return log_prior_density(params) + log_marginal(params)

        sampler = sampling_method(log_post_dens,
                                  step_size,
                                  inverse_mass_matrix,
                                  num_integration_steps = num_integration_steps)

        init_state = sampler.init(init_params)
        step = jax.jit(sampler.step)

        burn_key, it_key = rand.split(key,2)
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
        
        return (param_post_sample,info)

    def fit_nuts_sqkf(
        self,
        key,
        obs_data: st_data,
        X_obs: ArrayLike,
        m_0=None,
        U_0=None,
        likelihood="full",
        fixed_ind=[],
        burnin=10,
        n=10,
    ):

        obs_data_wide = obs_data.as_wide()
        obs_locs = jnp.column_stack([obs_data_wide["x"], obs_data_wide["y"]])
        PHI_obs = self.process_basis.mfun(obs_locs)
        nbasis = self.process_basis.nbasis

        if m_0 is None:
            m_0 = jnp.zeros(nbasis)
        if U_0 is None:
            U_0 = 10 * jnp.eye(nbasis)

        ks0 = (
            jnp.log(self.kernel.params[0]),
            jnp.log(self.kernel.params[1]),
            self.kernel.params[2],
            self.kernel.params[3],
        )

        params0 = (
            jnp.log(self.Sigma_eta[0, 0]),
            jnp.log(self.sigma2_eps),
            ks0,
            self.beta,
        )

        print(f"Initial Parameters:\n\n{format_params(params0)}\n")

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

            # yes, this looks bad BUT after jit-compilation,
            # these ifs will be compiled away.
            if "sigma2_eta" in fixed_ind:
                sigma2_eta = self.Sigma_eta[0, 0]
            if "sigma2_eps" in fixed_ind:
                sigma2_eps = self.sigma2_eps
            if "ks1" in fixed_ind:
                ks1 = self.kernel.params[0]
            if "ks2" in fixed_ind:
                ks2 = self.kernel.params[1]
            if "ks3" in fixed_ind:
                ks3 = self.kernel.params[2]
            if "ks4" in fixed_ind:
                ks4 = self.kernel.params[3]
            if "beta" in fixed_ind:
                beta = self.beta

            M = self.con_M((ks1, ks2, ks3, ks4))

            # CHECK BELOW
            ztildes = obs_data_wide["z"] - (X_obs @ beta)[:, None]

            ll, _, _, _, _, _ = filt.sqrt_filter_indep(
                m_0,
                U_0,
                M,
                PHI_obs,
                sigma2_eta,
                sigma2_eps,
                ztildes,
                likelihood=likelihood,
            )
            return ll

        step_size = 1e-3
        nparams = sum(leaf.size for leaf in jax.tree.leaves(params0))
        inverse_mass_matrix = jnp.ones(nparams)

        nuts = blackjax.nuts(log_marginal, step_size, inverse_mass_matrix)
        init_state = nuts.init(params0)

        step = jax.jit(nuts.step)
        
        burn_key, it_key = rand.split(key, 2)
        @scan_tqdm(burnin, desc='Burn-in...')
        def body_fn_burnin(carry, i):
            nuts_key = jax.random.fold_in(burn_key, i)
            new_state, info = step(nuts_key, carry)
            return new_state, (new_state, info)
        @scan_tqdm(n, desc='Sampling...')
        def body_fn_sample(carry, i):
            nuts_key = jax.random.fold_in(it_key, i)
            new_state, info = step(nuts_key, carry)
            return new_state, (new_state, info)

        # Burnin
        start_state, _ = jax.lax.scan(body_fn_burnin, init_state, jnp.arange(burnin))
        # Sample
        _, (param_post_sample,_) = jl.scan(body_fn_sample, start_state, jnp.arange(n))
        # print("Done!")

        post_mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), param_post_sample.position)

        logks1, logks2, ks3, ks4 = post_mean[2]

        ks1 = jnp.exp(logks1)
        ks2 = jnp.exp(logks2)

        new_kernel_params = (ks1, ks2, ks3, ks4)
        print(new_kernel_params)

        new_fitted_model = IDEM(
            process_basis=self.process_basis,
            kernel=param_exp_kernel(self.kernel.basis, new_kernel_params),
            process_grid=self.process_grid,
            Sigma_eta=jnp.exp(post_mean[0]) * jnp.eye(self.process_basis.nbasis),
            sigma2_eps=jnp.exp(post_mean[1]),
            beta=post_mean[3],
        )

        return new_fitted_model, post_mean


@partial(jax.jit, static_argnames=["T"])
def sim_idem(
    key: ArrayLike,
    T: int,
    M: ArrayLike,
    PHI_proc: ArrayLike,
    PHI_obs: ArrayLike,
    obs_locs: ArrayLike,
    beta: ArrayLike,
    alpha_0: ArrayLike,
    Sigma_eta: ArrayLike,
    Sigma_eps: ArrayLike,
    process_grid: Grid = create_grid(bounds, ngrids),
    int_grid: Grid = create_grid(bounds, ngrids),
) -> ArrayLike:
    """
    Simulates from a IDE model.
    For jit-ability, this only takes in certain parameters. For ease of use,
    use IDEM.simulate.

    Parameters
    ----------
    key: ArrayLike (2,)
        PRNG key
    T: int
        The number of time points to simulate
    M: ArrayLike (r,r)
        The transition matrix of the proces
    PHI_proc: ArrayLike (ngrid,r)
        The process basis coefficient matrix of the points on the process grid
    PHI_obs: ArrayLike (n*T,r*T)
        The process basis coefficient matrices of the observation points, in
        block-diagonal form
    beta: ArrayLike (p,)
        The covariate coefficients for the data
    sigma2_eta: float
        The variance of the process noise (currently iid, will be a covariance
        matrix in the future)
    sigma2_eps: float
        The variance of the observation noise
    alpha_0: ArrayLike (r,)
        The initial value for the process basis coefficients
    process_grid: Grid
        The grid at which to expand the process basis coefficients to the
        process function
    int_grid: Grid
        The grid to compute the Riemann integrals over (will be replaced with a
        better method soon)
    Returns
    ----------
    A tuple containing the values of the process and the values of the
    observation.
    """

    # key setup
    keys = rand.split(key, 5)

    nbasis = PHI_proc.shape[1]

    # nobs = obs_locs.shape[1]
    nobs = obs_locs.shape[0]

    # times = jnp.unique(obs_locs[:, 0], size=T)

    U_eta = jnp.linalg.cholesky(Sigma_eta)

    @jax.jit
    def step(carry, key):
        nextstate = M @ carry + U_eta @ rand.normal(key, shape=(nbasis,))
        return (nextstate, nextstate)

    alpha_keys = rand.split(keys[3], T)

    alpha = jl.scan(step, alpha_0, alpha_keys)[1]

    @jax.jit
    def get_process(alpha):
        return PHI_proc @ alpha

    vget_process = jax.vmap(get_process)

    process_vals = vget_process(alpha)
    X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

    chol_Sigma_eps = jnp.linalg.cholesky(Sigma_eps)
    
    obs_vals = (
        X_obs @ beta
        + PHI_obs @ alpha.flatten()
        + chol_Sigma_eps @ rand.normal(key, shape=(nobs,))
    )

    return (process_vals, obs_vals)


@partial(jax.jit, static_argnames=["con_M"])
def Q(
    obs_data,
    PHI_obs,
    m_0,
    Sigma_eps,
    Sigma_eta,
    Sigma_0,
    ks,
    beta,
    X_obs,
    con_M: Callable,  # function to take kernel params and give M
):
    M = con_M(ks)
    # nbasis = m_0.shape[0]

    # obs_locs = jnp.column_stack((obs_data.x, obs_data.y))

    _, seq = kalman_filter(
        m_0, Sigma_0, M, PHI_obs, Sigma_eta, Sigma_eps, beta, obs_data, X_obs
    )

    # In hindsight, storing all Ks is a waste of memory when only K_T is needed
    # (ms, Ps, mpreds, Ppreds,
    # Ks) = seq[0], seq[1], seq[2][1:], seq[3][1:], seq[5][1:]
    #
    # _, seq2 = kalman_smoother(ms, Ps, mpreds, Ppreds, M)

    # m_tTs, P_tTs, Js = (
    #   jnp.vstack([jnp.flip(seq[0], axis=1), ms[-1]]),
    #   jnp.concatenate(
    #       [jnp.flip(seq[1], axis=1), jnp.reshape(
    #           Ps[-1], (1, nbasis, nbasis))]
    #   ),
    #   jnp.flip(seq[2], axis=1),
    # )
    #
    #    _, seq3 = lag1_smoother(Ps, Js, Ks[-1], PHI_obs, M)
    #    P_TTmT = (jnp.eye(nbasis) - Ks[-1] @ PHI_obs) @ M @ Ps[-2]
    #
    #    P_ttmTs = jnp.concatenate(
    #        [jnp.flip(seq, axis=1), jnp.reshape(P_TTmT, (1, nbasis, nbasis))]
    #    )
    #
    #    xi_z = jnp.sum(obs_locs, axis=0)

def gen_example_idem(
    key: ArrayLike,
    k_spat_inv: bool = True,
    ngrid: ArrayLike = jnp.array([41, 41]),
    nints: ArrayLike = jnp.array([100, 100]),
    process_basis: Basis = None,
    sigma2_eta=0.05**2,
    sigma2_eps=0.1**2,
    beta=jnp.array([0.0, 0.0, 0.0]),
):
    """
    Creates an example IDE model, with randomly generated kernel on the
    domain [0,1]x[0,1]. Intial value of the process is simply some of the
    coefficients for the process basis are set to 1. The kernel has a
    Gaussian shape, with parameters defined as basis expansions in order to
    allow for spatial variance.

    Parameters
    ----------
    key: ArrayLike
        PRNG key
    k_spat_inv: Bool
        Whether or not the generated kernel should be spatially invarian.
    ngrid: ArrayLike
        The resolution of the grid at which the process is computed.
        Should have shape (2,).
    nints: ArrayLike
        The resolution of the grid at which Riemann integrals are computed.
        Should have shape (2,)

    Returns
    ----------
    A model of type IDEM.
    """


    keys = rand.split(key, 2)

    process_grid = create_grid(jnp.array([[0, 1], [0, 1]]), ngrid)

    if process_basis is None:
        process_basis = place_basis()
    if k_spat_inv:
        K_basis = (
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
        )
        k = (
            jnp.array([150.0]),
            jnp.array([0.002]),
            jnp.array([-0.1]),
            jnp.array([0.1]),
        )
        kernel = param_exp_kernel(K_basis, k)
    else:
        K_basis = (
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1),
            place_basis(nres=1),
        )
        k = (
            jnp.array([200]),
            jnp.array([0.002]),
            0.1 * rand.normal(keys[0], shape=(K_basis[2].nbasis,)),
            0.1 * rand.normal(keys[1], shape=(K_basis[3].nbasis,)),
        )
        kernel = param_exp_kernel(K_basis, k)

    nbasis = process_basis.nbasis
        
    if sigma2_eta is None:
        sigma2_eta = 0.05**2
    elif sigma2_eta == "random":
        A = rand.normal(keys[2], shape=(nbasis, nbasis))
        sigma2_eta = A.T @ A
        
    return Model(
        process_basis=process_basis,
        kernel=kernel,
        process_grid=process_grid,
        sigma2_eta=sigma2_eta,
        sigma2_eps=sigma2_eps,
        beta=beta,
    )

def init_model(data, n_process_grid=41, n_int_grid=100):
    #initial variances
    sigma2_eta = jnp.var(data.z)/2
    sigma2_eps = jnp.var(data.z)/2
    beta = jnp.array([0.]) # only intercept, no covariates

    # stations are stationary and there is no missing data, so there is the same number of observations per time period
    nobs = data.wide['x'].size 
    X_obs = jnp.ones((nobs,1)) # intercept
    
    process_basis = utils.place_basis(data = data.coords,
                                      nres = 2,
                                      min_knot_num = 3,) # defaults to bisquare basis functions
    
    process_grid = utils.create_grid(data.bounds, jnp.array([41, 41]))
    int_grid = utils.create_grid(data.bounds, jnp.array([100, 100]))

    const_basis = utils.constant_basis
    K_basis = (const_basis,const_basis,const_basis,const_basis)
    k = (jnp.array([150]) / (process_grid.area*process_grid.ngrid), # kernel scale
         jnp.array([0.002]) * (process_grid.area*process_grid.ngrid), # kernel shape
         jnp.array([0.]), # x drift
         jnp.array([0.])) # y drift
    kernel = idem.param_exp_kernel(K_basis, k)

    model = idem.Model(process_basis=process_basis,
                       kernel=kernel,
                       process_grid=process_grid,
                       sigma2_eta=sigma2_eta,
                       sigma2_eps=sigma2_eps,
                       beta=beta,
                       int_grid=int_grid)
    
    
def basis_params_to_st_data(alphas, process_basis, process_grid, times=None):
    """
    Converts the process expansion coefficients back into the original process
    $Y_t(s)$ on the inputted process grid.

    Params
    ----------
    alphas: ArrayLike (T, r)
      The basis coefficients of the process
    process_basis: Basis
      The basis to use in the expansion
    process_grid: Grid
      The grid points on which to evaluate $Y$
    times: ArrayLike (T,)
      (optional) The array of times which the processes correspond to
    """

    PHI_proc = process_basis.mfun(process_grid.coords)

    T = alphas.shape[0]
    if times is None:
        times = jnp.arange(T)

    assert T == len(times)

    @jax.jit
    def get_process(alpha):
        return PHI_proc @ alpha  # Could I not just multiply by PHI_proc?

    vget_process = jax.vmap(get_process)
    vals = vget_process(alphas)  # process values
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
    data = st_data(x=pdata[:, 1], y=pdata[:, 2], t=pdata[:, 0], z=pdata[:, 3])
    return data


def format_params(params):
    kernel_string = f"Kernel Parameters: \n\t shape:{jnp.exp(params[2][0]).tolist()}\n\t scale: {jnp.exp(params[2][1]).tolist()}\n\t offsets {params[2][2].tolist()}, {params[2][3].tolist()}"
    var_string = f"Variance Parameters: {jnp.exp(params[0]).tolist()}, {jnp.exp(params[1]).tolist()}"
    coeff_string = f"Coefficient Parameters: {params[3].tolist()}"
    return "\n".join([kernel_string, var_string, coeff_string])


def print_params(params: IdemParams):
    print("Parameters:")
    print(f"  sigma2_eps: {jnp.exp(params.log_sigma2_eps).tolist()}")
    print(f"  sigma2_eta: {jnp.exp(params.log_sigma2_eta).tolist()}")
    print(f"  Kernel Parameters:")
    print(f"    Scale: {jnp.exp(params.trans_kernel_params[0]).tolist()}")
    print(f"    Shape: {jnp.exp(params.trans_kernel_params[1]).tolist()}")
    print(f"    Offset X: {params.trans_kernel_params[2].tolist()}")
    print(f"    Offset Y: {params.trans_kernel_params[3].tolist()}")
    print(f"  beta: {params.beta.tolist()}")
    

if __name__ == "__main__":
    print("IDEM loaded as main. Simulating a simple example.")

    key = jax.random.PRNGKey(1)
    keys = rand.split(key, 2)

    model = gen_example_idem(keys[0], k_spat_inv=False)

    # Simulation
    T = 9
    nobs = 50

    process_data, obs_data = model.simulate(key, nobs=nobs)
    # Show all the plots generated
    # Plots are stored in the process_data, obs_data and model.kernel objects.
    process_data.show_plot()

  

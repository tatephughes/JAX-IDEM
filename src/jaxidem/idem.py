#!/usr/bin/env .venv/bin/python
# JAX imports
import jax.random as rand
import jax
import jax.numpy as jnp
import jax.lax as jl
from jax.scipy.linalg import solve
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
import jaxidem.utils as utils

import jaxidem.filters as filts

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
    """
    Creates a kernel in the style of AZM's R-IDE package. For details, see [here](../site/mathematics.html)
    """

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


# BEING DEPRECIATED just using a dictionary now.
class IdemParams(NamedTuple):
    """
    The parameters of an IDEM, as described in .
    Some parameters are log-transformed to force everything to be in $\\mathbf{R}$.
    """

    log_S2_eps: Union[
        Float[Array, "()"],
        PyTree[Float[Array, "(nobs[i],)"]],
        PyTree[Float[Array, "(nobs[i], nobs[i])"]],
    ]
    log_S2_eta: Union[Float[Array, "()"], Float[Array, "(r,)"], Float[Array, "(r, r)"]]
    trans_kernel_params: PyTree[Array]
    beta: ArrayLike


class Model:
    """
    The Integro-differential Equation Model.
    Unlike R-IDE, this does not take in data as part of the model, so the
    process grid and all involved bases must be manually made to be the domain
    of interest.
    To build a model using a dataset as a base, use init_model.

    Example
    ----------

    ```{python}
    #| output: false
    #| eval: false

    import jax
    import jaxidem.idem as idem
    import jaxidem.utils as utils
    import pandas as pd

    radar_df = pd.read_csv('./data/radar_df.csv')
    radar_data = utils.pd_to_st(radar_df, 's2', 's1', 'time', 'z')

    # Use a cosine basis with 20 frequencies in each axis (400 frequencies total)
    model = idem.init_model(data=radar_data, basis_type = 'cosine', basis_args=[20])

    ```

    We can get the negative log-likelihood function of the model (using square-root information filter) using `get_log_likelihood`

    ```{python}
    #| eval: false
    nll = model.get_log_like(radar_data, method="sqinf", likelihood='full')

    # this function takes arguments similar to `model.params`
    idem.print_params(model.params)
    # >>> Parameters:
    #       S2_eps: 49.88551330566406         <- noise of the basis coefficients
    #       S2_eta: 49.88551330566406         <- noise of the observation
    #       Kernel Parameters:                <- defines flow and diffusion of the process
    #         Scale: [150.00001525878906]     <- these parameters can themselves be basis
    #         Shape: [0.13500000536441803]       coefficients, allowing for spatiially
    #         Offset X: [0.0]                    invariant kernels
    #         Offset Y: [0.0]
    #       beta: [0.0]                       <- linear coefficients

    print(nll(model.params))
    # >>> -84666.09
    ```

    From here, we can do anything with this likelihood; for example, using optax to estimate the MLE

    ```{python}
    #| eval: false
    import optax

    nll_val_grad = jax.value_and_grad(nll)
    optimizer=optax.adam(1e-1)
    params = model.params
    opt_state = optimizer.init(model.params)

    for i in range(10): # for reasonable results, need a lot more than this
        val, grad = nll_val_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params=params)
        params = optax.apply_updates(params, updates)

    idem.print_params(params)
    # >>> Parameters:
    #       S2_eps: 18.12257957458496
    #       S2_eta: 18.876611709594727
    #       Kernel Parameters:
    #         Scale: [407.7012939453125]
    #         Shape: [0.28592854738235474]
    #         Offset X: [-0.050182048231363297]
    #         Offset Y: [-0.02744264155626297]
    #       beta: [-1.0158119201660156]

    # A new fitted model can be created using `Model.update`
    fitted_model = model.update(params)
    ```

    This functionality is also built-in to `Model.fit_mle`.
    MCMC using BlackJAX is also built-in to `Model.sample_posterior`.


    """

    def __init__(
        self,
        process_basis,
        kernel,
        process_grid,
        S2_eta,
        S2_eps,
        beta=jnp.array([0]),
        covariate_labels=["Intercept"],
        int_grid=create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([100, 100])),
    ):
        self.process_basis = process_basis
        self.kernel = kernel
        self.process_grid = process_grid
        self.S2_eta = jnp.array(S2_eta)
        self.int_grid = int_grid
        self.PHI_proc = process_basis.mfun(process_grid.coords)
        self.GRAM = (self.PHI_proc.T @ self.PHI_proc) * process_grid.area
        self.M = self.con_M(kernel.params)
        self.beta = beta
        self.covariate_labels = covariate_labels
        if len(beta) != len(covariate_labels):
            warnings.warn(
                "beta and covariate_names must have the same length; covariate names is only there to make it clear what variables are covariates, so assuming covariates inputted are of the correct length, things will still work; however, please make sure that the correct variables are being used. It is recommended to put a panda dataframe into idem.init_model to avoid these issues."
            )

        self.nbasis = process_basis.nbasis

        # should be a dictionary
        trans_kernel_params = (
            jnp.log(self.kernel.params[0]),
            jnp.log(self.kernel.params[1]),
            self.kernel.params[2],
            self.kernel.params[3],
        )

        #self.params = IdemParams(
        #    log_S2_eps=jnp.log(S2_eps),
        #    log_S2_eta=jnp.log(S2_eta),
        #    trans_kernel_params=trans_kernel_params,
        #    beta=self.beta,
        #)

        self.params = {
            "log_S2_eps": jnp.log(S2_eps),
            "log_S2_eta": jnp.log(S2_eta),
            "trans_kernel_params": trans_kernel_params,
            "beta": self.beta,
        }

        self.nparams = sum(arr.size for arr in jax.tree.leaves(self.params))

        self.S2_eta_shape = len(self.S2_eta.shape)

        match S2_eps:
            case jnp.ndarray():
                self.S2_eps = S2_eps
                self.S2_eps_shape = len(self.S2_eps.shape)
                self.eps_type = "array"
            case _ if isinstance(S2_eps, float):
                self.S2_eps = jnp.array(S2_eps)
                self.S2_eps_shape = 0
                self.eps_type = "array"
            case _ if isinstance(jax.tree.flatten(S2_eps_tree)[0][0], jnp.ndarray):
                self.S2_eps = S2_eps
                self.S2_eps_shape = len(jax.tree.flatten(S2_eps_tree)[0][0].shape)
                self.eps_type = "pytree"
                
    # @partial(jax.jit, static_argnames=["self", "alpha_0"])
    def simulate_basis(self, key, T, alpha_0=None):
        if alpha_0 is None:
            alpha_0 = jnp.zeros(self.nbasis)

        M = self.M
        PHI_proc = self.PHI_proc

        # Check that M is not explosive
        if jnp.max(jnp.absolute(jnp.linalg.eig(M)[0])) > 1.0:
            warnings.warn(
                """Eigenvalue above the absolute value of 1. Result
                will be explosive."""
            )

        match self.S2_eta_shape:
            case 0:
                U_eta = jnp.sqrt(self.S2_eta) * jnp.eye(self.nbasis)
            case 1:
                U_eta = jnp.diag(jnp.sqrt(self.S2_eta))
            case 2:
                U_eta = jnp.linalg.cholesky(self.S2_eta)

        @jax.jit
        def step(carry, key):
            nextstate = M @ carry + U_eta @ rand.normal(key, shape=(self.nbasis,))
            return (nextstate, nextstate)

        alpha_keys = rand.split(key, T)

        alphas = jl.scan(step, alpha_0, alpha_keys)[1]

        return alphas

    def simulate_process(self, alphas):
        return self.PHI_proc @ alphas.T

    def simulate_observations(self, key, alphas, data):
        T = len(data.full_times)

        coords_tree = data.coords_tree
        X_obs_tree = data.X_obs_tree

        nobs_tree = [obs.shape[0] for obs in coords_tree]

        match self.S2_eps_shape:
            case 0:
                U_eps_tree = [
                    jnp.sqrt(self.S2_eps) * jnp.eye(nobs_tree[t]) for t in range(T)
                ]
            case 1:
                U_eps_tree = jax.tree.map(
                    lambda sig: jnp.diag(jnp.sqrt(sig)), self.S2_eps
                )
            case 2:
                U_eps_tree = jax.tree.map(jnp.linalg.cholesky, self.S2_eps)

        PHI_obs_tree = jax.tree.map(self.process_basis.mfun, coords_tree)

        def get_observation(t):
            return (
                PHI_obs_tree[t] @ alphas[t, :]
                + X_obs_tree[t] @ self.beta
                + U_eps_tree[t] @ rand.normal(keys[t], shape=(nobs_tree[t],))
            )

        keys = jax.random.split(key, T)

        return jax.tree.map(get_observation, list(range(T)))

    def simulate(
        self,
        key,
        x: ArrayLike,
        y: ArrayLike,
        times,
        covariates: ArrayLike = None,
        alpha_0=None,
        dt=None,
    ):
        """
        Runs a simulation of the IDEM, generating the process at the models process_grid, and observations at given x, y, and times,
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

        keys = rand.split(key, 3)

        unique_times = jnp.unique(times)  # automatically sorted
        if dt is None:
            dt = jnp.min(jnp.abs(jnp.diff(unique_times)))

        full_times = jnp.arange(jnp.min(unique_times), jnp.max(unique_times) + dt, dt)
        T = len(full_times)

        alphas = self.simulate_basis(keys[1], T, alpha_0)

        process_values = self.simulate_process(alphas).T.reshape(
            (T * process_grid.ngrid,)
        )

        obs_data_nan = utils.st_data(
            x,
            y,
            times,
            z=jnp.full(x.shape, jnp.nan),
            dt=None,
            covariates=covariates,
            covariate_labels=self.covariate_labels,
        )
        (jnp.ones_like(x),)
        obs_vals = jnp.concatenate(
            self.simulate_observations(keys[2], alphas, obs_data_nan)
        )

        obs_data = utils.st_data(
            x,
            y,
            times,
            z=obs_vals,
            dt=None,
            covariates=covariates,
            covariate_labels=self.covariate_labels,
        )

        times = jnp.repeat(jnp.arange(1, T + 1), process_grid.ngrid)
        rep_coords = jnp.tile(process_grid.coords, (T, 1))
        x = rep_coords[:, 0]
        y = rep_coords[:, 1]

        process_data = utils.st_data(x=x, y=y, times=times, z=process_values)

        return (process_data, obs_data)

    def resimulate(self, key, data, alpha_0=None):
        return self.simulate(key, data.x, data.y, data.times, alpha_0)

    def get_log_like(
        self,
        obs_data,
        method="sqrt",
        m_0=None,
        P_0=None,
        likelihood="partial",
        negative=False,
    ):
        nbasis = self.nbasis

        if method in ("sqrt", "kalman", "skalman"):
            zs_tree = obs_data.zs_tree
            # ADD A CHECK THAT DATA N AND LOCATION IS CONSTANT
            obs_locs = obs_data.coords_tree[0]
            PHI_obs = self.process_basis.mfun(obs_locs)

            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            match method:
                case "sqrt":
                    init_mat = jnp.linalg.cholesky(P_0, upper=True)
                    filterer = filts.sqrt_filter
                case "kalman":
                    init_mat = P_0
                    filterer = filts.kal_filter
                case "skalman":
                    init_mat = P_0
                    filterer = filts.skal_filter

            @jax.jit
            def objective(params):
                #(
                #    log_S2_eps,
                #    log_S2_eta,
                #    ks,
                #    beta,
                    #) = params
                log_S2_eps = params['log_S2_eps']
                log_S2_eta = params['log_S2_eta']
                beta = params['beta']
                ks = params['trans_kernel_params']
                ztildes_tree = obs_data.tildify(beta)
                logks1, logks2, ks3, ks4 = ks
                ks1 = jnp.exp(logks1)
                ks2 = jnp.exp(logks2)
                S2_eta = jnp.exp(log_S2_eta)
                S2_eps = jnp.exp(log_S2_eps)
                M = self.con_M((ks1, ks2, ks3, ks4))
                filt_results = filterer(
                    m_0,
                    init_mat,
                    M,
                    PHI_obs,
                    S2_eta,
                    S2_eps,
                    ztildes_tree,
                    likelihood=likelihood,
                    S2_eta_shape=self.S2_eta_shape,
                    S2_eps_shape=self.S2_eps_shape,
                )
                if negative:
                    return -filt_results["ll"]
                else:
                    return filt_results["ll"]

            return objective

        elif method in ("inf", "sqinf", "parallel", "sikalman", "ikalman", "psqrt"):
            zs_tree = obs_data.zs_tree

            obs_locs_tree = obs_data.coords_tree

            PHI_obs_tree = jax.tree.map(self.process_basis.mfun, obs_locs_tree)

            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            if self.S2_eps_shape != 0:
                raise ValueError(
                    "Non-iid measurement errors are not supported for method='inf' or 'sqinf'. Please use methof='kalman' or 'sqrt'."
                )

            match method:
                case "sqinf":
                    init_vec = jnp.linalg.solve(P_0, m_0)
                    init_mat = jnp.linalg.cholesky(jnp.linalg.inv(P_0), upper=True)
                    filterer = filts.sqinf_filter
                case "inf":
                    init_vec = jnp.linalg.solve(P_0, m_0)
                    init_mat = jnp.linalg.inv(P_0)
                    filterer = filts.inf_filter
                case "parallel":
                    init_vec = m_0
                    init_mat = P_0
                    filterer = filts.pkal_filter
                case "psqrt":
                    init_vec = m_0
                    init_mat = jnp.linalg.cholesky(P_0)
                    filterer = filts.psqrt_filter
                case "ikalman":
                    init_vec = m_0
                    init_mat = P_0
                    filterer = filts.ikal_filter
                case "sikalman":
                    init_vec = m_0
                    init_mat = P_0
                    filterer = filts.sikal_filter


            @jax.jit
            def objective(params):
                #(
                #    log_S2_eps,
                #    log_S2_eta,
                #    ks,
                #    beta,
                #) = params
                log_S2_eps = params['log_S2_eps']
                log_S2_eta = params['log_S2_eta']
                beta = params['beta']
                ks = params['trans_kernel_params']
                ztildes_tree = obs_data.tildify(beta)
                logks1, logks2, ks3, ks4 = ks
                ks1 = jnp.exp(logks1)
                ks2 = jnp.exp(logks2)
                S2_eta = jnp.exp(log_S2_eta)
                S2_eps = jnp.exp(log_S2_eps)
                M = self.con_M((ks1, ks2, ks3, ks4))
                filt_results = filterer(
                    init_vec,
                    init_mat,
                    M,
                    PHI_obs_tree,
                    S2_eta,
                    [S2_eps for _ in range(obs_data.T)],
                    ztildes_tree,
                    likelihood=likelihood,
                    S2_eta_shape=self.S2_eta_shape,
                    S2_eps_shape=0,
                )
                if negative:
                    return -filt_results["ll"]
                else:
                    return filt_results["ll"]

            return objective
        else:
            raise ValueError(
                f"Invalid method, {method}, Please select one of ['kalman', 'sqrt', 'inf', 'sqinf', 'parallel']."
            )

    def get_filter(
        self,
        obs_data,
        method="sqrt",
        m_0=None,
        P_0=None,
        likelihood="partial",
        negative=False,
    ):
        nbasis = self.nbasis

        if method in ("sqrt", "kalman", "skalman"):
            zs_tree = obs_data.zs_tree
            # ADD A CHECK THAT DATA N AND LOCATION IS CONSTANT
            obs_locs = obs_data.coords_tree[0]
            PHI_obs = self.process_basis.mfun(obs_locs)

            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            match method:
                case "sqrt":
                    init_mat = jnp.linalg.cholesky(P_0, upper=True)
                    filterer = filts.sqrt_filter
                case "kalman":
                    init_mat = P_0
                    filterer = filts.kal_filter
                case "skalman":
                    init_mat = P_0
                    filterer = filts.skal_filter

            @jax.jit
            def objective(params):
                #(
                #    log_S2_eps,
                #    log_S2_eta,
                #    ks,
                #    beta,
                #) = params
                log_S2_eps = params['log_S2_eps']
                log_S2_eta = params['log_S2_eta']
                beta = params['beta']
                ks = params['trans_kernel_params']
                ztildes_tree = obs_data.tildify(beta)
                logks1, logks2, ks3, ks4 = ks
                ks1 = jnp.exp(logks1)
                ks2 = jnp.exp(logks2)
                S2_eta = jnp.exp(log_S2_eta)
                S2_eps = jnp.exp(log_S2_eps)
                M = self.con_M((ks1, ks2, ks3, ks4))
                filt_results = filterer(
                    m_0,
                    init_mat,
                    M,
                    PHI_obs,
                    S2_eta,
                    S2_eps,
                    ztildes_tree,
                    likelihood=likelihood,
                    S2_eta_shape=self.S2_eta_shape,
                    S2_eps_shape=self.S2_eps_shape,
                )
                return filt_results
            return objective

        elif method in ("inf", "sqinf", "parallel", "sikalman", "ikalman", "psqrt"):
            zs_tree = obs_data.zs_tree

            obs_locs_tree = obs_data.coords_tree

            PHI_obs_tree = jax.tree.map(self.process_basis.mfun, obs_locs_tree)

            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            if self.S2_eps_shape != 0:
                raise ValueError(
                    "Non-iid measurement errors are not supported for method='inf' or 'sqinf'. Please use methof='kalman' or 'sqrt'."
                )

            match method:
                case "sqinf":
                    init_vec = jnp.linalg.solve(P_0, m_0)
                    init_mat = jnp.linalg.cholesky(jnp.linalg.inv(P_0), upper=True)
                    filterer = filts.sqinf_filter
                case "inf":
                    init_vec = jnp.linalg.solve(P_0, m_0)
                    init_mat = jnp.linalg.inv(P_0)
                    filterer = filts.inf_filter
                case "parallel":
                    init_vec = m_0
                    init_mat = P_0
                    filterer = filts.pkal_filter
                case "psqrt":
                    init_vec = m_0
                    init_mat = jnp.linalg.cholesky(P_0)
                    filterer = filts.psqrt_filter
                case "ikalman":
                    init_vec = m_0
                    init_mat = P_0
                    filterer = filts.ikal_filter
                case "sikalman":
                    init_vec = m_0
                    init_mat = P_0
                    filterer = filts.sikal_filter


            @jax.jit
            def objective(params):
                #(
                #    log_S2_eps,
                #    log_S2_eta,
                #    ks,
                #    beta,
                #) = params
                log_S2_eps = params['log_S2_eps']
                log_S2_eta = params['log_S2_eta']
                beta = params['beta']
                ks = params['trans_kernel_params']
                ztildes_tree = obs_data.tildify(beta)
                logks1, logks2, ks3, ks4 = ks
                ks1 = jnp.exp(logks1)
                ks2 = jnp.exp(logks2)
                S2_eta = jnp.exp(log_S2_eta)
                S2_eps = jnp.exp(log_S2_eps)
                M = self.con_M((ks1, ks2, ks3, ks4))
                filt_results = filterer(
                    init_vec,
                    init_mat,
                    M,
                    PHI_obs_tree,
                    S2_eta,
                    [S2_eps for _ in range(obs_data.T)],
                    ztildes_tree,
                    likelihood=likelihood,
                    S2_eta_shape=self.S2_eta_shape,
                    S2_eps_shape=0,
                )
                return filt_results
            return objective
        else:
            raise ValueError(
                f"Invalid method, {method}, Please select one of ['kalman', 'sqrt', 'inf', 'sqinf', 'parallel']."
            )

        
    def filter(
        self,
        obs_data,
        forecast=0,
        method="sqrt",
        m_0=None,
        P_0=None,
        likelihood="partial",
    ):
        nbasis = self.nbasis

        if method in ("sqrt", "kalman", "skalman"):
            zs_tree = obs_data.zs_tree
            # ADD A CHECK THAT DATA N AND LOCATION IS CONSTANT
            obs_locs = obs_data.coords_tree[0]
            PHI_obs = self.process_basis.mfun(obs_locs)

            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            match method:
                case "sqrt":
                    init_mat = jnp.linalg.cholesky(P_0)
                    filterer = filts.sqrt_filter
                case "kalman":
                    init_mat = P_0
                    filterer = filts.kal_filter
                case "skalman":
                    init_mat = P_0
                    filterer = filts.skal_filter

            #(
            #    log_S2_eta,
            #    log_S2_eps,
            #    ks,
            #    beta,
            #) = self.params
            log_S2_eps = params['log_S2_eps']
            log_S2_eta = params['log_S2_eta']
            beta = params['beta']
            ks = params['trans_kernel_params']
            logks1, logks2, ks3, ks4 = ks
            ks1 = jnp.exp(logks1)
            ks2 = jnp.exp(logks2)
            S2_eta = jnp.exp(log_S2_eta)
            S2_eps = jnp.exp(log_S2_eps)
            M = self.con_M((ks1, ks2, ks3, ks4))
            ztildes_tree = obs_data.tildify(beta)
            filt_results = filterer(
                m_0,
                init_mat,
                M,
                PHI_obs,
                S2_eta,
                S2_eps,
                ztildes_tree,
                likelihood=likelihood,
                S2_eta_shape=self.S2_eta_shape,
                S2_eps_shape=self.S2_eps_shape,
                forecast=forecast,
            )

            ms = filt_results["ms"]
            filt_data = basis_params_to_st_data(
                ms, self.process_basis, self.process_grid
            )

            return (filt_data, filt_results)

        elif method in ("inf", "sqinf", "parallel", "ikalman", "sikalman", "psqrt"):
            zs_tree = obs_data.zs_tree

            obs_locs_tree = obs_data.coords_tree

            PHI_obs_tree = jax.tree.map(self.process_basis.mfun, obs_locs_tree)

            if m_0 is None:
                m_0 = jnp.zeros(nbasis)
            if P_0 is None:
                P_0 = 100 * jnp.eye(nbasis)

            if self.S2_eps_shape != 0:
                raise ValueError(
                    "Non-iid measurement errors are not supported for method='inf' or 'sqinf'. Please use methof='kalman' or 'sqrt'."
                )
            nu_0 = jnp.linalg.solve(P_0, m_0)
            match method:
                case "sqinf":
                    init_vec = jnp.linalg.solve(P_0, m_0)
                    init_mat = jnp.linalg.cholesky(jnp.linalg.inv(P_0), upper=True)
                    filterer = filts.sqinf_filter
                case "inf":
                    init_vec = jnp.linalg.solve(P_0, m_0)
                    init_mat = jnp.linalg.inv(P_0)
                    filterer = filts.inf_filter
                case "parallel":
                    init_vec = m_0
                    init_mat = P_0
                    filterer = filts.pkal_filter
                case "psqrt":
                    init_vec = m_0
                    init_mat = jnp.linalg.cholesky(P_0)
                    filterer = filts.psqrt_filter
                case "ikalman":
                    init_vec = m_0
                    init_mat = P_0
                    filterer = filts.ikal_filter
                case "sikalman":
                    init_vec = m_0
                    init_mat = P_0
                    filterer = filts.sikal_filter

                    
            #(
            #    log_S2_eta,
            #    log_S2_eps,
            #    ks,
            #    beta,
            #) = self.params
            log_S2_eps = params['log_S2_eps']
            log_S2_eta = params['log_S2_eta']
            beta = params['beta']
            ks = params['trans_kernel_params']
            ztildes_tree = obs_data.tildify(beta)
            logks1, logks2, ks3, ks4 = ks
            ks1 = jnp.exp(logks1)
            ks2 = jnp.exp(logks2)
            S2_eta = jnp.exp(log_S2_eta)
            S2_eps = jnp.exp(log_S2_eps)
            M = self.con_M((ks1, ks2, ks3, ks4))
            filt_results = filterer(
                nu_0,
                init_mat,
                M,
                PHI_obs_tree,
                S2_eta,
                [S2_eps for _ in range(obs_data.T)],
                ztildes_tree,
                likelihood=likelihood,
                S2_eta_shape=self.S2_eta_shape,
                S2_eps_shape=0,
                forecast=forecast,
            )

            nus = filt_results["nus"]
            nufores = filt_results["nu_forecast"]

            match method:
                case "sqinf":
                    Rs = (filt_results["Rs"], False)
                    ms = jax.scipy.linalg.cho_solve(Rs, nus[..., None]).squeeze(-1)
                    Rfores = (filt_results["R_forecast"], False)
                    mfores = jax.scipy.linalg.cho_solve(
                        Rfores, nufores[..., None]
                    ).squeeze(-1)
                case "inf":
                    Qs = filt_results["Qs"]
                    ms = jnp.linalg.solve(Qs, nus[..., None]).squeeze(-1)
                    Qfores = filt_results["Q_forecast"]
                    mfores = jnp.linalg.solve(Qfores, nufores[..., None]).squeeze(-1)

            filt_data = basis_params_to_st_data(
                ms, self.process_basis, self.process_grid
            )
            fore_data = basis_params_to_st_data(
                mfores, self.process_basis, self.process_grid
            )

            return (filt_data, fore_data, filt_results)

        else:
            raise ValueError(
                f"Invalid method, {method}, Please select one of ['kalman', 'sqrt', 'inf', 'sqinf']."
            )

    def smooth(self,
               filt_results,
               method="kalman"):

        if method == "kalman":
            ms = filt_results['ms']
            Ps = filt_results['Ps']
            (
                log_S2_eta,
                log_S2_eps,
                ks,
                beta,
            ) = self.params
            
            logks1, logks2, ks3, ks4 = ks
            ks1 = jnp.exp(logks1)
            ks2 = jnp.exp(logks2)
            S2_eta = jnp.exp(log_S2_eta)
            S2_eps = jnp.exp(log_S2_eps)
            
            M = self.con_M((ks1, ks2, ks3, ks4))

            m_post, P_post = filts.kal_smoother(ms, Ps, S2_eta, self.S2_eta_shape, M)

            smooth_data = basis_params_to_st_data(
                m_post, self.process_basis, self.process_grid
            )

            return smooth_data
            
        else:
            raise ValueError(
                f"Not implemented!"
            )
        
    def update(self, params):
        (
            log_S2_eta,
            log_S2_eps,
            ks,
            beta,
        ) = params
        logks1, logks2, ks3, ks4 = ks
        ks1 = jnp.exp(logks1)
        ks2 = jnp.exp(logks2)
        S2_eta = jnp.exp(log_S2_eta)
        S2_eps = jnp.exp(log_S2_eps)

        ker_params = (ks1, ks2, ks3, ks4)

        new_kernel = self.kernel.update(ker_params)

        newmodel = Model(
            self.process_basis,
            new_kernel,
            self.process_grid,
            S2_eta=S2_eta,
            S2_eps=S2_eps,
            beta=beta,
            int_grid=self.int_grid,
        )

        return newmodel

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

        def kernel_func(s, r):
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

        vec_ker = jax.vmap(jax.vmap(kernel_func, in_axes=(None, 0)), in_axes=(0, None))
        K = vec_ker(self.process_grid.coords, self.process_grid.coords)
        # TODO: Investigate better, faster, more accurate ways to compute this?
        # The assumption that the GRAM matrix is pdef breaks often in 32bit.
        return (solve(self.GRAM, self.PHI_proc.T @ K @ self.PHI_proc, assume_a='pos') * self.process_grid.area**2)

    def fit_mle(
        self,
        obs_data: st_data,
        fixed_ind: list = [],
        optimizer=optax.adam(1e-3),
        debug=False,
        max_its: int = 100,
        target_nll: ArrayLike = -jnp.inf,
        eps=None,
        loading_bar=True,
        method="sqrt",
    ):
        """
        Fits a new model by maximum likelihood estimation, maximizing the
        data likelihood, computed by the standard Kalman filter, using a given
        OPTAX optimiser.

        Parameters
        ----------
        obs_data: st_data
          The observed data, as an st_data object containing the data to be fit
          to.
        X_obs: ArrayLike (nobs, p)
          Matrix of covariate data, where p is the number of covariates
          (including a column of 1s)
        fixed_ind: list = []
          List of strings representing the variables to keep fixed at the value
          in ```self```. Possible values; "S2_eps", "S2_eta", "ks1",
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

        Returns
        ----------
        A tuple containing a new, fitted idem.IDEM object and the corresponding
        parameters.
        """

        bound_di = jnp.max(self.process_grid.ngrids * self.process_grid.deltas)

        print(f"Initial Parameters:\n\n{format_params(self.params)}\n")

        nll_val_grad = jax.value_and_grad(
            self.get_log_like(
                obs_data, method=method, likelihood="partial", negative=True
            )
        )

        nll, _ = nll_val_grad(self.params)
        params = self.params
        opt_state = optimizer.init(params)

        if loading_bar:
            progress = tqdm(range(max_its), desc="Optimising")
        else:
            progress = range(max_its)

        for i in progress:
            nllprev = nll
            nll, grad = nll_val_grad(params)
            updates, opt_state = optimizer.update(grad, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            # params = optax.projections.projection_box(params, lower, upper)

            if eps is not None and (jnp.isclose(nll, nllprev, atol=eps)):
                print("Likelihood stopped improving. Stopping early...")
                break
            if nll < target_nll:
                print("Achieved target likelihood. Stopping early...")
                break
            if loading_bar:
                progress.set_postfix_str(
                    f"ll: {-round(nll)}, offsets: {[round(params[2][2].tolist()[0], 4), round(params[2][3].tolist()[0], 4)]}"
                )
            if debug & loading_bar:
                progress.write(f"\nIteration: {i}")
                progress.write(format_params(params))
                progress.write(f"Current log-likelihood {-nll.tolist()}")
            elif debug:
                print(f"\nIteration: {i}")
                print(format_params(params))
                print(f"Current log-likelihood {-nll.tolist()}")

        new_fitted_model = self.update(params)

        print(
            f"""The log likelihood (up to a constant) of the initial model is
               {-nll}"""
        )
        print(
            f"""The final log likelihood (up to a constant) of the fit model is
               {-nll}"""
        )

        return (new_fitted_model, params)

    def sample_posterior(
        self,
        key,
        obs_data,
        n,
        # burnin, # nto implemented
        init=None,
        sampling_kernel=None,
    ):
        nparams = sum(arr.size for arr in jax.tree.leaves(self.params))

        if sampling_kernel is None:
            log_marginal = model.get_log_like(
                obs_data,
                method="sqinf",
                likelihood="partial",
                P_0=1000 * jnp.eye(self.process_basis.nbasis),
            )

            imm = jnp.ones(nparams)
            num_int = 5
            samp = blackjax.hmc(log_marginal, 1e-3, imm, num_int)
            step = samp.step
            init = samp.init(model.params)

            def sampling_kernel(carry, i):
                nuts_key = jax.random.fold_in(key, i)
                new_state, info = step(nuts_key, carry)
                return new_state, (new_state, info)

        _, (sample, info) = jax.lax.scan(sampling_kernel, init, jnp.arange(n))

        return (sample, info)


def gen_example_idem(
    key: ArrayLike,
    k_spat_inv: bool = True,
    ngrid: ArrayLike = jnp.array([41, 41]),
    nints: ArrayLike = jnp.array([100, 100]),
    process_basis: Basis = None,
    S2_eta=0.05**2,
    S2_eps=0.1**2,
    beta=None,
    kernel=None,
    covariate_labels=["Intercept"],
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

    if beta is None:
        beta = jnp.zeros(len(covariate_labels))

    keys = rand.split(key, 2)

    process_grid = create_grid(jnp.array([[0, 1], [0, 1]]), ngrid)

    if process_basis is None:
        process_basis = place_basis()
    if kernel is None:
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

    if S2_eta is None:
        S2_eta = 0.05**2
    elif S2_eta == "random":
        A = rand.normal(keys[2], shape=(nbasis, nbasis))
        S2_eta = A.T @ A

    return Model(
        process_basis=process_basis,
        kernel=kernel,
        process_grid=process_grid,
        S2_eta=S2_eta,
        S2_eps=S2_eps,
        beta=beta,
        covariate_labels=covariate_labels,
    )


def init_model(
    data,
    n_process_grid=41,
    n_int_grid=100,
    basis_type="cosine",
    basis_args=[10],
    k_spat_inv=True,
    k_basis_args=[[1, 1], [3, 3]],
):
    # minimum width of the space
    width = min([jnp.max(data.x) - jnp.min(data.x), jnp.max(data.y) - jnp.min(data.y)])

    # initial variances
    S2_eta = jnp.var(data.z) / 2
    S2_eps = jnp.var(data.z) / 2
    beta = jnp.zeros(data.covariates.shape[1])

    xmin = jnp.min(data.coords[:, 0])
    xmax = jnp.max(data.coords[:, 0])
    ymin = jnp.min(data.coords[:, 1])
    ymax = jnp.max(data.coords[:, 1])

    bounds = jnp.array([[xmin, xmax], [ymin, ymax]])

    if basis_type == "cosine":
        process_basis = utils.place_cosine_basis(bounds=bounds, N=basis_args[0])
    elif basis_type == "bisquare":
        process_basis = utils.place_basis(
            bounds=bounds,
            nres=basis_args[0],
            min_knot_num=basis_args[1],
        )  # defaults to bisquare basis functions
    else:
        raise ValueError(
            f"Invalid basis_type, {basis_type}, Please select one of ['bisquare', 'cosine'] (only these currently implemented)."
        )

    process_grid = utils.create_grid(
        data.bounds, jnp.array([n_process_grid, n_process_grid])
    )
    int_grid = utils.create_grid(data.bounds, jnp.array([n_int_grid, n_int_grid]))

    const_basis = utils.constant_basis

    b = 0.5 * width
    a = 1/(jnp.sqrt(2)*jnp.pi*b)

    if k_spat_inv:
        
        K_basis = (
            const_basis,
            const_basis,
            const_basis,
            const_basis,
        )
        k = (
            jnp.array([a]),
            jnp.array([b]),
            jnp.array([0.0]),
            jnp.array([0.0]),
        )
        kernel = param_exp_kernel(K_basis, k)
    else:
        K_basis = (
            const_basis,
            const_basis,
            place_basis(
                bounds=bounds,
                nres=k_basis_args[0][0],
                min_knot_num=k_basis_args[0][1],
            ),
            place_basis(
                bounds=bounds,
                nres=k_basis_args[1][0],
                min_knot_num=k_basis_args[1][1],
            ),
        )
        k = (
            jnp.array([a]),
            jnp.array([b]),
            0.1 * rand.normal(keys[0], shape=(K_basis[2].nbasis,)),
            0.1 * rand.normal(keys[1], shape=(K_basis[3].nbasis,)),
        )
        kernel = param_exp_kernel(K_basis, k)

    model = Model(
        process_basis=process_basis,
        kernel=kernel,
        process_grid=process_grid,
        S2_eta=S2_eta,
        S2_eps=S2_eps,
        beta=beta,
        covariate_labels=data.covariate_labels,
        int_grid=int_grid,
    )

    return model


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
    data = st_data(x=pdata[:, 1], y=pdata[:, 2], times=pdata[:, 0], z=pdata[:, 3])
    return data


def format_params(params):
    kernel_string = f"Kernel Parameters: \n\t shape:{jnp.exp(params[2][0]).tolist()}\n\t scale: {jnp.exp(params[2][1]).tolist()}\n\t offsets {params[2][2].tolist()}, {params[2][3].tolist()}"
    var_string = f"Variance Parameters: {jnp.exp(params[0]).tolist()}, {jnp.exp(params[1]).tolist()}"
    coeff_string = f"Coefficient Parameters: {params[3].tolist()}"
    return "\n".join([kernel_string, var_string, coeff_string])


def print_params(params: IdemParams):
    print("Parameters:")
    print(f"  S2_eps: {jnp.exp(params.log_S2_eps).tolist()}")
    print(f"  S2_eta: {jnp.exp(params.log_S2_eta).tolist()}")
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

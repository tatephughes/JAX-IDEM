#!/usr/bin/env python3

# JAX imports
import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
from jax.numpy.linalg import vector_norm, solve

# Plotting imports

import matplotlib.pyplot as plt

# Statistics and data handling imports
# import pandas as pd
# import patsy

# typing imports
from jax.typing import ArrayLike
from typing import Callable, NamedTuple  # , Union
from functools import partial

import warnings

# In-Module imports
from utilities import (
    create_grid,
    place_basis,
    outer_op,
    Basis,
    Grid,
    ST_Data_Long,
    ST_Data_Wide,
    plot_st_long,
    ST_towide,
)

ngrids = jnp.array([41, 41])
bounds = jnp.array([[0, 1], [0, 1]])


class Kernel:
    """Class defining a kernel, or a basis expansion of a kernel or its parameters."""

    def __init__(
        self,
        function: Callable,
        basis: tuple = None,
        params: tuple = None,
    ):
        self.basis = basis
        self.params = params
        self.function = function


def param_exp_kernel(K_basis: tuple, k: tuple):
    """Creates a kernel in the style of AZM's R-IDE packagess"""

    @jax.jit
    def kernel(s, r):
        """Generates the kernel function from the kernel basis and basis coefficients"""
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


class IDEM:
    """The Integro-differential Equation Model."""

    def __init__(
        self,
        process_basis,
        kernel,
        process_grid,
        sigma2_eta,
        sigma2_eps,
        beta,
        int_grid=create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([41, 41])),
        m_0=None,
        sigma2_0=None,
    ):
        self.process_basis = process_basis
        self.kernel = kernel
        self.process_grid = process_grid
        self.sigma2_eta = sigma2_eta
        self.sigma2_eps = sigma2_eps
        self.int_grid = int_grid
        self.M = construct_M(kernel, process_basis.mfun, int_grid)
        self.PHI_proc = process_basis.mfun(process_grid.coords)
        # self.PHI_obs = jl.map(process_basis.mfun, obs_locs)
        self.beta = beta
        self.m_0 = m_0
        self.sigma2_0 = sigma2_0

    def get_sim_params(self, int_grid: Grid = create_grid(bounds, ngrids)):
        """Helper function to grab the relevant parameters for simulation"""

        M = self.M
        PHI_proc = self.PHI_proc
        beta = self.beta
        sigma2_eta = self.sigma2_eta
        sigma2_eps = self.sigma2_eps
        process_grid = self.process_grid

        return (
            M,
            PHI_proc,
            beta,
            sigma2_eta,
            sigma2_eps,
            process_grid,
            int_grid,
        )

    # def get_filt_params(self):
    #    m_0 = self.m_0
    #    sigma_0 = se

    def simulate(
        self,
        key,
        obs_locs=None,
        fixed_data=True,
        nobs=100,
        T=9,
        int_grid: Grid = create_grid(bounds, ngrids),
    ):
        """Simulates from the model, using the jit-able function simIDEM.

        Parameters
        ----------
        key: ArrayLike
          PRNG key
        obs_locs: ArrayLike
          the observation locations in long format. This should be a (3, *) array where the first column corresponds to time, and the last two to spatial coordinates. If this is not provided, 50 random points per time are chosen in the domain of interest.
        int_grid: ArrayLike
          The grid over which to compute the Riemann integral.

        Returns
        ----------
        A tuple containing the Process data and the Observed data, both in long format in the ST_Data_Long type (see [utilities](/.env.example))
        """
        (
            M,
            PHI_proc,
            beta,
            sigma2_eta,
            sigma2_eps,
            process_grid,
            int_grid,
        ) = self.get_sim_params()

        m_0 = self.m_0
        P_0 = self.sigma2_0 * jnp.eye(m_0.shape[0])

        # Check that M is not explosive
        if jnp.max(jnp.absolute(jnp.linalg.eig(M)[0])) > 1.0:
            warnings.warn(
                "Eigenvalue above the absolute value of 1. Result will be explosive."
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
                        jnp.repeat(jnp.arange(T), nobs),
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
                        jnp.repeat(jnp.arange(T), nobs),
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

        # X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

        obs_locs_tree = jax.tree.map(
            lambda t: obs_locs[jnp.where(obs_locs[:, 0] == t)][:, 1:], list(times)
        )
        PHI_tree = jax.tree.map(self.process_basis.mfun, obs_locs_tree)

        alpha0 = jax.random.multivariate_normal(keys[1], m_0, P_0)

        # really should consider exploring a sparse matrix solution!
        PHI_obs = jax.scipy.linalg.block_diag(*PHI_tree)

        process_vals, obs_vals = simIDEM(
            key=keys[2],
            T=T,
            M=M,
            PHI_proc=PHI_proc,
            PHI_obs=PHI_obs,
            beta=beta,
            alpha0=alpha0,
            obs_locs=obs_locs,
            process_grid=process_grid,
            int_grid=int_grid,
        )

        # Create ST_Data_Long object
        process_grids = jnp.tile(process_grid.coords, (T, 1, 1))

        t_process_locs = jnp.vstack(
            jl.map(
                lambda i: jnp.column_stack(
                    [jnp.tile(i, process_grids[i].shape[0]), process_grids[i]]
                ),
                jnp.arange(T),
            )
        )

        pdata = jnp.column_stack([t_process_locs, jnp.concatenate(process_vals)])

        process_data = ST_Data_Long(
            x=pdata[:, 1], y=pdata[:, 2], t=pdata[:, 0], z=pdata[:, 3]
        )

        obs_data = ST_Data_Long(
            x=obs_locs[:, 1], y=obs_locs[:, 2], t=obs_locs[:, 0], z=obs_vals
        )

        return (process_data, obs_data)

    def filter(self, obs_data_wide, X_obs):
        obs_locs = jnp.column_stack([obs_data_wide.x, obs_data_wide.y])

        m_0 = self.m_0
        P_0 = self.sigma2_0 * jnp.eye(m_0.shape[0])
        M = self.M
        PHI_obs = self.process_basis.mfun(obs_locs)
        nbasis = self.process_basis.nbasis
        nobs = obs_locs.shape[0]

        Sigma_eta = self.sigma2_eta * jnp.eye(nbasis)
        Sigma_eps = self.sigma2_eps * jnp.eye(nobs)

        beta = self.beta

        carry, seq = kalman_filter(
            m_0,
            P_0,
            M,
            PHI_obs,
            Sigma_eta,
            Sigma_eps,
            beta,
            obs_data_wide,
            X_obs,
        )

        return carry[4], seq[0], seq[1], seq[2][1:], seq[3][1:], seq[5][1:]

    def smooth(self, ms, Ps, mpreds, Ppreds):
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
        M = self.M
        nbasis = Ps[0].shape[0]

        carry, seq = lag1_smoother(Ps, Js, K_T, PHI_obs, M)

        P_TTmT = (jnp.eye(nbasis) - K_T @ PHI_obs) @ M @ Ps[-2]

        return jnp.concatenate(
            [jnp.flip(seq, axis=1), jnp.reshape(P_TTmT, (1, nbasis, nbasis))]
        )

    def data_mle_fit(self, obs_data, X_obs):
        obs_data_wide = ST_towide(obs_data)
        obs_locs = jnp.column_stack([obs_data_wide.x, obs_data_wide.y])
        m_0 = self.m_0
        sigma2_0 = 0.1
        M = self.M
        beta = self.beta
        sigma2_eps = self.sigma2_eps
        sigma2_eta = self.sigma2_eta
        PHI_obs = self.process_basis.mfun(obs_locs)
        nbasis = self.process_basis.nbasis
        nobs = obs_locs.shape[0]
        grid = self.process_grid
        PHI = self.process_basis.mfun(grid.coords)
        GRAM = (PHI.T @ PHI) * grid.area

        if self.K_basis is None:
            raise "Please equip the model with a kernel basis"

        K_basis = self.K_basis
        k = self.k

        @jax.jit
        def con_M(k):
            @jax.jit
            def kernel(s, r):
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
                return theta[0] * jnp.exp(
                    -(jnp.sum((r - s - theta[2]) ** 2)) / theta[1]
                )

            K = outer_op(grid.coords, grid.coords, kernel)
            return solve(GRAM, PHI.T @ K @ PHI) * grid.area**2

        @jax.jit
        def objective(m_0, sigma2_0, sigma2_eps, sigma2_eta, ks, beta):
            M = con_M(ks)
            Sigma_eta = sigma2_eta * jnp.eye(nbasis)
            Sigma_eps = sigma2_eps * jnp.eye(nobs)
            P_0 = sigma2_0 * jnp.eye(nbasis)
            carry, _ = kalman_filter(
                m_0,
                P_0,
                M,
                PHI_obs,
                Sigma_eta,
                Sigma_eps,
                beta,
                obs_data_wide,
                X_obs,
            )
            return carry[4]


@partial(jax.jit, static_argnames=["T"])
def simIDEM(
    key: ArrayLike,
    T: int,
    M: ArrayLike,
    PHI_proc: ArrayLike,
    PHI_obs: ArrayLike,
    obs_locs: ArrayLike,
    beta: ArrayLike,
    alpha0: ArrayLike,
    sigma2_eta: float = 0.01**2,
    sigma2_eps: float = 0.01**2,
    process_grid: Grid = create_grid(bounds, ngrids),
    int_grid: Grid = create_grid(bounds, ngrids),
) -> ArrayLike:
    """
    Simulates from a given IDE model.
    For jit-ability, this only takes in certain parameters. For ease of use, use IDEM.simulate.

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
      The process basis coefficient matrices of the observation points, in block-diagonal form
    beta: ArrayLike (p,)
      The covariate coefficients for the data
    sigma2_eta: float
      The variance of the process noise (currently iid, will be a covariance matrix in the future)
    sigma2_eps: float
      The variance of the observation noise (currently iid, will be a covariance matrix in the future)
    alpha0: ArrayLike (r,)
      The initial value for the process basis coefficients
    process_grid: Grid
      The grid at which to expand the process basis coefficients to the process function
    int_grid: Grid
      The grid to compute the Riemann integrals over (will be replaced with a better method soon)
    Returns
    ----------
    A tuple containing the values of the process and the values of the observation.
    """

    # key setup
    keys = rand.split(key, 5)

    nbasis = PHI_proc.shape[1]

    # nobs = obs_locs.shape[1]
    nobs = obs_locs.shape[0]

    # times = jnp.unique(obs_locs[:, 0], size=T)

    @jax.jit
    def step(carry, key):
        nextstate = M @ carry + jnp.sqrt(sigma2_eta) * rand.normal(key, shape=(nbasis,))
        return (nextstate, nextstate)

    alpha_keys = rand.split(keys[3], T - 1)

    alpha = jnp.vstack((alpha0, jl.scan(step, alpha0, alpha_keys)[1]))

    @jax.jit
    def get_process(alpha):
        return PHI_proc @ alpha

    vget_process = jax.vmap(get_process)

    process_vals = vget_process(alpha)

    # X_proc = jnp.column_stack([jnp.ones(s_grid.shape[0]), s_grid])
    # beta = jnp.array([0.2, 0.2, 0.2])

    # X_obs = jl.map(
    #    lambda arr: jnp.column_stack([jnp.ones(arr.shape[0]), arr]), obs_locs
    # )

    # X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])
    X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

    # @jax.jit
    # def get_obs(X_obs_1, PHI_obs_1, alpha_1):
    #    return (
    #        X_obs_1 @ beta
    #        + PHI_obs_1 @ alpha_1
    #        + jnp.sqrt(sigma2_eps) * rand.normal(key, shape=(nobs,))
    #    )

    obs_vals = (
        X_obs @ beta
        + PHI_obs @ alpha.flatten()
        + jnp.sqrt(sigma2_eps) * rand.normal(key, shape=(nobs,))
    )

    return (process_vals, obs_vals)


# ONLY SUPPORTS FIXED OBSERVATION LOCATIONS
@jax.jit
def kalman_filter(
    m_0: ArrayLike,
    P_0: ArrayLike,
    M: ArrayLike,
    PHI_obs: ArrayLike,
    Sigma_eta: ArrayLike,
    Sigma_eps: ArrayLike,
    beta: ArrayLike,
    obs_data: ST_Data_Wide,  # Wide implies fixed data locs, assuming no missing data
    X_obs: ArrayLike,
) -> tuple:
    """
    Applies the Kalman Filter to a wide-format matrix of data.
    For jit-ability, this only allows full (no missing) data in a wide format.
    I hypothesise that, with a temporally parallelised filter, this will both be quicker and have this limitation removed.

    Parameters
    ----------
    m_0: ArrayLike (r,)
      The initial means of the process vector
    P_0: ArrayLike (r,r)
      The initial Covariance matrix of the process vector
    M: ArrayLike (r,r)
      The transition matrix of the process
    PHI_obs: ArrayLike (r,n)
      The process-to-data matrix
    Sigma_eta: Arraylike (r,r)
      The Covariance matrix of the process noise
    Sigma_eps: ArrayLike (n,n)
      The Covariance matrix of the observation noise
    beta: ArrayLike (p,)
      The covariate coefficients for the data
    obs_data: ST_Data_Wide
      The observed data to be filtered
    X_obs: ArrayLike (n,p)
      The matrix of covariate values
    Returns
    ----------
    A tuple containing:
      ll: The log (data) likelihood of the data
      ms: (T,r) The posterior means $m_{t mid t}$ of the process given the data 1:t
      Ps: (T,r,r) The posterior covariance matrices $P_{t mid t}$ of the process given the data 1:t
      mpreds: (T-1,r) The predicted next-step means $m_{t mid t-1}$ of the process given the data 1:t-1
      Ppreds: (T-1,r,r) The predicted next-step covariances $P_{t mid t-1}$ of the process given the data 1:t-1
      Ks: (n,r) The Kalman Gains at each time step
    """

    nbasis = m_0.shape[0]
    nobs = obs_data.z.shape[0]

    @jax.jit
    def step(carry, z_t):
        m_tt, P_tt, _, _, ll, _ = carry

        # predict
        m_pred = M @ m_tt
        P_pred = M @ P_tt @ M.T + Sigma_eta

        # Update
        eps_t = z_t - PHI_obs @ m_pred - X_obs @ beta
        # K_t = (
        #    P_pred
        #    @ PHI_obs.T
        #    @ solve(PHI_obs @ P_pred @ PHI_obs.T + Sigma_eps, jnp.eye(nobs))
        # )

        K_t = (solve(PHI_obs @ P_pred @ PHI_obs.T + Sigma_eps, PHI_obs) @ P_pred.T).T

        m_up = m_pred + K_t @ eps_t

        P_up = (jnp.eye(nbasis) - K_t @ PHI_obs) @ P_pred

        Sigma_t = PHI_obs @ P_pred @ PHI_obs.T + Sigma_eps
        ll_new = (
            ll
            + -0.5 * jnp.linalg.slogdet(Sigma_t)[1]
            - 0.5 * eps_t.T @ solve(Sigma_t, eps_t)
        )
        return (m_up, P_up, m_pred, P_pred, ll_new, K_t), (
            m_up,
            P_up,
            m_pred,
            P_pred,
            ll_new,
            K_t,
        )

    result = jl.scan(
        step,
        (m_0, P_0, m_0, P_0, 0, jnp.zeros((nbasis, nobs))),
        obs_data.z.T,
    )

    return result


@jax.jit
def kalman_smoother(ms, Ps, mpreds, Ppreds, M):
    # not implemented
    nbasis = ms[0].shape[0]

    @jax.jit
    def step(carry, y):
        m_tmtm = y[0]
        P_tmtm = y[1]
        m_ttm = y[2]
        P_ttm = y[3]

        m_tT = carry[0]
        P_tT = carry[1]

        J_tm = P_tmtm @ M.T @ solve(P_ttm, jnp.eye(nbasis))

        m_tmT = m_tmtm - J_tm @ (m_tT - m_ttm)
        P_tmT = P_tmtm - J_tm @ (P_tT - P_ttm) @ J_tm.T

        return (m_tmT, P_tmT, J_tm), (m_tmT, P_tmT, J_tm)

    ys = (
        jnp.flip(ms[0:-1], axis=1),
        jnp.flip(Ps[0:-1], axis=1),
        jnp.flip(mpreds, axis=1),
        jnp.flip(Ppreds, axis=1),
    )
    init = (ms[-1], Ps[-1], jnp.zeros((nbasis, nbasis)))

    result = jl.scan(step, init, ys)

    return result


@jax.jit
def lag1_smoother(Ps, Js, K_T, PHI_obs: ArrayLike, M: ArrayLike):
    nbasis = Ps[0].shape[0]
    P_TTmT = (jnp.eye(nbasis) - K_T @ PHI_obs) @ M @ Ps[-2]

    @jax.jit
    def step(carry, y):
        P_tt = y[0]
        P_tmtm = y[1]
        J_t = y[2]
        J_tm = y[3]

        P_tptT = carry

        P_ttmT = P_tt @ J_tm.T + J_t @ (P_tptT - M @ P_tmtm) @ J_tm.T

        return P_ttmT, P_ttmT

    ys = (
        jnp.flip(Ps[1:-1], axis=1),
        jnp.flip(Ps[0:-2], axis=1),
        jnp.flip(Js[1:], axis=1),
        jnp.flip(Js[0:-1], axis=1),
    )

    init = P_TTmT

    result = jl.scan(step, init, ys)

    return result


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
    nbasis = m_0.shape[0]

    obs_locs = jnp.column_stack((obs_data.x, obs_data.y))

    _, seq = kalman_filter(
        m_0, Sigma_0, M, PHI_obs, Sigma_eta, Sigma_eps, beta, obs_data, X_obs
    )

    # In hindsight, storing all Ks is a waste of memory when only K_T is needed
    ms, Ps, mpreds, Ppreds, Ks = seq[0], seq[1], seq[2][1:], seq[3][1:], seq[5][1:]

    _, seq2 = kalman_smoother(ms, Ps, mpreds, Ppreds, M)

    m_tTs, P_tTs, Js = (
        jnp.vstack([jnp.flip(seq[0], axis=1), ms[-1]]),
        jnp.concatenate(
            [jnp.flip(seq[1], axis=1), jnp.reshape(Ps[-1], (1, nbasis, nbasis))]
        ),
        jnp.flip(seq[2], axis=1),
    )

    _, seq3 = lag1_smoother(Ps, Js, Ks[-1], PHI_obs, M)
    P_TTmT = (jnp.eye(nbasis) - Ks[-1] @ PHI_obs) @ M @ Ps[-2]

    P_ttmTs = jnp.concatenate(
        [jnp.flip(seq, axis=1), jnp.reshape(P_TTmT, (1, nbasis, nbasis))]
    )

    xi_z = jnp.sum(obs_locs, axis=0)


def gen_example_idem(
    key: ArrayLike,
    k_spat_inv: bool = True,
    ngrid: ArrayLike = jnp.array([41, 41]),
    nints: ArrayLike = jnp.array([100, 100]),
    nobs: int = 50,
    m_0: ArrayLike = None,
    sigma2_0: ArrayLike = None,
    process_basis: Basis = None,
    sigma2_eta=0.05**2,
    sigma2_eps=0.01**2,
):
    """Creates an example IDE model, with randomly generated kernel on the domain [0,1]x[0,1]. Intial value of the process is simply some of the coefficients for the process basis are set to 1. The kernel has a Gaussian shape, with parameters defined as basis expansions in order to allow for spatial variance.

    Parameters
    ----------
    key: ArrayLike
      PRNG key
    k_spat_inv: Bool
      Whether or not the generated kernel should be spatially invariant.
    ngrid: ArrayLike
      The resolution of the grid at which the process is computed. Should have shape (2,).
    nints: ArrayLike
      The resolution of the grid at which Riemann integrals are computed. Should have shape (2,)
    nobs: int
      The number of observations per time interval.

    Returns
    ----------
    A model of type IDEM.
    """

    keys = rand.split(key, 2)

    process_grid = create_grid(jnp.array([[0, 1], [0, 1]]), ngrid)

    if process_basis is None:
        process_basis = place_basis()

    nbasis = process_basis.nbasis

    if m_0 is None:
        m_0 = jnp.zeros(nbasis)
    if sigma2_0 is None:
        sigma2_0 = 0.1

    if k_spat_inv:
        K_basis = (
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
        )
        k = (
            jnp.array([150]),
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

    beta = jnp.array([0, 0, 0])

    return IDEM(
        process_basis=process_basis,
        kernel=kernel,
        process_grid=process_grid,
        sigma2_eta=sigma2_eta,
        sigma2_eps=sigma2_eps,
        beta=beta,
        m_0=m_0,
        sigma2_0=sigma2_0,
    )


@partial(jax.jit, static_argnames=["kernel", "basis_mfun"])
def construct_M(kernel: Kernel, basis_mfun: Callable, grid: Grid):
    """Constructs the progression matrix, M, defining how the basis parameters evolve linearly with time. Integration is done by Rieamann sum for now, on the grid provided.

    Parameters
    ----------
      kernel: Arraylike, ArrayLike -> ArrayLike; kernel function defining the progression of the process. The first argument is s, the variable being integrated over, and the second object should be the parameters; an array of shape (3, ) containing the x offset, y offset and scale of the kernel.
      process_basis: Basis; the basis for the process
      grid: Grid; the grid object to be integrated over

    """

    PHI = basis_mfun(grid.coords)

    GRAM = (PHI.T @ PHI) * grid.area

    K = outer_op(grid.coords, grid.coords, kernel.function)

    return solve(GRAM, PHI.T @ K @ PHI) * grid.area**2


def basis_params_to_st_data(alphas, process_basis, process_grid):
    PHI_proc = process_basis.mfun(process_grid.coords)

    T = alphas.shape[0]

    @jax.jit
    def get_process(alpha):
        return PHI_proc @ alpha

    vget_process = jax.vmap(get_process)
    vals = vget_process(alphas)  # process values
    grids = jnp.tile(process_grid.coords, (T, 1, 1))
    t_locs = jnp.vstack(
        jl.map(
            lambda i: jnp.column_stack([jnp.tile(i, grids[i].shape[0]), grids[i]]),
            jnp.arange(T),
        )
    )
    pdata = jnp.column_stack([t_locs, jnp.concatenate(vals)])
    data = ST_Data_Long(x=pdata[:, 1], y=pdata[:, 2], t=pdata[:, 0], z=pdata[:, 3])
    return data


def param_wrap(params: tuple):
    m_0 = params[0]
    sigma2_0 = jnp.array([params[1]])
    sigma2_eps = jnp.array([params[2]])
    sigma2_eta = jnp.array([params[3]])
    k1 = params[4][0]
    k2 = params[4][1]
    k3 = params[4][2]
    k4 = params[4][3]

    nbasis = m_0.shape[0]
    kbasis1 = k1.shape[0]
    kbasis2 = k2.shape[0]
    kbasis3 = k3.shape[0]
    kbasis4 = k4.shape[0]

    param_wrapped = jnp.concatenate(
        (m_0, sigma2_0, sigma2_eps, sigma2_eta, k1, k2, k3, k4)
    )

    return (param_wrapped, (nbasis, kbasis1, kbasis2, kbasis3, kbasis4))


@partial(jax.jit, static_argnames=["dims"])
def param_unwrap(params_wrapped: ArrayLike, dims: tuple):
    nbasis = dims[0]
    kbasis1 = dims[1]
    kbasis2 = dims[2]
    kbasis3 = dims[3]
    kbasis4 = dims[4]

    m_0 = params_wrapped[:nbasis]
    sigma2_0 = params_wrapped[nbasis]
    sigma2_eps = params_wrapped[nbasis + 1]
    sigma2_eta = params_wrapped[nbasis + 2]
    k = (
        params_wrapped[nbasis + 3 : nbasis + 3 + kbasis1],
        params_wrapped[nbasis + 3 + kbasis1 : nbasis + 3 + kbasis1 + kbasis2],
        params_wrapped[
            nbasis + 3 + kbasis1 + kbasis2 : nbasis + 3 + kbasis1 + kbasis2 + kbasis3
        ],
        params_wrapped[
            nbasis + 3 + kbasis1 + kbasis2 + kbasis3 : nbasis
            + 3
            + kbasis1
            + kbasis2
            + kbasis3
            + kbasis4
        ],
    )

    return (m_0, sigma2_0, sigma2_eps, sigma2_eta, k)


if __name__ == "__main__":
    print("IDEM loaded as main. Simulating a simple example.")

    key = jax.random.PRNGKey(1)
    keys = rand.split(key, 2)

    model = gen_example_idem(keys[0], k_spat_inv=False)

    # Simulation
    T = 9
    nobs = 50

    process_data, obs_data = model.simulate(key, nobs=nobs)

    # plot the object
    plot_st_long(process_data)
    plt.show()
    plot_st_long(obs_data)
    plt.show()

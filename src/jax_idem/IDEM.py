#!/usr/bin/env python3

# JAX imports
import jax.random as rand
import jax
import jax.numpy as jnp
import jax.lax as jl
from jax.numpy.linalg import solve

from tqdm import tqdm
import optax

# Plotting imports
import matplotlib.pyplot as plt

# typing imports
from jax.typing import ArrayLike
from typing import Callable
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

from filter_smoother_functions import (
    kalman_filter,
    information_filter,
)

ngrids = jnp.array([41, 41])
bounds = jnp.array([[0, 1], [0, 1]])


class Kernel:
    """
    Generic class defining a kernel, or a basis expansion of a kernel,
    and its parameters.
    """

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

        return theta[0] * jnp.exp(-(jnp.sum((r - s - theta[2]) ** 2))
                                  / theta[1])

    return Kernel(basis=K_basis, params=k, function=kernel)


class IDEM_Model:
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
        self.PHI_proc = process_basis.mfun(process_grid.coords)
        self.GRAM = (self.PHI_proc.T @ self.PHI_proc) * process_grid.area
        self.M = self.con_M(kernel.params)
        # self.PHI_obs = jl.map(process_basis.mfun, obs_locs)
        self.beta = beta
        self.m_0 = m_0
        self.sigma2_0 = sigma2_0

        if self.sigma2_0 is None:
            self.sigma2_0 = 1000

    def get_sim_params(self, int_grid: Grid = create_grid(bounds, ngrids)):
        """Helper function to grab the relevant parameters for simulation"""

        # This is likely to be removed in the future.

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

    def simulate(
        self,
        key,
        obs_locs=None,
        fixed_data=True,
        nobs=100,
        T=9,
        int_grid: Grid = create_grid(bounds, ngrids),
    ):
        """
        Simulates from the model, using the jit-able function simIDEM.

        Parameters
        ----------
        key: ArrayLike
          PRNG key
        obs_locs: ArrayLike
          the observation locations in long format. This should be a
          (3, *) array where the first column corresponds to time, and
          the last two to spatial coordinates. If this is not
          provided, 50 random points per time are chosen in the domain
          of interest.d
        int_grid: ArrayLike
          The grid over which to compute the Riemann integral.

        Returns
        ----------
        A tuple containing the Process data and the Observed data, both
        in long format in the ST_Data_Long type (see
        [utilities](/.env.example))
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

        if T != len(times):
            raise ValueError("The times in obs_locs does not match inputted T")

        obs_locs_tree = jax.tree.map(
            lambda t: obs_locs[jnp.where(
                obs_locs[:, 0] == t)][:, 1:], list(times)
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

        pdata = jnp.column_stack(
            [t_process_locs, jnp.concatenate(process_vals)])

        process_data = ST_Data_Long(
            x=pdata[:, 1], y=pdata[:, 2], t=pdata[:, 0], z=pdata[:, 3]
        )

        obs_data = ST_Data_Long(
            x=obs_locs[:, 1], y=obs_locs[:, 2], t=obs_locs[:, 0], z=obs_vals
        )

        return (process_data, obs_data)

    def filter(self, obs_data_wide: ST_Data_Wide, X_obs):
        obs_locs = jnp.column_stack([obs_data_wide.x, obs_data_wide.y])

        m_0 = self.m_0

        M = self.M
        PHI_obs = self.process_basis.mfun(obs_locs)
        nbasis = self.process_basis.nbasis
        nobs = obs_locs.shape[0]

        if m_0 is None:
            PHI_obs_0 = self.process_basis.mfun(obs_locs)
            m_0 = PHI_obs_0.T @ obs_data_wide.z[:, 0]

        P_0 = self.sigma2_0 * jnp.eye(m_0.shape[0])

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
            obs_data_wide.z,
            X_obs,
        )

        return (carry[4], seq[0], seq[1], seq[2][1:], seq[3][1:], seq[5][1:])

    def filter_information(
        self,
        obs_data: ST_Data_Long,
        X_obs,
        nu_0=None,
        Q_0=None,
    ):

        nbasis = self.process_basis.nbasis

        if nu_0 is None:
            nu_0 = jnp.zeros(nbasis)
        if Q_0 is None:
            Q_0 = jnp.zeros((nbasis, nbasis))

        obs_locs = jnp.column_stack(
            jnp.column_stack((obs_data.x, obs_data.y))).T
        X_obs = jnp.column_stack(
            [jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

        unique_times = jnp.unique(obs_data.t)

        if unique_times.shape[0] <= 1:
            raise ValueError(
                """To filter, there needs to be more that one time
                               point."""
            )

        time_inds = tuple(jnp.arange(unique_times.shape[0]))
        obs_locs_tuple = jax.tree.map(
            lambda t: obs_locs[obs_data.t == t], time_inds)

        PHI_obs_tuple = jax.tree.map(self.process_basis.mfun, obs_locs_tuple)

        carry, seq = information_filter(
            nu_0,
            Q_0,
            self.M,
            PHI_obs_tuple,
            self.sigma2_eta,
            self.sigma2_eps,
            self.beta,
            obs_data,
            X_obs,
        )

        nus, Qs = seq

        return (nus, Qs)

    def smooth(self, ms, Ps, mpreds, Ppreds):
        M = self.M
        nbasis = ms[-1].shape[0]

        carry, seq = kalman_smoother(ms, Ps, mpreds, Ppreds, M)

        return (
            jnp.vstack([jnp.flip(seq[0], axis=1), ms[-1]]),
            jnp.concatenate(
                [jnp.flip(seq[1], axis=1), jnp.reshape(
                    Ps[-1], (1, nbasis, nbasis))]
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

    @partial(jax.jit, static_argnames=["self"])
    def con_M(self, ks):
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
            return theta[0] * jnp.exp(-(jnp.sum((r - s - theta[2]) ** 2))
                                      / theta[1])

        K = outer_op(self.process_grid.coords,
                     self.process_grid.coords, kernel)
        return (
            solve(self.GRAM, self.PHI_proc.T @ K @ self.PHI_proc)
            * self.process_grid.area**2
        )

    def data_mle_fit(
        self,
        obs_data: ST_Data_Long,
        X_obs: ArrayLike,
        fixed_ind: list = [],
        lower=None,
        upper=None,
        optimizer=optax.adam(1e-3),
        nits: int = 10,
    ):
        obs_data_wide = ST_towide(obs_data)
        obs_locs = jnp.column_stack([obs_data_wide.x, obs_data_wide.y])
        PHI_obs = self.process_basis.mfun(obs_locs)

        bound_di = jnp.max(self.process_grid.ngrids * self.process_grid.deltas)

        if upper is None:
            upper = (
                jnp.full(self.process_basis.nbasis, jnp.inf),
                jnp.array(jnp.log(1000)),
                jnp.array(jnp.log(1000)),
                jnp.array(jnp.log(1000)),
                (
                    jnp.full(
                        self.kernel.params[0].shape,
                        150 / (self.process_grid.area *
                               self.process_grid.ngrid) * 10,
                    ),
                    jnp.full(self.kernel.params[1].shape,
                             ((bound_di / 2) ** 2) / 10),
                    jnp.full(self.kernel.params[2].shape, jnp.inf),
                    jnp.full(self.kernel.params[3].shape, jnp.inf),
                ),
                jnp.full(self.beta.shape, jnp.inf),
            )

        # trying to match these bounds with andrewzm's,
        # but this is messy. Clean these up later please
        if lower is None:
            lower = (
                jnp.full(self.process_basis.nbasis, -jnp.inf),
                jnp.array(jnp.log(0.0001)),
                jnp.array(jnp.log(0.0001)),
                jnp.array(jnp.log(0.0001)),
                (
                    jnp.full(
                        self.kernel.params[0].shape,
                        150
                        / (self.process_grid.area * self.process_grid.ngrid)
                        * 0.01
                        / 1000,
                    ),
                    jnp.full(self.kernel.params[1].shape,
                             (bound_di / 2) ** 2 * 0.001),
                    jnp.full(self.kernel.params[2].shape, -jnp.inf),
                    jnp.full(self.kernel.params[3].shape, -jnp.inf),
                ),
                jnp.full(self.beta.shape, -jnp.inf),
            )

        # self.m_0 = PHI_obs.T @ obs_data_wide.z[:,0]

        # sigma2_0 = 3.4e38

        ks0 = (
            jnp.log(self.kernel.params[0]),
            jnp.log(self.kernel.params[1]),
            self.kernel.params[2],
            self.kernel.params[3],
        )

        params0 = (
            self.m_0,
            jnp.log(self.sigma2_0),
            jnp.log(self.sigma2_eta),
            jnp.log(self.sigma2_eps),
            ks0,
            self.beta,
        )

        nbasis = self.process_basis.nbasis
        nobs = obs_locs.shape[0]

        @jax.jit
        def objective(params):
            (m_0,
             log_sigma2_0,
             log_sigma2_eta,
             log_sigma2_eps,
             ks,
             beta, ) = params

            logks1, logks2, ks3, ks4 = ks

            ks1 = jnp.exp(logks1)
            ks2 = jnp.exp(logks2)

            sigma2_0 = jnp.exp(log_sigma2_0)
            sigma2_eta = jnp.exp(log_sigma2_eta)
            sigma2_eps = jnp.exp(log_sigma2_eps)

            # yes, this looks bad BUT after jit-compilation,
            # these ifs will be compiled away.
            if "m_0" in fixed_ind:
                m_0 = self.m_0
            if "sigma2_0" in fixed_ind:
                sigma2_0 = self.sigma2_0
            if "sigma2_eta" in fixed_ind:
                sigma2_eta = self.sigma2_eta
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

            Sigma_eta = sigma2_eta * jnp.eye(nbasis)
            Sigma_eps = sigma2_eps * jnp.eye(nobs)
            P_0 = sigma2_0 * jnp.eye(nbasis)

            carry, seq = kalman_filter(
                m_0,
                P_0,
                M,
                PHI_obs,
                Sigma_eta,
                Sigma_eps,
                beta,
                obs_data_wide.z,
                X_obs,
            )
            return -carry[4]

        obj_grad = jax.grad(objective)

        params = params0
        opt_state = optimizer.init(params)

        for i in tqdm(range(nits), desc="Optimising"):
            grad = obj_grad(params)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = optax.projections.projection_box(params, lower, upper)

        # please make sure this is all the arguments necessary
        new_fitted_model = IDEM_Model(
            process_basis=self.process_basis,
            kernel=param_exp_kernel(self.kernel.basis, params[4]),
            process_grid=self.process_grid,
            sigma2_eta=jnp.exp(params[2]),
            sigma2_eps=jnp.exp(params[3]),
            beta=params[5],
            m_0=params[0],
            sigma2_0=jnp.exp(params[1]),
        )

        init_ll, _, _, _, _, _ = self.filter(obs_data_wide, X_obs)

        print(
            f"""The log likelihood (up to a constant) of the initial model is
               {init_ll}"""
        )
        print(
            f"""The final log likelihood (up to a constant) of the fit model is
               {-objective(params)}"""
        )

        print(
            f"""\nthe final offset parameters are {params[4][2]} and
               {params[4][3]}\n\n"""
        )
        print(
            f"""\nthe final variance parameters are {jnp.exp(params[1])},
               {jnp.exp(params[2])}, and {jnp.exp(params[3])}\n\n"""
        )

        return new_fitted_model

    def fit_information_filter(
        self,
        obs_data: ST_Data_Long,
        X_obs: ArrayLike,
        fixed_ind: list = [],
        lower=None,
        upper=None,
        optimizer=optax.adam(1e-3),
        nits: int = 10,
    ):
        obs_data_wide = ST_towide(obs_data)
        obs_locs = jnp.column_stack([obs_data_wide.x, obs_data_wide.y])
        PHI_obs = self.process_basis.mfun(obs_locs)

        bound_di = jnp.max(self.process_grid.ngrids * self.process_grid.deltas)

        if upper is None:
            upper = (
                jnp.full(self.process_basis.nbasis, jnp.inf),
                jnp.array(jnp.log(1000)),
                jnp.array(jnp.log(1000)),
                jnp.array(jnp.log(1000)),
                (
                    jnp.full(
                        self.kernel.params[0].shape,
                        150 / (self.process_grid.area *
                               self.process_grid.ngrid) * 10,
                    ),
                    jnp.full(self.kernel.params[1].shape,
                             ((bound_di / 2) ** 2) / 10),
                    jnp.full(self.kernel.params[2].shape, jnp.inf),
                    jnp.full(self.kernel.params[3].shape, jnp.inf),
                ),
                jnp.full(self.beta.shape, jnp.inf),
            )

        # trying to match these bounds with andrewzm's,
        # but this is messy. Clean these up later please
        if lower is None:
            lower = (
                jnp.full(self.process_basis.nbasis, -jnp.inf),
                jnp.array(jnp.log(0.0001)),
                jnp.array(jnp.log(0.0001)),
                jnp.array(jnp.log(0.0001)),
                (
                    jnp.full(
                        self.kernel.params[0].shape,
                        150
                        / (self.process_grid.area * self.process_grid.ngrid)
                        * 0.01
                        / 1000,
                    ),
                    jnp.full(self.kernel.params[1].shape,
                             (bound_di / 2) ** 2 * 0.001),
                    jnp.full(self.kernel.params[2].shape, -jnp.inf),
                    jnp.full(self.kernel.params[3].shape, -jnp.inf),
                ),
                jnp.full(self.beta.shape, -jnp.inf),
            )

        # self.m_0 = PHI_obs.T @ obs_data_wide.z[:,0]

        # sigma2_0 = 3.4e38

        ks0 = (
            jnp.log(self.kernel.params[0]),
            jnp.log(self.kernel.params[1]),
            self.kernel.params[2],
            self.kernel.params[3],
        )

        params0 = (
            self.m_0,
            jnp.log(self.sigma2_0),
            jnp.log(self.sigma2_eta),
            jnp.log(self.sigma2_eps),
            ks0,
            self.beta,
        )

        nbasis = self.process_basis.nbasis
        nobs = obs_locs.shape[0]

        @jax.jit
        def objective(params):
            (m_0, log_sigma2_0, log_sigma2_eta,
             log_sigma2_eps, ks, beta, ) = params

            logks1, logks2, ks3, ks4 = ks

            ks1 = jnp.exp(logks1)
            ks2 = jnp.exp(logks2)

            sigma2_0 = jnp.exp(log_sigma2_0)
            sigma2_eta = jnp.exp(log_sigma2_eta)
            sigma2_eps = jnp.exp(log_sigma2_eps)

            if "m_0" in fixed_ind:
                m_0 = self.m_0
            if "sigma2_0" in fixed_ind:
                sigma2_0 = self.sigma2_0
            if "sigma2_eta" in fixed_ind:
                sigma2_eta = self.sigma2_eta
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

            Sigma_eta = sigma2_eta * jnp.eye(nbasis)
            Sigma_eps = sigma2_eps * jnp.eye(nobs)
            P_0 = sigma2_0 * jnp.eye(nbasis)

            carry, seq = kalman_filter(
                m_0,
                P_0,
                M,
                PHI_obs,
                Sigma_eta,
                Sigma_eps,
                beta,
                obs_data_wide.z,
                X_obs,
            )
            return -carry[4]

        obj_grad = jax.grad(objective)

        params = params0
        opt_state = optimizer.init(params)

        for i in tqdm(range(nits), desc="Optimising"):
            grad = obj_grad(params)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = optax.projections.projection_box(params, lower, upper)

        # please make sure this is all the arguments necessary
        new_fitted_model = IDEM_Model(
            process_basis=self.process_basis,
            kernel=param_exp_kernel(self.kernel.basis, params[4]),
            process_grid=self.process_grid,
            sigma2_eta=jnp.exp(params[2]),
            sigma2_eps=jnp.exp(params[3]),
            beta=params[5],
            m_0=params[0],
            sigma2_0=jnp.exp(params[1]),
        )

        init_ll, _, _, _, _, _ = self.filter(obs_data_wide, X_obs)

        print(
            f"""The log likelihood (up to a constant) of the initial model is
             {init_ll}"""
        )
        print(
            f"""The final log likelihood (up to a constant) of the fit model
               is {-objective(params)}"""
        )

        print(
            f"""\nthe final offset parameters are {params[4][2]} and
             {params[4][3]}\n\n"""
        )
        print(
            f"""\nthe final variance parameters are {jnp.exp(params[1])},
             {jnp.exp(params[2])}, and {jnp.exp(params[3])}\n\n"""
        )

        return new_fitted_model


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
    alpha0: ArrayLike (r,)
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

    @jax.jit
    def step(carry, key):
        nextstate = M @ carry + \
            jnp.sqrt(sigma2_eta) * rand.normal(key, shape=(nbasis,))
        return (nextstate, nextstate)

    alpha_keys = rand.split(keys[3], T)

    alpha = jl.scan(step, alpha0, alpha_keys)[1]

    @jax.jit
    def get_process(alpha):
        return PHI_proc @ alpha

    vget_process = jax.vmap(get_process)

    process_vals = vget_process(alpha)
    X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

    obs_vals = (
        X_obs @ beta
        + PHI_obs @ alpha.flatten()
        + jnp.sqrt(sigma2_eps) * rand.normal(key, shape=(nobs,))
    )

    return (process_vals, obs_vals)


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
    nobs: int = 50,
    m_0: ArrayLike = None,
    sigma2_0: ArrayLike = None,
    process_basis: Basis = None,
    sigma2_eta=0.05**2,
    sigma2_eps=0.01**2,
):
    """Creates an example IDE model, with randomly generated kernel on the
      domain [0,1]x[0,1]. Intial value of the process is simply some of the
      coefficients for the process basis are set to 1. The kernel has a
      Gaussian shape, with parameters defined as basis expansions in order to
      allow for spatial variance.

    Parameters
    ----------
    key: ArrayLike
      PRNG key
    k_spat_inv: Bool
      Whether or not the generated kernel should be spatially invariant.
    ngrid: ArrayLike
      The resolution of the grid at which the process is computed.
      Should have shape (2,).
    nints: ArrayLike
      The resolution of the grid at which Riemann integrals are computed.
      Should have shape (2,)
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

    print(kernel.params)

    beta = jnp.array([0.0, 0.0, 0.0])

    return IDEM_Model(
        process_basis=process_basis,
        kernel=kernel,
        process_grid=process_grid,
        sigma2_eta=sigma2_eta,
        sigma2_eps=sigma2_eps,
        beta=beta,
        m_0=m_0,
        sigma2_0=sigma2_0,
    )


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
            lambda i: jnp.column_stack(
                [jnp.tile(i, grids[i].shape[0]), grids[i]]),
            jnp.arange(T),
        )
    )
    pdata = jnp.column_stack([t_locs, jnp.concatenate(vals)])
    data = ST_Data_Long(x=pdata[:, 1], y=pdata[:, 2],
                        t=pdata[:, 0], z=pdata[:, 3])
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
        params_wrapped[nbasis + 3: nbasis + 3 + kbasis1],
        params_wrapped[nbasis + 3 + kbasis1: nbasis + 3 + kbasis1 + kbasis2],
        params_wrapped[
            nbasis
            + 3
            + kbasis1
            + kbasis2: nbasis
            + 3
            + kbasis1
            + kbasis2
            + kbasis3
        ],
        params_wrapped[
            nbasis
            + 3
            + kbasis1
            + kbasis2
            + kbasis3: nbasis
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

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
)

ngrids = jnp.array([41, 41])
bounds = jnp.array([[0, 1], [0, 1]])


class IDEM:
    """The Integro-differential Equation Model."""

    def __init__(
        self,
        process_basis,
        kernel,
        process_grid,
        sigma2_eta,
        sigma2_eps,
        alpha0,
        beta,
        int_grid=create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([41, 41])),
    ):
        self.process_basis = process_basis
        self.kernel = kernel
        self.process_grid = process_grid
        self.sigma2_eta = sigma2_eta
        self.sigma2_eps = sigma2_eps
        self.alpha0 = alpha0
        self.int_grid = int_grid
        self.M = construct_M(kernel, process_basis, int_grid)
        self.PHI_proc = process_basis.mfun(process_grid.coords)
        # self.PHI_obs = jl.map(process_basis.mfun, obs_locs)
        self.beta = beta

    def get_sim_params(self, int_grid: Grid = create_grid(bounds, ngrids)):
        """Helper function to grab the relevant parameters for simulation"""

        M = self.M
        PHI_proc = self.PHI_proc
        beta = self.beta
        sigma2_eta = self.sigma2_eta
        sigma2_eps = self.sigma2_eps
        alpha0 = self.alpha0
        process_grid = self.process_grid

        return (
            M,
            PHI_proc,
            beta,
            sigma2_eta,
            sigma2_eps,
            alpha0,
            process_grid,
            int_grid,
        )

    def simulate(
        self, key, obs_locs=None, T=9, int_grid: Grid = create_grid(bounds, ngrids)
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
            alpha0,
            process_grid,
            int_grid,
        ) = self.get_sim_params()

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

        keys = rand.split(key, 2)

        if obs_locs is None:
            nobs = 100

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

        # really should consider exploring a sparse matrix solution!
        PHI_obs = jax.scipy.linalg.block_diag(*PHI_tree)

        process_vals, obs_vals = simIDEM(
            key=keys[1],
            T=T,
            M=M,
            PHI_proc=PHI_proc,
            PHI_obs=PHI_obs,
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


@partial(jax.jit, static_argnames=["T"])
def simIDEM(
    key: ArrayLike,
    T: int,
    M: ArrayLike,
    PHI_proc: ArrayLike,
    PHI_obs: ArrayLike,
    obs_locs: ArrayLike,
    sigma2_eta: float = 0.01**2,
    sigma2_eps: float = 0.01**2,
    alpha0: ArrayLike = jnp.zeros(90).at[jnp.array([64])].set(1),
    process_grid: Grid = create_grid(bounds, ngrids),
    int_grid: Grid = create_grid(bounds, ngrids),
) -> ArrayLike:
    """
    Simulates from a given IDE model.
    For jit-ability, this only takes in certain parameters. For ease of use, use IDEM.simulate.

    Parameters
    ----------


    Returns
    ----------
    A tuple containing the values of the process and the values of the observation.
    """

    # key setup
    keys = rand.split(key, 5)

    nbasis = PHI_proc.shape[1]

    # nobs = obs_locs.shape[1]
    nobs = obs_locs.shape[0]

    times = jnp.unique(obs_locs[:, 0], size=T)

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
    beta = jnp.array([0.2, 0.2, 0.2])

    # X_obs = jl.map(
    #    lambda arr: jnp.column_stack([jnp.ones(arr.shape[0]), arr]), obs_locs
    # )

    # X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])
    X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

    @jax.jit
    def get_obs(X_obs_1, PHI_obs_1, alpha_1):
        return (
            X_obs_1 @ beta
            + PHI_obs_1 @ alpha_1
            + jnp.sqrt(sigma2_eps) * rand.normal(key, shape=(nobs,))
        )

    obs_vals = (
        X_obs @ beta
        + PHI_obs @ alpha.flatten()
        + jnp.sqrt(sigma2_eps) * rand.normal(key, shape=(nobs,))
    )

    return (process_vals, obs_vals)


def gen_example_idem(
    key: ArrayLike,
    k_spat_inv: bool = True,
    ngrid: ArrayLike = jnp.array([41, 41]),
    nints: ArrayLike = jnp.array([100, 100]),
    nobs: int = 50,
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
    process_basis = place_basis()
    nbasis = process_basis.nbasis

    # int_grid = create_grid(jnp.array([[0, 1], [0, 1]]), nints)

    # Other Coefficients
    sigma2_eta = 0.01**2
    sigma2_eps = 0.01**2
    # Q_eta = jnp.eye(nbasis) / sigma2_eta
    # Q_eps = jnp.eye(nobs * T) / sigma2_eps

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
        alpha0 = jnp.zeros(nbasis).at[jnp.array([64])].set(1)
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
        alpha0 = (
            jnp.zeros(nbasis)
            .at[jnp.array([77, 66, 19, 1, 34, 75, 31, 35, 46, 88])]
            .set(1)
        )

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

    # M = construct_M(kernel, process_basis, int_grid)

    beta = jnp.array([0.2, 0.2, 0.2])

    return IDEM(
        process_basis=process_basis,
        kernel=kernel,
        process_grid=process_grid,
        sigma2_eta=sigma2_eta,
        sigma2_eps=sigma2_eps,
        alpha0=alpha0,
        beta=beta,
    )


def construct_M(kernel: Callable, process_basis: Basis, grid: Grid):
    """Constructs the progression matrix, M, defining how the basis parameters evolve linearly with time. Integration is done by Rieamann sum for now, on the grid provided.

    Parameters
    ----------
      kernel: Arraylike, ArrayLike -> ArrayLike; kernel function defining the progression of the process. The first argument is s, the variable being integrated over, and the second object should be the parameters; an array of shape (3, ) containing the x offset, y offset and scale of the kernel.
      process_basis: Basis; the basis for the process
      grid: Grid; the grid object to be integrated over

    """

    PHI = process_basis.mfun(grid.coords)

    GRAM = (PHI.T @ PHI) * grid.area

    K = outer_op(grid.coords, grid.coords, kernel)

    return solve(GRAM, PHI.T @ K @ PHI) * grid.area**2


if __name__ == "__main__":
    print("IDEM loaded as main. Simulating a simple example.")

    key = jax.random.PRNGKey(1)
    keys = rand.split(key, 2)

    model = gen_example_idem(keys[0], k_spat_inv=False)

    # Simulation
    T = 9
    nobs = 50

    process_data, obs_data = model.simulate(key)

    # plot the object
    plot_st_long(process_data)
    plt.show()
    plot_st_long(obs_data)
    plt.show()

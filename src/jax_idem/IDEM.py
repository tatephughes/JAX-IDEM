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
from utilities import create_grid, place_basis, outer_op, Basis, Grid

ngrids = jnp.array([41, 41])
bounds = jnp.array([[0, 1], [0, 1]])


class IDEM(NamedTuple):
    process_basis: Basis
    K_basis: tuple
    ki: ArrayLike
    obs_grids: ArrayLike
    process_grid: Grid
    sigma2_eta: float
    sigma2_eps: float
    Q_eta: ArrayLike
    Q_eps: ArrayLike
    M: ArrayLike
    alpha0: ArrayLike


def gen_example_idem(
    key: ArrayLike,
    k_spat_inv: bool = True,
    ngrid: ArrayLike = jnp.array([41, 41]),
    nint: ArrayLike = jnp.array([100, 100]),
    nobs: int = 50,
):
    process_grid = create_grid(jnp.array([[0, 1], [0, 1]]), ngrid)
    process_basis = place_basis()

    int_grid = create_grid(jnp.array([[0, 1], [0, 1]]), nints)

    # Other Coefficients
    sigma2_eta = 0.01**2
    sigma2_eps = 0.01**2
    Q_eta = jnp.eye(nbasis) / sigma2_eta
    Q_eps = jnp.eye(nobs * T) / sigma2_eps

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

    M = construct_M(kernel, process_basis, int_grid)
    obs_grids = rand.uniform(keys[3], shape=(T, nobs, 2))

    return IDEM(
        process_basis,
        K_basis,
        k,
        obs_grids,
        process_grid,
        sigma2_eta,
        sigma2_eps,
        Q_eta,
        Q_eps,
        M,
        alpha0,
    )


def construct_M(kernel: Callable, process_basis: Basis, grid: Grid):
    """Constructs the progression matrix, M, defining how the basis parameters evolve linearly with time. Integration is done by Rieamann sum for now, on the grid provided.

    Parameters
    ----------
      kernel: Arraylike, ArrayLike -> ArrayLike; kernel function defining the progression of the process. The first argument is s, the variable being integrated over, and the second object should be the parameters; an array of shape (3, ) containing the x offset, y offset and scale of the kernel.
      process_basis: Basis; the basis for the process
      grid: ArrayLike; the grid to be integrated over
      gridarea: float: the area of a single cell of the grid

    """

    PHI = process_basis.mfun(grid.coords)

    GRAM = (PHI.T @ PHI) * grid.area

    K = outer_op(grid.coords, grid.coords, kernel)

    return solve(GRAM, PHI.T @ K @ PHI) * grid.area**2


@partial(jax.jit, static_argnames=["T"])
def simIDEM(
    key: ArrayLike,
    T: int,
    M: ArrayLike,
    PHI_proc: ArrayLike,
    PHI_obs: ArrayLike,
    obs_grids: ArrayLike,
    alpha0: ArrayLike = jnp.zeros(90).at[jnp.array([64])].set(1),
    process_grid: Grid = create_grid(bounds, ngrids),
    int_grid: Grid = create_grid(bounds, ngrids),
) -> ArrayLike:
    """
    Simulates from a given IDE model.
    Partially implemented, for now this will use a pre-defined model, similar to AZM's package

    Parameters:
      T: Int; the number of time increments to simulate
      nobs: Int; the number of observation points to simulate
      nres: Int; the grid size of the discretised process
      nint: Int; The resolution at which to compute the Riemann integrals
    """

    # key setup
    keys = rand.split(key, 5)

    nbasis = PHI_proc.shape[1]

    nobs = obs_grids.shape[1]

    @jax.jit
    def step(carry, key):
        nextstate = M @ carry + jnp.sqrt(sigma2_eta) * rand.normal(key, shape=(nbasis,))
        return (nextstate, nextstate)

    alpha_keys = rand.split(keys[3], T)

    alpha = jl.scan(step, alpha0, alpha_keys)[1]

    @jax.jit
    def get_process(alpha):
        return PHI_proc @ alpha

    vget_process = jax.vmap(get_process)

    process_vals = vget_process(alpha)

    # X_proc = jnp.column_stack([jnp.ones(s_grid.shape[0]), s_grid])
    beta = jnp.array([0.2, 0.2, 0.2])

    X_obs = jl.map(
        lambda arr: jnp.column_stack([jnp.ones(arr.shape[0]), arr]), obs_grids
    )

    @jax.jit
    def get_obs(X_obs_1, PHI_obs_1, alpha_1):
        return (
            X_obs_1 @ beta
            + PHI_obs_1 @ alpha_1
            + jnp.sqrt(sigma2_eps) * rand.normal(key, shape=(nobs,))
        )

    obs_vals = jax.vmap(get_obs)(X_obs, PHI_obs, alpha)

    return (process_vals, obs_vals)


if __name__ == "__main__":
    print(
        "IDEM loaded as main. Simulating a simple spatially-invariant kernel example."
    )

    key = jax.random.PRNGKey(5)
    keys = rand.split(key, 3)

    T = 9

    nobs = 50
    ngrids = jnp.array([41, 41])
    nints = jnp.array([100, 100])
    process_grid = create_grid(jnp.array([[0, 1], [0, 1]]), ngrids)
    obs_grids = rand.uniform(keys[3], shape=(T, nobs, 2))
    int_grid = create_grid(jnp.array([[0, 1], [0, 1]]), nints)

    process_basis = place_basis()
    nbasis = process_basis.nbasis

    k_spat_inv = 0

    if k_spat_inv == 1:
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

    M = construct_M(kernel, process_basis, int_grid)

    PHI_proc = process_basis.mfun(process_grid.coords)
    PHI_obs = jl.map(process_basis.mfun, obs_grids)

    # Other Coefficients
    sigma2_eta = 0.01**2
    sigma2_eps = 0.01**2
    Q_eta = jnp.eye(nbasis) / sigma2_eta
    Q_eps = jnp.eye(nobs * T) / sigma2_eps

    sim_keys = rand.split(keys[2], 50)

    process_vals_sample, obs_vals_sample = jl.map(
        lambda key: simIDEM(
            key=key,
            T=T,
            M=M,
            PHI_proc=PHI_proc,
            PHI_obs=PHI_obs,
            alpha0=alpha0,
            obs_grids=obs_grids,
            process_grid=process_grid,
            int_grid=int_grid,
        ),
        sim_keys,
    )

    process_vals = process_vals_sample[1]
    obs_vals = obs_vals_sample[1]

    vmin = jnp.min(process_vals)
    vmax = jnp.max(process_vals)

    fig, axes = plt.subplots(3, 3, figsize=(8, 5))

    for i in range(T):
        ax = axes[i // 3, i % 3]
        scatter = ax.scatter(
            process_grid.coords.T[0],
            process_grid.coords.T[1],
            c=process_vals[i],
            cmap="viridis",
            marker="s",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"T = {i+1}")
        ax.set_title(f"T = {i+1}", fontsize=5)  # Set title font size
        ax.tick_params(
            axis="both", which="major", labelsize=4
        )  # Set tick labels font size
        fig.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(3, 3, figsize=(8, 5))

    for i in range(9):
        ax = axes[i // 3, i % 3]
        scatter = ax.scatter(
            obs_grids[i].T[0],
            obs_grids[i].T[1],
            c=obs_vals[i],
            cmap="viridis",
            marker="s",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"T = {i+1}")
        ax.set_title(f"T = {i+1}", fontsize=5)  # Set title font size
        ax.tick_params(
            axis="both", which="major", labelsize=4
        )  # Set tick labels font size
        fig.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.show()

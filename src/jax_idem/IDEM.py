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

# In-Module imports
from utilities import create_grid, place_basis, outer_op


def forward_step(beta, kernel, locations, M, key):
    sigma_eta = 0.1 * jnp.exp(
        -outer_op(locations, locations, lambda s, x: vector_norm(s - x)) / 0.1
    )
    eta = rand.multivariate_normal(key, jnp.zeros(locations.shape[0]), sigma_eta)


def construct_M(kernel, process_basis, grid, gridarea):
    basis_vfunc, eval_basis, nbasis = process_basis

    PHI = eval_basis(grid)

    GRAM = (PHI.T @ PHI) * gridarea

    K = outer_op(grid, grid, kernel)

    return solve(GRAM, PHI.T @ K @ PHI) * gridarea**2


def simIDEM(T, k, K_basis, ngrids=jnp.array([41, 41]), nobs=100, nres=2, nint=200):
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
    key = jax.random.PRNGKey(seed=628)
    keys = rand.split(key, 3)

    # Place 90 basis functions across two resolutions (default values are fine here)
    # the original R code bases the grid 100 uniform points across the unit square,
    # which can also be done here by setting data=rand.uniform(<key>, shape=(100,2))
    process_basis = place_basis(nres=2)
    nbasis = process_basis[2]

    # now, for example, ```eval_basis(<list of points>, params)``` will work

    # Other Coefficients
    beta = jnp.array([0.2, 0.2, 0.2])
    sigma2_eta = 0.01**2
    sigma2_eps = 0.01**2

    Q_eta = jnp.eye(nbasis) / sigma2_eta
    Q_eps = jnp.eye(nobs * T) / sigma2_eps

    ngrids = jnp.array([41, 41])
    bounds = jnp.array([[0, 1], [0, 1]])

    # Process nodes
    s_grid, griddeltas = create_grid(bounds, ngrids)
    gridarea = jnp.prod(griddeltas)

    def kernel(s, r):
        theta = (
            k[0] @ K_basis[0][0](s),
            k[1] @ K_basis[1][0](s),
            jnp.array(
                [
                    k[2] @ K_basis[2][0](s),
                    k[3] @ K_basis[3][0](s),
                ]
            ),
        )

        return theta[0] * jnp.exp(-(jnp.sum((r - s - theta[2]) ** 2)) / theta[1])

    # alpha0 = jnp.zeros(nbasis).at[jnp.array([64])].set(1)
    alpha0 = (
        jnp.zeros(nbasis).at[jnp.array([77, 66, 19, 1, 34, 75, 31, 35, 46, 88])].set(1)
    )

    M = construct_M(kernel, process_basis, s_grid, gridarea)

    def step(carry, key):
        nextstate = M @ carry + jnp.sqrt(sigma2_eta) * rand.normal(key, shape=(nbasis,))
        return (nextstate, nextstate)

    alpha_keys = rand.split(keys[3], T)

    alpha = jl.scan(step, alpha0, alpha_keys)[1]

    def get_process(alpha, s_grid):
        return process_basis[1](s_grid) @ alpha

    vget_process = jax.vmap(get_process, in_axes=(0, None))

    process_vals = vget_process(alpha, s_grid)

    return process_vals


if __name__ == "__main__":
    print(
        "IDEM loaded as main. Simulating a simple spatially-invariant kernel example."
    )

    key = jax.random.PRNGKey(1)
    keys = rand.split(key, 3)

    ngrids = jnp.array([41, 41])
    s_grid, griddeltas = create_grid(jnp.array([[0, 1], [0, 1]]), ngrids)

    k_spat_inv = 1

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
    else:
        K_basis = (
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
            place_basis(nres=1),
            place_basis(nres=1),
        )
        k = (
            jnp.array([200]),
            jnp.array([0.2]),
            0.01 * rand.normal(keys[0], shape=(K_basis[2][2],)),
            0.01 * rand.normal(keys[2], shape=(K_basis[3][2],)),
        )

    process_vals = simIDEM(T=9, k=k, K_basis=K_basis, ngrids=ngrids)

    vmin = jnp.min(process_vals)
    vmax = jnp.max(process_vals)

    fig, axes = plt.subplots(3, 3, figsize=(8, 5))

    for i in range(9):
        ax = axes[i // 3, i % 3]
        scatter = ax.scatter(
            s_grid.T[0],
            s_grid.T[1],
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

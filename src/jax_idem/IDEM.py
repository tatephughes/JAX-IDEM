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
    def psi(s, params):
        squarenorm = jnp.array([jnp.sum((s - params[0:2]) ** 2)])
        return ((2 - squarenorm) ** 2 * jnp.where(squarenorm < params[2], 1, 0))[0]

    # consider a sparse matrix approach here!
    eval_basis = jax.vmap(jax.vmap(psi, in_axes=(None, 0)), in_axes=(0, None))

    PHI = eval_basis(grid, process_basis)

    # griddelta? shouldnt it be the area?
    GRAM = (PHI.T @ PHI) * gridarea

    K = outer_op(grid, grid, kernel)

    return solve(GRAM, PHI.T @ K @ PHI) * gridarea**2


def simIDEM(T, k, K_basis, griddelta=0.01, nobs=100, nres=2, nint=200):
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

    # Bisquare basis functions
    def psi(s, params):
        squarenorm = jnp.array([jnp.sum((s - params[0:2]) ** 2)])
        return ((2 - squarenorm) ** 2 * jnp.where(squarenorm < params[2], 1, 0))[0]

    vec_phi = jax.vmap(psi, in_axes=(None, 0))

    eval_basis = jax.vmap(jax.vmap(psi, in_axes=(None, 0)), in_axes=(0, None))

    # Place 90 basis functions across two resolutions (default values are fine here)
    # the original R code bases the grid 100 uniform points across the unit square,
    # which can also be done here by setting data=rand.uniform(<key>, shape=(100,2))
    process_basis = place_basis()
    nbasis = process_basis.shape[0]

    # now, for example, ```eval_basis(<list of points>, params)``` will work

    # Other Coefficients
    beta = jnp.array([0.2, 0.2, 0.2])
    sigma2_eta = 0.01**2
    sigma2_eps = 0.01**2

    Q_eta = jnp.eye(nbasis) / sigma2_eta
    Q_eps = jnp.eye(nobs * T) / sigma2_eps

    # Process nodes
    s_grid = create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([griddelta, griddelta]))

    def kernel(s, r):
        theta = (
            k[0],
            k[1],
            jnp.array([k[2] @ vec_phi(s, K_basis[2]), k[3] @ vec_phi(s, K_basis[3])]),
        )

        return theta[0] * jnp.exp(-(jnp.sum((r - s - theta[2]) ** 2)) / theta[1])

    vector = jnp.zeros(nbasis)
    alpha0 = vector.at[jnp.array([1, 10, 8, 65, 20, 24])].set(1)

    M = construct_M(kernel, process_basis, s_grid, griddelta**2)

    def step(carry, key):
        nextstate = M @ carry + jnp.sqrt(sigma2_eta) * rand.normal(key, shape=(nbasis,))
        return (nextstate, nextstate)

    alpha_keys = rand.split(keys[3], T)

    alpha = jl.scan(step, alpha0, alpha_keys)[1]

    def get_process(alpha, s_grid):
        return eval_basis(s_grid, process_basis) @ alpha

    vget_process = jax.vmap(get_process, in_axes=(0, None))

    process_vals = vget_process(alpha, s_grid)

    return process_vals


if __name__ == "__main__":
    print(
        "IDEM loaded as main. There will be an example simulation here in future versions."
    )

    key = jax.random.PRNGKey(1)
    keys = rand.split(key, 3)

    griddelta = 0.01
    s_grid = create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([griddelta, griddelta]))

    K_basis = (jnp.array(1), jnp.array(1), place_basis(nres=1), place_basis(nres=1))
    k = (
        jnp.array(200),
        jnp.array(0.002),
        0.01 * rand.normal(keys[0], shape=(K_basis[2].shape[0],)),
        0.01 * rand.normal(keys[2], shape=(K_basis[3].shape[0],)),
    )

    process_vals = simIDEM(T=9, k=k, K_basis=K_basis, griddelta=griddelta)

    fig, axes = plt.subplots(3, 3, figsize=(8, 5))

    for i in range(9):
        ax = axes[i // 3, i % 3]
        scatter = ax.scatter(
            s_grid.T[0], s_grid.T[1], c=process_vals[i], cmap="viridis", marker="s"
        )
        ax.set_title(f"T = {i+1}")
        fig.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.show()

#!/usr/bin/env python3

# JAX imports
import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
from jax.numpy.linalg import vector_norm, solve

# Statistics and data handling imports
#import pandas as pd
#import patsy

import utilities
from utilities import *

def kernel(s, x, thetap):

    """
    Gaussian kernel function

    Parameters:
    s: The first coordinate point, should be a JAX array of shape [d,]
    x: The second coordinate point, should be a JAX array of shape [d,]
    thetap: an array of the kernel parameters, of the form
            [scale, shape, *shifts]
            should be a JAX array of shape [d+2,]
    """

    scale = thetap[0]
    shape = thetap[1]
    shifts = thetap[2:]

    return scale*jnp.exp(-jnp.sum((x - s - shifts)**2)/shape)

def forward_step(beta, kernel, locations, M, key):

    sigma_eta = 0.1 * jnp.exp(-outer_operation(
        locations, locations, lambda s,x: vector_norm(s-x))
                              /0.1)
    eta = rand.multivariate_normal(key, jnp.zeros(locations.shape[0]), sigma_eta)

def construct_M(kernel, process_basis, grid, griddelta):

    def psi(s, params):
        squarenorm = jnp.array([jnp.sum((s-params[0:2])**2)])
        return ((2 - squarenorm)**2 * jnp.where(squarenorm < params[2], 1, 0))[0]

    # consider a sparse matrix approach here!
    eval_basis = jax.vmap(jax.vmap(psi, in_axes=(None, 0)), in_axes=(0, None))
    
    PHI = eval_basis(grid, process_basis)

    GRAM = (PHI.T @ PHI) * griddelta

    K = outer_op(grid, grid, kernel)
    
    return solve(GRAM, PHI.T @ K @ PHI) * griddelta**2

    
    
def simIDEM(T, k, nobs = 100, nres = 2, nint=200):

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
        squarenorm = jnp.array([jnp.sum((s-params[0:2])**2)])
        return ((2 - squarenorm)**2 * jnp.where(squarenorm < params[2], 1, 0))[0]
    vec_phi = jax.vmap(psi, in_axes=(None, 0))

    eval_basis = jax.vmap(jax.vmap(psi, in_axes=(None, 0)), in_axes=(0, None))
    
    # Place 90 basis functions across two resolutions (default values are fine here)
    # the original R code bases the grid 100 uniform points across the unit square,
    # which can also be done here by setting data=rand.uniform(<key>, shape=(100,2))
    process_basis = place_basis()
    r = process_basis.shape[0]

    # now, for example, ```eval_basis(<list of points>, params)``` will work

    # Other Coefficients
    beta = jnp.array([0.2,0.2,0.2])
    sigma2_eta = 0.01**2
    sigma2_eps = 0.01**2

    Q_eta = jnp.eye(r) / sigma2_eta
    Q_eps = jnp.eye(nobs*T) / sigma2_eps
    
    # Process nodes
    griddelta = 0.01
    s_grid = create_grid(jnp.array([[0,1],[0,1]]),
                         jnp.array([griddelta, griddelta]))


    # Basis for spatially variying kernel
    K_basis = (1, 1, place_basis(nres=2), place_basis(nres=2))
    #k = (200, 0.002, 0.1*rand.normal(keys[0], shape=(K_basis[2].shape[0], )),
    #                 0.1*rand.normal(keys[1], shape=(K_basis[3].shape[0], )))

    # experimental values
    #k = (jnp.array(200),
    #     jnp.array(0.2),
    #     0.01*rand.normal(keys[0], shape=(K_basis[2].shape[0], )),
    #     0.01*rand.normal(keys[1], shape=(K_basis[3].shape[0], )))
    
    def kernel(s,r):
        theta = (k[0], k[1],
                 jnp.array([k[2] @ vec_phi(s, K_basis[2]),
                            k[3] @ vec_phi(s, K_basis[3])]))
    
        return theta[0] * jnp.exp(-(jnp.sum((r-s-theta[2])**2)) / theta[1])

    vector = jnp.zeros(r)
    alpha0 = vector.at[jnp.array([1,10,8, 65, 20, 24])].set(1)

    M = construct_M(kernel, process_basis, s_grid, griddelta)

    def step(carry, key):
        nextstate = M @ carry + jnp.sqrt(sigma2_eta) * rand.normal(key, shape = (r,))
        return(nextstate, nextstate)

    alpha_keys = rand.split(keys[3], T)

    alpha = jl.scan(step, alpha0, alpha_keys)[1]

    def get_process(alpha, s_grid):
        return eval_basis(s_grid, process_basis) @ alpha
    vget_process = jax.vmap(get_process, in_axes=(0,None))

    process_vals = vget_process(alpha, s_grid)
    
    return process_vals

#!/usr/bin/env python3

# JAX imports
import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
from jax.numpy.linalg import vector_norm

# Statistics and data handling imports
#import pandas as pd
#import patsy

import utilities
from utilities import create_grid, outer_op

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

def eval_basis(basis_func, locations, knots, alpha):

    """
    Evaluates the basis expansion defined by the coefficients alpha at vector of locations.

    Parameters:
    basis_func: Double, Double -> Double;  the basis functions, taking a location and a
                                           knot to the 
    locations: Array[Array[Double]]; Array of coordinates on which to evaluate the function
    knots: The centres of the basis functions

    Returns: Array[Double]
    The array of values of the expanded function at each coordinate point.
    """

    PHIalpha = outer_op(stations, knots, basis_func) @ alpha

def simIDEM(T, nobs = 100, nres = 40, nint=200):

    """
    Simulates from a given IDE model.
    Partially implemented, for now this will use a pre-defined model, similar to AZM's package

    Parameters:
    T: Int; the number of time increments to simulate
    nobs: Int; the number of observation points to simulate
    nres: Int; the grid size of the discretised process
    nint: Int; The resolution at which to compute the Riemann integrals
    """

    # Bisquare basis functions
    def psi(s, knot, w=1):
        squarenorm = jnp.array([jnp.sum((s-knot)**2)])
        return ((2 - squarenorm)**2 * jnp.where(squarenorm < w, 1, 0))[0]
    
    return "WOW"

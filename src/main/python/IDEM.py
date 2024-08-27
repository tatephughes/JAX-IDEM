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

def forward_step(Y, kernel, locations, key):

    sigma_eta = 0.1 * jnp.exp(-outer_operation(
        locations, locations, lambda s,x: vector_norm(s-x))
                              /0.1)
    eta = rand.multivariate_normal(key, jnp.zeros(locations.shape[0]), sigma_eta)

    

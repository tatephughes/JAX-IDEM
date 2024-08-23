#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
from jax.numpy.linalg import solve

def create_grid(bounds,
                deltas,
                ):

    """
    Creates an n-dimensional grid based on the given bounds and deltas.

    Parameters:
    bounds: JAX array of shape 2xdimension, the bounds for each dimension
    deltas: JAX array of the increment for each dimension.

    Returns:
    JAX array of coordinate arrays for the grid.
    """
    
    dimension = jnp.size(bounds, axis=0)

    def gen_range(i):
        return jnp.arange(bounds[i][0], bounds[i][1], deltas[i])

    # This is a traditional loop. I imagine there must be a better,
    # more parallelisable way to do this, but arange is being a bit
    # funky. since it is a reasonably simple operation, it should be
    # fine for now.
    axis_linspaces = [gen_range(i) for i in range(dimension)]

    grid = jnp.stack(jnp.meshgrid(*axis_linspaces, indexing='ij'),
                     axis=-1).reshape(-1,dimension)
    
    return grid

#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
from jax.numpy.linalg import solve

def create_grid(bounds,
                deltas):

    """
    Creates an n-dimensional grid based on the given bounds and deltas.

    Parameters:
    bounds: Array[[Double, Double]]; The bounds for each dimension
    deltas: Array[Double]; The increment for each dimension.

    Returns: Array[Array[Double]]
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

def outer_op(a, b, op = lambda x,y: x*y):

    """
    Computes the outer operation of two vectors, a generalisation of the outer product.

    Parameters:
    vec1: Array[A]; array of the first vector
    vec2: Array[B]; array of the second vector
    operation: A, B -> C; A function acting on an element of vec1 and an element of vec2.
               By default, this is the outer product.

    Returns: Array[C]
    The matrix of the result of applying operation to every pair of elements from the
    two vectors.
    """

    def ascan(aval, bvec):
        return jl.map(lambda bval: op(aval,bval), bvec)
    
    return jl.map(lambda aval: ascan(aval, b), a)

    
  

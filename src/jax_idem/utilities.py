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

    Returns: Array[Array[C]]
    The matrix of the result of applying operation to every pair of elements from the
    two vectors.
    """

    def ascan(aval, bvec):
        return jl.map(lambda bval: op(aval,bval), bvec)
    
    return jl.map(lambda aval: ascan(aval, b), a)

def place_basis(data = jnp.array([[0,0],[1,1]]),
                                         nres = 2,
                                         aperture = 1.25,
                                         min_knot_num = 3
                                        ):

    '''
    Distributes knots and scales for basis functions over a number of resolutions,
    similar to auto_basis from the R package FRK.
    This function must be run outside of a jit loop, since it involves varying the
    length of arrays.

    NOTE: This function does not directly give basis functions, like in FRK for
    example. Instead, it returns an array of parameters for a basis function,
    in the form [x_coord, y_coord, scale].
    
    
    '''

    xmin = jnp.min(data[:,0])
    xmax = jnp.max(data[:,0])
    ymin = jnp.min(data[:,1])
    ymax = jnp.max(data[:,1])
    
    asp_ratio = (ymax-ymin)/(xmax-xmin)
    
    if asp_ratio < 1:
        ny = min_knot_num
        nx = jnp.round(ny/asp_ratio).astype(int)
    else:
        nx = min_knot_num
        ny = jnp.round(asp_ratio * nx).astype(int)     
        
    def basis_at_res(res):

        x_range = jnp.linspace(xmin, xmax, nx*3**res)
        y_range = jnp.linspace(ymin, ymax, ny*3**res)

        knots = jnp.stack(jnp.meshgrid(x_range, y_range, indexing='ij'),
                 axis=-1).reshape(-1,2)

        def pairwise_distances(points):
            diff = points[:, None, :] - points[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
            return dist

        distances = pairwise_distances(knots)

        # the diagonal here is to ensure the 0s on the diagonal are ignored
        min_distance = jnp.min(distances + jnp.eye(knots.shape[0]) * jnp.max(distances))

        aperture_res = min_distance * aperture

        return  jnp.hstack([knots, jnp.full((knots.shape[0], 1), aperture_res)])

    # You cant abstract over array lengths, this must(?) be done like this
    return jnp.vstack([basis_at_res(res) for res in range(nres)])

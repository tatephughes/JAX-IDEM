#!/usr/bin/env python3

import jax
import jax.numpy as jnp
# import jax.lax as jl
# import jax.random as rand
# from jax.numpy.linalg import solve


def create_grid(bounds, ngrids):
    """
    Creates an n-dimensional grid based on the given bounds and deltas.

    Parameters:
    bounds: Array[[Double, Double]]; The bounds for each dimension
    ngrid: Array[Int]; The number of columns/rows/hyper-column in each dimension

    Returns: (Array[Array[Double]], Array[Double])
    Tuple of a JAX array of coordinate arrays for the grid and the length between grid points in each dimension.
    """

    dimension = jnp.size(bounds, axis=0)

    def gen_range(i):
        return jnp.linspace(bounds[i][0], bounds[i][1], ngrids[i])

    # This is a traditional loop. I imagine there must be a better,
    # more parallelisable way to do this, but arange is being a bit
    # funky. since it is a reasonably simple operation, it should be
    # fine for now.
    axis_linspaces = [gen_range(i) for i in range(dimension)]

    grid = jnp.stack(jnp.meshgrid(*axis_linspaces, indexing="ij"), axis=-1).reshape(
        -1, dimension
    )

    # I'm actually reasonably certain that this delta is wrong. it should be divided by ```ngrids-1```, but this is how it is in the IDE R package. With fine integration grids, this shouldn't matter much, but if it is a problem I will put up a github issue on IDE.
    return (
        grid.at[:, [0, 1]].set(grid[:, [1, 0]]),
        (bounds[:, 1] - bounds[:, 0]) / ngrids,
    )


def outer_op(a, b, op=lambda x, y: x * y):
    """
    Computes the outer operation of two vectors, a generalisation of the outer product.

    Parameters:
    vec1: ArrayLike[A]; array of the first vector
    vec2: ArrayLike[B]; array of the second vector
    operation: A, B -> C; A function acting on an element of vec1 and an element of vec2.
               By default, this is the outer product.

    Returns: Array[Array[C]]
    The matrix of the result of applying operation to every pair of elements from the
    two vectors.
    """

    vec_op = jax.vmap(jax.vmap(op, in_axes=(None, 0)), in_axes=(0, None))

    return vec_op(a, b)


def bisquare(s, params):
    squarenorm = jnp.array([jnp.sum((s - params[0:2]) ** 2)])
    return (
        (1 - squarenorm / (params[2] ** 2)) ** 2
        * jnp.where(squarenorm < params[2] ** 2, 1, 0)
    )[0]


def place_basis(
    data=jnp.array([[0, 0], [1, 1]]),
    nres=2,
    aperture=1.25,
    min_knot_num=3,
    basis_fun=bisquare,
):
    """
            Distributes knots (centroids) and scales for basis functions over a number of resolutions,
            similar to auto_basis from the R package FRK.
            This function must be run outside of a jit loop, since it involves varying the
            length of arrays.

    Parameters:
      data: Arraylike[ArrayLike[Double]]; array of 2D points defining the space on which to put the basis functions
      nres: Int; The number of resolutions at which to place basis functions
      aperture: Double; Scaling factor for the scale parameter (scale parameter will be w=aperture * d, where d is the minimum distance between any two of the knots)
      min_knot_num: Int; The number of basis functions to place in each dimension at the coursest resolution
      basis_fun: (ArrayLike[Double], ArrayLike[Double]) -> Double; the basis functions being used. The basis function's second argument must be an array with three doubles; the first coordinate for the centre, the second coordinate for the centre, and the scale/aperture of the function.

    Returns:
      A tuple of two functions and an integer, the first evaluating the basis functions at a point, and the second evaluating the basis functions on an array of points.
    """

    xmin = jnp.min(data[:, 0])
    xmax = jnp.max(data[:, 0])
    ymin = jnp.min(data[:, 1])
    ymax = jnp.max(data[:, 1])

    asp_ratio = (ymax - ymin) / (xmax - xmin)

    if asp_ratio < 1:
        ny = min_knot_num
        nx = jnp.round(ny / asp_ratio).astype(int)
    else:
        nx = min_knot_num
        ny = jnp.round(asp_ratio * nx).astype(int)

    def basis_at_res(res):
        bounds = jnp.array([[xmin, xmax], [ymin, ymax]])
        ngrids = jnp.array([nx, ny]) * 3**res

        knots, deltas = create_grid(bounds, ngrids)
        w = jnp.min(deltas) * aperture * 1.5

        return jnp.hstack([knots, jnp.full((knots.shape[0], 1), w)])

    params = jnp.vstack([basis_at_res(res) for res in range(nres)])
    nbasis = params.shape[0]

    def basis_vfun(s):
        return jax.vmap(basis_fun, in_axes=(None, 0))(s, params)

    def eval_basis(s_array):
        return jax.vmap(jax.vmap(basis_fun, in_axes=(None, 0)), in_axes=(0, None))(
            s_array, params
        )

    return (basis_vfun, eval_basis, nbasis)

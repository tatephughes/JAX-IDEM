#!/usr/bin/env python3

import jax
import jax.numpy as jnp
# from typing import Callable
# import jax.lax as jl
# import jax.random as rand
# from jax.numpy.linalg import solve

from jax.typing import ArrayLike

from typing import Callable, NamedTuple  # , Union
from jax import Array
import matplotlib.pyplot as plt


class Grid(NamedTuple):
    coords: ArrayLike
    deltas: ArrayLike
    ngrids: ArrayLike
    ngrid: int
    dim: int
    area: float


class Basis(NamedTuple):
    vfun: Callable
    mfun: Callable
    params: ArrayLike
    nbasis: int


def create_grid(bounds: ArrayLike, ngrids: ArrayLike) -> (ArrayLike, float):
    """Creates an n-dimensional grid based on the given bounds and deltas.

    Parameters
    ----------
      bounds: Array[[Double, Double]]; The bounds for each dimension
      ngrid: Array[Int]; The number of columns/rows/hyper-column in each dimension

    Returns: Grid
    Grid Object (NamedTuple) containing the coordinates, deltas, grid numbers, areas, etc. See Grid.
    """

    dimension = jnp.size(bounds, axis=0)

    def gen_range(i):
        return jnp.linspace(bounds[i][0], bounds[i][1], ngrids[i])

    # This is a traditional loop. I imagine there must be a better,
    # more jittable way to do this, but linspace is being a bit
    # funky. This is because the 'num' argument must be a concrete value
    # rather than a traced one.since it is a reasonably simple operation,
    # it should be fine for now, and it won't run inside of loops.
    axis_linspaces = [gen_range(i) for i in range(dimension)]

    grid = jnp.stack(jnp.meshgrid(*axis_linspaces, indexing="ij"), axis=-1).reshape(
        -1, dimension
    )

    deltas = (bounds[:, 1] - bounds[:, 0]) / (ngrids - 1)

    return Grid(
        coords=grid.at[:, [0, 1]].set(grid[:, [1, 0]]),
        deltas=deltas,
        ngrids=ngrids,
        dim=dimension,
        ngrid=jnp.prod(ngrids),
        area=jnp.prod(deltas),
    )


def outer_op(a: ArrayLike, b: ArrayLike, op: Callable = lambda x, y: x * y) -> Array:
    """Computes the outer operation of two vectors, a generalisation of the outer product.

    Parameters
    ----------
    a: ArrayLike[A]; array of the first vector
    b: ArrayLike[B]; array of the second vector
    op: A, B -> C; A function acting on an element of vec1 and an element of vec2.
               By default, this is the outer product.

    Returns: Array[Array[C]]
    ----------
    The matrix of the result of applying operation to every pair of elements from the two vectors.
    """

    if not isinstance(a, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {a}")
    if not isinstance(b, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {b}")

    vec_op = jax.vmap(jax.vmap(op, in_axes=(None, 0)), in_axes=(0, None))

    return vec_op(a, b)


@jax.jit
def bisquare(s: ArrayLike, params: ArrayLike) -> ArrayLike:
    squarenorm = jnp.array([jnp.sum((s - params[0:2]) ** 2)])
    w2 = params[2] ** 2
    return (jnp.where(squarenorm < w2, (1 - squarenorm / w2) ** 2, 0))[0]


def place_basis(
    data=jnp.array([[0, 0], [1, 1]]),
    nres=2,
    aperture=1.25,
    min_knot_num=3,
    basis_fun=bisquare,
):
    """Distributes knots (centroids) and scales for basis functions over a number of resolutions,similar to auto_basis from the R package FRK.  This function must be run outside of a jit loop, since it involves varying the length of arrays.

    Parameters
    ----------
      data: Arraylike[ArrayLike[Double]]; array of 2D points defining the space on which to put the basis functions
      nres: Int; The number of resolutions at which to place basis functions
      aperture: Double; Scaling factor for the scale parameter (scale parameter will be w=aperture * d, where d is the minimum distance between any two of the knots)
      min_knot_num: Int; The number of basis functions to place in each dimension at the coursest resolution
      basis_fun: (ArrayLike[Double], ArrayLike[Double]) -> Double; the basis functions being used. The basis function's second argument must be an array with three doubles; the first coordinate for the centre, the second coordinate for the centre, and the scale/aperture of the function.

    Returns
    ----------
    A Basis object (NamedTuple) with the vector and matrix functions, and the parameters associated to the basis functions.
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

        grid = create_grid(bounds, ngrids)
        w = jnp.min(grid.deltas) * aperture * 1.5

        return jnp.hstack([grid.coords, jnp.full((grid.ngrid, 1), w)])

    params = jnp.vstack([basis_at_res(res) for res in range(nres)])
    nbasis = params.shape[0]

    @jax.jit
    def basis_vfun(s):
        return jax.vmap(basis_fun, in_axes=(None, 0))(s, params)

    @jax.jit
    def eval_basis(s_array):
        return jax.vmap(jax.vmap(basis_fun, in_axes=(None, 0)), in_axes=(0, None))(
            s_array, params
        )

    return Basis(basis_vfun, eval_basis, params, nbasis)
    # return {"vfun": basis_vfun, "mfun": eval_basis, "r": nbasis, "pars": params}


class ST_Data_Wide(NamedTuple):
    x: ArrayLike  # x coordinates
    y: ArrayLike  # y coordinates
    z: ArrayLike  # matrix of the process, each row corresponds to a spatial location and each column is a time


class ST_Data_Long(NamedTuple):
    """This could be a full class with plotting methods!"""

    x: ArrayLike
    y: ArrayLike
    t: ArrayLike
    z: ArrayLike


def ST_tolong(data: ST_Data_Wide):
    return "not implemented"


def ST_towide(data: ST_Data_Long):
    return "not implemented"


def plot_st_long(data: ST_Data_Long):
    # in the future, you should be able to directly pass in figsize and nrows/cols.

    # column_stack doesn't care about named tuples
    data_array = jnp.column_stack(data)

    T = int(jnp.max(data.t) - jnp.min(data.t)) + 1

    vmin = jnp.min(data.z)
    vmax = jnp.max(data.z)

    nrows = int(jnp.ceil(T / 3))

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(nrows, 3, figsize=(6, nrows * 1.5))
    axes = axes.flatten()

    # Loop through each time point and create a scatter plot
    for t in range(T):
        # fairly sure this should use jnp.where or similar
        time_data = data_array[data_array[:, 2] == t]
        x = time_data[:, 0]
        y = time_data[:, 1]
        values = time_data[:, 3]

        scatter = axes[t].scatter(
            x,
            y,
            c=values,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axes[t].set_title(f"Time = {t}", fontsize=11)
        # axes[t].set_xlabel("x", fontsize=9)
        # axes[t].set_ylabel("y", fontsize=9)
        axes[t].tick_params(
            axis="both", which="major", labelsize=5
        )  # Set tick labels font size

        # Add color bar
        fig.colorbar(scatter, ax=axes[t])

    plt.tight_layout()


def plot_st_wide(data: ST_Data_Wide):
    print("WARNING plot_st_wide is not fully implemented")
    vmin = jnp.min(data.z)
    vmax = jnp.max(data.z)

    T = data.z.shape[0]

    fig, axes = plt.subplots(3, int(jnp.ceil(T / 3)), figsize=(8, 5))

    for i in range(T):
        ax = axes[i // 3, i % (int(jnp.ceil(T / 3)))]
        scatter = ax.scatter(
            data.x,
            data.y,
            c=data.z[i],
            cmap="viridis",
            marker="s",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"T = {i+1}")
        ax.set_title(f"T = {i+1}", fontsize=11)  # Set title font size
        ax.tick_params(
            axis="both", which="major", labelsize=9
        )  # Set tick labels font size
        fig.colorbar(scatter, ax=ax)

    plt.tight_layout()

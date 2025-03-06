#!/usr/bin/env python3

# JAX imports
import jax
import jax.numpy as jnp
import jax.lax as jl
# Typing imports
from jax.typing import ArrayLike
from typing import Callable, NamedTuple  # , Union

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from PIL import Image
import io

class Grid(NamedTuple):
    """
    A simple grid class to store (currently exclusively regular) grids, along
    with some key quantities such as the lenth between grid points, the
    number of grid points and the area/volume of each grid square/cube.
    Supports arbitrarily high dimension.
    Ideally, in the future, this will support non-regular grids with any
    necessary quantities to do, for example, integration over the points on the
    grid.
    """

    coords: ArrayLike
    deltas: ArrayLike
    ngrids: ArrayLike
    ngrid: int
    dim: int
    area: float


class Basis(NamedTuple):
    """
    A simple class for spatial basis expansions.

    Attributes
    ----------
    vfun: ArrayLike (ndim,) -> ArrayLike (nbasis,)
        Applying to a single spatial location, evaluates all the basis functions
        on the single location and returns the result as a vector.
    mfun: ArrayLike (ndim, n) -> ArrayLike (nbasis, n)
        Applying to a array of spatial points, evaluates all the basis functions
        on each point and returns the results in a matrix.
    params: ArrayLike(nparams, nbasis)
        The parameters defining each basis function. For example, for bisquare
        basis functions, the parameters are the locations of the centers of each
        function, and the function's scale.
    nbasis: int
        The number of basis functions in the expansion.
    """

    vfun: Callable
    mfun: Callable
    params: ArrayLike
    nbasis: int


def create_grid(bounds: ArrayLike, ngrids: ArrayLike) -> Grid:
    """
    Creates an n-dimensional grid based on the given bounds and deltas.

    Parameters
    ----------
    bounds: ArrayLike (2, n)
        The bounds for each dimension
    ngrids: ArrayLike (n, )
        The number of columns/rows/hyper-column in each dimension

    Returns
    ----------
    Grid Object (NamedTuple) containing the coordinates, deltas, grid
    numbers, areas, etc. See the Grid class.
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


def outer_op(
    a: ArrayLike, b: ArrayLike, op: Callable = lambda x, y: x * y
) -> ArrayLike:
    """
    Computes the outer operation of two vectors, a generalisation of the outer
    product.

    Parameters
    ----------
    a: ArrayLike[A] (n, )
        Array of the first vector
    b: ArrayLike[B] (m, )
        Array of the second vector
    op: A, B -> C
        A jit-function acting on an element of vec1 and an element of vec2.
        By default, this is the outer product.

    Returns
    ----------
    ArrayLike[C] (n, m):
        The matrix of the result of applying operation to every pair of elements
        from the two vectors.
    """

    if not isinstance(a, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {a}")
    if not isinstance(b, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {b}")

    # NOTE: this likely goes against the principles of JIT.
    # Although it feels logical in my head, the operation being used would
    # be recompiled on each use of this function; it would be much better to
    # directly vectorise the function using a line akin to the one below.

    vec_op = jax.vmap(jax.vmap(op, in_axes=(None, 0)), in_axes=(0, None))

    return vec_op(a, b)


@jax.jit
def bisquare(s: ArrayLike, params: ArrayLike) -> ArrayLike:
    """Generic bisquare function"""
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
    """
    Distributes knots (centroids) and scales for basis functions over a
    number of resolutions,similar to auto_basis from the R package FRK.
    This function must be run outside of a jit loop, since it involves
    varying the length of arrays.

    Parameters
    ----------
    data: Arraylike (ndim, npoints)
        Array of 2D points defining the space on which to put the basis functions
    nres: Int
        The number of resolutions at which to place basis functions
    aperture: Double
        Scaling factor for the scale parameter (scale parameter will be
        w=aperture * d, where d is the minimum distance between any two of the
        knots)
    min_knot_num: Int
        The number of basis functions to place in each dimension at the coursest
        resolution
    basis_fun: ArrayLike (ndim,), ArrayLike (nparams) -> Double
        The basis functions being used. The basis function's second argument must
        be an array with three doubles; the first coordinate for the centre, the
        second coordinate for the centre, and the scale/aperture of the function.
    Returns
    ----------
    A Basis object (NamedTuple) with the vector and matrix functions, and the
    parameters associated to the basis functions.
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

def random_basis(
        key,
        knot_num=3,
        data=jnp.array([[0, 0], [1, 1]]),
        aperture=1.25,
        basis_fun=bisquare,
):
    """
    Randomly distributes knots (centroids) and scales for basis functions.
    This function must be run outside of a jit loop, since it involves
    varying the length of arrays.

    Parameters
    ----------
    key: ArrayLike
        PRNG key
    knot_num: Int
        The number of knots at which to place basis functions
    data: Arraylike (ndim, npoints)
        Array of 2D points defining the space on which to put the basis functions
    aperture: Double
        Scaling factor for the scale parameter (scale parameter will be
        w=aperture * d, where d is the minimum distance between any two of the
        knots)
    basis_fun: ArrayLike (ndim,), ArrayLike (nparams) -> Double
        The basis functions being used. The basis function's second argument must
        be an array with three doubles; the first coordinate for the centre, the
        second coordinate for the centre, and the scale/aperture of the function.
    Returns
    ----------
    A Basis object (NamedTuple) with the vector and matrix functions, and the
    parameters associated to the basis functions.
    """

    xmin = jnp.min(data[:, 0])
    xmax = jnp.max(data[:, 0])
    ymin = jnp.min(data[:, 1])
    ymax = jnp.max(data[:, 1])

    keys = jax.random.split(key, 2)

    w = ((xmax - xmin) * (ymax - ymin)) / knot_num
    
    params=jnp.hstack([jax.random.uniform(keys[0], shape=(knot_num,1), minval=xmin, maxval=xmax),
                       jax.random.uniform(keys[1], shape=(knot_num,1), minval=ymin, maxval=ymax),
                       jnp.full((knot_num, 1), w)])
    
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


def place_fourier_basis(data=jnp.array([[0, 0], [1, 1]]), N: int = 20):
    """Not currently fully implemented."""

    # Note that the (N**2/2 + N/2)th coefficient must be real

    if N % 2 != 0:
        raise Exception("Only even N for convenience")

    N = float(N)

    @jax.jit
    def phi(k, s):
        return jnp.sqrt(1 / (2 * N)) * jnp.exp(2 * jnp.pi * 1.0j * (jnp.dot(k, s)))

    x, y = jnp.meshgrid(jnp.arange(N + 1) - N / 2, jnp.arange(N + 1) - N / 2)
    pairs = jnp.stack([x.flatten(), y.flatten()], axis=-1)

    @jax.jit
    def basis_vfun(s):
        return jax.vmap(phi, in_axes=(0, None))(pairs, s)

    @jax.jit
    def eval_basis(s_array):
        return jax.vmap(jax.vmap(phi, in_axes=(0, None)), in_axes=(None, 0))(
            pairs,
            s_array,
        )

    return Basis(basis_vfun, eval_basis, pairs, int(N**2))


def plot_kernel(kernel, output_file="kernel.png"):
    """Will be replaced with the show_plt method in the kernel class."""

    grid = create_grid(jnp.array([[0, 1], [0, 1]]), jnp.array([10, 10])).coords

    def offset(s):
        return -jnp.array(
            [
                kernel.params[2] @ kernel.basis[2].vfun(s),
                kernel.params[3] @ kernel.basis[3].vfun(s),
            ]
        )

    vecoffset = jax.vmap(offset)

    offsets = vecoffset(grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.quiver(grid[:, 0], grid[:, 1], offsets[:, 0], offsets[:, 1])
    # ax.quiverkey(q, X=0.3, Y=1.1, U=10)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.title("Kernel Direction")

    plt.savefig(output_file)
    plt.close()


class st_data:
    """
    For storing spatio-temporal data and appropriate methods for plotting such
    data, and converting between long and wide formats.
    """

    def __init__(self, x: ArrayLike, y: ArrayLike, t: ArrayLike, z: ArrayLike):
        self.x = x
        self.y = y
        self.t = t
        self.z = z

        self.times = jnp.unique(t)
        self.T = len(self.times)

    def as_wide(self):
        """
        Gives the data in wide format. Any missing data will be represented in
        the returned matrix as NaN.

        Returns
        ----------
        A dictionary containing the x coordinates and y coordinates as JAX
        arrays, and a matrix corresponding to the value of the process at
        each time point (columns) and spatial point (rows).
        """
        data_array = jnp.column_stack((self.x, self.y, self.t, self.z))

        times = self.times
        xycoords = jnp.unique(data_array[:, 0:2], axis=0)
        nlocs = data_array.shape[0]

        @jax.jit
        def extract(array):  # (from a generative model, dont trust!)
            array_no_nan = jax.numpy.nan_to_num(array, nan=0.0)
            float_value = jnp.sum(array_no_nan)
            return float_value

        @jax.jit
        def getval(xy, t):
            xyt = jnp.hstack((xy, t))
            mask = jnp.all(data_array[:, 0:3] == xyt, axis=1)
            masked = jnp.where(mask, data_array[:, 3], jnp.tile(jnp.nan, nlocs))
            return jl.cond(
                jnp.all(jnp.isnan(masked)),
                lambda x: jnp.nan,
                lambda x: extract(masked),
                0,
            )

        return {
            "x": xycoords[:, 0],
            "y": xycoords[:, 1],
            "z": outer_op(xycoords, times, getval),
        }

    def show_plot(self):
        nrows = int(jnp.ceil(self.T / 3))

        # Create a figure and axes for the subplots
        with plt.style.context("seaborn-v0_8-dark-palette"):
            fig, axes = plt.subplots(nrows, 3, figsize=(6, nrows * 1.5))
            axes = axes.flatten()

            data_array = jnp.column_stack([self.x, self.y, self.t, self.z])

            vmin = jnp.min(self.z)
            vmax = jnp.max(self.z)

            # Loop through each time point and create a scatter plot
            for i, time in enumerate(self.times):
                # fairly sure this should use jnp.where or similar
                time_data = data_array[data_array[:, 2] == time]
                x = time_data[:, 0]
                y = time_data[:, 1]
                values = time_data[:, 3]

                scatter = axes[i].scatter(
                    x,
                    y,
                    c=values,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[i].set_title(f"Time = {time}", fontsize=11)
                # axes[t].set_xlabel("x", fontsize=9)
                # axes[t].set_ylabel("y", fontsize=9)
                axes[i].tick_params(
                    axis="both", which="major", labelsize=5
                )  # Set tick labels font size

                # Add color bar
                fig.colorbar(scatter, ax=axes[i])

            fig.tight_layout()
            fig.show()

    def save_plot(self, filename, width=6, height=1.5, dpi=300):
        nrows = int(jnp.ceil(self.T / 3))

        # Create a figure and axes for the subplots
        with plt.style.context("seaborn-v0_8-dark-palette"):
            fig, axes = plt.subplots(nrows, 3, figsize=(width, nrows * height))
            axes = axes.flatten()

            data_array = jnp.column_stack([self.x, self.y, self.t, self.z])

            vmin = jnp.min(self.z)
            vmax = jnp.max(self.z)

            # Loop through each time point and create a scatter plot
            for i, time in enumerate(self.times):
                # fairly sure this should use jnp.where or similar
                time_data = data_array[data_array[:, 2] == time]
                x = time_data[:, 0]
                y = time_data[:, 1]
                values = time_data[:, 3]

                scatter = axes[i].scatter(
                    x,
                    y,
                    c=values,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[i].set_title(f"Time = {time}", fontsize=11)
                # axes[t].set_xlabel("x", fontsize=9)
                # axes[t].set_ylabel("y", fontsize=9)
                axes[i].tick_params(
                    axis="both", which="major", labelsize=5
                )  # Set tick labels font size

                # Add color bar
                fig.colorbar(scatter, ax=axes[i])

            fig.tight_layout()
            fig.savefig(filename, dpi=dpi)

    def save_gif(self):
        """UNIMPLEMENTED"""
        return None


def gif_st_grid(
    data: st_data,
    output_file="spatio_temporal.gif",
    interval=100,
    width=5,
    height=4,
    dpi=300,
):
    data_array = jnp.column_stack([data.x, data.y, data.t, data.z])
    vmin = jnp.min(data.z)
    vmax = jnp.max(data.z)

    frames = []

    grid = int(jnp.sqrt(data_array[data_array[:, 2] == jnp.min(data.times)].shape[0]))

    for t in data.times:
        time_data = data_array[data_array[:, 2] == t]
        values = time_data[:, 3]
        valmat = jnp.flipud(values.reshape(grid, grid))

        plt.figure(figsize=(width, height))

        sns.heatmap(
            valmat,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
        )

        buf = io.BytesIO()

        plt.title(f"Time: {t}")
        # plt.set_xlabel(data.x)
        # plt.set_ylabel(data.y)

        plt.savefig(buf, format="png", dpi=dpi)
        plt.close()

        frames.append(Image.open(buf))

    frames[0].save(
        output_file, save_all=True, append_images=frames[1:], duration=interval, loop=0
    )


def gif_st_pts(
    data: st_data,
    output_file="spatio_temporal.gif",
    interval=100,
    width=5,
    height=4,
    dpi=300,
):
    data_array = jnp.column_stack([data.x, data.y, data.t, data.z])
    vmin = jnp.min(data.z)
    vmax = jnp.max(data.z)

    frames = []

    T = int(jnp.max(data.t) - jnp.min(data.t)) + 1

    for t in range(T):
        time_data = data_array[data_array[:, 2] == t]
        x = time_data[:, 0]
        y = time_data[:, 1]
        values = time_data[:, 3]

        fig, ax = plt.subplots(figsize=(width, height))

        # plt.scatter(x, y, c=values, vmin=vmin, vmax=vmax, cmap="viridis")

        sns.scatterplot(
            x=x,
            y=y,
            hue=values,
            c=values,
            size=values,
            sizes=(20, 200),
            norm=Normalize(vmin=vmin, vmax=vmax),
            legend=False,
            ax=ax,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()

        plt.title(f"Time: {t}")
        # plt.set_xlabel(data.x)
        # plt.set_ylabel(data.y)

        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)

        plt.savefig(buf, format="png", dpi=dpi)
        plt.close()

        frames.append(Image.open(buf))

    frames[0].save(
        output_file, save_all=True, append_images=frames[1:], duration=interval, loop=0
    )

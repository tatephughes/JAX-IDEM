#!/usr/bin/env python3

# JAX imports
import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand

# Typing imports
from jaxtyping import ArrayLike, PyTree, Array, Float, Int
from typing import Callable, NamedTuple, Union, List

from functools import partial
import warnings

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import seaborn as sns
from PIL import Image
import io

# nice loading bars
from tqdm.auto import tqdm
import time

# Panda (maybe should be made optional!)
import pandas as pd
from pandas.api.types import is_string_dtype, is_float_dtype, is_integer_dtype


class Grid(NamedTuple):
    """
    A simple grid class to store (currently exclusively regular) grids, along
    with some key quantities such as the lenth between grid points, the
    number of grid points and the area/volume of each grid square/cube.
    Supports arbitrarily high dimension.

    Attributes
    ----------
    coords : Float[Array, "n dim"]
        The coordinates of all grid points, stored as an array of shape (n, dim).
    deltas : Float[Array, "dim"]
        The spacing between grid points along each dimension.
    ngrids : Int[Array, "dim"]
        The number of grid points along each axis.
    ngrid : int
        The total number of grid points (i.e., n = product of ngrids).
    dim : int
        The spatial dimension of the grid.
    area : float
        The area (in 2D) or volume (in 3D) of each grid cell.

    Notes
    -----
    Symbolic dimensions:

        - `dim` : the space dimension
        - `n` : the number of points on the grid

    Ideally, in the future, this will support non-regular grids with any
    necessary quantities to do, for example, integration over the points on the
    grid.
    """

    coords: Float[Array, "n dim"]
    deltas: Float[Array, "dim"]
    ngrids: Int[Array, "dim"]
    ngrid: int
    dim: int
    area: float


def create_grid(bounds: Float[Array, "d 2"], ngrids: Float[Array, "d"]) -> Grid:
    """
    Creates an n-dimensional grid based on the given bounds and deltas.

    Parameters
    ----------
    bounds: Float[Array, "d 2"]
        The bounds for each dimension.
    ngrids: Float[Array, "d"]
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
    # it should be fine for now, and it won't run inside of jax loops.
    axis_linspaces = [gen_range(i) for i in range(dimension)]

    grid = jnp.stack(jnp.meshgrid(*axis_linspaces, indexing="ij"), axis=-1).reshape(
        -1, dimension
    )
    sort_grid = grid[jnp.argsort(grid[:, 1]), :]

    deltas = (bounds[:, 1] - bounds[:, 0]) / (ngrids - 1)
    # for the purposes of testing against R-ide, we can purposefully get this a little wrong (since R-IDE does)
    # deltas = (bounds[:, 1] - bounds[:, 0]) / (ngrids)

    return Grid(
        # coords=grid.at[:, [0, 1]].set(grid[:, [1, 0]]),
        coords=sort_grid,
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
    DEPRECIATED, not used any more, i think. leaving it here for now juuust in case.

    Computes the outer operation of two vectors, a generalisation of the outer
    product. Wowza

    Parameters
    ----------
    a:
        Array of the first vector.  Assumed shape (n,)
    b:
        Array of the second vector. Assumed shape (m,)
    op:
        A jit-function acting on an element of vec1 and an element of vec2.
        By default, this is the outer product.

    Returns
    ----------
    result : ArrayLike
        The matrix of the result of applying operation to every pair of elements
        from the two vectors. Return shape (n, m).
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

    return vec_op(a, op)


@jax.jit
def bisquare(s: Float[Array, "2"], params: Float[Array, "3"]) -> float:
    
    """Generic 2D bisquare function
    Parameters
    ----------
    s: Float[Array, "2"]
        The point at which to evauluate the bisquare function
    params: Float[Array, "3"]
        Parameters of the function; the first two elements are the offset parameter, and the last is the shape parameter.
        Yes, this should _probably_ be a dictionary or something.
    """
    squarenorm = jnp.array([jnp.sum((s - params[0:2]) ** 2)])
    w2 = params[2] ** 2
    return (jnp.where(squarenorm < w2, (1 - squarenorm / w2) ** 2, 0))[0]


class Basis(NamedTuple):
    """
    A simple class for spatial basis expansions.

    Attributes
    ----------
    vfun: Float[Array, "ndim"] -> Float[Array, "nbasis"]
        Applying to a single spatial location, evaluates all the basis functions
        on the single location and returns the result as a vector.
    mfun: Float[Array, "ndim n"] -> Float[Array, "nbasis n"]
        Applying to a array of spatial points, evaluates all the basis functions
        on each point and returns the results in a matrix.
    params: Float[Array, "nparams nbasis"]
        The parameters defining each basis function. For example, for bisquare
        basis functions, the parameters are the locations of the centers of each
        function, and the function's scale.
    nbasis: int
        The number of basis functions in the expansion.
    """

    vfun: Callable[[Float[Array, "ndim"]], Float[Array, "nbasis"]]
    mfun: Callable[[Float[Array, "ndim n"]], Float[Array, "nbasis n"]]
    params: Float[Array, "nparams nbasis"]
    nbasis: int


def place_basis(
    bounds: Float[Array, "ndim 2"] = jnp.array([[0, 1], [0, 1]]),
    nres: int = 2,
    aperture: float = 1.25,
    min_knot_num: int = 3,
    basis_fun: Callable[
        [Float[Array, "ndim"], Float[Array, "nparams"]], float
    ] = bisquare,
) -> Basis:
    """
    Distributes knots (centroids) and scales for basis functions over a
    number of resolutions,similar to auto_basis from the R package FRK.
    This function must be run outside of a jit loop, since it involves
    varying the length of arrays.

    Parameters
    ----------
    bounds: Float[Array, "ndim 2"]
        The bounds for each dimension for the space we want to cover with basis functions
    nres: int
        The number of resolutions at which to place basis functions
    aperture: float
        Scaling factor for the scale parameter (scale parameter will be
        w=aperture * d, where d is the minimum distance between any two of the
        knots)
    min_knot_num: int
        The number of basis functions to place in each dimension at the coursest
        resolution
    basis_fun: Float[Array, "ndim"], Float[Array, "nparams"] -> float
        The basis functions being used. The basis function's second argument must
        be an array with three doubles; the first coordinate for the centre, the
        second coordinate for the centre, and the scale/aperture of the function.
    Returns
    ----------
    A Basis object (NamedTuple) with the vector and matrix functions, and the
    parameters associated to the basis functions.
    """

    xmin = bounds[0, 0]
    xmax = bounds[0, 1]
    ymin = bounds[1, 0]
    ymax = bounds[1, 1]

    asp_ratio = (ymax - ymin) / (xmax - xmin)

    if asp_ratio < 1:
        ny = jnp.array(min_knot_num)
        # nx = jnp.round(ny / asp_ratio).astype(int)
        nx = ny / asp_ratio
    else:
        nx = jnp.array(min_knot_num)
        # ny = jnp.round(asp_ratio * nx).astype(int)
        ny = asp_ratio * nx

    def basis_at_res(res):
        # small point; isn't the 3 here arbitrary?
        ngrids = jnp.round(jnp.array([nx, ny]) * 3**res).astype(jnp.int32)

        grid = create_grid(bounds, ngrids)

        w = jnp.min(grid.deltas) * aperture * 1.5

        # maybe adjust for create_grid being wrong
        # w = jnp.min(grid.deltas*ngrids/(ngrids-1)) * aperture *1.5

        return jnp.hstack([grid.coords, jnp.full((grid.ngrid, 1), w)])

    params = jnp.vstack([basis_at_res(res) for res in range(nres)])
    nbasis = params.shape[0]

    @jax.jit
    def basis_vfun(s: Float[Array, "ndim"]) -> Float[Array, "nbasis"]:
        return jax.vmap(basis_fun, in_axes=(None, 0))(s, params)

    @jax.jit
    def eval_basis(s_array: Float[Array, "ndim n"]) -> Float[Array, "n nbasis"]:
        return jax.vmap(jax.vmap(basis_fun, in_axes=(None, 0)), in_axes=(0, None))(
            s_array, params
        )

    return Basis(basis_vfun, eval_basis, params, nbasis)


def random_basis(
    key: Int[Array, "2"],
    knot_num: int = 3,
    bounds: Float[Array, "ndim 2"] = jnp.array([[0, 1], [0, 1]]),
    aperture: float = 10,
    basis_fun: Callable[
        [Float[Array, "ndim"], Float[Array, "nparams"]], float
    ] = bisquare,
):
    """
    Randomly distributes knots (centroids) and scales for basis functions.
    This function must be run outside of a jit loop, since it involves
    varying the length of arrays.

    Parameters
    ----------
    key: Int[Array, "2"]
        PRNG key
    knot_num: int
        The number of knots at which to place basis functions
    bounds: Float[Array, "ndim 2"]
        The bounds for each dimension for the space we want to cover with basis functions
    aperture: float
        Scaling factor for the scale parameter (scale parameter will be
        w=aperture * d, where d is the minimum distance between any two of the
        knots)
    basis_fun: Float[Array, "ndim"], Float[Array, "nparams"] -> float
        The basis functions being used. The basis function's second argument must
        be an array with three doubles; the first coordinate for the centre, the
        second coordinate for the centre, and the scale/aperture of the function.
    Returns
    ----------
    A Basis object (NamedTuple) with the vector and matrix functions, and the
    parameters associated to the basis functions.
    """

    xmin = bounds[0, 0]
    xmax = bounds[0, 1]
    ymin = bounds[1, 0]
    ymax = bounds[1, 1]

    keys = jax.random.split(key, 2)

    w = ((xmax - xmin) * (ymax - ymin)) / knot_num

    params = jnp.hstack(
        [
            jax.random.uniform(keys[0], shape=(knot_num, 1), minval=xmin, maxval=xmax),
            jax.random.uniform(keys[1], shape=(knot_num, 1), minval=ymin, maxval=ymax),
            jnp.full((knot_num, 1), w) * aperture,
        ]
    )

    nbasis = params.shape[0]

    @jax.jit
    def basis_vfun(s: Float[Array, "ndim"]) -> Float[Array, "nbasis"]:
        return jax.vmap(basis_fun, in_axes=(None, 0))(s, params)

    @jax.jit
    def eval_basis(s_array: Float[Array, "ndim n"]) -> Float[Array, "n nbasis"]:
        return jax.vmap(jax.vmap(basis_fun, in_axes=(None, 0)), in_axes=(0, None))(
            s_array, params
        )

    return Basis(basis_vfun, eval_basis, params, nbasis)


def place_cosine_basis(
    bounds: Float[Array, "ndim 2"] = jnp.array([[0, 1], [0, 1]]), N: int = 20
):
    """
    Creates cosine (fourier) basis functions at N frequencies at each dimension.

    Parameters
    ----------
    bounds: Float[Array, "ndim 2"]
        The bounds for each dimension for the space we want to cover with basis functions
    N: int
        The number of frequencies across each dimension (so in total there will be N^2 basis functions)
    Returns
    ----------
    A Basis object (NamedTuple) with the vector and matrix functions, and the
    parameters associated to the basis functions.
    """

    xmin = bounds[0, 0]
    xmax = bounds[0, 1]
    ymin = bounds[1, 0]
    ymax = bounds[1, 1]

    N = float(N)

    L_1 = jnp.abs(xmin - xmax)
    L_2 = jnp.abs(ymin - ymax)

    @jax.jit
    def phi(ks, s):
        return jnp.cos(ks[0] * jnp.pi * s[0] / L_1) * jnp.cos(
            ks[1] * jnp.pi * s[1] / L_2
        )

    k1s, k2s = jnp.meshgrid(jnp.arange(N), jnp.arange(N))
    pairs = jnp.stack([k1s.flatten(), k2s.flatten()], axis=-1)

    @jax.jit
    def basis_vfun(s: Float[Array, "ndim"]) -> Float[Array, "nbasis"]:
        return jax.vmap(phi, in_axes=(0, None))(pairs, s)

    @jax.jit
    def eval_basis(s_array: Float[Array, "ndim n"]) -> Float[Array, "n nbasis"]:
        return jax.vmap(jax.vmap(phi, in_axes=(0, None)), in_axes=(None, 0))(
            pairs,
            s_array,
        )

    return Basis(basis_vfun, eval_basis, pairs, int(N**2))


class st_data:
    """
    Stores spatio-temporal data for use with IDEMs.

    Supports:
    - Regular spatial grids with arbitrary covariates
    - Time alignment on a regular lattice

    Limitations:
    - Assumes regular time spacing
    - Assumes spatial coordinates are 2D
    - Wide-format conversion may fail for sparse or irregular data

    Attributes:
    - x, y, times, z: raw data vectors
    - covariates: design matrix without intercept (intercept is added in __init__)
    - full_times: regular time lattice
    - coords_tree, zs_tree, X_obs_tree: time-sliced views
    """

    def __init__(
        self,
        x: Float[Array, "n"],
        y: Float[Array, "n"],
        times: Float[Array, "n"],
        z: Float[Array, "n"],
        dt: float = None,
        covariates: Float[Array, "n ncovs"] = None,
        covariate_labels: List[str] = None,
    ):
        self.x = jnp.array(x)
        self.y = jnp.array(y)
        self.times = jnp.array(times, dtype=self.x.dtype)
        self.z = jnp.array(z) # huh?
        if covariates is None:
            self.covariates = jnp.ones((x.size, 1))
            self.covariate_labels = ["Intercept"]
        else:
            self.covariates = jnp.column_stack(
                [jnp.ones_like(x), jnp.array(covariates)]
            )
            self.covariate_labels = ["Intercept"] + list(covariate_labels)

        self.data_array = jnp.column_stack(
            (self.x, self.y, self.times, self.z, self.covariates)
        )
        sorted_indices = jnp.argsort(self.data_array[:, 2])  # always sort by time
        self.data_array = self.data_array[sorted_indices]

        # the logic below works, but _probably_ isn't as efficient as is
        # could be. doesn't really need to be though.
        unique_times = jnp.unique(times)  # automatically sorted
        self.unique_times = unique_times
        # self.T = len(self.times)
        if dt is None:
            dt = jnp.min(jnp.abs(jnp.diff(unique_times)))

        full_times = jnp.arange(jnp.min(unique_times), jnp.max(unique_times) + dt, dt)
        self.full_times = full_times
        time_indices = jnp.searchsorted(full_times, times)
        #time_indices = [0]
        #for time in full_times[1:]:
        #    i = 0
        #    while times[0] + i * dt <= unique_times[-1]:
        #        if jnp.allclose(time, times[0] + i * dt):
        #            time_indices.append(i)
        #        i = i + 1
        if times.size != time_indices.size:
            raise ValueError(
                "Not all times where found on the regular lattice using the smallest time difference. st_data is only for spatial data that can be places on a regular time lattice with the mimimum difference between two time points. Providing a custom dt can fix this, but the data set will not be ideal for discrete-time modelling."
            )

        #def associate(time):
        #    index = jnp.argwhere(
        #        jnp.isclose(full_times, time), size=1, fill_value=jnp.nan
        #    )
        #    return index[0][0]

        #self.t = jl.map(associate, times)
        self.t = time_indices

        self.T = self.full_times.size

        self.coords = jnp.unique(self.data_array[:, 0:2], axis=0)

        xmin = jnp.min(self.coords[:, 0])
        xmax = jnp.max(self.coords[:, 0])
        ymin = jnp.min(self.coords[:, 1])
        ymax = jnp.max(self.coords[:, 1])
        self.bounds = jnp.array([[xmin, xmax], [ymin, ymax]])


        # These lines can take a while...
        self.zs_tree = jax.tree.map(lambda i: z[jnp.where(self.t == i)], list(range(self.T)))
        self.X_obs_tree = jax.tree.map(lambda i: self.covariates[jnp.where(self.t == i)], list(range(self.T)))
        self.coords_tree = jax.tree.map(lambda i: self.data_array[:, 0:2][jnp.where(self.t == i)], list(range(self.T)))
        self.tilding_elts = jax.tree.map(lambda a, b: (a, b), self.zs_tree, self.X_obs_tree)



    def __repr__(self):
         return f"st_data({self.data_array})"

    def show_table(self, n=5, float_format="{:.4g}"):
        arr = jax.device_get(self.data_array)   # host NumPy array
        
        colnames = ["x", "y", "times", "z"] + self.covariate_labels
        df = pd.DataFrame(arr, columns=colnames)
        top = df.head(n)
        bot = df.tail(n)

        
        
        if len(df) > 2*n:
            top_string = top.to_string(index=False, float_format=lambda x: float_format.format(x))
            bot_string = bot.to_string(index=False, float_format=lambda x: float_format.format(x))

            return top_string + "\n ...\n" + bot_string
        else:
            return df.to_string(index=False, float_format=lambda x: float_format.format(x))

     
    def __str__(self):

        return self.show_table()
        
        
    @partial(jax.jit, static_argnames=["self"])
    def tildify(
        self,
        beta,
    ):
        def entilden(tup):
            z, X_obs = tup
            return z - X_obs @ beta

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 2

        # ztildes = self.z - self.X_obs_stacked @ beta
        # ztildes_tree = [ztildes[jnp.where(self.times==time)] for time in self.full_times]
        ztildes_tree = jax.tree.map(entilden, self.tilding_elts, is_leaf=is_leaf)
        return ztildes_tree
    
    def __getitem__(self, key):

        if not isinstance(key, tuple):
            time_key, space_key = key, slice(None)
        else:
            time_key, space_key = key

        # force full_times to be a float
        full_times = jnp.array(self.full_times[time_key], dtype=self.x.dtype)
        full_coords = self.coords[space_key]

        data_array_timesliced = self.data_array[jnp.isin(self.data_array[:,2], full_times)]

        eq = data_array_timesliced[:,:2][:, None, :] == full_coords[None, :, :]
        
        # row_matches shape (N, M): True where both coords match
        row_matches = eq.all(axis=2)
        # mask shape (N,): True if row matches any allowed coord
        mask = row_matches.any(axis=1) 
        
        data_array = data_array_timesliced[mask]
        
        return st_data(x = data_array[:,0],
                       y = data_array[:,1],
                       times = data_array[:,2],
                       z = data_array[:,3],
                       covariates = data_array[:,5:],
                       covariate_labels = self.covariate_labels[1:])
    
#    def select(
#            self,
#            indices: slice
#    ):
#
#        x = self.x[]
#
#        return st_data(x, y, times, z, dt, covariates, covariate_labels)
    
def pd_to_st(df: pd.DataFrame, xlabel, ylabel, tlabel, zlabel, covariate_labels=None):
    """
    Converts a pandas DataFrame into a `st_data` object for spatio-temporal modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing spatio-temporal data.
    xlabel : str
        Column name for the x spatial coordinate.
    ylabel : str
        Column name for the y spatial coordinate.
    tlabel : str
        Column name for the time variable. Can be datetime, string, float, or int.
        This will be converted into seconds for datetime or string types.
    zlabel : str
        Column name for the observed process value.
    covariate_labels : list of str, optional
        List of column names for covariates. An intercept column is automatically added.

    Returns
    -------
    st_data
        A structured spatio-temporal data object with time-sliced views and covariate matrix.

    Notes
    -----
    - If `covariate_labels` is None, a single intercept column is used.
    - Only supports 2D coordinates, and times should fit on a lattice.
    """

    if pd.api.types.is_datetime64_any_dtype(df[tlabel]):
        print(
            "Time inputted is of datetime type. This will be converted to a number corresponding to the seconds since the earliest time."
        )
        times = (df[tlabel] - df[tlabel].min()).dt.total_seconds()
    elif is_string_dtype(df[tlabel]):
        print(
            "Time inputted are strings. Attempting to coerce into datetime objects, then into floats..."
        )
        time_dt = pd.to_datetime(df[tlabel])
        times = jnp.array((time_dt - time_dt.min()).dt.total_seconds())
    elif is_float_dtype(df[tlabel]) or is_integer_dtype(df[tlabel]):
        times = jnp.array(df[tlabel])
    else:
        warnings.warn(
            """Times inputted not of a familiar type (datetime, string, float or int). Attempting to coerce into a JAX array anyway, but this will likely give an error."""
        )
        times = jnp.array(df[tlabel])

    if covariate_labels is None:
        covariates = None
    else:
        covariates = jnp.column_stack([jnp.array(df[col]) for col in covariate_labels])

    return st_data(
        x=jnp.array(df[xlabel]),
        y=jnp.array(df[ylabel]),
        times=times,
        z=jnp.array(df[zlabel]),
        covariates=covariates,
        covariate_labels=covariate_labels,
    )


def noise(key, tree, noise_scale=1e-5):
    """
    Adds gaussian noise to each leaf of a PyTree.
    """
    keys = rand.split(key, num=len(jax.tree.leaves(tree)))
    return jax.tree.map(
        lambda x, k: x + noise_scale * jax.random.normal(k, shape=x.shape)
        if isinstance(x, jnp.ndarray)
        else x,
        tree,
        jax.tree.unflatten(jax.tree.structure(tree), keys),
    )


def check_nans(tree):
    def check(x):
        if isinstance(x, jnp.ndarray) and jnp.any(jnp.isnan(x)):
            return True
        else:
            return False

    return any(jax.tree.leaves(jax.tree.map(check, tree)))


class TimeResults(NamedTuple):
    compile_time: float
    average_time: float
    total_time: float
    value: float


def time_jit(key, func, inp_tree, n, noise_scale=1e-5, desc="", device=None):
    """
    Timer function that uses random noise to properly time jit-compiled functions.
    Only takes functions with PyTrees of JAX arrays as inputs.

    Returns a tuple containing the compile time (attribute `compile_time`) and the average run time (attribute `average_time`, and the result of the computation at its original input (attribute `value`).
    """

    func_jit = jax.jit(func)

    failed = False

    tot_time = 0
    progress_bar = tqdm(range(n + 1), desc=desc)
    for i in progress_bar:
        noise_key = jax.random.fold_in(key, i)
        inp_noise = noise(noise_key, inp_tree, noise_scale=noise_scale)

        start_time = time.time()
        result = func_jit(inp_noise).block_until_ready()
        elapsed = time.time() - start_time
        # print(f"i: {i}, Value: {round(result,5)}, Elapsed: {round(elapsed, 5)}s")

        if i != 0:
            tot_time = tot_time + elapsed
        else:
            compile_time = elapsed

        if check_nans(result):
            warnings.warn("The function has returned a PyTree/array with nan.")
            failed = True

    if failed:
        value = jnp.nan
        average_time = jnp.nan
        average_val = jnp.nan
    else:
        value = func_jit(inp_tree)
        average_time = tot_time / n

    # print(f"Compile time: {compile_time}s")
    # print(f"Average time: {av_time}s")
    return TimeResults(compile_time, average_time, tot_time, value)


constant_basis = place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1)


def flatten(tree: PyTree):
    """
        Flattens a PyTree into a single 1D JAX array, preserving structure metadata for reconstruction.

        Parameters
        ----------
        tree : PyTree
            A nested structure of JAX arrays (e.g. list, tuple, dict of arrays).

        Returns
        -------
        flat_array : jax.Array
            A 1D array containing all leaf values concatenated in order.
        leaves : list of jax.Array
            The individual leaf arrays extracted from the tree.
        treedef : jax.tree_util.PyTreeDef
            The structure definition needed to reconstruct the original tree.
        cumsum : jax.Array
            Cumulative offsets of each leaf within the flat array (excluding final leaf),
    useful for slicing or mapping back to leaf boundaries.

        Notes
        -----
        - Use `jax.tree_unflatten(treedef, leaves)` to reconstruct the original tree.
        - This function assumes all leaves are array-like and compatible with `jnp.ravel`.
    """
    leaves, treedef = jax.tree.flatten(tree)
    flat_array = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
    sizes = jnp.array([leaf.size for leaf in leaves])
    cumsum = jnp.cumsum(sizes[:-1])
    return flat_array, leaves, treedef, cumsum


@partial(jax.jit, static_argnames=["treedef", "cumsum"])
def unflatten(flat_array, leaves, treedef, cumsum):
    """
        Reconstructs a PyTree from a flattened 1D array and shape metadata.

        This function reverses the operation performed by `flatten()`, restoring the original
    PyTree structure and leaf shapes using the provided `treedef` and `leaves`. It assumes
    that `flat_array` was created by concatenating the raveled leaves in order.

        Parameters
        ----------
        flat_array : jax.Array
            A 1D array containing all leaf values concatenated.
        leaves : list of jax.Array
            The original leaf arrays (used to recover shapes).
        treedef : jax.tree_util.PyTreeDef
            The structure definition of the original PyTree.
        cumsum : jax.Array
            Cumulative offsets used to split `flat_array` into leaf segments.

        Returns
        -------
        tree : PyTree
            The reconstructed PyTree with original structure and leaf shapes.

        Notes
        -----
        - Assumes `utils.flatten()` was used to generate `flat_array`, `leaves`, `treedef`, and `cumsum`.
        - Leaf shapes are restored using `.reshape(original.shape)` from the original leaves.
    """

    splits = jnp.split(flat_array, cumsum)
    reshaped_leaves = [
        leaf.reshape(original.shape) for leaf, original in zip(splits, leaves)
    ]
    return jax.tree.unflatten(treedef, reshaped_leaves)


def flatten_and_unflatten(tree: PyTree):
    """
        Flattens a PyTree into a 1D JAX array and returns a JIT-compatible function to reconstruct it.

        This utility is useful for optimization, serialization, or transformation workflows where
    structured data must be flattened into a single array, but later reconstructed with full shape
    and structure fidelity.

        Parameters
        ----------
        tree : PyTree
            A nested structure of JAX arrays (e.g. list, tuple, dict of arrays).

        Returns
        -------
        flat_array : jax.Array
            A 1D array containing all leaf values concatenated in order.
        unflatten : Callable[[jax.Array], PyTree]
            A JIT-compiled function that reconstructs the original PyTree from a flat array.

        Notes
        -----
        - Leaf shapes are preserved using `.reshape()` during reconstruction.
        - The returned `unflatten` function is statically compiled and safe to use inside JAX transforms.
        - Useful for parameter packing/unpacking in custom optimizers or probabilistic models.

        WARNING; written with the aid of a generative model. Does the trick, and is simple enough.
    """

    # Flatten the PyTree
    flat_leaves, treedef = jax.tree.flatten(tree)

    # Convert to a 1D array
    flat_array = jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves])

    # **Ensure sizes and split indices are concrete**
    sizes = [leaf.size for leaf in flat_leaves]  # Extract sizes concretely
    split_indices = list(
        jnp.cumsum(jnp.array(sizes[:-1]))
    )  # Precompute split indices outside JIT

    # JIT-compatible unflatten function
    @jax.jit
    def unflatten(flat_array):
        # Dynamically reshape leaves
        splits = [
            flat_array[start:end]
            for start, end in zip(
                [0] + split_indices, split_indices + [len(flat_array)]
            )
        ]
        reshaped_leaves = [
            split.reshape(leaf.shape) for split, leaf in zip(splits, flat_leaves)
        ]
        return jax.tree.unflatten(treedef, reshaped_leaves)

    return flat_array, unflatten


@partial(
    jax.jit,
    static_argnames=["shape_code"],
)
def add_variance(
    P: Float[Array, "n n"],
    S2: Union[Float[Array, "1"], Float[Array, "n"], Float[Array, "n n"]],
    shape_code: int,
) -> Float[Array, "n n"]:
    match shape_code:
        case 0 | 1:
            return jnp.fill_diagonal(P, S2 + jnp.diag(P), inplace=False)
        case 2:
            return P + S2
        case _:
            raise ValueError(f"Invalid shape code: {shape_code}")


@partial(
    jax.jit,
    static_argnames=["shape_code"],
)
def mult_variance(
    P: Float[Array, "n n"],
    S2: Union[Float[Array, "1"], Float[Array, "n"], Float[Array, "n n"]],
    shape_code: int,
):
    # returns S2 @ P.T in a hopefully efficient way, regardless of the shape of S2
    match shape_code:
        case 0:
            return S2 * P.T
        case 1:
            return jnp.diag(S2) @ P.T
        case 2:
            return S2 @ P.T


@jax.jit
def qr_R(A, B):
    """Wrapper for the stacked-QR decompositon"""
    return jnp.linalg.qr(jnp.vstack([A, B]), mode="r")


@jax.jit
def ql_L(A, B):
    """Wrapper for the stacked-QL decompositon"""
    A_flipped = jnp.flip(jnp.vstack([A, B]), axis=1)
    R = jnp.linalg.qr(A_flipped, mode="r")
    L = jnp.flip(R, axis=(0, 1))

    return L

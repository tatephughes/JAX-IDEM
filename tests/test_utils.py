import pytest
import jax.numpy as jnp
from jaxtyping import Float, Array
import jax.random as jr

from jaxidem.utils import create_grid, Grid


def test_create_grid_2d():
    bounds: Float[Array, "2 2"] = jnp.array([[0.0, 0.0], [1.0, 2.0]])
    ngrids: Float[Array, "2"] = jnp.array([3, 5])  # 3 x 5 grid

    grid = create_grid(bounds, ngrids)

    # Check dimension
    assert grid.dim == 2

    # Check number of grid points
    expected_ngrid = int(jnp.prod(ngrids))
    assert grid.coords.shape == (expected_ngrid, 2)
    assert grid.ngrid == expected_ngrid

    # Check deltas
    expected_deltas = (bounds[:, 1] - bounds[:, 0]) / (ngrids - 1)
    assert jnp.allclose(grid.deltas, expected_deltas)

    # Check area
    expected_area = jnp.prod(expected_deltas)
    assert jnp.isclose(grid.area, expected_area)

    # Check that coords span the bounds
    mins = jnp.min(grid.coords, axis=0)
    maxs = jnp.max(grid.coords, axis=0)
    assert jnp.allclose(mins, bounds[:, 0])
    assert jnp.allclose(maxs, bounds[:, 1])


def test_create_grid_1d():
    bounds: Float[Array, "1 2"] = jnp.array([[0.0, 1.0]])
    ngrids: Float[Array, "1"] = jnp.array([4])

    grid = create_grid(bounds, ngrids)

    assert grid.dim == 1
    assert grid.coords.shape == (4, 1)
    assert jnp.allclose(grid.deltas, jnp.array([1.0 / 3]))
    assert jnp.isclose(grid.area, 1.0 / 3)


def test_create_grid_nonuniform_bounds():
    bounds = jnp.array([[2.0, 5.0], [-1.0, 3.0]])
    ngrids = jnp.array([4, 3])

    grid = create_grid(bounds, ngrids)

    # Check that deltas match expected spacing
    expected_deltas = (bounds[:, 1] - bounds[:, 0]) / (ngrids - 1)
    assert jnp.allclose(grid.deltas, expected_deltas)

    # Check that grid spans correct range
    mins = jnp.min(grid.coords, axis=0)
    maxs = jnp.max(grid.coords, axis=0)
    assert jnp.allclose(mins, bounds[:, 0])
    assert jnp.allclose(maxs, bounds[:, 1])


from jaxidem.utils import place_basis, Basis


def test_place_basis_basic_structure():
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    basis = place_basis(bounds, nres=2, min_knot_num=2)

    # Check Basis object structure
    assert isinstance(basis, Basis)
    assert callable(basis.vfun)
    assert callable(basis.mfun)
    assert basis.params.ndim == 2
    assert basis.params.shape[1] == 3  # x, y, scale
    assert basis.nbasis == basis.params.shape[0]


def test_place_basis_vfun_consistency():
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    basis = place_basis(bounds, nres=1, min_knot_num=2)

    # Evaluate at a single point
    s = jnp.array([0.5, 0.5])
    result = basis.vfun(s)

    assert result.shape == (basis.nbasis,)
    assert jnp.all(result >= 0.0)
    assert jnp.any(result > 0.0)  # At least one basis should be active


def test_place_basis_mfun_consistency():
    bounds = jnp.array([[-2.0, 2.0], [-2.0, 2.0]])
    basis = place_basis(bounds, nres=1, min_knot_num=2)

    # Evaluate at multiple points
    s_array = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    result = basis.mfun(s_array)

    assert result.shape == (s_array.shape[0], basis.nbasis)
    assert jnp.all(result >= 0.0)
    assert jnp.any(result > 0.0)


def test_place_basis_resolution_scaling():
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    basis_low = place_basis(bounds, nres=1, min_knot_num=2)
    basis_high = place_basis(bounds, nres=2, min_knot_num=2)

    # Higher resolution should produce more basis functions
    assert basis_high.nbasis > basis_low.nbasis


from jaxidem.utils import random_basis, Basis


def test_random_basis_structure_and_shape():
    key = jr.PRNGKey(0)
    knot_num = 5
    bounds = jnp.array([[0.0, 1.0], [0.0, 2.0]])
    basis = random_basis(key, knot_num=knot_num, bounds=bounds)

    # Check Basis structure
    assert isinstance(basis, Basis)
    assert callable(basis.vfun)
    assert callable(basis.mfun)
    assert basis.params.shape == (knot_num, 3)
    assert basis.nbasis == knot_num


def test_random_basis_param_bounds():
    key = jr.PRNGKey(42)
    bounds = jnp.array([[0.0, 1.0], [0.0, 2.0]])
    basis = random_basis(key, knot_num=10, bounds=bounds)

    centers = basis.params[:, :2]
    assert jnp.all(centers[:, 0] >= bounds[0, 0]) and jnp.all(
        centers[:, 0] <= bounds[0, 1]
    )
    assert jnp.all(centers[:, 1] >= bounds[1, 0]) and jnp.all(
        centers[:, 1] <= bounds[1, 1]
    )


def test_random_basis_aperture_scaling():
    key = jr.PRNGKey(123)
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    knot_num = 4
    aperture = 5.0
    basis = random_basis(key, knot_num=knot_num, bounds=bounds, aperture=aperture)

    expected_w = ((1.0 * 1.0) / knot_num) * aperture
    actual_ws = basis.params[:, 2]
    assert jnp.allclose(actual_ws, expected_w)


def test_random_basis_vfun_behavior():
    key = jr.PRNGKey(7)
    basis = random_basis(key, knot_num=6)

    s = jnp.array([0.5, 0.5])
    result = basis.vfun(s)

    assert result.shape == (basis.nbasis,)
    assert jnp.all(result >= 0.0)
    assert jnp.any(result > 0.0)


def test_random_basis_mfun_behavior():
    key = jr.PRNGKey(99)
    basis = random_basis(key, knot_num=4)

    s_array = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    result = basis.mfun(s_array)

    assert result.shape == (s_array.shape[0], basis.nbasis)
    assert jnp.all(result >= 0.0)
    assert jnp.any(result > 0.0)


from jaxidem.utils import place_cosine_basis, Basis


def test_cosine_basis_structure():
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    N = 4
    basis = place_cosine_basis(bounds, N)

    assert isinstance(basis, Basis)
    assert callable(basis.vfun)
    assert callable(basis.mfun)
    assert basis.params.shape == (N**2, 2)
    assert basis.nbasis == N**2


def test_cosine_basis_vfun_behavior():
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    basis = place_cosine_basis(bounds, N=4)

    s = jnp.array([0.5, 0.5])
    result = basis.vfun(s)

    assert result.shape == (basis.nbasis,)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.any(result != 0.0)


def test_cosine_basis_mfun_behavior():
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    basis = place_cosine_basis(bounds, N=2)

    s_array = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    result = basis.mfun(s_array)

    assert result.shape == (s_array.shape[0], basis.nbasis)
    assert jnp.all(jnp.isfinite(result))


from jaxidem.utils import st_data


def generate_sample_data(n=6):
    x = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    y = jnp.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5])
    times = jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    z = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    covariates = jnp.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
    labels = ["cov1"]
    return x, y, times, z, covariates, labels


def test_st_data_initialization():
    x, y, times, z, covs, labels = generate_sample_data()
    data = st_data(x, y, times, z, covariates=covs, covariate_labels=labels)

    assert data.x.shape == (6,)
    assert data.covariates.shape == (6, 2)  # intercept + 1 covariate
    assert data.data_array.shape[1] == 6  # x, y, time, z, cov1, intercept
    assert data.T == 3
    assert data.coords.shape[1] == 2
    assert jnp.allclose(data.bounds[:, 0], jnp.array([0.0, 0.0]))
    assert jnp.allclose(data.bounds[:, 1], jnp.array([1.0, 1.0]))


def test_st_data_tildify_behavior():
    x, y, times, z, covs, labels = generate_sample_data()
    data = st_data(x, y, times, z, covariates=covs, covariate_labels=labels)

    beta = jnp.array([1.0, 2.0])  # intercept + cov1
    ztildes_tree = data.tildify(beta)

    assert isinstance(ztildes_tree, list)
    assert len(ztildes_tree) == data.T
    for zt in ztildes_tree:
        assert zt.ndim == 1
        assert jnp.all(jnp.isfinite(zt))


def test_st_data_time_alignment():
    x, y, times, z, covs, labels = generate_sample_data()
    data = st_data(x, y, times, z, covariates=covs, covariate_labels=labels)

    # Check that time indices match expected lattice
    expected_times = jnp.array([0.0, 1.0, 2.0])
    assert jnp.allclose(data.full_times, expected_times)
    assert jnp.all(jnp.isin(data.times, data.full_times))

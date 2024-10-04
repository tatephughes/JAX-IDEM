import jax.numpy as jnp

import sys

sys.path.append("../../main/jax_idem")
# import pytest
import utilities


class TestCreateGrid:
    def test_shape(self, ndim=3, ngrid=50):
        bounds = jnp.stack([jnp.array([0, 1])] * ndim)
        ngrids = jnp.tile(ngrid, ndim)

        grid, deltas = utilities.create_grid(bounds, ngrids)

        assert grid.shape == (
            int(jnp.prod(ngrids)),
            ndim,
        ), f"Expected shape {(jnp.prod(ngrids), ndim)}, got shape {grid.shape}"


class TestOuterOp:
    def test_1D_case(self):
        vec1 = jnp.array([1])
        vec2 = jnp.array([0])

        result = utilities.outer_op(vec1, vec2, lambda x, y: x - y)

        assert result == jnp.array([[1]]), "Failed 1D case"

    def test_shape(self):
        vec1 = jnp.array([1, 2])
        vec2 = jnp.array([4, 5, 3])

        result = utilities.outer_op(vec1, vec2, lambda x, y: x + y)

        new_shape = (vec1.shape[0], vec2.shape[0])

        assert result.shape == new_shape, "outer_op gives wrong return shape"

    def test_array_shape(self):
        def sq_norm(s, r):
            return (s[0] - r[0]) ** 2 + (s[1] - r[1]) ^ 2

        vec1 = jnp.array([[1, 2], [2, 3]])
        vec2 = jnp.array([[1, 0], [0, 1]])

        result = utilities.outer_op(vec1, vec2, sq_norm)

        assert result.shape == (2, 2)

    def test_val(self):
        vec1 = jnp.array([1, 2])
        vec2 = jnp.array([4, 5, 3])

        result = utilities.outer_op(vec1, vec2, lambda x, y: x + y)

        assert result[0, 0] == 5

    def test_array_val(self):
        def sq_norm(s, r):
            return (s[0] - r[0]) ** 2 + (s[1] - r[1]) ** 2

        vec1 = jnp.array([[1, 2], [2, 3]])
        vec2 = jnp.array([[1, 0], [0, 1]])

        result = utilities.outer_op(vec1, vec2, sq_norm)

        first_result = sq_norm(vec1[0], vec2[0])

        assert result[0, 0] == first_result

    def test_grid(self):
        bounds = jnp.stack([jnp.array([0, 1])] * 2)
        ngrids = jnp.tile(50, 2)

        grid, deltas = utilities.create_grid(bounds, ngrids)

        def psi(s, r):
            squarenorm = jnp.array([jnp.sum((s - r) ** 2)])
            return ((2 - squarenorm) ** 2 * jnp.where(squarenorm < 1, 0.5, 0))[0]

        result = utilities.outer_op(grid, grid, psi)

        assert result.shape == (grid.shape[0], grid.shape[0])


class TestPlaceBasis:
    min_knot_num = 3
    nres = 2
    basis = utilities.place_basis(nres=nres, min_knot_num=min_knot_num)

    s = jnp.array([0.5, 0.5])  # example point
    bounds = jnp.stack([jnp.array([0, 1])] * 2)
    ngrids = jnp.tile(42, 2)

    s_grid, deltas = utilities.create_grid(bounds, ngrids)  # example set of points

    vphi = basis.vfun(s)  # basis vector on s
    mPHI = basis.mfun(s_grid)  # basis matrix on s_grid

    def test_shape_vfun(self):
        assert self.vphi.shape == (self.basis.nbasis,)

    def test_shape_eval(self):
        assert self.mPHI.shape == (jnp.prod(self.ngrids), self.basis.nbasis)

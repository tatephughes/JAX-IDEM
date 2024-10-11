import jax.numpy as jnp
import jax
import jax.random as rand

import sys

sys.path.append("../../main/jax_idem")
# import pytest
import IDEM
import utilities


class TestConstructM:
    key = jax.random.PRNGKey(seed=628)
    keys = rand.split(key, 3)

    process_basis = utilities.place_basis()

    def kernel(self, s, r):
        return jnp.exp(-(jnp.sum((r - s) ** 2)))

    bounds = jnp.array([[0, 1], [0, 1]])

    grid = utilities.create_grid(bounds, jnp.array([41, 41]))  # example set of points

    def test_shape(self):
        M = IDEM.construct_M(self.kernel, self.process_basis, self.grid)
        nbasis = self.process_basis.nbasis

        assert M.shape == (nbasis, nbasis)

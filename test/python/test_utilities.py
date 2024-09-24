import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
from jax.numpy.linalg import solve

import pytest
import utilities

class TestCreateGrid:

    def test_shape(self, ndim=3, griddelta=1/jnp.pi**2):
    
        bounds = jnp.stack([jnp.array([0, 1])] * ndim)
        deltas = jnp.tile(griddelta, ndim)
        
        numpoints = int((1/griddelta)+1)**ndim
        
        assert (utilities.create_grid(bounds, deltas).shape
                == (numpoints, ndim))

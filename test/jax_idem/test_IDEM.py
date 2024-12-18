import jax.numpy as jnp
import jax
import jax.random as rand
import pytest

import sys
import warnings

sys.path.append("../../main/jax_idem")
import IDEM
import utilities


class testSimIDEM:
    @pytest.fixture(autouse=True)
    def setup(self):
        # this key gives non-explosive kernel
        key = jax.random.PRNGKey(3)
        self.keys = rand.split(key, 3)

        # make a super simple example model
        self.model = IDEM.gen_example_idem(
            keys[0],
            k_spat_inv=False,
            ngrid=jnp.array([10, 10]),
            nints=jnp.array([10, 10]),
            nobs=3,
            process_basis=utilities.place_basis(nres=1, min_knot_num=2),
        )

        self.nobs = 4

        self.T = 3

        if jnp.max(jnp.absolute(jnp.linalg.eig(self.model.M)[0])) > 1.0:
            warnings.warn(
                "Eigenvalue above the absolute value of 1. Result will be explosive."
            )

        self.obs_locs = jnp.column_stack(
            [
                jnp.repeat(jnp.arange(T), nobs) + 1,
                rand.uniform(
                    keys[0],
                    shape=(T * nobs, 2),
                    minval=0,
                    maxval=1,
                ),
            ]
        )

        self.times = jnp.unique(obs_locs[:, 0])

    def test_shape(self):
        obs_locs_tree = jax.tree.map(
            lambda t: self.obs_locs[jnp.where(self.obs_locs[:, 0] == t)][:, 1:],
            list(self.times),
        )
        PHI_tree = jax.tree.map(self.model.process_basis.mfun, obs_locs_tree)

        PHI_obs = jax.scipy.linalg.block_diag(*PHI_tree)

        process_vals, obs_vals = IDEM.simIDEM(
            key=self.keys[1],
            T=self.T,
            M=self.model.M,
            PHI_proc=self.model.PHI_proc,
            PHI_obs=PHI_obs,
            beta=self.model.beta,
            alpha0=self.model.m_0,
            obs_locs=self.obs_locs,
            process_grid=self.model.process_grid,
            int_grid=self.model.int_grid,
        )

        assert (process_vals.shape == (self.T, self.model.process_grid.ngrid)) & (
            obs_vals.shape == (self.T * self.nobs)
        )

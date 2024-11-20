import jax.numpy as jnp
import jax
import jax.random as rand

import sys
import warnings

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


class testSimIDEM:
    key = jax.random.PRNGKey(1)
    keys = rand.split(key, 3)

    # make a super simple example model
    model = IDEM.gen_example_idem(
        keys[0],
        k_spat_inv=False,
        ngrid=jnp.array([10, 10]),
        nints=jnp.array([10, 10]),
        nobs=3,
    )

    (
        M,
        PHI_proc,
        beta,
        sigma2_eta,
        sigma2_eps,
        alpha0,
        process_grid,
        int_grid,
    ) = model.get_sim_params()

    nobs = 4

    T = 3

    if jnp.max(jnp.absolute(jnp.linalg.eig(M)[0])) > 1.0:
        warnings.warn(
            "Eigenvalue above the absolute value of 1. Result will be explosive."
        )

    bounds = jnp.array(
        [
            [
                jnp.min(process_grid.coords[:, 0]),
                jnp.max(process_grid.coords[:, 0]),
            ],
            [
                jnp.min(process_grid.coords[:, 1]),
                jnp.max(process_grid.coords[:, 1]),
            ],
        ]
    )

    obs_locs = jnp.column_stack(
        [
            jnp.repeat(jnp.arange(T), nobs),
            rand.uniform(
                keys[0],
                shape=(T * nobs, 2),
                minval=bounds[:, 0],
                maxval=bounds[:, 1],
            ),
        ]
    )

    times = jnp.unique(obs_locs[:, 0])

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
            M=self.M,
            PHI_proc=self.PHI_proc,
            PHI_obs=self.PHI_obs,
            beta=self.beta,
            alpha0=self.alpha0,
            obs_locs=self.obs_locs,
            process_grid=self.process_grid,
            int_grid=self.int_grid,
        )

        assert (process_vals.shape == (self.T, self.process_grid.ngrid)) & (
            obs_vals.shape == (self.T * self.nobs)
        )

import filter_smoother_functions as fsf
import utilities
import idem
import jax.numpy as jnp
import jax
import jax.random as rand
import pytest

import sys
import warnings

sys.path.append("../../main/jax_idem")


class TestInformationFilter:

    @pytest.fixture(autouse=True)
    def setup(self):

        key = jax.random.PRNGKey(1)
        keys = rand.split(key, 3)

        process_basis = utilities.place_basis(nres=2, min_knot_num=3)

        self.model = idem.gen_example_idem(
            keys[0],
            k_spat_inv=True,
            process_basis=process_basis,
        )

        T = 10

        # Simulation
        self.process_data, self.obs_data = self.model.simulate(
            nobs=50, T=T, key=keys[1]
        )

        self.nbasis = self.model.process_basis.nbasis
        self.nu_0 = jnp.zeros(nbasis)
        self.Q_0 = 0.01*jnp.eye(nbasis)

        obs_locs_wide = jnp.array(
            [obs_data.as_wide()['x'], obs_data.as_wide()['y']])
        self.unique_times = jnp.unique(self.obs_data.t)
        self.time_inds = tuple(jnp.arange(unique_times.shape[0]))
        self.obs_locs_tuple = jax.tree.map(
            lambda t: obs_locs[self.obs_data.t == t], tuple(self.unique_times)
        )

        self.PHI_obs_tuple = jax.tree.map(
            self.model.process_basis.mfun, self.obs_locs_tuple)

        self.X_obs = jnp.column_stack(
            [jnp.ones(obs_locs_wide.shape[1]), obs_locs_wide.T])

        self.Sigma_eta = self.model.sigma2_eta * jnp.eye(self.nbasis)
        self.Sigma_eps = self.model.sigma2_eps * jnp.eye(50)
        self.Sigma_eps_tuple = [self.Sigma_eps for _ in range(T)]

        zs = [obs_data.z[obs_data.t == t] -
              self.X_obs @ self.model.beta for t in unique_times]

        self.ll, self.nus, self.Qs, self.nupreds, self.Qpreds = fsf.information_filter(
            self.nu_0,
            self.Q_0,
            self.model.M,
            self.PHI_obs_tuple,
            self.Sigma_eta,
            self.Sigma_eps,
            self.obs_data.z,
        )

    def test_shape(self):

        nus = self.nus

        assert (nus.shape == (nbasis,)) & (
            carry[1].shape == (nbasis, nbasis)
        ), f"Expeced shapes {nbasis} and {(nbasis, nbasis)}, got {carry[0].shape} and {carry[1].shape}."

    def test_againt_kf(self):

        return None

    def test_against_naive(self):

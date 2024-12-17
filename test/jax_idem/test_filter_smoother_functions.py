import jax.numpy as jnp
import jax
import jax.random as rand
import pytest

import sys
import warnings

sys.path.append("../../main/jax_idem")
import IDEM
import utilities
import filter_smoother_functions


class TestInformationFilter:

    @pytest.fixture(autouse=True)
    def setup(self):

        key = jax.random.PRNGKey(1)
        keys = rand.split(key, 3)

        process_basis = utilities.place_basis(nres=2, min_knot_num=3)

        self.model = IDEM.gen_example_idem(
            keys[0],
            k_spat_inv=True,
            process_basis=process_basis,
        )

        T = 10

        nobs = jax.random.randint(keys[1], (T,), 10, 51)

        locs_keys = jax.random.split(keys[2], T)

        obs_locs = jnp.vstack(
            [
                jnp.column_stack(
                    [
                        jnp.repeat(t + 1, n),
                        rand.uniform(
                            locs_keys[t],
                            shape=(n, 2),
                            minval=0,
                            maxval=1,
                        ),
                    ]
                )
                for t, n in enumerate(nobs)
            ]
        )

        # Simulation
        self.process_data, self.obs_data = self.model.simulate(
            nobs=50, T=T, key=keys[1], obs_locs=obs_locs
        )

    def test_shape(self):

        nbasis = self.model.process_basis.nbasis
        nu_0 = jnp.zeros(nbasis)
        Q_0 = jnp.zeros((nbasis, nbasis))

        obs_locs = jnp.column_stack(
            jnp.column_stack((self.obs_data.x, self.obs_data.y))
        ).T
        unique_times = jnp.unique(self.obs_data.t)
        time_inds = tuple(jnp.arange(unique_times.shape[0]))
        obs_locs_tuple = jax.tree.map(
            lambda t: obs_locs[self.obs_data.t == t], time_inds
        )

        PHI_obs_tuple = jax.tree.map(self.model.process_basis.mfun, obs_locs_tuple)

        X_obs = jnp.column_stack([jnp.ones(obs_locs.shape[0]), obs_locs[:, -2:]])

        carry, seq = filter_smoother_functions.information_filter(
            nu_0,
            Q_0,
            self.model.M,
            PHI_obs_tuple,
            self.model.sigma2_eta,
            self.model.sigma2_eps,
            self.model.beta,
            self.obs_data,
            X_obs,
        )

        assert (carry[0].shape == (nbasis,)) & (
            carry[1].shape == (nbasis, nbasis)
        ), f"Expeced shapes {nbasis} and {(nbasis,nbasis)}, got {carry[0].shape} and {carry[1].shape}."

    def test_againt_kf(self):

        return None

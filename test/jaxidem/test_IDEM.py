import jax.numpy as jnp
import jax
import jax.random as rand
import pytest

import sys
import warnings

sys.path.append("../src/jaxidem")
sys.path.append("./src/jaxidem")
import idem
import utilities

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

class TestSimIdem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # this key gives non-explosive kernel
        self.key = jax.random.PRNGKey(3)
        self.keys = rand.split(self.key, 4)

        # make a super simple example model
        self.model = idem.gen_example_idem(
            self.keys[0],
            k_spat_inv=False,
            ngrid=jnp.array([10, 10]),
            nints=jnp.array([10, 10]),
            process_basis=utilities.place_basis(nres=1, min_knot_num=2),
        )

        self.nobs = 4
        self.T = 3
        self.nbasis = self.model.process_basis.nbasis

        match len(self.model.sigma2_eta.shape):
            case 0:
                Sigma_eta = self.model.sigma2_eta * jnp.eye(self.model.nbasis)
            case 1:
                Sigma_eta = jnp.diag(self.model.sigma2_eta)
            case 2:
                Sigma_eta = self.model.sigma2_eta

        match len(self.model.sigma2_eps.shape):
            case 0:
                Sigma_eps = self.model.sigma2_eps * jnp.eye(self.nobs*self.T)
            case 1:
                Sigma_eps = jnp.diag(self.model.sigma2_eps)
            case 2:
                Sigma_eps = self.model.sigma2_eps

        
        obs_locs = jnp.column_stack(
            [
                jnp.repeat(jnp.arange(self.T), self.nobs) + 1,
                rand.uniform(
                    self.keys[1],
                    shape=(self.T * self.nobs, 2),
                    minval=0,
                    maxval=1,
                ),
            ]
        )

        self.unique_times = jnp.unique(obs_locs[:, 0])
        
        obs_locs_tree = jax.tree.map(
            lambda t: obs_locs[jnp.where(obs_locs[:, 0] == t)][:, 1:],
            list(self.unique_times),
        )
        
        PHI_tree = jax.tree.map(self.model.process_basis.mfun, obs_locs_tree)
        PHI_obs = jax.scipy.linalg.block_diag(*PHI_tree)

        alpha_0 = jax.random.multivariate_normal(
                self.keys[2], jnp.zeros(self.nbasis), 0.1 * jnp.eye(self.nbasis)
            )
        
        self.process_vals, self.obs_vals = idem.sim_idem(
            key=self.keys[3],
            T=self.T,
            M=self.model.M,
            PHI_proc=self.model.PHI_proc,
            PHI_obs=PHI_obs,
            beta=self.model.beta,
            alpha_0=alpha_0,
            obs_locs=obs_locs,
            process_grid=self.model.process_grid,
            int_grid=self.model.int_grid,
            Sigma_eps=Sigma_eps,
            Sigma_eta=Sigma_eta,
        )
    def test_shape(self):

        assert (self.process_vals.shape == (self.T, self.model.PHI_proc.shape[0])) & (
            self.obs_vals.shape == (self.T * self.nobs,)
        )

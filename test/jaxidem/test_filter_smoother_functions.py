import filter_smoother_functions as fsf
import utilities
import idem
import jax.numpy as jnp
import jax
import jax.random as rand
import pytest

import sys
import warnings

# sys.path.append("../src/jaxidem")
# sys.path.append("./src/jaxidem")

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')


class TestFilters:

    @pytest.fixture(autouse=True)
    def setup(self):

        key = jax.random.PRNGKey(1)
        keys = rand.split(key, 3)

        process_basis = utilities.place_basis(nres=2, min_knot_num=3)

        model = idem.gen_example_idem(
            keys[0],
            k_spat_inv=True,
            process_basis=process_basis,
        )

        beta = model.beta
        M = model.M
        sigma2_eps = model.sigma2_eps
        sigma2_eta = model.Sigma_eta[0,0]
        nbasis = model.process_basis.nbasis
        self.nbasis = nbasis

        T = 10
        self.T = T
        alpha_0 = jnp.zeros(nbasis).at[81].set(10)
        process_data, obs_data = model.simulate(
            nobs=100, T=T, key=keys[1], alpha_0=alpha_0)

        unique_times = jnp.unique(obs_data.t)

        obs_data_wide = obs_data.as_wide()
        nobs = obs_data_wide['x'].shape[0]

        Sigma_eta = sigma2_eta * jnp.eye(nbasis)
        Sigma_eps = sigma2_eps * jnp.eye(nobs)
        Sigma_eps_tuple = [Sigma_eps for _ in range(T)]

        obs_locs = jnp.column_stack([obs_data_wide["x"], obs_data_wide["y"]])
        obs_locs_long = jnp.column_stack(
            jnp.column_stack((obs_data.x, obs_data.y))).T

        obs_locs_tuple = [obs_locs_long[obs_data.t == t][:, 0:]
                          for t in unique_times]

        PHI_obs = model.process_basis.mfun(obs_locs)
        PHI_obs_tuple = jax.tree.map(model.process_basis.mfun, obs_locs_tuple)

        X_obs = jnp.column_stack(
            [jnp.ones(obs_data_wide['x'].shape[0]), obs_data_wide['x'], obs_data_wide['y']])
        X_obs_tuple = jax.tree.map(
            lambda locs: jnp.column_stack((jnp.ones(len(locs)), locs)), obs_locs_tuple)

        m_0 = jnp.zeros(nbasis)
        P_0 = 100 * jnp.eye(nbasis)
        nu_0 = jnp.zeros(nbasis)
        Q_0 = 0.01 * jnp.eye(nbasis)

        ztildes = obs_data_wide["z"] - (X_obs @ model.beta)[:, None]

        zs_tuple = [obs_data.z[obs_data.t == t] for t in unique_times]
        ztildes_tuple = [zs_tuple[i] - X_obs_tuple[i]@beta for i in range(T)]

        self.ll_kf, self.ms_kf, self.Ps_kf, _, self.Ppreds_kf, _ = fsf.kalman_filter(
            m_0,
            P_0,
            model.M,
            PHI_obs,
            Sigma_eta,
            Sigma_eps,
            ztildes,
            likelihood="full"
        )
        self.ll_kfi, self.ms_kfi, self.Ps_kfi, _, self.Ppreds_kfi, _ = fsf.kalman_filter_indep(
            m_0,
            P_0,
            model.M,
            PHI_obs,
            model.Sigma_eta[0,0],
            model.sigma2_eps,
            ztildes,
            likelihood="full",
        )
        self.ll_sq, self.ms_sq, self.Us_sq, _, self.Upreds_sq, _ = fsf.sqrt_filter(
            m_0,
            jnp.linalg.cholesky(P_0),
            model.M,
            PHI_obs,
            model.Sigma_eta,
            model.sigma2_eps*jnp.eye(nobs),
            ztildes,
            likelihood="full",
        )
        self.ll_sqi, self.ms_sqi, self.Us_sqi, _, self.Upreds_sqi, _ = fsf.sqrt_filter_indep(
            m_0,
            jnp.linalg.cholesky(P_0),
            model.M,
            PHI_obs,
            model.Sigma_eta[0,0],
            model.sigma2_eps,
            ztildes,
            likelihood="full",
        )
        self.Ppreds_sqi = jnp.matmul(jnp.transpose(
            self.Upreds_sqi, (0, 2, 1)), self.Upreds_sqi)

        self.ll_if, self.nus_if, self.Qs_if, _, _ = fsf.information_filter(
            nu_0,
            Q_0,
            model.M,
            PHI_obs_tuple,
            Sigma_eta,
            Sigma_eps_tuple,
            ztildes_tuple,
            likelihood = "full",
        )
        self.ms_if = jnp.linalg.solve(
            self.Qs_if, self.nus_if[..., None]).squeeze(-1)
        self.Qpreds_kf = jnp.linalg.inv(self.Ppreds_kf)
        self.ll_ifi, self.nus_ifi, self.Qs_ifi, _, self.Qpreds_ifi = fsf.information_filter_indep(
            nu_0,
            Q_0,
            model.M,
            PHI_obs_tuple,
            sigma2_eta,
            sigma2_eps,
            ztildes_tuple,
            likelihood = "full",
        )
        self.ms_ifi = jnp.linalg.solve(
            self.Qs_ifi, self.nus_ifi[..., None]).squeeze(-1)
        
        self.ll_sqif, self.nus_sqif, self.Rs_sqif, _, _ = fsf.sqrt_information_filter(
            nu_0,
            jnp.linalg.cholesky(Q_0),
            model.M,
            PHI_obs_tuple,
            Sigma_eta,
            Sigma_eps_tuple,
            ztildes_tuple,
            likelihood = "full",
        )
        self.Qs_sqif = jnp.matmul(jnp.transpose(
            self.Rs_sqif, (0, 2, 1)), self.Rs_sqif)
        self.ms_sqif = jnp.linalg.solve(
            self.Qs_sqif, self.nus_sqif[..., None]).squeeze(-1)
        
        self.ll_sqifi, self.nus_sqifi, self.Rs_sqifi, _, self.Rpreds_sqifi = fsf.sqrt_information_filter_indep(
            nu_0,
            jnp.linalg.cholesky(Q_0),
            model.M,
            PHI_obs_tuple,
            sigma2_eta,
            sigma2_eps,
            ztildes_tuple,
            likelihood = "full",
        )
        self.Qs_sqifi = jnp.matmul(jnp.transpose(
            self.Rs_sqifi, (0, 2, 1)), self.Rs_sqifi)
        self.ms_sqifi = jnp.linalg.solve(
            self.Qs_sqifi, self.nus_sqifi[..., None]).squeeze(-1)
        self.Qpreds_sqifi = jnp.matmul(jnp.transpose(
            self.Rpreds_sqifi, (0, 2, 1)), self.Rpreds_sqifi)

    def test_shape(self):

        assert (self.nus_if.shape == (self.T, self.nbasis)) & (
            self.Qs_if.shape == (self.T, self.nbasis, self.nbasis)
        ), f"Expected shapes {self.T, self.nbasis} and {(self.T, self.nbasis, self.nbasis)}, got {self.nus.shape} and {self.Qs.shape}."

    def test_means_across_filters(self):

         assert (jnp.allclose(self.ms_kf, self.ms_kfi, atol=1e-03) &
                jnp.allclose(self.ms_kfi, self.ms_sqi, atol=1e-03) &
                jnp.allclose(self.ms_sqi, self.ms_if, atol=1e-03) &
                jnp.allclose(self.ms_if, self.ms_ifi, atol=1e-03) &
                jnp.allclose(self.ms_ifi, self.ms_sqifi, atol=1e-03) &
                jnp.allclose(self.ms_sqifi, self.ms_sq, atol=1e-03) &
                jnp.allclose(self.ms_sq, self.ms_sqif, atol=1e-03)
                )

    def test_Ppreds_across_filters(self):

        assert (jnp.allclose(self.Ppreds_kfi, self.Ppreds_sqi,
                             atol=1e-03))

    def test_Qs_across_ifilters(self):

        assert (jnp.allclose(self.Qpreds_ifi, self.Qpreds_sqifi,
                             atol=1e-03))

    def test_lls(self):

        assert (jnp.allclose(self.ll_kf, self.ll_kfi, atol=1e-03) &
                jnp.allclose(self.ll_kfi, self.ll_sqi, atol=1e-03) &
                jnp.allclose(self.ll_sqi, self.ll_if, atol=1e-03) &
                jnp.allclose(self.ll_if, self.ll_ifi, atol=1e-03) &
                jnp.allclose(self.ll_ifi, self.ll_sqifi, atol=1e-03)
                )

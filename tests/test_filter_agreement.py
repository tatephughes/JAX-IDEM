import pytest
import jax.numpy as jnp

from jaxidem.filters.kal_filter import kal_filter
from jaxidem.filters.skal_filter import skal_filter
from jaxidem.filters.ikal_filter import ikal_filter
from jaxidem.filters.inf_filter import inf_filter
from jaxidem.filters.sqrt_filter import sqrt_filter
from jaxidem.filters.sqinf_filter import sqinf_filter
from jaxidem.filters.pkal_filter import pkal_filter
from jaxidem.filters.spkal_filter import spkal_filter


def invert_information(nus, Qs):
    """Convert information form to state form: m = Q⁻¹ ν, P = Q⁻¹"""
    ms = [jnp.linalg.solve(Q, nu) for nu, Q in zip(nus, Qs)]
    Ps = [jnp.linalg.inv(Q) for Q in Qs]
    return jnp.stack(ms), jnp.stack(Ps)


@pytest.fixture
def shared_inputs():
    r, n, T = 2, 3, 5
    m_0 = jnp.zeros(r)
    P_0 = jnp.eye(r)
    nu_0 = jnp.zeros(r)
    Q_0 = jnp.eye(r)
    M = jnp.eye(r)
    PHI = jnp.ones((n, r))
    zs_tree = [jnp.ones(n) * t for t in range(T)]
    S2_eta = jnp.array([0.1])
    S2_eps = jnp.array([0.05])
    return dict(
        m_0=m_0,
        P_0=P_0,
        nu_0=nu_0,
        Q_0=Q_0,
        M=M,
        PHI=PHI,
        PHI_tree=[PHI] * T,
        zs_tree=zs_tree,
        S2_eta=S2_eta,
        S2_eps=S2_eps,
        S2_eps_tree=[S2_eps] * T,
        S2_eta_shape=0,
        S2_eps_shape=0,
        forecast=0,
        likelihood="full",
    )


def test_all_filters_agree(shared_inputs):
    # 1) Standard Kalman
    kf = kal_filter(
        m_0=shared_inputs["m_0"],
        P_0=shared_inputs["P_0"],
        M=shared_inputs["M"],
        PHI=shared_inputs["PHI"],
        S2_eta=shared_inputs["S2_eta"],
        S2_eps=shared_inputs["S2_eps"],
        zs_tree=shared_inputs["zs_tree"],
        S2_eta_shape=shared_inputs["S2_eta_shape"],
        S2_eps_shape=shared_inputs["S2_eps_shape"],
        forecast=shared_inputs["forecast"],
        likelihood=shared_inputs["likelihood"],
    )
    ms_kf, Ps_kf = kf["ms"], kf["Ps"]
    ll_kf = kf["ll"]

    # 1.5) Stabilised Kalman
    skf = skal_filter(
        m_0=shared_inputs["m_0"],
        P_0=shared_inputs["P_0"],
        M=shared_inputs["M"],
        PHI=shared_inputs["PHI"],
        S2_eta=shared_inputs["S2_eta"],
        S2_eps=shared_inputs["S2_eps"],
        zs_tree=shared_inputs["zs_tree"],
        S2_eta_shape=shared_inputs["S2_eta_shape"],
        S2_eps_shape=shared_inputs["S2_eps_shape"],
        forecast=shared_inputs["forecast"],
        likelihood=shared_inputs["likelihood"],
    )
    ms_skf, Ps_skf = skf["ms"], skf["Ps"]
    ll_skf = skf["ll"]

    # 1.5.5) Pseudo-information filter
    ikf = ikal_filter(
        m_0=shared_inputs["m_0"],
        P_0=shared_inputs["P_0"],
        M=shared_inputs["M"],
        PHI_tree=shared_inputs["PHI_tree"],
        S2_eta=shared_inputs["S2_eta"],
        S2_eps_tree=shared_inputs["S2_eps_tree"],
        zs_tree=shared_inputs["zs_tree"],
        S2_eta_shape=shared_inputs["S2_eta_shape"],
        S2_eps_shape=shared_inputs["S2_eps_shape"],
        forecast=shared_inputs["forecast"],
        likelihood=shared_inputs["likelihood"],
    )
    ms_ikf, Ps_ikf = ikf["ms"], ikf["Ps"]
    ll_ikf = ikf["ll"]

    # 2) Information Filter
    inf = inf_filter(
        nu_0=shared_inputs["nu_0"],
        Q_0=shared_inputs["Q_0"],
        M=shared_inputs["M"],
        PHI_tree=shared_inputs["PHI_tree"],
        S2_eta=shared_inputs["S2_eta"],
        S2_eps_tree=shared_inputs["S2_eps_tree"],
        zs_tree=shared_inputs["zs_tree"],
        S2_eta_shape=shared_inputs["S2_eta_shape"],
        S2_eps_shape=shared_inputs["S2_eps_shape"],
        forecast=shared_inputs["forecast"],
        likelihood=shared_inputs["likelihood"],
    )
    ms_inf, Ps_inf = invert_information(inf["nus"], inf["Qs"])
    ll_inf = inf["ll"]

    # 3) Square-Root Kalman Filter
    sq = sqrt_filter(
        m_0=shared_inputs["m_0"],
        U_0=jnp.linalg.cholesky(shared_inputs["P_0"]),
        M=shared_inputs["M"],
        PHI=shared_inputs["PHI"],
        S2_eta=shared_inputs["S2_eta"],
        S2_eps=shared_inputs["S2_eps"],
        zs_tree=shared_inputs["zs_tree"],
        S2_eta_shape=shared_inputs["S2_eta_shape"],
        S2_eps_shape=shared_inputs["S2_eps_shape"],
        forecast=shared_inputs["forecast"],
        likelihood=shared_inputs["likelihood"],
    )
    ms_sq, Us = sq["ms"], sq["Us"]
    # Reconstruct covariances: P = U @ U.T
    # Ps_sq = jnp.einsum("tij,tkj->tik", Us, Us)
    # Ps_sq = Us.T @ Us
    Ps_sq = jnp.stack([U.T @ U for U in Us])
    ll_sq = sq["ll"]

    # 4) Square-Root Information Filter
    sqinf = sqinf_filter(
        nu_0=shared_inputs["nu_0"],
        R_0=jnp.linalg.cholesky(shared_inputs["Q_0"]),
        M=shared_inputs["M"],
        PHI_tree=shared_inputs["PHI_tree"],
        S2_eta=shared_inputs["S2_eta"],
        S2_eps_tree=shared_inputs["S2_eps_tree"],
        zs_tree=shared_inputs["zs_tree"],
        S2_eta_shape=shared_inputs["S2_eta_shape"],
        S2_eps_shape=shared_inputs["S2_eps_shape"],
        forecast=shared_inputs["forecast"],
        likelihood=shared_inputs["likelihood"],
    )
    # Convert R (Cholesky of info) to Q = R @ R.T, then invert
    Rs = sqinf["Rs"]
    # Qs_sqinf = jnp.einsum("tij,tjk->tik", Rs, Rs)
    # Qs_sqinf = Rs.T @ Rs
    Qs_sqinf = jnp.stack([R.T @ R for R in Rs])
    ms_sqinf, Ps_sqinf = invert_information(sqinf["nus"], Qs_sqinf)
    ll_sqinf = sqinf["ll"]

    # 5) Parallel Kalman Filter
    pk = pkal_filter(
        m_0=shared_inputs["m_0"],
        P_0=shared_inputs["P_0"],
        M=shared_inputs["M"],
        PHI_tree=shared_inputs["PHI_tree"],
        S2_eta=shared_inputs["S2_eta"],
        S2_eps_tree=shared_inputs["S2_eps_tree"],
        zs_tree=shared_inputs["zs_tree"],
        S2_eta_shape=shared_inputs["S2_eta_shape"],
        S2_eps_shape=shared_inputs["S2_eps_shape"],
        likelihood=shared_inputs["likelihood"],
    )
    ms_pk, Ps_pk = pk["ms"], pk["Ps"]
    ll_pk = pk["ll"]

    # 5) Parallel Kalman Filter
    spk = spkal_filter(
        m_0=shared_inputs["m_0"],
        P_0=shared_inputs["P_0"],
        M=shared_inputs["M"],
        PHI_tree=shared_inputs["PHI_tree"],
        S2_eta=shared_inputs["S2_eta"],
        S2_eps_tree=shared_inputs["S2_eps_tree"],
        zs_tree=shared_inputs["zs_tree"],
        S2_eta_shape=shared_inputs["S2_eta_shape"],
        S2_eps_shape=shared_inputs["S2_eps_shape"],
        likelihood=shared_inputs["likelihood"],
    )
    ms_spk, Ps_spk = spk["ms"], spk["Ps"]
    ll_spk = spk["ll"]

    # Now compare all to the standard Kalman outputs
    for name, (ms_i, Ps_i, ll_i) in {
        "skf": (ms_skf, Ps_skf, ll_skf),
        "ikf": (ms_ikf, Ps_ikf, ll_ikf),
        "inf": (ms_inf, Ps_inf, ll_inf),
        "sqrt": (ms_sq, Ps_sq, ll_sq),
        "sqinf": (ms_sqinf, Ps_sqinf, ll_sqinf),
        "pkal": (ms_pk, Ps_pk, ll_pk),
        "spkal": (ms_spk, Ps_spk, ll_spk),
    }.items():
        # means
        assert jnp.allclose(ms_kf, ms_i, atol=1e-5), f"{name} means differ"
        # covariances
        assert jnp.allclose(Ps_kf, Ps_i, atol=1e-5), f"{name} covariances differ"
        # log-likelihood
        assert jnp.allclose(ll_kf, ll_i, atol=1e-5), f"{name} log-likelihood differs"

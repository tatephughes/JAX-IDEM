import pytest
import jax.numpy as jnp
from jaxidem.filters import kal_filter


@pytest.fixture
def simple_kf_inputs():
    r, n, T = 2, 3, 5
    m_0 = jnp.zeros(r)
    P_0 = jnp.eye(r)
    M = jnp.eye(r)
    PHI = jnp.ones((n, r))
    S2_eta = jnp.array([0.1])  # scalar process noise
    S2_eps = jnp.array([0.05])  # scalar obs noise
    zs_tree = [jnp.ones(n) * t for t in range(T)]
    return dict(
        m_0=m_0,
        P_0=P_0,
        M=M,
        PHI=PHI,
        S2_eta=S2_eta,
        S2_eps=S2_eps,
        zs_tree=zs_tree,
        S2_eta_shape=0,
        S2_eps_shape=0,
    )


@pytest.mark.parametrize("likelihood", ["none", "partial", "full"])
def test_kalman_filter_shapes_and_types(simple_kf_inputs, likelihood):
    forecast_steps = 2
    results = kal_filter(
        **simple_kf_inputs,
        forecast=forecast_steps,
        likelihood=likelihood,
    )

    # Required keys
    expected_keys = {"ll", "ms", "Ps", "m_preds", "P_preds", "m_forecast", "P_forecast"}
    assert expected_keys == set(results.keys())

    # ll type
    assert isinstance(results["ll"], (float, jnp.ndarray))

    # ms: PyTree[Array, "T r"]
    assert isinstance(results["ms"], jnp.ndarray)
    assert results["ms"].shape[1] == simple_kf_inputs["PHI"].shape[1]  # r

    # Ps: PyTree[Array, "T r r"]
    assert results["Ps"].shape[1:] == (simple_kf_inputs["PHI"].shape[1],) * 2


@pytest.mark.parametrize("forecast_steps", [0, 2])
def test_kal_filter_forecast_behavior(simple_kf_inputs, forecast_steps):
    results = kal_filter(
        **simple_kf_inputs,
        forecast=forecast_steps,
        likelihood="partial",
    )

    assert results["m_forecast"].shape[0] == forecast_steps
    assert results["P_forecast"].shape[0] == forecast_steps


@pytest.mark.parametrize(
    "shape_code, S2_shape",
    [
        (0, (1,)),  # scalar
        (1, (2,)),  # vector
        (2, (2, 2)),  # matrix
    ],
)
def test_kalman_filter_noise_shapes(simple_kf_inputs, shape_code, S2_shape):
    n = simple_kf_inputs["PHI"].shape[1]
    S2_eta = jnp.ones(S2_shape) * 0.1
    S2_eps = jnp.ones((1,)) * 0.05
    results = kal_filter(
        **{**simple_kf_inputs, "S2_eta": S2_eta, "S2_eta_shape": shape_code},
        forecast=1,
        likelihood="partial",
    )
    assert "ms" in results
    assert "Ps" in results

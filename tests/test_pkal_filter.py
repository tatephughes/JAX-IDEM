import pytest
import jax.numpy as jnp
from jaxidem.filters import pkal_filter


@pytest.fixture
def simple_pkal_inputs():
    r = 2
    T = 4
    m_0 = jnp.zeros(r)
    P_0 = jnp.eye(r)
    M = jnp.eye(r)

    # Variable observation dimensions
    PHI_tree = [jnp.ones((n, r)) for n in [1, 2, 1, 3]]
    zs_tree = [jnp.ones(n) * t for t, n in enumerate([1, 2, 1, 3])]
    S2_eps_tree = [jnp.ones((1,)) for _ in zs_tree]  # scalar noise

    S2_eta = jnp.array([0.1])  # scalar process noise

    return dict(
        m_0=m_0,
        P_0=P_0,
        M=M,
        PHI_tree=PHI_tree,
        S2_eta=S2_eta,
        S2_eps_tree=S2_eps_tree,
        zs_tree=zs_tree,
        S2_eta_shape=0,
        S2_eps_shape=0,
    )


@pytest.mark.parametrize("likelihood", ["none", "partial", "full"])
def test_pkal_filter_shapes_and_keys(simple_pkal_inputs, likelihood):
    results = pkal_filter(
        **simple_pkal_inputs,
        likelihood=likelihood,
    )

    expected_keys = {"ll", "ms", "Ps"}
    assert expected_keys == set(results.keys())

    # ll type
    assert isinstance(results["ll"], (float, jnp.ndarray))

    # ms: PyTree[Array, "T r"]
    assert isinstance(results["ms"], jnp.ndarray)
    assert results["ms"].shape[1] == simple_pkal_inputs["M"].shape[1]  # r

    # Ps: PyTree[Array, "T r r"]
    assert results["Ps"].shape[1:] == (simple_pkal_inputs["M"].shape[1],) * 2


@pytest.mark.parametrize(
    "shape_code, S2_shape",
    [
        (0, (1,)),  # scalar
        (1, (2,)),  # vector
        (2, (2, 2)),  # matrix
    ],
)
def test_pkal_filter_process_noise_shapes(simple_pkal_inputs, shape_code, S2_shape):
    r = simple_pkal_inputs["m_0"].shape[0]
    S2_eta = jnp.ones(S2_shape) * 0.1
    results = pkal_filter(
        **{**simple_pkal_inputs, "S2_eta": S2_eta, "S2_eta_shape": shape_code},
        likelihood="partial",
    )
    assert "ms" in results
    assert "Ps" in results


@pytest.mark.parametrize("likelihood", ["none", "partial", "full"])
def test_pkal_filter_with_empty_observation_step(simple_pkal_inputs, likelihood):
    """
    Edge case: one time point has no data at all (n = 0).
    """
    r = simple_pkal_inputs["m_0"].shape[0]

    # Variable observation dimensions, including one with n = 0
    PHI_tree = [jnp.ones((n, r)) for n in [1, 2, 0, 1, 3]]
    zs_tree = [jnp.ones(n) * t for t, n in enumerate([1, 2, 0, 1, 3])]
    S2_eps_tree = [jnp.ones((1,)) for _ in zs_tree]  # scalar noise for each step

    inputs = {
        **simple_pkal_inputs,
        "PHI_tree": PHI_tree,
        "zs_tree": zs_tree,
        "S2_eps_tree": S2_eps_tree,
    }

    results = pkal_filter(
        **inputs,
        likelihood=likelihood,
    )

    # Basic key checks
    expected_keys = {"ll", "ms", "Ps"}
    assert expected_keys == set(results.keys())

    # Shape checks
    assert results["ms"].shape[0] == len(zs_tree)
    assert results["Ps"].shape[0] == len(zs_tree)

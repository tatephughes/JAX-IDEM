import pytest
import jax.numpy as jnp
from jaxidem.filters import sqinf_filter


@pytest.fixture
def simple_sqinf_inputs():
    r = 2
    T = 4
    nu_0 = jnp.zeros(r)
    R_0 = jnp.eye(r)  # Cholesky factor of information matrix
    M = jnp.eye(r)

    # Variable observation dimensions
    PHI_tree = [jnp.ones((n, r)) for n in [1, 2, 1, 3]]
    zs_tree = [jnp.ones(n) * t for t, n in enumerate([1, 2, 1, 3])]
    S2_eps_tree = [jnp.ones((1,)) for _ in zs_tree]  # scalar noise

    S2_eta = jnp.array([0.1])  # scalar process noise

    return dict(
        nu_0=nu_0,
        R_0=R_0,
        M=M,
        PHI_tree=PHI_tree,
        S2_eta=S2_eta,
        S2_eps_tree=S2_eps_tree,
        zs_tree=zs_tree,
        S2_eta_shape=0,
        S2_eps_shape=0,
    )

# Aight so this pytest does NOT check for nans, which absolutely happen here.
# Problem is, sqinf filter downright FAILS for n<r, which is EXPECTED BEHAVIOUR

@pytest.mark.parametrize("likelihood", ["none", "partial", "full"])
def test_sqinf_filter_shapes_and_keys(simple_sqinf_inputs, likelihood):
    forecast_steps = 2
    results = sqinf_filter(
        **simple_sqinf_inputs,
        forecast=forecast_steps,
        likelihood=likelihood,
    )

    expected_keys = {
        "ll",
        "nus",
        "Rs",
        "nu_preds",
        "R_preds",
        "nu_forecast",
        "R_forecast",
    }
    assert expected_keys == set(results.keys())

    # ll type
    assert isinstance(results["ll"], (float, jnp.ndarray))

    # nus: PyTree[Array, "T r"]
    assert isinstance(results["nus"], jnp.ndarray)
    assert results["nus"].shape[1] == simple_sqinf_inputs["M"].shape[1]  # r

    # Rs: PyTree[Array, "T r r"]
    assert results["Rs"].shape[1:] == (simple_sqinf_inputs["M"].shape[1],) * 2


@pytest.mark.parametrize("forecast_steps", [0, 2])
def test_sqinf_filter_forecast_behavior(simple_sqinf_inputs, forecast_steps):
    results = sqinf_filter(
        **simple_sqinf_inputs,
        forecast=forecast_steps,
        likelihood="partial",
    )

    assert results["nu_forecast"].shape[0] == forecast_steps
    assert results["R_forecast"].shape[0] == forecast_steps


@pytest.mark.parametrize(
    "shape_code, S2_shape",
    [
        (0, (1,)),  # scalar
        (1, (2,)),  # vector
        (2, (2, 2)),  # matrix
    ],
)
def test_sqinf_filter_process_noise_shapes(simple_sqinf_inputs, shape_code, S2_shape):
    r = simple_sqinf_inputs["nu_0"].shape[0]
    S2_eta = jnp.ones(S2_shape) * 0.1
    results = sqinf_filter(
        **{**simple_sqinf_inputs, "S2_eta": S2_eta, "S2_eta_shape": shape_code},
        forecast=1,
        likelihood="partial",
    )
    assert "nus" in results
    assert "Rs" in results


@pytest.mark.parametrize("likelihood", ["none", "partial", "full"])
def test_sqinf_filter_with_empty_observation_step(simple_sqinf_inputs, likelihood):
    """
    Edge case: one time point has no data at all (n = 0).
    """
    r = simple_sqinf_inputs["nu_0"].shape[0]

    # Variable observation dimensions, including one with n = 0
    PHI_tree = [jnp.ones((n, r)) for n in [1, 2, 0, 1, 3]]
    zs_tree = [jnp.ones(n) * t for t, n in enumerate([1, 2, 0, 1, 3])]
    S2_eps_tree = [jnp.ones((1,)) for _ in zs_tree]  # scalar noise for each step

    inputs = {
        **simple_sqinf_inputs,
        "PHI_tree": PHI_tree,
        "zs_tree": zs_tree,
        "S2_eps_tree": S2_eps_tree,
    }

    results = sqinf_filter(
        **inputs,
        forecast=1,
        likelihood=likelihood,
    )

    # Basic key checks
    expected_keys = {
        "ll",
        "nus",
        "Rs",
        "nu_preds",
        "R_preds",
        "nu_forecast",
        "R_forecast",
    }
    assert expected_keys == set(results.keys())

    # Shape checks
    assert results["nus"].shape[0] == len(zs_tree)
    assert results["Rs"].shape[0] == len(zs_tree)
    assert results["nu_forecast"].shape[0] == 1
    assert results["R_forecast"].shape[0] == 1

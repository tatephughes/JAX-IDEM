# JAX imports
import jax
import jax.numpy as jnp
import jax.scipy as jsc
import jax.lax as jl
from jax.scipy.linalg import solve_triangular as st
from jax.scipy.linalg import solve

# Typing imports
from jaxtyping import ArrayLike, PyTree, Array, Float
from typing import Tuple, Union, Literal, TypedDict

# Utility imports
from functools import partial
from jaxidem.utils import add_variance


class KalmanResults(TypedDict):
    ll: float
    ms: PyTree[Float[Array, "T r"]]
    Ps: PyTree[Float[Array, "T r r"]]
    m_preds: PyTree[Float[Array, "T r"]]
    P_preds: PyTree[Float[Array, "T r r"]]
    m_forecast: Float[Array, "forecast r"]
    P_forecast: Float[Array, "forecast r r"]


@partial(
    jax.jit,
    static_argnames=["S2_eta_shape", "S2_eps_shape", "forecast", "likelihood"],
)
def skal_filter(
    m_0: Float[Array, "r"],
    P_0: Float[Array, "r r"],
    M: Float[Array, "r r"],
    PHI: Float[Array, "n r"],
    S2_eta: Union[Float[Array, "1"], Float[Array, "r"], Float[Array, "r r"]],
    S2_eps: Union[Float[Array, "1"], Float[Array, "n"], Float[Array, "n n"]],
    zs_tree: PyTree[Float[Array, "n"]],
    S2_eta_shape: int,
    S2_eps_shape: int,
    forecast: int = 0,
    likelihood: Literal["none", "partial", "full"] = "partial",
) -> KalmanResults:
    """
    The Joseph-stabilised Kalman Filter.

    Small change to the update step of the Kalman filter can reduce numerical instability.

    Parameters
    ----------
    m_0: Float[Array, "r"]
        Prior mean of the filter at T=0.
    P_0: Float[Array, "r r"]
        Prior Variance of the filter at T=0.
    M: Float[Array, "r r"]
        State-Transition Matrix.
    PHI: Float[Array, "n r"]
        Observation Matrix.
    S2_eta: Float[Array, "1"] | Float[Array, "r"] | Float[Array, "r r"]
        Process noise variance. Can be of shape (1,) (for scalar/i.i.d.), shape (r,) (for uncorrelated noise), or shape (r, r) (for correlated noise).
    S2_eps: Float[Array, "1"] | Float[Array, "n"] | Float[Array, "n n"]
        Observation noise variance. Can be of shape (1,) (for scalar/i.i.d. noise), shape (n,) (for uncorrelated noise), or shape (n, n) (for correlated noise).
    zs_tree: PyTree[Float[Array, "n"]]
        The observed data values. For the Kalman filter, each observation must have the same dimension and use the same observation matrix. For varying observation dimension and observation matrix, use the information filter, square-root information filter, of temporally parallel Kalman filter.
    S2_eta_shape: int
        Shape code for `S2_eta` (needed for JIT compilation); 0 is scalar/i.i.d., 1 is vector/uncorrelated, 2 is matrix/correlated. Must match the dimensionality of `S2_eta`.
    S2_eps_shape: int
        Shape code for `S2_eps` (needed for JIT compilation); 0 is scalar/i.i.d., 1 is vector/uncorrelated, 2 is matrix/correlated. Must match the dimensionality of `S2_eps`.
    forecast: int
        The number of time points to forecast ahead.
    likelihood: Literal["none", "partial", "full"]
        Log-likelihood computation mode. Must be one of 'none'; No likelihood is computed, 'partial'; Likelihood computed up to an additive constant 'full'; Full likelihood is computed
        Default is 'partial'.

    Returns
    ----------
    filt_results: KalmanResults
        For details see notes below

    Notes
    -----

    The return type is KalmanResults, containing:
        - `ll` : float
            - Log-likelihood value (possibly up to an additive constant).
        - `ms` : PyTree[Float[Array, "T r"]]
            - Posterior means at each time step.
        - `Ps` : PyTree[Float[Array, "T r r"]]
            - Posterior covariance matrices at each time step.
        - `mpreds` : PyTree[Float[Array, "T r"]]
            - Predicted means at each time step.
        - `Ppreds` : PyTree[Float[Array, "T r r"]]
            - Predicted covariance matrices at each time step.
        - `mforecast` : Float[Array, "forecast r"]
            - Forecast means.
        - `Pforecast` : Float[Array, "forecast r r"]
            - Forecast covariance matrices.


    Symbolic dimensions:

       - `r` : state dimension
       - `n` : observation dimension
       - `T` : number of time steps
       - `forecast` : number of forecast steps
    """

    r = m_0.size
    n = zs_tree[0].size

    @jax.jit
    def step(carry, z_t):
        m_tt, P_tt, _, _, ll, _ = carry

        m_pred = M @ m_tt
        P_pred = add_variance(M @ P_tt @ M.T, S2_eta, S2_eta_shape)

        e_t = z_t - PHI @ m_pred
        Sigma_t = add_variance(PHI @ P_pred @ PHI.T, S2_eps, S2_eps_shape)
        K_t = (solve(Sigma_t, PHI, assume_a="pos") @ P_pred.T).T

        m_up = m_pred + K_t @ e_t

        match S2_eps_shape:
            case 0:
                ksk = S2_eps * K_t @ K_t.T
            case 1:
                ksk = K_t @ jnp.diag(S2_eps) @ K_t.T
            case 2:
                ksk = K_t @ S2_eps @ K_t.T

        P_up = (jnp.eye(r) - K_t @ PHI) @ P_pred @ (jnp.eye(r) - K_t @ PHI).T + ksk

        if likelihood == "full":
            Ui_t = jnp.linalg.cholesky(Sigma_t)
            s = st(Ui_t, e_t, lower=True)
            ll_new = (
                ll
                - jnp.sum(jnp.log(jnp.diag(Ui_t)))
                - 0.5 * jnp.dot(s, s)
                - 0.5 * n * jnp.log(2 * jnp.pi)
            )
        elif likelihood == "partial":
            Ui_t = jnp.linalg.cholesky(Sigma_t)
            s = st(Ui_t, e_t, lower=True)
            ll_new = ll - jnp.sum(jnp.log(jnp.diag(Ui_t))) - 0.5 * jnp.dot(s, s)
        elif likelihood == "none":
            ll_new = jnp.nan
        else:
            raise ValueError(
                "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
            )

        return (m_up, P_up, m_pred, P_pred, ll_new, K_t), (
            m_up,
            P_up,
            m_pred,
            P_pred,
            ll_new,
            K_t,
        )

    carry, seq = jl.scan(
        step,
        (m_0, P_0, m_0, P_0, 0, jnp.zeros((r, n))),
        jnp.column_stack(zs_tree).T,
    )

    ll, ms, Ps, mpreds, Ppreds, Ks = (carry[4], seq[0], seq[1], seq[2], seq[3], seq[5])

    def forecast_step(carry, x):
        m_tt, P_tt = carry

        # predict
        m_pred = M @ m_tt
        P_prop = M @ P_tt @ M.T

        match S2_eta_shape:
            case 0 | 1:
                P_pred = jnp.fill_diagonal(
                    P_prop, S2_eta + jnp.diag(P_prop), inplace=False
                )
            case 2:
                P_pred = P_prop + S2_eta

        return (m_pred, P_pred), (m_pred, P_pred)

    carry_fc, seq_fc = jl.scan(
        forecast_step,
        (ms[-1], Ps[-1]),
        jnp.arange(forecast),
    )

    filt_results = KalmanResults(
        ll=ll,
        ms=ms,
        Ps=Ps,
        m_preds=mpreds,
        P_preds=Ppreds,
        m_forecast=seq_fc[0],
        P_forecast=seq_fc[1],
    )

    return filt_results

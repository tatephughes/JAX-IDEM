# JAX imports
import jax
import jax.numpy as jnp
import jax.scipy as jsc
import jax.lax as jl
from jax.scipy.linalg import solve_triangular as st

# Typing imports
from jaxtyping import ArrayLike, PyTree, Array, Float
from typing import Tuple, Union, Literal, TypedDict

# Utility imports
from functools import partial
from jaxidem.utils import qr_R


class SqrtResults(TypedDict):
    ll: float
    ms: PyTree[Float[Array, "T r"]]
    Us: PyTree[Float[Array, "T r r"]]
    m_preds: PyTree[Float[Array, "T r"]]
    U_preds: PyTree[Float[Array, "T r r"]]
    m_forecast: Float[Array, "forecast r"]
    U_forecast: Float[Array, "forecast r r"]


@partial(
    jax.jit,
    static_argnames=["S2_eta_shape", "S2_eps_shape", "forecast", "likelihood"],
)
def sqrt_filter(
    m_0: Float[Array, "r"],
    U_0: Float[Array, "r r"],
    M: Float[Array, "r r"],
    PHI: Float[Array, "n r"],
    S2_eta: Float[Array, "1"] | Float[Array, "r"] | Float[Array, "r r"],
    S2_eps: Float[Array, "1"] | Float[Array, "n"] | Float[Array, "n n"],
    zs_tree: PyTree[Float[Array, "n"]],
    S2_eta_shape: int,
    S2_eps_shape: int,
    forecast: int = 0,
    likelihood: Literal["none", "partial", "full"] = "full",
) -> SqrtResults:
    """
    The Square-Root Kalman Filter.

    Parameters
    ----------
    m_0: Float[Array, "r"]
        Prior mean of the filter at T=0.

    U_0: Float[Array, "r r"]
        Prior upper-triangular Cholesky factor of the state covariance at T=0.

    M: Float[Array, "r r"]
        State-transition matrix.

    PHI: Float[Array, "n r"]
        Observation matrix.

    S2_eta: Float[Array, "1"] | Float[Array, "r"] | Float[Array, "r r"]
        Process noise variance. Can be of shape (1,) (for scalar/i.i.d.), shape (r,) (for uncorrelated noise), or shape (r, r) (for correlated noise).

    S2_eps: Float[Array, "1"] | Float[Array, "n"] | Float[Array, "n n"]
        Observation noise variance. Can be of shape (1,) (for scalar/i.i.d. noise), shape (n,) (for uncorrelated noise), or shape (n, n) (for correlated noise).

    zs_tree: PyTree[Float[Array, "n"]]
        Observed data values. For the square-root filter, each observation must have the same dimension and use the same observation matrix.

    S2_eta_shape: int
        Shape code for `S2_eta` (needed for JIT compilation); 0 is scalar/i.i.d., 1 is vector/uncorrelated, 2 is matrix/correlated. Must match the dimensionality of `S2_eta`.

    S2_eps_shape: int
        Shape code for `S2_eps` (needed for JIT compilation); 0 is scalar/i.i.d., 1 is vector/uncorrelated, 2 is matrix/correlated. Must match the dimensionality of `S2_eps`.

    forecast: int
        Number of time points to forecast ahead.

    likelihood: Literal["none", "partial", "full"]
        Log-likelihood computation mode. Must be one of 'none' (no likelihood is computed), 'partial' (likelihood computed up to an additive constant), or 'full' (full likelihood is computed). Default is 'full'.

    Returns
    ----------
    filt_results: SqrtResults
        For details see Notes below.

    Notes
    -----
    The return type is SqrtResults, containing:

        - `ll` : float
          Log-likelihood value (possibly up to an additive constant).

        - `ms` : PyTree[Float[Array, "T r"]]
          Posterior means at each time step.

        - `Us` : PyTree[Float[Array, "T r r"]]
          Posterior square-roots of the state covariance at each time step.
          (Small note; these are not necessarily the cholesky factors!)

        - `m_preds` : PyTree[Float[Array, "T r"]]
          Predicted means at each time step.

        - `U_preds` : PyTree[Float[Array, "T r r"]]
          Predicted square-roots of the state covariance at each time step.

        - `m_forecast` : Float[Array, "forecast r"]
          Forecast means.

        - `U_forecast` : Float[Array, "forecast r r"]
          Forecast square-roots of the state covariance.

    Symbolic dimensions:

        - `r` : state dimension
        - `n` : observation dimension
        - `T` : number of time steps
        - `forecast` : number of forecast steps
    """

    r = m_0.shape[0]
    nobs = zs_tree[0].size

    # Force the variances to be matrices.
    # Might not be the best way to do this, of course.
    # But feeding these into qr_R is easiest for now.
    match S2_eta_shape:
        case 0:
            sigma_eta = jnp.sqrt(S2_eta) * jnp.eye(r)
        case 1:
            sigma_eta = jnp.diag(jnp.sqrt(S2_eta))
        case 2:
            sigma_eta = jnp.linalg.cholesky(S2_eta)

    match S2_eps_shape:
        case 0:
            sigma_eps = jnp.sqrt(S2_eps) * jnp.eye(nobs)
        case 1:
            sigma_eps = jnp.diag(jnp.sqrt(S2_eps))
        case 2:
            sigma_eps = jnp.linalg.cholesky(S2_eps)

    @jax.jit
    def step(carry, z_t):
        m_tt, U_tt, _, _, ll, _ = carry

        m_pred = M @ m_tt
        U_pred = qr_R(U_tt @ M.T, sigma_eta)

        e_t = z_t - PHI @ m_pred

        # Ui_t = jnp.linalg.qr(jnp.vstack([U_pred @ PHI.T, U_eps]), mode="r")
        Ui_t = qr_R(U_pred @ PHI.T, sigma_eps)
        K_t = (
            st(
                Ui_t,
                st(Ui_t.T, PHI, lower=True) @ U_pred.T @ U_pred,
            )
        ).T

        m_up = m_pred + K_t @ e_t
        U_up = qr_R(U_pred @ (jnp.eye(r) - K_t @ PHI).T, sigma_eps @ K_t.T)

        s = st(Ui_t.T, e_t, lower=True)

        match likelihood:
            case "full":
                ll_new = (
                    ll
                    - jnp.sum(jnp.log(jnp.abs(jnp.diag(Ui_t))))
                    - 0.5 * jnp.dot(s, s)
                    - 0.5 * nobs * jnp.log(2 * jnp.pi)
                )
            case "partial":
                ll_new = (
                    ll - jnp.sum(jnp.log(jnp.abs(jnp.diag(Ui_t)))) - 0.5 * jnp.dot(s, s)
                )
            case "none":
                ll_new = jnp.nan
            case _:
                raise ValueError(
                    "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
                )

        return (m_up, U_up, m_pred, U_pred, ll_new, K_t), (
            m_up,
            U_up,
            m_pred,
            U_pred,
            ll_new,
            K_t,
        )

    carry, seq = jl.scan(
        step,
        (m_0, U_0, m_0, U_0, 0, jnp.zeros((r, nobs))),
        jnp.column_stack(zs_tree).T,
    )

    ll, ms, Us, mpreds, Upreds, Ks = (carry[4], seq[0], seq[1], seq[2], seq[3], seq[5])

    def forecast_step(carry, x):
        m_tt, U_tt = carry

        m_pred = M @ m_tt
        U_pred = qr_R(U_tt @ M.T, sigma_eta)

        return (m_pred, U_pred), (m_pred, U_pred)

    carry_fc, seq_fc = jl.scan(
        forecast_step,
        (ms[-1], Us[-1]),
        jnp.arange(forecast),
    )

    filt_results = SqrtResults(
        ll=ll,
        ms=jnp.vstack([m_0[None,:], ms]),
        Us=jnp.vstack([U_0[None,:,:], Us]),
        m_preds=mpreds,
        U_preds=Upreds,
        m_forecast=seq_fc[0],
        U_forecast=seq_fc[1],
    )

    return filt_results


def sqrt_smoother(ms,
                  Us,
                  S2_eta,
                  S2_eta_shape,
                  M,
                  ):

    r = ms.shape[1]
    
    match S2_eta_shape:
        case 0:
            sigma_eta = jnp.sqrt(S2_eta) * jnp.eye(r)
        case 1:
            sigma_eta = jnp.diag(jnp.sqrt(S2_eta))
        case 2:
            sigma_eta = jnp.linalg.cholesky(S2_eta)

    I = jnp.eye(r)
            
    def backsmooth(carry, x):

        m_tpT, U_tpT = carry
        m_tt, U_tt = x

        mpred = M @ m_tt
        Upred = qr_R(U_tt@M.T, sigma_eta)

        C = (st(Upred, st(Upred.T, M @ U_tt.T @ U_tt, lower=True), lower=False)).T

        U_t = qr_R(Upred, sigma_eta)
        
        m_tT = C @ (m_tpT - mpred)
        U_tT = qr_R(U_t @ C.T, U_tt @ (I - C@M).T)

        return (m_tT, U_tT), (m_tT, U_tT)

    xs = (jnp.flip(ms[:-2], axis=0),
          jnp.flip(Us[:-2], axis=0),)
    
    carry, seq = jl.scan(
        backsmooth,
        (ms[-1], Us[-1]),
        xs,
    )

    m_post, U_post = (jnp.vstack([jnp.flip(seq[0], axis=0), ms[None, -1]]),
                      jnp.vstack([jnp.flip(seq[1], axis=0), Us[None, -1]]))

    return m_post, U_post

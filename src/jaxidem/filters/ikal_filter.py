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
def ikal_filter(
    m_0: Float[Array, "r"],
    P_0: Float[Array, "r r"],
    M: Float[Array, "r r"],
    PHI_tree: PyTree[Float[Array, "?n r"]],
    S2_eta: Union[Float[Array, "1"], Float[Array, "r"], Float[Array, "r r"]],
    S2_eps_tree: PyTree[
        Union[Float[Array, "1"], Float[Array, "?n"], Float[Array, "?n ?n"]]
    ],
    zs_tree: PyTree[Float[Array, "n"]],
    S2_eta_shape: int,
    S2_eps_shape: int,
    forecast: int = 0,
    likelihood: Literal["none", "partial", "full"] = "partial",
) -> KalmanResults:
    
    r = m_0.size
    n = zs_tree[0].size

    def informationify(z_k, PHI_k, S2_eps_k):

        match S2_eps_shape:
            case 0 | 1:
                i_k = PHI_k.T @ (z_k / S2_eps_k)
                I_k = PHI_k.T / S2_eps_k @ PHI_k
            case 2:
                i_k = PHI_k.T @ solve(S2_eps_k, z_k, assume_a="pos")
                I_k = PHI_k.T @ solve(S2_eps_k, PHI_k, assume_a="pos")

        return jnp.vstack((i_k, I_k))

    scan_elts = jnp.stack(jax.tree.map(informationify, zs_tree, PHI_tree, S2_eps_tree))

    @jax.jit
    def step(carry, scan_elt):
        m_tt, P_tt, _, _, _ = carry

        i_tp = scan_elt[0, :]
        I_tp = scan_elt[1:, :]

        m_pred = M @ m_tt
        P_pred = add_variance(M @ P_tt @ M.T, S2_eta, S2_eta_shape)

        e_t = i_tp - I_tp @ m_pred
        # Sigma_t = I_tp @ P_pred @ I_tp.T + I_tp
        # K_t = (solve(Sigma_t, I_tp, assume_a="pos") @ P_pred.T).T
        K_t = (solve(P_pred @ I_tp.T + jnp.eye(r), P_pred.T)).T

        m_up = m_pred + K_t @ e_t
        P_up = (jnp.eye(r) - K_t @ I_tp) @ P_pred

        return (m_up, P_up, m_pred, P_pred, K_t), (
            m_up,
            P_up,
            m_pred,
            P_pred,
            K_t,
        )

    carry, seq = jl.scan(
        step,
        (m_0, P_0, m_0, P_0, jnp.zeros((r, r))),
        scan_elts,
    )

    ms, Ps, mpreds, Ppreds, Ks = (seq[0], seq[1], seq[2], seq[3], seq[4])

    if likelihood in ("full", "partial"):

        @jax.jit
        def get_ll(z, PHI, S2_eps, m_pred, P_pred):
            n = z.size

            e = z - PHI @ m_pred

            Sigma_t = add_variance(PHI @ P_pred @ PHI.T, S2_eps, S2_eps_shape)

            Ui_t = jnp.linalg.cholesky(Sigma_t)
            s = st(Ui_t, e, lower=True)

            match likelihood:
                case "full":
                    ll = (
                        -jnp.sum(jnp.log(jnp.diag(Ui_t)))
                        - 0.5 * jnp.dot(s, s)
                        - 0.5 * n * jnp.log(2 * jnp.pi)
                    )
                case "partial":
                    ll = -jnp.sum(jnp.log(jnp.diag(Ui_t))) - 0.5 * jnp.dot(s, s)

            return ll

        lls = jnp.array(jax.tree.map(get_ll, zs_tree, PHI_tree, S2_eps_tree, list(mpreds), list(Ppreds)))
        ll = jnp.sum(lls)

    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

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

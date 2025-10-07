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
from jaxidem.utils import add_variance
from jaxidem.utils import qr_R
from jaxidem.utils import ql_L


class SqinfResults(TypedDict):
    ll: float
    nus: PyTree[Float[Array, "T r"]]
    Rs: PyTree[Float[Array, "T r r"]]
    nu_preds: PyTree[Float[Array, "T r"]]
    R_preds: PyTree[Float[Array, "T r r"]]
    nu_forecast: Float[Array, "forecast r"]
    R_forecast: Float[Array, "forecast r r"]


@partial(
    jax.jit,
    static_argnames=["S2_eta_shape", "S2_eps_shape", "forecast", "likelihood"],
)
def sqinf_filter(
    nu_0: Float[Array, "r"],
    R_0: Float[Array, "r r"],
    M: Float[Array, "r r"],
    PHI_tree: PyTree[Float[Array, "?n r"]],
    S2_eta: Float[Array, "1"] | Float[Array, "r"] | Float[Array, "r r"],
    S2_eps_tree: PyTree[Float[Array, "1"] | Float[Array, "?n"] | Float[Array, "?n ?n"]],
    zs_tree: PyTree[Float[Array, "n"]],
    S2_eta_shape: int,
    S2_eps_shape: int,
    forecast: int = 0,
    likelihood: Literal["none", "partial", "full"] = "partial",
) -> SqinfResults:
    """
    The Square-Root Information Filter.

    Parameters
    ----------
    nu_0: Float[Array, "r"]
        Initial information vector at time T=0.

    R_0: Float[Array, "r r"]
        Initial upper-triangular Cholesky factor of the information matrix at T=0.

    M: Float[Array, "r r"]
        State-transition matrix.

    PHI_tree: PyTree[Float[Array, "?n r"]]
        Observation matrices for each time step. Each leaf corresponds to a time step and may have a different observation dimension `n`.

    S2_eta: Float[Array, "1"] | Float[Array, "r"] | Float[Array, "r r"]
        Process noise variance. Can be of shape (1,) (for scalar/i.i.d.), shape (r,) (for uncorrelated noise), or shape (r, r) (for correlated noise).

    S2_eps_tree: PyTree[Float[Array, "1"] | Float[Array, "?n"] | Float[Array, "?n ?n"]]
        Observation noise variance for each time step. Each leaf may vary in dimension: shape (1,) for scalar/i.i.d. noise, shape (n,) for uncorrelated noise, or shape (n, n) for correlated noise.

    zs_tree: PyTree[Float[Array, "n"]]
        Observed data values. Each leaf corresponds to a time step and may have a different observation dimension `n`.

    S2_eta_shape: int
        Shape code for `S2_eta` (needed for JIT compilation); 0 is scalar/i.i.d., 1 is vector/uncorrelated, 2 is matrix/correlated.

    S2_eps_shape: int
        Shape code for `S2_eps` (needed for JIT compilation); 0 is scalar/i.i.d., 1 is vector/uncorrelated, 2 is matrix/correlated. Must match the dimensionality of each leaf of `S2_eps`.

    forecast: int
        Number of time points to forecast ahead.

    likelihood: Literal["none", "partial", "full"]
        Log-likelihood computation mode. Must be one of 'none' (no likelihood is computed), 'partial' (likelihood computed up to an additive constant), or 'full' (full likelihood is computed). Default is 'partial'.

    Returns
    ----------
    filt_results: SqinfResults
        For details see Notes below.

    Notes
    -----
    The return type is SqinfResults, containing:

        - `ll` : float
          Log-likelihood value (possibly up to an additive constant).

        - `nus` : PyTree[Float[Array, "T r"]]
          Information vectors at each time step.

        - `Rs` : PyTree[Float[Array, "T r r"]]
          Squareroots of the information matrices at each time step.

        - `nupreds` : PyTree[Float[Array, "T r"]]
          Predicted information vectors.

        - `Rpreds` : PyTree[Float[Array, "T r r"]]
          Predicted square-roots of the information matrices.

        - `nuforecast` : Float[Array, "forecast r"]
          Forecasted information vectors.

        - `Rforecast` : Float[Array, "forecast r r"]
          Forecasted square-roots of the information matrices.

    Symbolic dimensions:

        - `r` : state dimension
        - `n` : observation dimension (may vary per time step)
        - `T` : number of time steps
        - `forecast` : number of forecast steps

    Note that PHI_tree must all have full row rank.
    """

    r = nu_0.shape[0]

    def informationify(z_k, PHI_k, S2_eps_k):

        # THIS WILL FAIL if S2_eps_shape!=0 and there is a time point with no data.
        match S2_eps_shape:
            case 0:
                i_k = PHI_k.T @ z_k / S2_eps_k
                # Below cholesky _should_ be faster, but is much less stable; why?
                # R_k = safe_qr(PHI_k) / jnp.sqrt(S2_eps_k)
                # R_k = jnp.linalg.qr((PHI_k), mode="r") / jnp.sqrt(S2_eps_k)
                # R_k = safe_cholesky(PHI_k.T @ PHI_k / S2_eps_k)
                R_k = jnp.linalg.cholesky(PHI_k.T @ PHI_k / S2_eps_k, upper=True)
            case 1:
                sigma_eps = jnp.diag(jnp.sqrt(S2_eps_k))
                i_k = PHI_k.T @ st(
                    sigma_eps, st(sigma_eps.T, z_k, lower=False), lower=True
                )
                R_k = jnRp.linalg.qr(st(sigma_eps.T, PHI_k, lower=False), mode="r")
            case 2:
                sigma_eps = jnp.linalg.cholesky(S2_eps_k)
                i_k = PHI_k.T @ st(
                    sigma_eps, st(sigma_eps.T, z_k, lower=False), lower=True
                )
                R_k = jnp.linalg.qr(st(sigma_eps.T, PHI_k, lower=False), mode="r")

        return jnp.vstack((i_k, R_k))

    scan_elts = jnp.array(jax.tree.map(informationify, zs_tree, PHI_tree, S2_eps_tree))

    # Minv = jnp.linalg.solve(M, jnp.eye(r))
    match S2_eta_shape:
        case 0:
            sigma_eta = jnp.sqrt(S2_eta) * jnp.eye(r)
        case 1:
            sigma_eta = jnp.diag(jnp.sqrt(S2_eta))
        case 2:
            sigma_eta = jnp.linalg.cholesky(S2_eta)

    def step(carry, scan_elt):
        nu_tt, R_tt, _, _ = carry

        i_tp = scan_elt[0, :]
        R_tp = scan_elt[1:, :]

        U_pred = ql_L(st(R_tt.T, M.T, lower=True), sigma_eta)
        R_pred = st(U_pred, jnp.eye(r), lower=True).T
        nu_pred = (
            R_pred.T @ R_pred @ M @ st(R_tt, st(R_tt.T, nu_tt, lower=True), lower=False)
        )

        nu_up = nu_pred + i_tp
        R_up = qr_R(R_pred, R_tp)

        return (nu_up, R_up, nu_pred, R_pred), (nu_up, R_up, nu_pred, R_pred)

    carry, seq = jl.scan(
        step,
        (nu_0, R_0, jnp.zeros(r), jnp.eye(r)),
        scan_elts,
    )

    nus, Rs, nupreds, Rpreds = (seq[0], seq[1], seq[2], seq[3])
    
    if likelihood in ("full", "partial"):
        
        def likelihood_func(z_k, PHI_k, S2_eps_k, nu_pred, R_pred):
            n_k = z_k.shape[0]
            
            match S2_eps_shape:
                case 0:
                    S_eps = jnp.sqrt(S2_eps_k) * jnp.eye(n_k)
                case 1:
                    S_eps = jnp.diag(jnp.sqrt(S2_eps_k))
                case 2:
                    S_eps = jnp.linalg.cholesky(S2_eps_k)

            e_k = z_k - PHI_k @ st(R_pred, st(R_pred.T, nu_pred, lower=True), lower=False)

            Ui_t = qr_R(st(R_pred.T, PHI_k.T, lower=True), S_eps)

            s = st(Ui_t.T, e_k, lower=True)

            match likelihood:
                case "full":
                    ll = (
                        -jnp.sum(jnp.log(jnp.abs(jnp.diag(Ui_t))))
                        - 0.5 * jnp.dot(s, s)
                        - 0.5 * n_k * jnp.log(2 * jnp.pi)
                    )
                case "partial":
                    ll = -jnp.sum(jnp.log(jnp.abs(jnp.diag(Ui_t)))) - 0.5 * jnp.dot(
                        s, s
                    )
                case "none":
                    ll = jnp.nan
                case _:
                    raise ValueError(
                        "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
                    )

            return ll

        lls = jnp.array(jax.tree.map(likelihood_func, zs_tree, PHI_tree, S2_eps_tree, list(nupreds), list(Rpreds)))
        ll = jnp.sum(lls)
    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

    # fc_scan_elts = jnp.repeat(
    #    jnp.expand_shapes(jnp.zeros((r + 1, r)), axis=0), forecast, axis=0
    # )
    fc_scan_elts = jnp.tile(jnp.zeros((r + 1, r)), (forecast, 1, 1))

    carry_pred, seq_pred = jl.scan(
        step,
        (nus[-1], Rs[-1], jnp.zeros(r), jnp.eye(r)),
        fc_scan_elts,
    )

    filt_results = SqinfResults(
        ll=ll,
        nus=nus,
        Rs=Rs,
        nu_preds=nupreds,
        R_preds=Rpreds,
        nu_forecast=seq_pred[0],
        R_forecast=seq_pred[1],
    )

    return filt_results

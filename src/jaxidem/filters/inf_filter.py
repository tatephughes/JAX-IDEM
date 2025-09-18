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


class InformationResults(TypedDict):
    ll: float
    nus: PyTree[Float[Array, "T r"]]
    Qs: PyTree[Float[Array, "T r r"]]
    nupreds: PyTree[Float[Array, "T r"]]
    Qpreds: PyTree[Float[Array, "T r r"]]
    nuforecast: Float[Array, "forecast r"]
    Qforecast: Float[Array, "forecast r r"]


@partial(
    jax.jit,
    static_argnames=["S2_eta_shape", "S2_eps_shape", "forecast", "likelihood"],
)
def inf_filter(
    nu_0: Float[Array, "r"],
    Q_0: Float[Array, "r r"],
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
) -> dict:
    """
    The Information Filter.

    Parameters
    ----------
    nu_0: Float[Array, "r"]
        Initial information vector at time T=0.

    Q_0: Float[Array, "r r"]
        Initial information matrix at time T=0.

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
        Shape code for `S2_eta` (needed for JIT compilation); 0 is scalar/i.i.d., 1 is vector/uncorrelated, 2 is matrix/correlated. Must match the dimensionality of `S2_eta`.

    S2_eps_shape: int
        Shape code for `S2_eps` (needed for JIT compilation); 0 is scalar/i.i.d., 1 is vector/uncorrelated, 2 is matrix/correlated. Must match the dimensionality of each leaf of `S2_eps`.

    forecast: int
        Number of time points to forecast ahead.

    likelihood: Literal["none", "partial", "full"]
        Log-likelihood computation mode. Must be one of 'none' (no likelihood is computed), 'partial' (likelihood computed up to an additive constant), or 'full' (full likelihood is computed). Default is 'partial'.

    Returns
    ----------
    filt_results: InformationResults
        For details see Notes below.

    Notes
    -----
    The return type is InformationResults, containing:

        - `ll` : float
          Log-likelihood value (possibly up to an additive constant).

        - `nus` : PyTree[Float[Array, "T r"]]
          Information vectors at each time step.

        - `Qs` : PyTree[Float[Array, "T r r"]]
          Information matrices at each time step.

        - `nu_preds` : PyTree[Float[Array, "T r"]]
          Predicted information vectors.

        - `Q_preds` : PyTree[Float[Array, "T r r"]]
          Predicted information matrices.

        - `nu_forecast` : Float[Array, "forecast r"]
          Forecasted information vectors.

        - `Q_forecast` : Float[Array, "forecast r r"]
          Forecasted information matrices.

    Symbolic dimensions:

        - `r` : state dimension
        - `n` : observation dimension (may vary per time step)
        - `T` : number of time steps
        - `forecast` : number of forecast steps
    """

    mapping_elts = jax.tree.map(
        lambda z, phi, eps: (z, phi, eps), zs_tree, PHI_tree, S2_eps_tree
    )

    r = nu_0.size

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_k = tup[1]
        S2_eps_k = tup[2]

        match S2_eps_shape:
            case 0 | 1:
                i_k = PHI_k.T @ (z_k / S2_eps_k)
                I_k = PHI_k.T / S2_eps_k @ PHI_k
            case 2:
                i_k = PHI_k.T @ solve(S2_eps_k, z_k, assume_a="pos")
                I_k = PHI_k.T @ solve(S2_eps_k, PHI_k, assume_a="pos")

        return jnp.vstack((i_k, I_k))

    def is_leaf(node):
        return jax.tree.structure(node).num_leaves == 3

    scan_elts = jnp.stack(jax.tree.map(informationify, mapping_elts, is_leaf=is_leaf))

    # This is one situation where I do not know how to avoid inverting
    # a matrix explicitly...
    Minv = jnp.linalg.inv(M)

    match S2_eta_shape:
        case 0 | 1:
            S2_eta_inv = 1 / S2_eta
        case 2:
            S2_eta_inv = solve(S2_eta, jnp.eye(r), assume_a="pos")

    def step(carry, scan_elt):
        nu_tt, Q_tt, _, _ = carry

        i_tp = scan_elt[0, :]
        I_tp = scan_elt[1:, :]

        S_t = Minv.T @ Q_tt @ Minv

        J_t = solve(
            add_variance(S_t, S2_eta_inv, S2_eta_shape).T, S_t.T, assume_a="pos"
        ).T

        # match S2_eta_shape:
        #    case 0:
        #        J_t = jnp.linalg.solve((S_t + S2_eta_inv * jnp.eye(r)).T, S_t.T).T
        #    case 1:
        #        J_t = jnp.linalg.solve((S_t + jnp.diag(S2_eta_inv)).T, S_t.T).T
        #    case 2:
        #        J_t = jnp.linalg.solve((S_t + S2_eta_inv).T, S_t.T).T

        nu_pred = (jnp.eye(r) - J_t) @ Minv.T @ nu_tt
        Q_pred = (jnp.eye(r) - J_t) @ S_t

        nu_up = nu_pred + i_tp
        Q_up = Q_pred + I_tp

        return (nu_up, Q_up, nu_pred, Q_pred), (
            nu_up,
            Q_up,
            nu_pred,
            Q_pred,
        )

    carry, seq = jl.scan(
        step,
        (nu_0, Q_0, jnp.zeros(r), jnp.eye(r)),
        scan_elts,
    )

    # mapping_elts = jax.tree.map(
    #    lambda t: (seq[0][t], PHI_tree[t], S2_eps_tree[t]),
    #    tuple(range(len(zs_tree))),
    # )

    if likelihood in ("full", "partial"):
        mapping_elts = jax.tree.map(
            lambda t: (
                zs_tree[t],
                PHI_tree[t],
                S2_eps_tree[t],
                seq[2][t],
                seq[3][t],
            ),
            tuple(range(len(zs_tree))),
        )

        def likelihood_func(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            S2_eps = tree[2]
            nu_pred = tree[3]
            Q_pred = tree[4]
            cholQ = jax.scipy.linalg.cho_factor(Q_pred)

            e = z - PHI @ jax.scipy.linalg.cho_solve(cholQ, nu_pred)
            Sigma_t = add_variance(
                PHI @ jax.scipy.linalg.cho_solve(cholQ, PHI.T), S2_eps, S2_eps_shape
            )
            Ui_t = jnp.linalg.cholesky(Sigma_t)

            # match S2_eps_shape:
            #    case 0:
            #        P_oprop = PHI @ jax.scipy.linalg.cho_solve(cholQ, PHI.T)
            #        Sigma_t = jnp.fill_diagonal(
            #            P_oprop, S2_eps + jnp.diag(P_oprop), inplace=False
            #        )
            #        chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
            #    case 1:
            #        P_oprop = PHI @ jax.scipy.linalg.cho_solve(cholQ, PHI.T)
            #        Sigma_t = jnp.fill_diagonal(
            #            P_oprop, jnp.diag(S2_eps) + jnp.diag(P_oprop), inplace=False
            #        )
            #        chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
            #    case 2:
            #        chol_Sigma_t = jnp.linalg.cholesky(
            #            PHI @ jax.scipy.linalg.cho_solve(cholQ, PHI.T) + S2_eps
            #        )

            s = st(Ui_t, e, lower=True)

            match likelihood:
                case "full":
                    ll = (
                        -jnp.sum(jnp.log(jnp.diag(Ui_t)))
                        - 0.5 * jnp.dot(s, s)
                        - 0.5 * nobs * jnp.log(2 * jnp.pi)
                    )
                case "partial":
                    ll = -jnp.sum(jnp.log(jnp.diag(Ui_t))) - 0.5 * jnp.dot(s, s)

            return ll

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 5

        lls = jnp.array(jax.tree.map(likelihood_func, mapping_elts, is_leaf=is_leaf))
        ll = jnp.sum(lls)
    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

    nus, Qs, nupreds, Qpreds = (seq[0], seq[1], seq[2], seq[3])

    fc_scan_elts = jnp.tile(jnp.zeros((r + 1, r)), (forecast, 1, 1))

    carry_pred, seq_pred = jl.scan(
        step,
        (nus[-1], Qs[-1], jnp.zeros(r), jnp.eye(r)),
        fc_scan_elts,
    )

    filt_results = InformationResults(
        ll=ll,
        nus=nus,
        Qs=Qs,
        nu_preds=nupreds,
        Q_preds=Qpreds,
        nu_forecast=seq_pred[0],
        Q_forecast=seq_pred[1],
    )

    return filt_results

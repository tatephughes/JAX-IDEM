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
from jaxidem.utils import mult_variance


class PKalmanResults(TypedDict):
    ll: float
    ms: PyTree[Float[Array, "T r"]]
    Ps: PyTree[Float[Array, "T r r"]]


@partial(jax.jit, static_argnames=["S2_eta_shape", "S2_eps_shape", "likelihood"])
def pkal_filter(
    m_0: Float[Array, "r"],
    P_0: Float[Array, "r r"],
    M: Float[Array, "r r"],
    PHI_tree: Union[Float[Array, "1"], Float[Array, "r"], Float[Array, "r r"]],
    S2_eta: Union[Float[Array, "1"], Float[Array, "r"], Float[Array, "r r"]],
    S2_eps_tree: PyTree[
        Union[Float[Array, "1"], Float[Array, "?n"], Float[Array, "?n ?n"]]
    ],
    zs_tree: PyTree[Float[Array, "n"]],
    S2_eta_shape: int,
    S2_eps_shape: int,
    likelihood: Literal["none", "partial", "full"] = "partial",
) -> PKalmanResults:
    """
    The Temporally Parallel Kalman Filter.

    Parameters
    ----------
    m_0: Float[Array, "r"]
        Prior mean of the filter at T=0.

    P_0: Float[Array, "r r"]
        Prior state covariance at T=0.

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
        Shape code for each leaf of `S2_eps_tree` (needed for JIT compilation); 0 is scalar/i.i.d., 1 is vector/uncorrelated, 2 is matrix/correlated.

    likelihood: Literal["none", "partial", "full"]
        Log-likelihood computation mode. Must be one of 'none' (no likelihood is computed), 'partial' (likelihood computed up to an additive constant), or 'full' (full likelihood is computed). Default is 'partial'.

    Returns
    ----------
    filt_results: PKalmanResults
        For details see Notes below.

    Notes
    -----
    The return type is PKalmanResults, containing:

        - `ll` : float
          Log-likelihood value (possibly up to an additive constant).

        - `ms` : PyTree[Float[Array, "T r"]]
          Posterior means at each time step.

        - `Ps` : PyTree[Float[Array, "T r r"]]
          Posterior covariance matrices at each time step.

    Symbolic dimensions:

        - `r` : state dimension
        - `n` : observation dimension (may vary per time step)
        - `T` : number of time steps
    """

    r = m_0.shape[0]

    # Get first filtering elements
    m1pred = M @ m_0
    P1pred = add_variance(M @ P_0 @ M.T, S2_eta, S2_eta_shape)

    S_1 = add_variance(
        PHI_tree[0] @ P1pred @ PHI_tree[0].T, S2_eps_tree[0], S2_eps_shape
    )

    # match S2_eta_shape:
    #    case 0:
    #        P1pred = M @ P_0 @ M.T + S2_eta * jnp.eye(r)
    #    case 1:
    #        P1pred = M @ P_0 @ M.T + jnp.diag(S2_eta)
    #    case 2:
    #        P1pred = M @ P_0 @ M.T + S2_eta

    # match S2_eps_shape:
    #    case 0 | 1:
    #        P_oprop = PHI_tree[0] @ P1pred @ PHI_tree[0].T
    #        S_1 = jnp.fill_diagonal(
    #            P_oprop, S2_eps_tree[0] + jnp.diag(P_oprop), inplace=False
    #        )
    #        # S1 = PHI_tree[0]@P1pred@PHI_tree[0].T + S2_eps_tree[0]*jnp.eye()
    #    case 2:
    #        S_1 = PHI_tree[0] @ P1pred @ PHI_tree[0].T + S2_eps_tree[0]

    # Possibly better to use cholesky since S is pdef
    # cholS = jsc.linalg.cho_factor(S_1)
    # D_1 = (jsc.linalg.cho_solve(cholS, PHI_tree[0])@P1pred.T).T
    D_1 = solve(S_1, PHI_tree[0] @ P1pred, assume_a="pos").T

    A_1 = jnp.zeros((r, r))
    b_1 = m1pred + D_1 @ (zs_tree[0] - PHI_tree[0] @ m1pred)
    C_1 = P1pred - D_1 @ S_1 @ D_1.T

    nu_1 = M.T @ PHI_tree[0].T @ solve(S_1, zs_tree[0], assume_a="pos")
    Q_1 = M.T @ PHI_tree[0].T @ solve(S_1, PHI_tree[0] @ M, assume_a="pos")

    first_elt = (A_1, b_1, C_1, nu_1, Q_1)

    # mapping_elts = jax.tree.map(
    #    lambda t: (zs_tree[t], PHI_tree[t], S2_eps_tree[t]),
    #    tuple(range(len(zs_tree))),
    # )
    mapping_elts = jax.tree.map(
        lambda z, phi, eps: (z, phi, eps), zs_tree, PHI_tree, S2_eps_tree
    )

    def get_element(mapping_elt: tuple):
        z_k = mapping_elt[0]
        PHI_k = mapping_elt[1]
        S2_eps_k = mapping_elt[2]

        # match S2_eta_shape:
        #    case 0:
        #        s2p = S2_eta * jnp.eye(PHI_k.shape[1]) @ PHI_k.T
        #    case 1:
        #        # probably wrong
        #        s2p = PHI_k.T @ jnp.diag(S2_eta)
        #    case 2:
        #        # probably wrong
        #        s2p = S2_eta @ PHI_k.T

        s2p = mult_variance(PHI_k, S2_eta, S2_eta_shape)
        ps2p = PHI_k @ s2p

        # match S2_eps_shape:
        #    case 0:
        #        S_k = jnp.fill_diagonal(ps2p, S2_eps_k + jnp.diag(ps2p), inplace=False)
        #    case 1:
        #        S_k = ps2p + jnp.diag(S2_eps_k)
        #    case 2:
        #        S_k = ps2p + S2_eps_k

        S_k = add_variance(ps2p, S2_eps_k, S2_eps_shape)
        cholS = jsc.linalg.cho_factor(S_k)

        D_k = (jsc.linalg.cho_solve(cholS, s2p.T)).T

        imdp = jnp.eye(r) - D_k @ PHI_k

        A_k = imdp @ M
        b_k = D_k @ z_k

        # match S2_eta_shape:
        #    case 0:
        #        # C_k = jnp.fill_diagonal(imdp, S2_eps_k * jnp.diag(imdp), inplace=False)
        #        C_k = imdp @ (jnp.eye(r) * S2_eps_k)
        #    case 1:
        #        C_k = imdp @ jnp.diag(S2_eta)
        #    case 2:
        #        C_k = imdp @ S2_eta

        C_k = mult_variance(imdp, S2_eta, S2_eta_shape).T

        nu_k = M.T @ PHI_k.T @ jsc.linalg.cho_solve(cholS, z_k)
        Q_k = M.T @ PHI_k.T @ jsc.linalg.cho_solve(cholS, PHI_k @ M)

        return (A_k, b_k, C_k, nu_k, Q_k)

    def is_leaf(node):
        return jax.tree.structure(node).num_leaves == 3

    elts = jax.tree.map(get_element, mapping_elts[1:], is_leaf=is_leaf)

    # this might lead to biig compile times. not 100% sure
    # all_elts = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *((first_elt,) + elts))
    all_elts = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), first_elt, *elts)

    @jax.vmap
    def compose(elt_i, elt_j):
        A_i, b_i, C_i, nu_i, Q_i = elt_i
        A_j, b_j, C_j, nu_j, Q_j = elt_j

        I = jnp.eye(r)

        ipcq = I + C_i @ Q_j
        ipqc = I + Q_j @ C_i

        # lots of cho solves, can be simplified.
        A_ij = A_j @ solve(ipcq, A_i, assume_a="pos")
        b_ij = A_j @ solve(ipcq, (b_i + C_i @ nu_j), assume_a="pos") + b_j
        C_ij = A_j @ solve(ipcq, C_i @ A_j.T, assume_a="pos") + C_j
        nu_ij = A_i.T @ solve(ipqc, nu_j - Q_j @ b_i, assume_a="pos") + nu_i
        Q_ij = A_i.T @ solve(ipqc, Q_j @ A_i, assume_a="pos") + Q_i

        return (A_ij, b_ij, C_ij, nu_ij, Q_ij)

    final_elts = jl.associative_scan(compose, all_elts)

    ms = final_elts[1]
    Ps = final_elts[2]
    # nu, Q = final_elts[3][-1], final_elts[4][-1]

    mpreds = jnp.einsum("ij,tj->ti", M, jnp.vstack([m_0, final_elts[1]])[:-1])

    vadd_eta = jax.vmap(lambda P: add_variance(P, S2_eta, S2_eta_shape))

    Ppreds = vadd_eta(
        jnp.einsum(
            "ij,tjk,kl->til", M.T, jnp.vstack([P_0[None, :, :], final_elts[2]])[:-1], M
        )
    )

    if likelihood in ("full", "partial"):

        @jax.jit
        def get_ll(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            S2_eps = tree[2]
            mpred = tree[3]
            Ppred = tree[4]

            e = z - PHI @ mpred
            # Sigma_t = PHI @ P_pred @ PHI.T + S2_eps
            Sigma_t = add_variance(PHI @ Ppred @ PHI.T, S2_eps, S2_eps_shape)

            Ui_t = jnp.linalg.cholesky(Sigma_t)
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

        mapping_elts = jax.tree.map(
            lambda t: (
                zs_tree[t],
                PHI_tree[t],
                S2_eps_tree[t],
                mpreds[t],
                Ppreds[t],
            ),
            tuple(range(len(zs_tree))),
        )

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 5

        lls = jnp.array(jax.tree.map(get_ll, mapping_elts, is_leaf=is_leaf))
        ll = jnp.sum(lls)

    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

    filt_results = PKalmanResults(ll=ll, ms=ms, Ps=Ps)

    return filt_results

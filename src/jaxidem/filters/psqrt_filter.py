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
from jaxidem.utils import mult_variance
from jaxidem.utils import qr_R


class PSqrtResults(TypedDict):
    ll: float
    ms: PyTree[Float[Array, "T r"]]
    Ps: PyTree[Float[Array, "T r r"]]


@partial(jax.jit, static_argnames=["S2_eta_shape", "S2_eps_shape", "likelihood"])
def psqrt_filter(
    m_0: Float[Array, "r"],
    U_0: Float[Array, "r r"],
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
) -> PSqrtResults:
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
    I = jnp.eye(r)
    
    match S2_eta_shape:
       case 0:
           S_eta = jnp.eye(r) * jnp.sqrt(S2_eta)
       case 1:
           S_eta = jnp.diag(jnp.sqrt(S2_eta))
       case 2:
           S_eta = jnp.linalg.cholesky(S2_eta)

    # Im not sure this is jit-compatible....
    #nobs = jax.tree.map(lambda z: z.size, zs_tree)
    match S2_eps_shape:
        case 0:
            S_eps_tree = jax.tree.map(lambda z, s2: jnp.eye(z.size)*jnp.sqrt(s2), zs_tree, S2_eps_tree)
        case 1:
            S_eps_tree = jax.tree.map(lambda s2: jnp.diag(jnp.sqrt(s2)), S2_eps_tree)
        case 2:
            S_eps_tree = jax.tree.map(jnp.linalg.cholesky, S2_eps_tree)
        
    # Get first filtering elements
    m1pred = M @ m_0
    U1pred = qr_R(U_0@M.T, S_eta)

    W_1 = qr_R(U1pred@PHI_tree[0].T, S_eps_tree[0])

    K_1 = st(W_1, st(W_1.T, PHI_tree[0]@U1pred.T@U1pred, lower=True), lower=False).T

    A_1 = jnp.zeros((r, r))
    b_1 = m1pred + K_1 @ (zs_tree[0] - PHI_tree[0] @ m1pred)
    Uc_1 = qr_R(U1pred@(I - K_1@PHI_tree[0]).T, S_eps_tree[0]@K_1.T)

    nu_1 = M.T @ PHI_tree[0].T @ st(W_1, st(W_1.T, zs_tree[0], lower=False),lower=True)
    #Uj_1 = qr_R(st(W_1.T, PHI_tree[0]@M, lower=True), jnp.zeros((zs_tree[0].size, r)))
    Uj_1 = jnp.linalg.qr(st(W_1.T, PHI_tree[0]@M, lower=True), mode='r')
    Uj_1f = jnp.pad(Uj_1,pad_width=((0,r-zs_tree[0].size),(0,0)), mode='empty')
    
    first_elt = (A_1, b_1, Uc_1, nu_1, Uj_1f)

    def get_element(z_k, PHI_k, S_eps_k):

        n = z_k.size

        W_k = qr_R(S_eta @ PHI_k.T, S_eps_k)

        K_k = st(W_k, st(W_k.T, PHI_k @ S_eta @ S_eta.T, lower=True), lower=False).T
        
        imkp = (I - K_k @ PHI_k)
        
        A_k = imkp @ M
        b_k = K_k @ z_k

        Uc_k = qr_R(S_eta @ imkp.T, S_eps_k @ K_k.T)

        nu_k = M.T @ PHI_k.T @ st(W_k, st(W_k.T, z_k, lower=True),lower=False)
        #Uj_k = qr_R(st(W_k.T, PHI_k@M, lower=True), jnp.zeros((n, r)))
        Uj_k = jnp.linalg.qr(st(W_k.T, PHI_k@M, lower=True), mode='r')
        Uj_kf = jnp.pad(Uj_k, pad_width=((0,r-z_k.size),(0,0)), mode='empty')
        
        return (A_k, b_k, Uc_k, nu_k, Uj_kf)

    elts = jax.tree.map(get_element, zs_tree[1:], PHI_tree[1:], S_eps_tree[1:])

    all_elts = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), first_elt, *elts)

    @jax.vmap
    def compose(elt_i, elt_j):
        A_i, b_i, Uc_i, nu_i, Uj_i = elt_i
        A_j, b_j, Uc_j, nu_j, Uj_j = elt_j

        # not happy about this
        C_i = Uc_i.T@Uc_i
        J_j = Uj_j.T@Uj_j
        
        scpcjc = qr_R(Uj_j@C_i, Uc_i)
        sjpjcj = qr_R(Uc_i@J_j, Uj_j)

        A_ij = A_j@C_i@st(scpcjc, st(scpcjc.T, A_i, lower=True), lower=False)
        b_ij = A_j@C_i@st(scpcjc, st(scpcjc.T, b_i + C_i @ nu_j, lower=True), lower=False) + b_j
        Uc_ij = qr_R(st(scpcjc.T, C_i@A_j.T, lower=True), Uc_j)

        # J_j is only invertible under specific circumstances!
        nu_ij = A_i.T @ J_j @ st(sjpjcj, st(sjpjcj.T, nu_j - J_j @ b_i, lower=True), lower=False) + nu_i

        # This line is likely the culprit of it currently not working.
        Uj_ij = qr_R(st(sjpjcj.T, J_j, lower=True) @ A_i, Uj_i)

        return (A_ij, b_ij, Uc_ij, nu_ij, Uj_ij)

    final_elts = jl.associative_scan(compose, all_elts)

    ms = final_elts[1]
    Us = final_elts[2]
    # nu, Q = final_elts[3][-1], final_elts[4][-1]

    mpreds = jnp.einsum("ij,tj->ti", M, jnp.vstack([m_0, final_elts[1]])[:-1])

    Upreds = jax.vmap(lambda U: qr_R(U@M.T, S_eta))(jnp.vstack([U_0[None, :, :], Us])[:-1])

    if likelihood in ("full", "partial"):
    
        @jax.jit
        def get_ll(z_k, PHI_k, S_eps_k, mpred, Upred):

            n = z_k.size
            
            e = z_k - PHI_k @ mpred
            # Sigma_t = PHI @ P_pred @ PHI.T + S2_eps
            #Sigma_t = add_variance(PHI_k @ Ppred @ PHI_k.T, S_eps_k, S2_eps_shape)

            Ui_t = qr_R(Upred @ PHI_k.T, S_eps_k)
            
            s = st(Ui_t.T, e, lower=True)

            match likelihood:
                case "full":
                    ll = (
                        -jnp.sum(jnp.log(jnp.abs(jnp.diag(Ui_t))))
                        - 0.5 * jnp.dot(s, s)
                        - 0.5 * n * jnp.log(2 * jnp.pi)
                    )
                case "partial":
                    ll = -jnp.sum(jnp.log(jnp.diag(Ui_t))) - 0.5 * jnp.dot(s, s)

            return ll

        lls = jnp.array(jax.tree.map(get_ll, zs_tree, PHI_tree, S_eps_tree, list(mpreds), list(Upreds)))
        ll = jnp.sum(lls)

    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

    filt_results = PSqrtResults(ll=ll, ms=ms, Us=Us)

    return filt_results

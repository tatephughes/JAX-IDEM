import jax
import jax.numpy as jnp
import jax.scipy as jsc
import jax.lax as jl
from jaxtyping import ArrayLike, PyTree
from typing import Tuple

from functools import partial
from jax.scipy.linalg import solve_triangular as st


@jax.jit
def qr_R(A, B):
    """Wrapper for the stacked-QR decompositon"""
    return jnp.linalg.qr(jnp.vstack([A, B]), mode="r")


@jax.jit
def ql_L(A, B):
    """Wrapper for the stacked-QL decompositon"""
    A_flipped = jnp.flip(jnp.vstack([A, B]), axis=1)
    R = jnp.linalg.qr(A_flipped, mode="r")
    L = jnp.flip(R, axis=(0, 1))

    return L


@partial(
    jax.jit,
    static_argnames=["sigma2_eta_dim", "sigma2_eps_dim", "forecast", "likelihood"],
)
def kalman_filter(
    m_0: ArrayLike,
    P_0: ArrayLike,
    M: ArrayLike,
    PHI: ArrayLike,
    sigma2_eta: ArrayLike,
    sigma2_eps: ArrayLike,
    zs_tree: PyTree,
    sigma2_eta_dim: int,
    sigma2_eps_dim: int,
    forecast: int = 0,
    likelihood: str = "partial",
) -> tuple:
    """
    The standard Kalman Filter.

    Parameters
    ----------
    m_0:
        Prior mean of the filter at the 0 time point.

    Returns
    ----------
    A namedtuple containing the results of the filter.
    """

    r = m_0.shape[0]
    nobs = zs_tree[0].size

    @jax.jit
    def step(carry, z_t):
        m_tt, P_tt, _, _, ll, _ = carry

        # predict
        m_pred = M @ m_tt
        P_prop = M @ P_tt @ M.T

        match sigma2_eta_dim:
            case 0 | 1:
                P_pred = jnp.fill_diagonal(
                    P_prop, sigma2_eta + jnp.diag(P_prop), inplace=False
                )
            case 2:
                P_pred = P_prop + sigma2_eta

        # Update

        # Prediction Errors
        e_t = z_t - PHI @ m_pred

        P_oprop = PHI @ P_pred @ PHI.T
        match sigma2_eps_dim:
            case 0 | 1:
                Sigma_t = jnp.fill_diagonal(
                    P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
                )
            case 2:
                Sigma_t = P_oprop + sigma2_eps

        # Kalman Gain
        K_t = (jnp.linalg.solve(Sigma_t, PHI) @ P_pred.T).T

        m_up = m_pred + K_t @ e_t

        P_up = (jnp.eye(r) - K_t @ PHI) @ P_pred

        # likelihood of epsilon, using cholesky decomposition
        chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
        z = st(chol_Sigma_t, e_t, lower=True)

        if likelihood == "full":
            ll_new = (
                ll
                - jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                - 0.5 * jnp.dot(z, z)
                - 0.5 * nobs * jnp.log(2 * jnp.pi)
            )
        elif likelihood == "partial":
            ll_new = ll - jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)
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

    def forecast_step(carry, x):
        m_tt, P_tt = carry

        # predict
        m_pred = M @ m_tt
        P_prop = M @ P_tt @ M.T

        match sigma2_eta_dim:
            case 0 | 1:
                P_pred = jnp.fill_diagonal(
                    P_prop, sigma2_eta + jnp.diag(P_prop), inplace=False
                )
            case 2:
                P_pred = P_prop + sigma2_eta

        return (m_pred, P_pred), (m_pred, P_pred)

    carry, seq = jl.scan(
        step,
        (m_0, P_0, m_0, P_0, 0, jnp.zeros((r, nobs))),
        jnp.column_stack(zs_tree).T,
    )

    ll, ms, Ps, mpreds, Ppreds, Ks = (carry[4], seq[0], seq[1], seq[2], seq[3], seq[5])

    carry_fc, seq_fc = jl.scan(
        forecast_step,
        (ms[-1], Ps[-1]),
        jnp.arange(forecast),
    )

    filt_results = {
        "ll": ll,
        "ms": ms,
        "Ps": Ps,
        "mpreds": mpreds,
        "Ppreds": Ppreds,
        "mforecast": seq_fc[0],
        "Pforecast": seq_fc[1],
    }

    return filt_results


@partial(
    jax.jit,
    static_argnames=["sigma2_eta_dim", "sigma2_eps_dim", "forecast", "likelihood"],
)
def information_filter(
    nu_0: ArrayLike,  # initial information vector
    Q_0: ArrayLike,  # initial information matrix
    M: ArrayLike,
    PHI_tree: tuple,
    sigma2_eta: ArrayLike,
    sigma2_eps_tree: tuple,
    zs_tree: tuple,
    sigma2_eta_dim: int,
    sigma2_eps_dim: int,
    forecast: int = 0,
    likelihood: str = "partial",
) -> tuple:
    mapping_elts = jax.tree.map(
        lambda t: (zs_tree[t], PHI_tree[t], sigma2_eps_tree[t]),
        tuple(range(len(zs_tree))),
    )

    r = nu_0.shape[0]

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_k = tup[1]
        sigma2_eps_k = tup[2]

        match sigma2_eps_dim:
            case 0 | 1:
                i_k = PHI_k.T @ (z_k / sigma2_eps_k)
                I_k = PHI_k.T / sigma2_eps_k @ PHI_k
            case 2:
                i_k = PHI_k.T @ jnp.linalg.solve(sigma2_eps_k, z_k)
                I_k = PHI_k.T @ jnp.linalg.solve(sigma2_eps_k, PHI_k)

        return jnp.vstack((i_k, I_k))

    def is_leaf(node):
        return jax.tree.structure(node).num_leaves == 3

    scan_elts = jnp.array(jax.tree.map(informationify, mapping_elts, is_leaf=is_leaf))

    # This is one situation where I do not know how to avoid inverting
    # a matrix explicitly...
    Minv = jnp.linalg.solve(M, jnp.eye(r))

    match sigma2_eta_dim:
        case 0 | 1:
            sigma2_eta_inv = 1 / sigma2_eta
        case 2:
            sigma2_eta_inv = jnp.linalg.solve(sigma2_eta, jnp.eye(r))

    def step(carry, scan_elt):
        nu_tt, Q_tt, _, _ = carry

        i_tp = scan_elt[0, :]
        I_tp = scan_elt[1:, :]

        S_t = Minv.T @ Q_tt @ Minv

        match sigma2_eta_dim:
            case 0:
                J_t = jnp.linalg.solve((S_t + sigma2_eta_inv * jnp.eye(r)).T, S_t.T).T
            case 1:
                J_t = jnp.linalg.solve((S_t + jnp.diag(sigma2_eta_inv)).T, S_t.T).T
            case 2:
                J_t = jnp.linalg.solve((S_t + sigma2_eta_inv).T, S_t.T).T

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
    #    lambda t: (seq[0][t], PHI_tree[t], sigma2_eps_tree[t]),
    #    tuple(range(len(zs_tree))),
    # )

    if likelihood in ("full", "partial"):
        mapping_elts = jax.tree.map(
            lambda t: (
                zs_tree[t],
                PHI_tree[t],
                sigma2_eps_tree[t],
                seq[2][t],
                seq[3][t],
            ),
            tuple(range(len(zs_tree))),
        )

        def likelihood_func(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            sigma2_eps = tree[2]
            nu_pred = tree[3]
            Q_pred = tree[4]
            cholQ = jax.scipy.linalg.cho_factor(Q_pred)
            e = z - PHI @ jax.scipy.linalg.cho_solve(cholQ, nu_pred)

            match sigma2_eps_dim:
                case 0:
                    P_oprop = PHI @ jax.scipy.linalg.cho_solve(cholQ, PHI.T)
                    Sigma_t = jnp.fill_diagonal(
                        P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
                    )
                    chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
                case 1:
                    P_oprop = PHI @ jax.scipy.linalg.cho_solve(cholQ, PHI.T)
                    Sigma_t = jnp.fill_diagonal(
                        P_oprop, jnp.diag(sigma2_eps) + jnp.diag(P_oprop), inplace=False
                    )
                    chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
                case 2:
                    chol_Sigma_t = jnp.linalg.cholesky(
                        PHI @ jax.scipy.linalg.cho_solve(cholQ, PHI.T) + sigma2_eps
                    )

            z = st(chol_Sigma_t, e, lower=True)

            match likelihood:
                case "full":
                    ll = (
                        -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                        - 0.5 * jnp.dot(z, z)
                        - 0.5 * nobs * jnp.log(2 * jnp.pi)
                    )
                case "partial":
                    ll = -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)

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

    fc_scan_elts = jnp.repeat(
        jnp.expand_dims(jnp.zeros((r + 1, r)), axis=0), forecast, axis=0
    )
    carry_pred, seq_pred = jl.scan(
        step,
        (nus[-1], Qs[-1], jnp.zeros(r), jnp.eye(r)),
        fc_scan_elts,
    )

    filt_results = {
        "ll": ll,
        "nus": nus,
        "Qs": Qs,
        "nupreds": nupreds,
        "Qpreds": Qpreds,
        "nuforecast": seq_pred[0],
        "Qforecast": seq_pred[1],
    }

    return filt_results


@partial(
    jax.jit,
    static_argnames=["sigma2_eta_dim", "sigma2_eps_dim", "forecast", "likelihood"],
)
def sqrt_filter(
    m_0: ArrayLike,
    U_0: ArrayLike,
    M: ArrayLike,
    PHI: ArrayLike,
    sigma2_eta: ArrayLike,
    sigma2_eps: ArrayLike,
    zs_tree: PyTree,
    sigma2_eta_dim: int,
    sigma2_eps_dim: int,
    forecast: int = 0,
    likelihood: str = "full",
) -> tuple:
    r = m_0.shape[0]
    nobs = zs_tree[0].size

    match sigma2_eta_dim:
        case 0:
            sigma_eta = jnp.sqrt(sigma2_eta) * jnp.eye(r)
        case 1:
            sigma_eta = jnp.diag(jnp.sqrt(sigma2_eta))
        case 2:
            sigma_eta = jnp.linalg.cholesky(sigma2_eta)

    match sigma2_eps_dim:
        case 0:
            sigma_eps = jnp.sqrt(sigma2_eps) * jnp.eye(nobs)
        case 1:
            sigma_eps = jnp.diag(jnp.sqrt(sigma2_eps))
        case 2:
            sigma_eps = jnp.linalg.cholesky(sigma2_eps)

    @jax.jit
    def step(carry, z_t):
        m_tt, U_tt, _, _, ll, _ = carry

        # predict
        m_pred = M @ m_tt

        U_pred = qr_R(U_tt @ M.T, sigma_eta)
        # Update

        # Prediction error
        e_t = z_t - PHI @ m_pred

        # Prediction deviation matrix
        # Ui_t = jnp.linalg.qr(jnp.vstack([U_pred @ PHI.T, U_eps]), mode="r")
        Ui_t = qr_R(U_pred @ PHI.T, sigma_eps)

        # Kalman Gain
        K_t = (
            st(
                Ui_t,
                st(Ui_t.T, PHI, lower=True) @ U_pred.T @ U_pred,
            )
        ).T

        m_up = m_pred + K_t @ e_t

        U_up = qr_R(U_pred @ (jnp.eye(r) - K_t @ PHI).T, sigma_eps @ K_t.T)

        # likelihood of epsilon, using cholesky decomposition
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

    def forecast_step(carry, x):
        m_tt, U_tt = carry

        m_pred = M @ m_tt
        U_pred = qr_R(U_tt @ M.T, sigma_eta)

        return (m_pred, U_pred), (m_pred, U_pred)

    carry, seq = jl.scan(
        step,
        (m_0, U_0, m_0, U_0, 0, jnp.zeros((r, nobs))),
        jnp.column_stack(zs_tree).T,
    )

    ll, ms, Us, mpreds, Upreds, Ks = (carry[4], seq[0], seq[1], seq[2], seq[3], seq[5])

    carry_fc, seq_fc = jl.scan(
        forecast_step,
        (ms[-1], Us[-1]),
        jnp.arange(forecast),
    )

    filt_results = {
        "ll": ll,
        "ms": ms,
        "Us": Us,
        "mpreds": mpreds,
        "Upreds": Upreds,
        "mforecast": seq_fc[0],
        "Uforecast": seq_fc[1],
    }

    return filt_results


def safe_cholesky(matrix):
    def zero_case(_):
        return jnp.zeros_like(matrix)

    def nonzero_case(matrix):
        return jnp.linalg.cholesky(matrix, upper=True)

    return jax.lax.cond(jnp.all(matrix == 0), zero_case, nonzero_case, operand=matrix)


def safe_qr(matrix):
    def empty_case(_):
        return jnp.zeros((matrix.shape[1], matrix.shape[1]))

    def nonempty_case(matrix):
        return jnp.linalg.qr(matrix, mode="r")

    return jax.lax.cond(matrix.size == 0, empty_case, nonempty_case, operand=matrix)


@partial(
    jax.jit,
    static_argnames=["sigma2_eta_dim", "sigma2_eps_dim", "forecast", "likelihood"],
)
def sqrt_information_filter(
    nu_0: ArrayLike,
    R_0: ArrayLike,
    M: ArrayLike,
    PHI_tree: tuple,
    sigma2_eta: ArrayLike,
    sigma2_eps_tree: tuple,
    zs_tree: tuple,
    sigma2_eta_dim: int,
    sigma2_eps_dim: int,
    forecast: int = 0,
    likelihood: str = "partial",
) -> tuple:
    r = nu_0.shape[0]

    mapping_elts = jax.tree.map(
        lambda t: (zs_tree[t], PHI_tree[t], sigma2_eps_tree[t]),
        tuple(range(len(zs_tree))),
    )

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_k = tup[1]
        sigma2_eps_k = tup[2]

        # THIS WILL FAIL if sigma2_eps_dim!=0 and there is a time point with no data.
        match sigma2_eps_dim:
            case 0:
                i_k = PHI_k.T @ z_k / sigma2_eps_k
                # Below cholseky _should_ be faster, but is much less stable; why?
                # R_k = safe_qr(PHI_k) / jnp.sqrt(sigma2_eps_k)
                # R_k = jnp.linalg.qr((PHI_k), mode="r") / jnp.sqrt(sigma2_eps_k)
                R_k = safe_cholesky(PHI_k.T @ PHI_k / sigma2_eps_k)
            case 1:
                sigma_eps = jnp.diag(jnp.sqrt(sigma2_eps_k))
                i_k = PHI_k.T @ st(
                    sigma_eps, st(sigma_eps.T, z_k, lower=False), lower=True
                )
                R_k = jnp.linalg.qr(st(sigma_eps.T, PHI_k, lower=False), mode="r")
            case 2:
                sigma_eps = jnp.linalg.cholesky(sigma2_eps_k)
                i_k = PHI_k.T @ st(
                    sigma_eps, st(sigma_eps.T, z_k, lower=False), lower=True
                )
                R_k = jnp.linalg.qr(st(sigma_eps.T, PHI_k, lower=False), mode="r")

        return jnp.vstack((i_k, R_k))

    def is_leaf(node):
        return jax.tree.structure(node).num_leaves == 3

    scan_elts = jnp.array(jax.tree.map(informationify, mapping_elts, is_leaf=is_leaf))

    # Minv = jnp.linalg.solve(M, jnp.eye(r))
    match sigma2_eta_dim:
        case 0:
            sigma_eta = jnp.sqrt(sigma2_eta) * jnp.eye(r)
        case 1:
            sigma_eta = jnp.diag(jnp.sqrt(sigma2_eta))
        case 2:
            sigma_eta = jnp.linalg.cholesky(sigma2_eta)

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

    if likelihood in ("full", "partial"):
        mapping_elts = jax.tree.map(
            lambda t: (
                zs_tree[t],
                PHI_tree[t],
                seq[2][t],
                seq[3][t],
                sigma2_eps_tree[t],
            ),
            tuple(range(len(zs_tree))),
        )

        def likelihood_func(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            nu_pred = tree[2]
            R_pred = tree[3]
            sigma2_eps = tree[4]

            match sigma2_eps_dim:
                case 0:
                    sigma_eps = jnp.sqrt(sigma2_eps) * jnp.eye(nobs)
                case 1:
                    sigma_eps = jnp.diag(jnp.sqrt(sigma2_eps))
                case 2:
                    sigma_eps = jnp.linalg.cholesky(sigma2_eps)

            # Q_pred = R_pred.T@R_pred
            # e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)
            e = z - PHI @ st(R_pred, st(R_pred.T, nu_pred, lower=True), lower=False)

            # P_oprop = PHI @ jnp.linalg.solve(Q_pred, PHI.T)
            # Sigma_t = P_oprop + sigma_eps.T@sigma_eps
            # chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)

            # or

            # CHANGE THIS TO THE CHOLESKY VERSION
            # Its way faster (on GPU) and seems about as stable
            Ui_t = qr_R(st(R_pred.T, PHI.T, lower=True), sigma_eps)
            # chol_Sigma_t = jnp.linalg.cholesky(R_t.T @ R_t)

            s = st(Ui_t.T, e, lower=True)

            match likelihood:
                case "full":
                    ll = (
                        -jnp.sum(jnp.log(jnp.abs(jnp.diag(Ui_t))))
                        - 0.5 * jnp.dot(s, s)
                        - 0.5 * nobs * jnp.log(2 * jnp.pi)
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

    nus, Rs, nupreds, Rpreds = (seq[0], seq[1], seq[2], seq[3])

    fc_scan_elts = jnp.repeat(
        jnp.expand_dims(jnp.zeros((r + 1, r)), axis=0), forecast, axis=0
    )

    carry_pred, seq_pred = jl.scan(
        step,
        (nus[-1], Rs[-1], jnp.zeros(r), jnp.eye(r)),
        fc_scan_elts,
    )

    filt_results = {
        "ll": ll,
        "nus": nus,
        "Rs": Rs,
        "nupreds": nupreds,
        "Rpreds": Rpreds,
        "nuforecast": seq_pred[0],
        "Rforecast": seq_pred[1],
    }

    return filt_results


@partial(jax.jit, static_argnames=["sigma2_eta_dim", "sigma2_eps_dim", "likelihood"])
def pkalman_filter(
    m_0: ArrayLike,
    P_0: ArrayLike,
    M: ArrayLike,
    PHI_tree: tuple,
    sigma2_eta: ArrayLike,
    sigma2_eps_tree: tuple,
    zs_tree: tuple,
    sigma2_eta_dim: int,
    sigma2_eps_dim: int,
    likelihood: str = "partial",
) -> tuple:
    r = m_0.shape[0]

    # Get first filtering elements
    m1pred = M @ m_0

    match sigma2_eta_dim:
        case 0:
            P1pred = M @ P_0 @ M.T + sigma2_eta * jnp.eye(r)
        case 1:
            P1pred = M @ P_0 @ M.T + jnp.diag(sigma2_eta)
        case 2:
            P1pred = M @ P_0 @ M.T + sigma2_eta

    match sigma2_eps_dim:
        case 0 | 1:
            P_oprop = PHI_tree[0] @ P1pred @ PHI_tree[0].T
            S_1 = jnp.fill_diagonal(
                P_oprop, sigma2_eps_tree[0] + jnp.diag(P_oprop), inplace=False
            )
            # S1 = PHI_tree[0]@P1pred@PHI_tree[0].T + sigma2_eps_tree[0]*jnp.eye()
        case 2:
            S_1 = PHI_tree[0] @ P1pred @ PHI_tree[0].T + sigma2_eps_tree[0]

    # Possibly better to use cholesky since S is pdef
    # cholS = jsc.linalg.cho_factor(S_1)
    # D_1 = (jsc.linalg.cho_solve(cholS, PHI_tree[0])@P1pred.T).T
    D_1 = (jnp.linalg.solve(S_1, PHI_tree[0] @ P1pred)).T

    A_1 = jnp.zeros((r, r))
    b_1 = m1pred + D_1 @ (zs_tree[0] - PHI_tree[0] @ m1pred)
    C_1 = P1pred - D_1 @ S_1 @ D_1.T

    nu_1 = M.T @ PHI_tree[0].T @ jsc.linalg.solve(S_1, zs_tree[0])
    Q_1 = M.T @ PHI_tree[0].T @ jsc.linalg.solve(S_1, PHI_tree[0] @ M)

    first_elt = (A_1, b_1, C_1, nu_1, Q_1)

    mapping_elts = jax.tree.map(
        lambda t: (zs_tree[t], PHI_tree[t], sigma2_eps_tree[t]),
        tuple(range(len(zs_tree))),
    )

    def get_element(mapping_elt: tuple):
        z_k = mapping_elt[0]
        PHI_k = mapping_elt[1]
        sigma2_eps_k = mapping_elt[2]

        match sigma2_eta_dim:
            case 0:
                s2p = sigma2_eta * jnp.eye(PHI_k.shape[1]) @ PHI_k.T
            case 1:
                # probably wrong
                s2p = PHI_k.T @ jnp.diag(sigma2_eta)
            case 2:
                # probably wrong
                s2p = sigma2_eta @ PHI_k.T

        ps2p = PHI_k @ s2p

        match sigma2_eps_dim:
            case 0:
                S_k = jnp.fill_diagonal(
                    ps2p, sigma2_eps_k + jnp.diag(ps2p), inplace=False
                )
            case 1:
                S_k = ps2p + jnp.diag(sigma2_eps_k)
            case 2:
                S_k = ps2p + sigma2_eps_k

        cholS = jsc.linalg.cho_factor(S_k)

        D_k = (jsc.linalg.cho_solve(cholS, s2p.T)).T

        imdp = jnp.eye(r) - D_k @ PHI_k

        A_k = imdp @ M
        b_k = D_k @ z_k

        match sigma2_eta_dim:
            case 0:
                # C_k = jnp.fill_diagonal(imdp, sigma2_eps_k * jnp.diag(imdp), inplace=False)
                C_k = imdp @ (jnp.eye(r) * sigma2_eps_k)
            case 1:
                C_k = imdp @ jnp.diag(sigma2_eta)
            case 2:
                C_k = imdp @ sigma2_eta

        nu_k = M.T @ PHI_k.T @ jsc.linalg.cho_solve(cholS, z_k)
        Q_k = M.T @ PHI_k.T @ jsc.linalg.cho_solve(cholS, PHI_k @ M)

        return (A_k, b_k, C_k, nu_k, Q_k)

    def is_leaf(node):
        return jax.tree.structure(node).num_leaves == 3

    elts = jax.tree.map(get_element, mapping_elts[1:], is_leaf=is_leaf)

    # this might lead to biig compile times. not 100% sure
    all_elts = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *((first_elt,) + elts))

    @jax.vmap
    def compose(elt_i, elt_j):
        A_i, b_i, C_i, nu_i, Q_i = elt_i
        A_j, b_j, C_j, nu_j, Q_j = elt_j

        I = jnp.eye(r)

        ipcq = I + C_i @ Q_j
        ipqc = I + Q_j @ C_i

        # lots of cho solves, can be simplified.
        A_ij = A_j @ jsc.linalg.solve(ipcq, A_i)
        b_ij = A_j @ jsc.linalg.solve(ipcq, (b_i + C_i @ nu_j)) + b_j
        C_ij = A_j @ jsc.linalg.solve(ipcq, C_i @ A_j.T) + C_j
        nu_ij = A_i.T @ jsc.linalg.solve(ipqc, nu_j - Q_j @ b_i) + nu_i
        Q_ij = A_i.T @ jsc.linalg.solve(ipqc, Q_j @ A_i) + Q_i

        return (A_ij, b_ij, C_ij, nu_ij, Q_ij)

    final_elts = jl.associative_scan(compose, all_elts)

    ms = final_elts[1]
    Ps = final_elts[2]

    nu, Q = final_elts[3][-1], final_elts[4][-1]

    if likelihood in ("full", "partial"):

        @jax.jit
        def get_ll(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            sigma2_eps = tree[2]
            m = tree[3]
            P = tree[4]

            m_pred = M @ m
            P_pred = M @ P @ M.T + sigma2_eta

            e = z - PHI @ m_pred
            S = PHI @ P_pred @ PHI.T + sigma2_eps

            cholS = jnp.linalg.cholesky(S)
            s = st(cholS, e, lower=True)

            match likelihood:
                case "full":
                    ll = (
                        -jnp.sum(jnp.log(jnp.diag(cholS)))
                        - 0.5 * jnp.dot(s, s)
                        - 0.5 * nobs * jnp.log(2 * jnp.pi)
                    )
                case "partial":
                    ll = -jnp.sum(jnp.log(jnp.diag(cholS))) - 0.5 * jnp.dot(s, s)

            return ll

        mapping_elts = jax.tree.map(
            lambda t: (
                zs_tree[t],
                PHI_tree[t],
                sigma2_eps_tree[t],
                ms[t],
                Ps[t],
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

    filt_results = {"ll": ll, "ms": ms, "Ps": Ps}

    return filt_results


@jax.jit
def kalman_smoother(ms, Ps, mpreds, Ppreds, M):
    """NOT FULLY IMPLEMENTED"""
    r = ms[0].shape[0]

    @jax.jit
    def step(carry, y):
        m_tmtm = y[0]
        P_tmtm = y[1]
        m_ttm = y[2]
        P_ttm = y[3]

        m_tT = carry[0]
        P_tT = carry[1]

        J_tm = P_tmtm @ M.T @ jnp.linalg.solve(P_ttm, jnp.eye(r))

        m_tmT = m_tmtm - J_tm @ (m_tT - m_ttm)
        P_tmT = P_tmtm - J_tm @ (P_tT - P_ttm) @ J_tm.T

        return (m_tmT, P_tmT, J_tm), (m_tmT, P_tmT, J_tm)

    ys = (
        jnp.flip(ms[0:-1], axis=1),
        jnp.flip(Ps[0:-1], axis=1),
        jnp.flip(mpreds, axis=1),
        jnp.flip(Ppreds, axis=1),
    )
    init = (ms[-1], Ps[-1], jnp.zeros((r, r)))

    result = jl.scan(step, init, ys)

    return result


@jax.jit
def lag1_smoother(Ps, Js, K_T, PHI: ArrayLike, M: ArrayLike):
    """CURRENTLY OUT-OF-DATE AND UNTESTED"""

    r = Ps[0].shape[0]
    P_TTmT = (jnp.eye(r) - K_T @ PHI) @ M @ Ps[-2]

    @jax.jit
    def step(carry, y):
        P_tt = y[0]
        P_tmtm = y[1]
        J_t = y[2]
        J_tm = y[3]

        P_tptT = carry

        P_ttmT = P_tt @ J_tm.T + J_t @ (P_tptT - M @ P_tmtm) @ J_tm.T

        return P_ttmT, P_ttmT

    ys = (
        jnp.flip(Ps[1:-1], axis=1),
        jnp.flip(Ps[0:-2], axis=1),
        jnp.flip(Js[1:], axis=1),
        jnp.flip(Js[0:-1], axis=1),
    )

    init = P_TTmT

    result = jl.scan(step, init, ys)

    return result

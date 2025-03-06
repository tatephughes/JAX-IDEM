import jax
import jax.numpy as jnp
import jax.lax as jl
from jax.typing import ArrayLike
from typing import Tuple
from functools import partial

st = jax.scipy.linalg.solve_triangular 

@partial(jax.jit, static_argnames=["likelihood"])
def kalman_filter(
    m_0: ArrayLike,
    P_0: ArrayLike,
    M: ArrayLike,
    PHI: ArrayLike,
    Sigma_eta: ArrayLike,
    Sigma_eps: ArrayLike,
    zs: ArrayLike,
    likelihood: str = "partial",
) -> tuple:
    """
    Applies the Kalman Filter to a wide-format matrix of data.
    For jit-ability, this only allows full (no missing) data in a wide format.
    For changing data locations or changing data dimension, see [`information_filter`](/reference/information_filter.qmd).
    For the Kalman filter with uncorrelated errors, see [`kalman_filter_indep`](/reference/kalman_filter_indep.qmd).
    For the square-root Kalman filter, [`sqrt_filter_indep`](/reference/sqrt_filter_indep.qmd).

    Computes posterior means and variances for a system
    $$\\begin{split}
        \\mathbf Z_t &= \\Phi \\boldsymbol\\alpha_t + \\boldsymbol \\epsilon_t, \\quad t = 1,\\dots, T,\\\\
        \\boldsymbol \\alpha_{t+1} &= M\\boldsymbol \\alpha_t + \\boldsymbol\\eta_t,\\quad t = 0,2,\\dots, T-1,\\\\
    \\end{split}
    $$
    with initial 'priors'
    $$\\begin{split}
        \\boldsymbol \\alpha_{0} \\sim \\mathcal N(\\mathbf m_0, \\mathbf P_0),\\\\
    \\end{split}
    $$
    where
    $$\\begin{split}
        \\boldsymbol \\epsilon_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0,\\Sigma_\\epsilon),
        \\boldsymbol \\eta_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0,\\Sigma_\\eta).
    \\end{split}
    $$

    Parameters
    ----------
    m_0: ArrayLike (r,)
        The initial means of the process vector
    P_0: ArrayLike (r,r)
        The initial Covariance matrix of the process vector
    M: ArrayLike (r,r)
        The transition matrix of the process
    PHI: ArrayLike (r,n)
        The process-to-data matrix
    Sigma_eta: Arraylike (r,r)
        The Covariance matrix of the process noise
    Sigma_eps: ArrayLike (n,n)
        The Covariance matrix of the observation noise
    zs: ArrayLike
        The observed data to be filtered, in matrix format
    likelihood: string = 'partial'
        (STATIC) The mode to compute the likelihood ('full' with constant terms, 'partial' without constant terms, or 'none'.)
    
    Returns
    ----------
    ll: ArrayLike (1,)
        The log (data) likelihood of the data
    ms: ArrayLike (T,r)
        The posterior means $m_{t \\mid t}$ of the process given the data 1:t
    Ps: ArrayLike (T,r,r)
        The posterior covariance matrices $P_{t \\mid t}$ of the process given the data 1:t
    mpreds: ArrayLike (T-1,r)
        The predicted next-step means $m_{t \\mid t-1}$ of the process given the data 1:t-1
    Ppreds: ArrayLike (T-1,r,r)
        The predicted next-step covariances $P_{t \\mid t-1}$ of the process given the data 1:t-1
    Ks: ArrayLike (T,n,r)
        The Kalman Gains at each time step
    """

    nbasis = m_0.shape[0]
    nobs = zs.shape[0]

    @jax.jit
    def step(carry, z_t):
        m_tt, P_tt, _, _, ll, _ = carry

        # predict
        m_pred = M @ m_tt
        P_pred = M @ P_tt @ M.T + Sigma_eta

        # Update

        # Prediction Errors
        e_t = z_t - PHI @ m_pred

        Sigma_t = PHI @ P_pred @ PHI.T + Sigma_eps

        # Kalman Gain
        K_t = (jnp.linalg.solve(Sigma_t, PHI) @ P_pred.T).T

        m_up = m_pred + K_t @ e_t

        P_up = (jnp.eye(nbasis) - K_t @ PHI) @ P_pred

        # likelihood of epsilon, using cholesky decomposition
        chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
        z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e_t, lower=True)

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

    carry, seq = jl.scan(
        step,
        (m_0, P_0, m_0, P_0, 0, jnp.zeros((nbasis, nobs))),
        zs.T,
    )

    return (carry[4], seq[0], seq[1], seq[2], seq[3], seq[5])


@partial(jax.jit, static_argnames=["likelihood"])
def kalman_filter_indep(
    m_0: ArrayLike,
    P_0: ArrayLike,
    M: ArrayLike,
    PHI: ArrayLike,
    sigma2_eta: float,
    sigma2_eps: float,
    zs: ArrayLike,  # data matrix, with time across columns
    likelihood: str = "partial",
) -> tuple:
    """
    Applies the Kalman Filter to a wide-format matrix of data.
    Additionally assumes that both the error terms $$\\boldsymbol \\epsilon_t$$ and the noise terms $$\\boldsymbol \\eta_t$$ both have uncorrelated componants.
    For jit-ability, this only allows full (no missing) data in a wide format.
    For changing data locations or changing data dimension, see [`information_filter_inder`](/reference/information_filter_indep.qmd).
    For the Kalman filter with correlated errors, see [`kalman_filter`](/reference/kalman_filter.qmd).
    For the square-root Kalman filter, [`sqrt_filter_indep`](/reference/sqrt_filter_indep.qmd).

    Computes posterior means and variances for a system
    $$\\begin{split}
        \\mathbf Z_t &= \\Phi \\boldsymbol\\alpha_t + \\boldsymbol \\epsilon_t, \\quad t = 1,\\dots, T,\\\\
        \\boldsymbol \\alpha_{t+1} &= M\\boldsymbol \\alpha_t + \\boldsymbol\\eta_t,\\quad t = 0,2,\\dots, T-1,\\\\
    \\end{split}
    $$
    with initial 'priors'
    $$\\begin{split}
        \\boldsymbol \\alpha_{0} \\sim \\mathcal N(\\mathbf m_0, \\mathbf P_0),\\\\
    \\end{split}
    $$
    where
    $$\\begin{split}
        \\boldsymbol \\epsilon_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\sigma^2_\\epsilon I),
        \\boldsymbol \\eta_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\sigma^2_\\eta I).
    \\end{split}
    $$

    Parameters
    ----------
    m_0: ArrayLike (r,)
        The initial means of the process vector
    P_0: ArrayLike (r,r)
        The initial Covariance matrix of the process vector
    M: ArrayLike (r,r)
        The transition matrix of the process
    PHI: ArrayLike (r,n)
        The process-to-data matrix
    sigma2_eta: float
        The variance of the process noise
    sigma2_eps: float
        The variance  of the observation noise
    zs: ArrayLike
        The observed data to be filtered, in matrix format
    likelihood: string = 'partial'
        (STATIC) The mode to compute the likelihood ('full' with constant terms, 'partial' without constant terms, or 'none'.)
    
    Returns
    ----------
    ll: ArrayLike (1,)
        The log (data) likelihood of the data
    ms: ArrayLike (T,r)
        The posterior means $m_{t \\mid t}$ of the process given the data 1:t
    Ps: ArrayLike (T,r,r)
        The posterior covariance matrices $P_{t \\mid t}$ of the process given the data 1:t
    mpreds: ArrayLike (T-1,r)
        The predicted next-step means $m_{t \\mid t-1}$ of the process given the data 1:t-1
    Ppreds: ArrayLike (T-1,r,r)
        The predicted next-step covariances $P_{t \\mid t-1}$ of the process given the data 1:t-1
    Ks: ArrayLike (T,n,r)
        The Kalman Gains at each time step
    """

    nbasis = m_0.shape[0]
    nobs = zs.shape[0]

    @jax.jit
    def step(carry, z_t):
        m_tt, P_tt, _, _, ll, _ = carry

        # predict
        m_pred = M @ m_tt

        # Add sigma2_eps to the diagonal intelligently
        P_prop = M @ P_tt @ M.T
        P_pred = jnp.fill_diagonal(P_prop, sigma2_eta + jnp.diag(P_prop), inplace=False)

        # Update

        # Prediction error
        e_t = z_t - PHI @ m_pred

        # Prediction Variance
        P_oprop = PHI @ P_pred @ PHI.T
        Sigma_t = jnp.fill_diagonal(
            P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
        )

        # Kalman Gain
        K_t = (jnp.linalg.solve(Sigma_t, PHI) @ P_pred.T).T

        m_up = m_pred + K_t @ e_t

        P_up = (jnp.eye(nbasis) - K_t @ PHI) @ P_pred

        # likelihood of epsilon, using cholesky decomposition
        chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
        z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e_t, lower=True)
        if likelihood == "full":
            ll_new = (
                ll
                - jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                - 0.5 * jnp.dot(z, z)
                - 0.5 * nobs * jnp.log(2 * jnp.pi)
            )
            # ll_new = ll + \
            #    jax.scipy.stats.multivariate_normal.logpdf(
            #        z_t, PHI @ m_pred, Sigma_t)
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

    carry, seq = jl.scan(
        step,
        (m_0, P_0, m_0, P_0, 0, jnp.zeros((nbasis, nobs))),
        zs.T,
    )

    return (carry[4], seq[0], seq[1], seq[2], seq[3], seq[5])


@partial(jax.jit, static_argnames=["likelihood"])
def sqrt_filter_indep(
    m_0: ArrayLike,
    U_0: ArrayLike,
    M: ArrayLike,
    PHI: ArrayLike,
    sigma2_eta: float,
    sigma2_eps: float,
    zs: ArrayLike,  # data matrix, with time across columns
    likelihood: str = "partial",
) -> tuple:
    """
    Applies square-root Kalman filter using QR decompositions (cite: Kevin S. Tracy, 2022)
    Additionally assumes that both the error terms $$\\boldsymbol \\epsilon_t$$ and the noise terms $$\\boldsymbol \\eta_t$$ both have uncorrelated componants.
    For jit-ability, this only allows full (no missing) data in a wide format.
    For changing data locations or changing data dimension, see [`information_filter_inder`](/reference/information_filter_indep.qmd).
    For the standard Kalman filter with uncorrelated errors, see [`kalman_filter`](/reference/kalman_filter.qmd).

    Computes posterior means and variances for a system
    $$\\begin{split}
        \\mathbf Z_t &= \\Phi \\boldsymbol\\alpha_t + \\boldsymbol \\epsilon_t, \\quad t = 1,\\dots, T,\\\\
        \\boldsymbol \\alpha_{t+1} &= M\\boldsymbol \\alpha_t + \\boldsymbol\\eta_t,\\quad t = 0,2,\\dots, T-1,\\\\
    \\end{split}
    $$
    with initial 'priors'
    $$\\begin{split}
        \\boldsymbol \\alpha_{0} \\sim \\mathcal N(\\mathbf m_0, \\mathbf P_0),\\\\
    \\end{split}
    $$
    where
    $$\\begin{split}
        \\boldsymbol \\epsilon_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\sigma^2_\\epsilon I),
        \\boldsymbol \\eta_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\sigma^2_\\eta I).
    \\end{split}
    $$

    Parameters
    ----------
    m_0: ArrayLike (r,)
        The initial means of the process vector
    U_0: ArrayLike (r,r)
        The initial square-root covariance matrix of the process vector
    M: ArrayLike (r,r)
        The transition matrix of the process
    PHI: ArrayLike (r,n)
        The process-to-data matrix
    sigma2_eta: float
        The variance of the process noise
    sigma2_eps: float
        The variance  of the observation noise
    zs: ArrayLike
        The observed data to be filtered, in matrix format
    likelihood: string = 'partial'
        (STATIC) The mode to compute the likelihood ('full' with constant terms, 'partial' without constant terms, or 'none'.)
    
    Returns
    ----------
    ll: ArrayLike (1,)
        The log (data) likelihood of the data
    ms: ArrayLike (T,r)
        The posterior means $m_{t \\mid t}$ of the process given the data 1:t
    Us: ArrayLike (T,r,r)
        The posterior square-root covariance matrices $U_{t \\mid t}$ of the process given the data 1:t
    mpreds: ArrayLike (T-1,r)
        The predicted next-step means $m_{t \\mid t-1}$ of the process given the data 1:t-1
    Upreds: ArrayLike (T-1,r,r)
        The predicted next-step square-root covariances $U_{t \\mid t-1}$ of the process given the data 1:t-1
    Ks: ArrayLike (T,n,r)
        The Kalman Gains at each time step
    """

    nbasis = m_0.shape[0]
    nobs = zs.shape[0]
    sigma_eps = jnp.sqrt(sigma2_eps)
    sigma_eta = jnp.sqrt(sigma2_eta)
    U_eps = sigma_eps * jnp.eye(nobs)
    U_eta = sigma_eta * jnp.eye(nbasis)

    @jax.jit
    def step(carry, z_t):
        m_tt, U_tt, _, _, ll, _ = carry

        # predict
        m_pred = M @ m_tt

        # Add sigma2_eps to the diagonal intelligently
        U_pred = jnp.linalg.qr(jnp.vstack([U_tt @ M.T, U_eta]), mode="r")

        # Update

        # Prediction error
        e_t = z_t - PHI @ m_pred

        # Prediction deviation matrix
        Ui_t = jnp.linalg.qr(jnp.vstack([U_pred @ PHI.T, U_eps]), mode="r")

        # Kalman Gain (can you work with the qr decomp of K_t? precision seems lost here.)
        K_t = (
            jax.scipy.linalg.solve_triangular(
                Ui_t,
                jax.scipy.linalg.solve_triangular(Ui_t.T, PHI, lower=True)
                @ U_pred.T
                @ U_pred,
            )
        ).T  # look at the U_pred.T@U_pred above; is it right?

        m_up = m_pred + K_t @ e_t

        #        P_up = (jnp.eye(nbasis) - K_t @ PHI) @ P_pred
        U_up = jnp.linalg.qr(
            jnp.vstack([U_pred @ (jnp.eye(nbasis) - K_t @ PHI).T, U_eps @ K_t.T]),
            mode="r",
        )

        # likelihood of epsilon, using cholesky decomposition
        chol_Sigma_t = Ui_t
        z = jax.scipy.linalg.solve_triangular(chol_Sigma_t.T, e_t, lower=True)
        if likelihood == "full":
            ll_new = (
                ll
                - jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_Sigma_t))))
                - 0.5 * jnp.dot(z, z)
                - 0.5 * nobs * jnp.log(2 * jnp.pi)
            )
        elif likelihood == "partial":
            ll_new = (
                ll
                - jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_Sigma_t))))
                - 0.5 * jnp.dot(z, z)
            )
        elif likelihood == "none":
            ll_new = jnp.nan
        else:
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
        (m_0, U_0, m_0, U_0, 0, jnp.zeros((nbasis, nobs))),
        zs.T,
    )

    return (carry[4], seq[0], seq[1], seq[2], seq[3], seq[5])


@partial(jax.jit, static_argnames=["likelihood"])
def information_filter(
    nu_0: ArrayLike,  # initial information vector
    Q_0: ArrayLike,  # initial information matrix
    M: ArrayLike,
    PHI_tree: tuple,
    Sigma_eta: ArrayLike,
    Sigma_eps_tree: tuple,
    zs_tree: tuple,
    likelihood: str = "partial",
) -> tuple:
    """
    WARNING: Very numerically unstable in this form
    
    Applies the information Filter (inverse-Kalman filter) to a PyTree of data points at a number of times.
    Unlike the Kalman filters, this allows for missing data and data changing shape, by taking a PyTree (most likely a list) of observations at each time (which can be jagged).
    For the standard Kalman filter with uncorrelated errors, see [`kalman_filter`](/reference/kalman_filter.qmd).
    For the square-root Kalman filter, [`sqrt_filter_indep`](/reference/sqrt_filter_indep.qmd).

    Computes posterior information vectors and information matrices for a system
    $$\\begin{split}
        \\mathbf Z_t &= \\Phi \\boldsymbol\\alpha_t + \\boldsymbol \\epsilon_t, \\quad t = 1,\\dots, T,\\\\
        \\boldsymbol \\alpha_{t+1} &= M\\boldsymbol \\alpha_t + \\boldsymbol\\eta_t,\\quad t = 0,2,\\dots, T-1,\\\\
    \\end{split}
    $$
    with initial 'priors'
    $$\\begin{split}
        \\boldsymbol \\alpha_{0} \\sim \\mathcal N(\\mathbf m_0, \\mathbf P_0),\\\\
    \\end{split}
    $$
    where
    $$\\begin{split}
        \\boldsymbol \\epsilon_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\Sigma_\\epsilon),
        \\boldsymbol \\eta_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\Sigma_\\eta).
    \\end{split}
    $$

    Parameters
    ----------
    nu_0: ArrayLike (r,)
        The initial information vector of the process vector
    Q_0: ArrayLike (r,r)
        The initial information matrix of the process vector
    M: ArrayLike (r,r)
        The transition matrix of the process
    PHI_tree: PyTree of ArrayLike (r,n_t)
        The process-to-data matrices at each time
    Sigma_eta: float
        The variance of the process noise
    Sigma_eps_tree: float
        The variance matrices of the observation noise
    zs_tree: PyTree of ArrayLike (n_t,)
        The observed data to be filtered
    likelihood: string = 'partial'
        (STATIC) The mode to compute the likelihood ('full' with constant terms, 'partial' without constant terms, or 'none'.)
    
    Returns
    ----------
    ll: ArrayLike (1,)
        The log (data) likelihood of the data
    nus: ArrayLike (T,r)
        The posterior means $m_{t \\mid t}$ of the process given the data 1:t
    Qs: ArrayLike (T,r,r)
        The posterior covariance matrices $P_{t \\mid t}$ of the process given the data 1:t
    nupreds: ArrayLike (T-1,r)
        The predicted next-step means $m_{t \\mid t-1}$ of the process given the data 1:t-1
    Qpreds: ArrayLike (T-1,r,r)
        The predicted next-step covariances $P_{t \\mid t-1}$ of the process given the data 1:t-1
    Ks: ArrayLike (T,n,r)
        The Kalman Gains at each time step
    """

    mapping_elts = jax.tree.map(
        lambda t: (zs_tree[t], PHI_tree[t], Sigma_eps_tree[t]),
        tuple(range(len(zs_tree))),
    )

    r = nu_0.shape[0]

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_k = tup[1]
        Sigma_eps_k = tup[2]
        i_k = PHI_k.T @ jnp.linalg.solve(Sigma_eps_k, z_k)
        I_k = PHI_k.T @ jnp.linalg.solve(Sigma_eps_k, PHI_k)
        return jnp.vstack((i_k, I_k))

    def is_leaf(node):
        # return len(node)==2 # IMPORTANT NOTE: what if T=2?
        return jax.tree.structure(node).num_leaves == 3
        # This works better, but could still be a problem if T=1. But
        # then, why would you even be filtering?
        # Added a raise to filter_information for this case

    scan_elts = jnp.array(jax.tree.map(informationify, mapping_elts, is_leaf=is_leaf))

    # This is one situation where I do not know how to avoid inverting
    # a matrix explicitly...
    Minv = jnp.linalg.solve(M, jnp.eye(r))
    Sigma_eta_inv = jnp.linalg.solve(Sigma_eta, jnp.eye(r))

    def step(carry, scan_elt):
        nu_tt, Q_tt, _, _ = carry

        i_tp = scan_elt[0, :]
        I_tp = scan_elt[1:, :]

        S_t = Minv.T @ Q_tt @ Minv
        J_t = jnp.linalg.solve((S_t + Sigma_eta_inv).T, S_t.T).T

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

    mapping_elts = jax.tree.map(
        lambda t: (seq[0][t], PHI_tree[t], Sigma_eps_tree[t]),
        tuple(range(len(zs_tree))),
    )

    if likelihood == "full":
        mapping_elts = jax.tree.map(
            lambda t: (
                zs_tree[t],
                PHI_tree[t],
                Sigma_eps_tree[t],
                seq[2][t],
                seq[3][t],
            ),
            tuple(range(len(zs_tree))),
        )

        def comp_full_likelihood_good(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            Sigma_eps = tree[2]
            nu_pred = tree[3]
            Q_pred = tree[4]
            e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)

            chol_Sigma_t = jnp.linalg.cholesky(
                PHI @ jnp.linalg.solve(Q_pred, PHI.T) + Sigma_eps
            )
            z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e, lower=True)

            ll = (
                -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                - 0.5 * jnp.dot(z, z)
                - 0.5 * nobs * jnp.log(2 * jnp.pi)
            )

            return ll

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 5

        lls = jnp.array(
            jax.tree.map(comp_full_likelihood_good, mapping_elts, is_leaf=is_leaf)
        )
        ll = jnp.sum(lls)
    elif likelihood == "partial":
        mapping_elts = jax.tree.map(
            lambda t: (
                zs_tree[t],
                PHI_tree[t],
                Sigma_eps_tree[t],
                seq[2][t],
                seq[3][t],
            ),
            tuple(range(len(zs_tree))),
        )

        def comp_full_likelihood_good(tree):
            z = tree[0]
            PHI = tree[1]
            Sigma_eps = tree[2]
            nu_pred = tree[3]
            Q_pred = tree[4]
            e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)

            chol_Sigma_t = jnp.linalg.cholesky(
                PHI @ jnp.linalg.solve(Q_pred, PHI.T) + Sigma_eps
            )
            z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e, lower=True)

            ll = -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)

            return ll

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 5

        lls = jnp.array(
            jax.tree.map(comp_full_likelihood_good, mapping_elts, is_leaf=is_leaf)
        )
        ll = jnp.sum(lls)
    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

    return ll, seq[0], seq[1], seq[2], seq[3]

@partial(jax.jit, static_argnames=["likelihood"])
def information_filter_indep(
    nu_0: ArrayLike,  # initial information vector
    Q_0: ArrayLike,  # initial information matrix
    M: ArrayLike,
    PHI_tree: tuple,
    sigma2_eta: float,
    sigma2_eps: float,
    zs_tree: tuple,
    likelihood: str = "partial",
) -> tuple:
    """
    Applies the information Filter (inverse-Kalman filter) to a PyTree of data points at a number of times.
    Additionally assumes that both the error terms $$\\boldsymbol \\epsilon_t$$ and the noise terms $$\\boldsymbol \\eta_t$$ both have uncorrelated componants.
    Unlike the Kalman filters, this allows for missing data and data changing shape, by taking a PyTree (most likely a list) of observations at each time (which can be jagged).
    For the standard Kalman filter with uncorrelated errors, see [`kalman_filter`](/reference/kalman_filter.qmd).
    For the square-root Kalman filter, [`sqrt_filter_indep`](/reference/sqrt_filter_indep.qmd).

    Computes posterior information vectors and information matrices for a system
    $$\\begin{split}
        \\mathbf Z_t &= \\Phi \\boldsymbol\\alpha_t + \\boldsymbol \\epsilon_t, \\quad t = 1,\\dots, T,\\\\
        \\boldsymbol \\alpha_{t+1} &= M\\boldsymbol \\alpha_t + \\boldsymbol\\eta_t,\\quad t = 0,2,\\dots, T-1,\\\\
    \\end{split}
    $$
    with initial 'priors'
    $$\\begin{split}
        \\boldsymbol \\alpha_{0} \\sim \\mathcal N(\\mathbf m_0, \\mathbf P_0),\\\\
    \\end{split}
    $$
    where
    $$\\begin{split}
        \\boldsymbol \\epsilon_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\sigma^2_\\epsilon I),
        \\boldsymbol \\eta_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\sigma^2_\\eta I).
    \\end{split}
    $$

    Parameters
    ----------
    nu_0: ArrayLike (r,)
        The initial information vector of the process vector
    Q_0: ArrayLike (r,r)
        The initial information matrix of the process vector
    M: ArrayLike (r,r)
        The transition matrix of the process
    PHI_tree: PyTree of ArrayLike (r,n_t)
        The process-to-data matrices at each time
    sigma2_eta: float
        The variance of the process noise
    sigma2_eps: float
        The variance the observation noise
    zs_tree: PyTree of ArrayLike (n_t,)
        The observed data to be filtered
    likelihood: string = 'partial'
        (STATIC) The mode to compute the likelihood ('full' with constant terms, 'partial' without constant terms, or 'none'.)
    
    Returns
    ----------
    ll: ArrayLike (1,)
        The log (data) likelihood of the data
    nus: ArrayLike (T,r)
        The posterior means $m_{t \\mid t}$ of the process given the data 1:t
    Qs: ArrayLike (T,r,r)
        The posterior covariance matrices $P_{t \\mid t}$ of the process given the data 1:t
    nupreds: ArrayLike (T-1,r)
        The predicted next-step means $m_{t \\mid t-1}$ of the process given the data 1:t-1
    Qpreds: ArrayLike (T-1,r,r)
        The predicted next-step covariances $P_{t \\mid t-1}$ of the process given the data 1:t-1
    Ks: ArrayLike (T,n,r)
        The Kalman Gains at each time step
    """

    mapping_elts = jax.tree.map(
        lambda t: (
            zs_tree[t],
            PHI_tree[t],
        ),
        tuple(range(len(zs_tree))),
    )

    r = nu_0.shape[0]

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_k = tup[1]
        i_k = PHI_k.T @ z_k / sigma2_eps
        I_k = PHI_k.T @ PHI_k / sigma2_eps
        return jnp.vstack((i_k, I_k))

    def is_leaf(node):
        # return len(node)==2 # IMPORTANT NOTE: what if T=2?
        return jax.tree.structure(node).num_leaves == 2
        # This works better, but could still be a problem if T=1. But
        # then, why would you even be filtering?
        # Added a raise to filter_information for this case

    scan_elts = jnp.array(jax.tree.map(informationify, mapping_elts, is_leaf=is_leaf))

    # This is one situation where I do not know how to avoid inverting
    # a matrix explicitely...
    # With the adjusted information filter, you should be able to fix this now
    Minv = jnp.linalg.solve(M, jnp.eye(r))
    sigma2_eta_inv = 1 / sigma2_eta

    def step(carry, scan_elt):
        nu_tt, Q_tt, _, _ = carry

        i_tp = scan_elt[0, :]
        I_tp = scan_elt[1:, :]

        S_t = Minv.T @ Q_tt @ Minv
        #R_t = jnp.fill_diagonal(S_t, jnp.diag(S_t) + sigma2_eta_inv, inplace=False)
        J_t = jnp.linalg.solve((S_t + sigma2_eta_inv*jnp.eye(r)).T, S_t.T).T

        #J_tmin = -jnp.fill_diagonal(J_t, jnp.diag(J_t - 1), inplace=False)

        nu_pred = (jnp.eye(r) - J_t) @ Minv.T @ nu_tt
        Q_pred = (jnp.eye(r) - J_t) @ S_t

        nu_up = nu_pred + i_tp
        Q_up = Q_pred + I_tp

        # chol_Sigma_iota = jnp.linalg.cholesky(
        #    I_tp @ jnp.linalg.solve(Q_pred, I_tp.T) + I_tp)
        # iota = i_tp - I_tp @ jnp.linalg.solve(Q_up, nu_up)
        # z = jax.scipy.linalg.solve_triangular(chol_Sigma_iota, iota, lower=True)
        # if full_likelihood:
        #    ll_new = ll - jnp.sum(jnp.log(jnp.diag(chol_Sigma_iota))) - \
        #        0.5 * jnp.dot(z, z) - 0.5 * r * jnp.log(2*jnp.pi)
        # else:
        #    ll_new = ll - \
        #        jnp.sum(jnp.log(jnp.diag(chol_Sigma_iota))) - 0.5 * jnp.dot(z, z)
        return (nu_up, Q_up, nu_pred, Q_pred), (nu_up, Q_up, nu_pred, Q_pred)

    carry, seq = jl.scan(
        step,
        (nu_0, Q_0, jnp.zeros(r), jnp.eye(r)),
        scan_elts,
    )

    if likelihood == "full":
        mapping_elts = jax.tree.map(
            lambda t: (zs_tree[t], PHI_tree[t], seq[2][t], seq[3][t]),
            tuple(range(len(zs_tree))),
        )

        def comp_full_likelihood_good(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            nu_pred = tree[2]
            Q_pred = tree[3]
            e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)

            P_oprop = PHI @ jnp.linalg.solve(Q_pred, PHI.T)
            Sigma_t = jnp.fill_diagonal(
                P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
            )

            chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
            z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e, lower=True)

            ll = (
                -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                - 0.5 * jnp.dot(z, z)
                - 0.5 * nobs * jnp.log(2 * jnp.pi)
            )

            return ll

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 4

        lls = jnp.array(
            jax.tree.map(comp_full_likelihood_good, mapping_elts, is_leaf=is_leaf)
        )
        ll = jnp.sum(lls)
    elif likelihood == "partial":
        mapping_elts = jax.tree.map(
            lambda t: (zs_tree[t], PHI_tree[t], seq[2][t], seq[3][t]),
            tuple(range(len(zs_tree))),
        )

        def comp_full_likelihood_good(tree):
            z = tree[0]
            PHI = tree[1]
            nu_pred = tree[2]
            Q_pred = tree[3]
            e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)

            P_oprop = PHI @ jnp.linalg.solve(Q_pred, PHI.T)
            Sigma_t = jnp.fill_diagonal(
                P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
            )

            chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
            z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e, lower=True)

            ll = -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)

            return ll

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 4

        lls = jnp.array(
            jax.tree.map(comp_full_likelihood_good, mapping_elts, is_leaf=is_leaf)
        )
        ll = jnp.sum(lls)
    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

    return ll, seq[0], seq[1], seq[2], seq[3]

@partial(jax.jit, static_argnames=["likelihood"])
def sqrt_information_filter_indep(
    nu_0: ArrayLike, 
    L_0: ArrayLike, 
    M: ArrayLike,
    PHI_tree: tuple,
    sigma2_eta: float,
    sigma2_eps: float,
    zs_tree: tuple,
    likelihood: str = "partial",
) -> tuple:
    """
    Applies the square-root information Filter to a PyTree of data points at a number of times.
    Additionally assumes that both the error terms $$\\boldsymbol \\epsilon_t$$ and the noise terms $$\\boldsymbol \\eta_t$$ both have uncorrelated componants.
    Unlike the Kalman filters, this allows for missing data and data changing shape, by taking a PyTree (most likely a list) of observations at each time (which can be jagged).
    For the standard Kalman filter with uncorrelated errors, see [`kalman_filter`](/reference/kalman_filter.qmd).
    For the square-root Kalman filter, [`sqrt_filter_indep`](/reference/sqrt_filter_indep.qmd).

    Computes posterior information vectors and information matrices for a system
    $$\\begin{split}
        \\mathbf Z_t &= \\Phi \\boldsymbol\\alpha_t + \\boldsymbol \\epsilon_t, \\quad t = 1,\\dots, T,\\\\
        \\boldsymbol \\alpha_{t+1} &= M\\boldsymbol \\alpha_t + \\boldsymbol\\eta_t,\\quad t = 0,2,\\dots, T-1,\\\\
    \\end{split}
    $$
    with initial 'priors'
    $$\\begin{split}
        \\boldsymbol \\alpha_{0} \\sim \\mathcal N(\\mathbf m_0, \\mathbf P_0),\\\\
    \\end{split}
    $$
    where
    $$\\begin{split}
        \\boldsymbol \\epsilon_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\sigma^2_\\epsilon I),
        \\boldsymbol \\eta_t \\overset{\\mathrm{iid}}{\\sim} \\mathcal N(0, \\sigma^2_\\eta I).
    \\end{split}
    $$

    Parameters
    ----------
    nu_0: ArrayLike (r,)
        The initial information vector of the process vector
    L_0: ArrayLike (r,r)
        The lower-triangular root of the initial information matrix of the process vector
    M: ArrayLike (r,r)
        The transition matrix of the process
    PHI_tree: PyTree of ArrayLike (r,n_t)
        The process-to-data matrices at each time
    sigma2_eta: float
        The variance of the process noise
    sigma2_eps: float
        The variance the observation noise
    zs_tree: PyTree of ArrayLike (n_t,)
        The observed data to be filtered
    likelihood: string = 'partial'
        (STATIC) The mode to compute the likelihood ('full' with constant terms, 'partial' without constant terms, or 'none'.)
    
    Returns
    ----------
    ll: ArrayLike (1,)
        The log (data) likelihood of the data
    nus: ArrayLike (T,r)
        The posterior means $m_{t \\mid t}$ of the process given the data 1:t
    Qs: ArrayLike (T,r,r)
        The posterior covariance matrices $P_{t \\mid t}$ of the process given the data 1:t
    nupreds: ArrayLike (T-1,r)
        The predicted next-step means $m_{t \\mid t-1}$ of the process given the data 1:t-1
    Qpreds: ArrayLike (T-1,r,r)
        The predicted next-step covariances $P_{t \\mid t-1}$ of the process given the data 1:t-1
    Ks: ArrayLike (T,n,r)
        The Kalman Gains at each time step
    """

    mapping_elts = jax.tree.map(
        lambda t: (
            zs_tree[t],
            PHI_tree[t],
        ),
        tuple(range(len(zs_tree))),
    )

    r = nu_0.shape[0]

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_k = tup[1]
        i_k = PHI_k.T @ z_k / sigma2_eps
        L_k = jnp.linalg.cholesky(PHI_k.T @ PHI_k) / sigma2_eps
        return jnp.vstack((i_k, L_k))

    def is_leaf(node): return jax.tree.structure(node).num_leaves == 2

    scan_elts = jnp.array(jax.tree.map(informationify, mapping_elts, is_leaf=is_leaf))

    #Minv = jnp.linalg.solve(M, jnp.eye(r))
    sigma2_eta_inv = 1 / sigma2_eta
    sigma_eta = jnp.sqrt(sigma2_eta)
    nbasis = nu_0.shape[0]

    
    def step(carry, scan_elt):
        nu_tt, L_tt, _, _ = carry

        i_tp = scan_elt[0, :]
        L_tp = scan_elt[1:, :]

        U_pred = qr_R(st(L_tt.T, M.T, lower=False), sigma_eta*jnp.eye(nbasis))
        L_pred = st(U_pred, jnp.eye(nbasis), lower=False)
        nu_pred = L_pred.T @ L_pred @ M @ st(L_tt, 
                      st(L_tt.T,
                         nu_tt,
                         lower=False), lower=True)

        nu_up = nu_pred + i_tp
        L_up = ql_L(L_pred, L_tp)

        return (nu_up, L_up, nu_pred, L_pred), (nu_up, L_up, nu_pred, L_pred)

    carry, seq = jl.scan(
        step,
        (nu_0, L_0, jnp.zeros(r), jnp.eye(r)),
        scan_elts,
    )

    if likelihood == "full":
        mapping_elts = jax.tree.map(
            lambda t: (zs_tree[t], PHI_tree[t], seq[2][t], seq[3][t]),
            tuple(range(len(zs_tree))),
        )

        def comp_full_likelihood_good(tree):
            z = tree[0]
            nobs = z.shape[0]
            PHI = tree[1]
            nu_pred = tree[2]
            L_pred = tree[3]

            Q_pred = L_pred.T@L_pred
            e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)

            P_oprop = PHI @ jnp.linalg.solve(Q_pred, PHI.T)
            Sigma_t = jnp.fill_diagonal(
                P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
            )

            chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
            z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e, lower=True)

            ll = (
                -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
                - 0.5 * jnp.dot(z, z)
                - 0.5 * nobs * jnp.log(2 * jnp.pi)
            )

            return ll

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 4

        lls = jnp.array(
            jax.tree.map(comp_full_likelihood_good, mapping_elts, is_leaf=is_leaf)
        )
        ll = jnp.sum(lls)
    elif likelihood == "partial":
        mapping_elts = jax.tree.map(
            lambda t: (zs_tree[t], PHI_tree[t], seq[2][t], seq[3][t]),
            tuple(range(len(zs_tree))),
        )

        def comp_full_likelihood_good(tree):
            z = tree[0]
            PHI = tree[1]
            nu_pred = tree[2]
            L_pred = tree[3]
            
            Q_pred = L_pred.T@L_pred
            e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)

            P_oprop = PHI @ jnp.linalg.solve(Q_pred, PHI.T)
            Sigma_t = jnp.fill_diagonal(
                P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
            )

            chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
            z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, e, lower=True)

            ll = -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)

            return ll

        def is_leaf(node):
            return jax.tree.structure(node).num_leaves == 4

        lls = jnp.array(
            jax.tree.map(comp_full_likelihood_good, mapping_elts, is_leaf=is_leaf)
        )
        ll = jnp.sum(lls)
    elif likelihood == "none":
        ll = jnp.nan
    else:
        raise ValueError(
            "Invalid option for 'likelihood'. Choose from 'full', 'partial', 'none' (default: 'partial')."
        )

    return ll, seq[0], seq[1], seq[2], seq[3]


@partial(jax.jit, static_argnames=["likelihood"])
def information_filter_indep_experimental(
    nu_0: ArrayLike,  # initial information vector
    Q_0: ArrayLike,  # initial information matrix
    M: ArrayLike,
    PHI_tree: tuple,
    sigma2_eta: float,
    sigma2_eps: float,
    zs_tree: tuple,
    likelihood: str = "partial",
) -> tuple:
    """
    Experimental modification of [`information_filter_inder`](/reference/information_filter_indep.qmd)
    """

    mapping_elts = jax.tree.map(
        lambda t: (
            zs_tree[t],
            PHI_tree[t],
        ),
        tuple(range(len(zs_tree))),
    )

    r = nu_0.shape[0]

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_k = tup[1]
        i_k = PHI_k.T @ z_k / sigma2_eps
        I_k = PHI_k.T @ PHI_k / sigma2_eps
        return jnp.vstack((i_k, I_k))

    def is_leaf(node):
        # return len(node)==2 # IMPORTANT NOTE: what if T=2?
        return jax.tree.structure(node).num_leaves == 2
        # This works better, but could still be a problem if T=1. But
        # then, why would you even be filtering?
        # Added a raise to filter_information for this case

    scan_elts = jnp.array(jax.tree.map(informationify, mapping_elts, is_leaf=is_leaf))

    # This is one situation where I do not know how to avoid inverting
    # a matrix explicitely...
    # With the adjusted information filter, you should be able to fix this now
    Minv = jnp.linalg.solve(M, jnp.eye(r))
    sigma2_eta_inv = 1 / sigma2_eta

    def step(carry, scan_elt):
        _, nu_tt, Q_tt, _, _ = carry

        i_tp = scan_elt[0, :]
        I_tp = scan_elt[1:, :]

        S_t = Minv.T @ Q_tt @ Minv
        R_t = jnp.fill_diagonal(S_t, jnp.diag(S_t) + sigma2_eta_inv, inplace=False)
        J_t = S_t @ jnp.linalg.solve(R_t, jnp.eye(r))

        J_tmin = -jnp.fill_diagonal(J_t, jnp.diag(J_t - 1), inplace=False)

        nu_pred = J_tmin @ Minv.T @ nu_tt
        Q_pred = J_tmin @ S_t

        nu_up = nu_pred + i_tp
        Q_up = Q_pred + I_tp

        chol_Sigma_iota = jnp.linalg.cholesky(
            I_tp @ jnp.linalg.solve(Q_pred, I_tp.T) + I_tp
        )
        iota = i_tp - I_tp @ jnp.linalg.solve(Q_up, nu_up)
        z = jax.scipy.linalg.solve_triangular(chol_Sigma_iota, iota, lower=True)

        if likelihood == "full":
            ll_k = (
                -jnp.sum(jnp.log(jnp.diag(chol_Sigma_iota)))
                - 0.5 * jnp.dot(z, z)
                - 0.5 * r * jnp.log(2 * jnp.pi)
            )
        elif likelihood == "partial":
            ll_k = -jnp.sum(jnp.log(jnp.diag(chol_Sigma_iota))) - 0.5 * jnp.dot(z, z)
        else:
            ll_k = jnp.nan

        return (ll_k, nu_up, Q_up, nu_pred, Q_pred), (
            ll_k,
            nu_up,
            Q_up,
            nu_pred,
            Q_pred,
        )

    carry, seq = jl.scan(
        step,
        (0, nu_0, Q_0, jnp.zeros(r), jnp.eye(r)),
        scan_elts,
    )

    mapping_elts = jax.tree.map(
        lambda t: (seq[0][t], PHI_tree[t]), tuple(range(len(zs_tree)))
    )

    sigma2_eps_inv = 1 / sigma2_eps

    # def detmult(tup):
    #    Phieps = tup[1] * sigma2_eps_inv
    #    R = jnp.linalg.cholesky(Phieps@Phieps.T)
    #    return jnp.sum(jnp.diag(R)) + tup[0]

    def detmult(tup):
        Phieps = tup[1] * sigma2_eps_inv
        _, svs, _ = jax.scipy.linalg.svd(Phieps)
        return jnp.sum(jnp.log(svs)) + tup[0]

    ll = jnp.sum(jnp.array(jax.tree.map(detmult, mapping_elts, is_leaf=is_leaf)))

    return ll, seq[1], seq[2], seq[3], seq[4]



@jax.jit
def kalman_smoother(ms, Ps, mpreds, Ppreds, M):
    """NOT FULLY IMPLEMENTED"""
    nbasis = ms[0].shape[0]

    @jax.jit
    def step(carry, y):
        m_tmtm = y[0]
        P_tmtm = y[1]
        m_ttm = y[2]
        P_ttm = y[3]

        m_tT = carry[0]
        P_tT = carry[1]

        J_tm = P_tmtm @ M.T @ jnp.linalg.solve(P_ttm, jnp.eye(nbasis))

        m_tmT = m_tmtm - J_tm @ (m_tT - m_ttm)
        P_tmT = P_tmtm - J_tm @ (P_tT - P_ttm) @ J_tm.T

        return (m_tmT, P_tmT, J_tm), (m_tmT, P_tmT, J_tm)

    ys = (
        jnp.flip(ms[0:-1], axis=1),
        jnp.flip(Ps[0:-1], axis=1),
        jnp.flip(mpreds, axis=1),
        jnp.flip(Ppreds, axis=1),
    )
    init = (ms[-1], Ps[-1], jnp.zeros((nbasis, nbasis)))

    result = jl.scan(step, init, ys)

    return result


@jax.jit
def lag1_smoother(Ps, Js, K_T, PHI: ArrayLike, M: ArrayLike):
    """CURRENTLY OUT-OF-DATE AND UNTESTED"""

    nbasis = Ps[0].shape[0]
    P_TTmT = (jnp.eye(nbasis) - K_T @ PHI) @ M @ Ps[-2]

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


def kalman_filter_indep_vd(
    m_0: ArrayLike,
    P_0: ArrayLike,
    M: ArrayLike,
    PHI_tree: Tuple[ArrayLike, ...],
    sigma2_eta: float,
    sigma2_eps: float,
    zs_tree: Tuple[ArrayLike, ...],
    likelihood: str = "partial",
) -> tuple:

    nbasis = m_0.shape[0]

    nobs = jnp.array(jax.tree.map(lambda z: z.shape[0], zs_tree))
    max_obs = jnp.max(nobs)
    
    zs = jnp.array(jax.tree.map(lambda z: jnp.pad(z, (0, max_len - z.shape[0]),constant_values=jnp.nan), zs_tree))
    PHIs = jnp.array(jax.tree.map(lambda PHI: jnp.pad(PHI, ((0, max_len - PHI.shape[0]), (0,0)),constant_values=jnp.nan), PHI_tree))
    scan_elts = jnp.concatenate([zs[:,:, None], PHIs], axis=-1)
    
    @jax.jit
    def step(carry, elt):
        m_tt, P_tt, _, _, ll = carry
        n, z_t, PHI_t = elt
        z_t = z_t[0:n]
        PHI_t = PHI_t[:n,:n]
        
        # predict
        m_pred = M @ m_tt

        # Add sigma2_eps to the diagonal intelligently
        P_prop = M @ P_tt @ M.T
        P_pred = jnp.fill_diagonal(P_prop, sigma2_eta + jnp.diag(P_prop), inplace=False)

        # Update
        # Prediction error
        e_t = z_t - PHI @ m_pred

        # Prediction Variance
        P_oprop = PHI @ P_pred @ PHI.T
        Sigma_t = jnp.fill_diagonal(
            P_oprop, sigma2_eps + jnp.diag(P_oprop), inplace=False
        )

        # Kalman Gain
        K_t = (jnp.linalg.solve(Sigma_t, PHI) @ P_pred.T).T

        m_up = m_pred + K_t @ e_t
        P_up = (jnp.eye(nbasis) - K_t @ PHI) @ P_pred
        ll_new=jnp.nan
        return (m_up, P_up, m_pred, P_pred, ll_new), (
            m_up,
            P_up,
            m_pred,
            P_pred,
            ll_new,
        )

    carry, seq = jl.scan(
        step,
        (m_0, P_0, m_0, P_0, 0),
        (nobs, zs, PHIs),
    )

    return (carry[4], seq[0], seq[1], seq[2], seq[3])

def qr_R(A,B):
    """Wrapper for the stacked-QR decompositon"""
    return jnp.linalg.qr(jnp.vstack([A, B]), mode="r")

@jax.jit
def ql_L(A,B):
    """Computes the QL decomposition of matrix A."""
    A_flipped = jnp.flip(jnp.vstack([A, B]), axis=1)
    R = jnp.linalg.qr(A_flipped, mode='r')
    L = jnp.flip(R, axis=(0, 1))
    
    return L

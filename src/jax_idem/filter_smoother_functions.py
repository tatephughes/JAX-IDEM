#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax.lax as jl
from jax.typing import ArrayLike

from functools import partial

# ONLY SUPPORTS FIXED OBSERVATION LOCATIONS


@partial(jax.jit, static_argnames=["full_likelihood"])
def kalman_filter(
    m_0: ArrayLike,
    P_0: ArrayLike,
    M: ArrayLike,
    PHI_obs: ArrayLike,
    Sigma_eta: ArrayLike,
    Sigma_eps: ArrayLike,
    ztildes: ArrayLike,  # data matrix, with time across columns
    full_likelihood: bool = False
) -> tuple:
    """
    Applies the Kalman Filter to a wide-format matrix of data.
    For jit-ability, this only allows full (no missing) data in a wide format.
    I hypothesise that, with a temporally parallelised filter, this will both
    be quicker and have this limitation removed.

    See also [the version for independant errors](kalman_filter_indep.qmd)

    Parameters
    ----------
    m_0: ArrayLike (r,)
        The initial means of the process vector
    P_0: ArrayLike (r,r)
        The initial Covariance matrix of the process vector
    M: ArrayLike (r,r)
        The transition matrix of the process
    PHI_obs: ArrayLike (r,n)
        The process-to-data matrix
    Sigma_eta: Arraylike (r,r)
        The Covariance matrix of the process noise
    Sigma_eps: ArrayLike (n,n)
        The Covariance matrix of the observation noise
    ztildes: ArrayLike
        The observed data to be filtered, in matrix format
    Returns
    ----------
    A tuple containing:
        ll: The log (data) likelihood of the data
        ms: (T,r) The posterior means $m_{t \mid t}$ of the process given
    the data 1:t
        Ps: (T,r,r) The posterior covariance matrices $P_{t \mid t}$ of
    the process given the data 1:t
        mpreds: (T-1,r) The predicted next-step means $m_{t \mid t-1}$
    of the process given the data 1:t-1
        Ppreds: (T-1,r,r) The predicted next-step covariances
    $P_{t \mid t-1}$ of the process given the data 1:t-1
        Ks: (n,r) The Kalman Gains at each time step
    """

    nbasis = m_0.shape[0]
    nobs = ztildes.shape[0]

    @jax.jit
    def step(carry, z_t):
        m_tt, P_tt, _, _, ll, _ = carry

        # predict
        m_pred = M @ m_tt
        P_pred = M @ P_tt @ M.T + Sigma_eta

        # Update

        # Prediction Errors
        eps_t = z_t - PHI_obs @ m_pred

        Sigma_t = PHI_obs @ P_pred @ PHI_obs.T + Sigma_eps

        # Kalman Gain
        K_t = (
            jnp.linalg.solve(Sigma_t, PHI_obs)
            @ P_pred.T
        ).T

        m_up = m_pred + K_t @ eps_t

        P_up = (jnp.eye(nbasis) - K_t @ PHI_obs) @ P_pred

        # likelihood of epsilon, using cholesky decomposition
        chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
        z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, eps_t, lower=True)

        if full_likelihood:
            ll_new = ll - jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - \
                0.5 * jnp.dot(z, z) - 0.5 * nobs * jnp.log(2*jnp.pi)
        else:
            ll_new = ll - \
                jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)

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
        ztildes.T,
    )

    return (carry[4], seq[0], seq[1], seq[2][:], seq[3][:], seq[5][:])


@partial(jax.jit, static_argnames=["full_likelihood"])
def kalman_filter_indep(
    m_0: ArrayLike,
    P_0: ArrayLike,
    M: ArrayLike,
    PHI_obs: ArrayLike,
    sigma2_eta: float,
    sigma2_eps: float,
    ztildes: ArrayLike,  # data matrix, with time across columns
    full_likelihood: bool = False
) -> tuple:
    """
    Applies the Kalman Filter to a wide-format matrix of data, with some
    added efficiency for independant errors.
    For jit-ability, this only allows full (no missing) data in a wide format.
    I hypothesise that, with a temporally parallelised filter, this will both
    be quicker and have this limitation removed.

    Parameters
    ----------
    m_0: ArrayLike (r,)
        The initial means of the process vector
    P_0: ArrayLike (r,r)
        The initial Covariance matrix of the process vector
    M: ArrayLike (r,r)
        The transition matrix of the process
    PHI_obs: ArrayLike (r,n)
        The process-to-data matrix
    sigma2_eta: floats
        The Covariance matrix of the process noise
    sigma2_eps: floats
        The Covariance matrix of the observation noise
    ztildes: ArrayLike
        The observed data to be filtered, in matrix format
    Returns
    ----------
    A tuple containing:
        ll: The log (data) likelihood of the data
        ms: (T,r) The posterior means $m_{t \mid t}$ of the process given
    the data 1:t
        Ps: (T,r,r) The posterior covariance matrices $P_{t \mid t}$ of
    the process given the data 1:t
        mpreds: (T-1,r) The predicted next-step means $m_{t \mid t-1}$
    of the process given the data 1:t-1
        Ppreds: (T-1,r,r) The predicted next-step covariances
    $P_{t \mid t-1}$ of the process given the data 1:t-1
        Ks: (n,r) The Kalman Gains at each time step
    """

    nbasis = m_0.shape[0]
    nobs = ztildes.shape[0]

    @jax.jit
    def step(carry, z_t):
        m_tt, P_tt, _, _, ll, _ = carry

        # predict
        m_pred = M @ m_tt

        # Add sigma2_eps to the diagonal intelligently
        P_prop = M @ P_tt @ M.T
        P_pred = jnp.fill_diagonal(
            P_prop, sigma2_eta + jnp.diag(P_prop), inplace=False)

        # Update

        # Prediction error
        eps_t = z_t - PHI_obs @ m_pred

        # Prediction Variance
        P_oprop = PHI_obs @ P_pred @ PHI_obs.T
        Sigma_t = jnp.fill_diagonal(
            P_oprop, sigma2_eps+jnp.diag(P_oprop), inplace=False)

        # Kalman Gain
        K_t = (
            jnp.linalg.solve(Sigma_t, PHI_obs)
            @ P_pred.T
        ).T

        m_up = m_pred + K_t @ eps_t

        P_up = (jnp.eye(nbasis) - K_t @ PHI_obs) @ P_pred

        # likelihood of epsilon, using cholesky decomposition
        chol_Sigma_t = jnp.linalg.cholesky(Sigma_t)
        z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, eps_t, lower=True)
        if full_likelihood:
            ll_new = ll - jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - \
                0.5 * jnp.dot(z, z) - 0.5 * nobs * jnp.log(2*jnp.pi)
        else:
            ll_new = ll - \
                jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)

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
        ztildes.T,
    )

    return (carry[4], seq[0], seq[1], seq[2][1:], seq[3][1:], seq[5][1:])


@partial(jax.jit, static_argnames=["full_likelihood"])
def information_filter(
    nu_0: ArrayLike,  # initial information vector
    Q_0: ArrayLike,  # initial information matrix
    M: ArrayLike,
    PHI_obs_tuple: tuple,
    Sigma_eta: ArrayLike,
    Sigma_eps_tuple: tuple,
    ztildes: tuple,
    full_likelihood: bool = False
) -> tuple:

    mapping_elts = jax.tree.map(
        lambda t: (ztildes[t], PHI_obs_tuple[t],
                   Sigma_eps_tuple[t]), tuple(range(len(ztildes)))
    )

    r = nu_0.shape[0]

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_obs_k = tup[1]
        Sigma_eps_k = tup[2]
        i_k = PHI_obs_k.T @ jnp.linalg.solve(Sigma_eps_k, z_k)
        I_k = PHI_obs_k.T @ jnp.linalg.solve(Sigma_eps_k, PHI_obs_k)
        return jnp.vstack((i_k, I_k))

    def is_leaf(node):
        # return len(node)==2 # IMPORTANT NOTE: what if T=2?
        return jax.tree.structure(node).num_leaves == 3
        # This works better, but could still be a problem if T=1. But
        # then, why would you even be filtering?
        # Added a raise to filter_information for this case

    scan_elts = jnp.array(jax.tree.map(
        informationify, mapping_elts, is_leaf=is_leaf))

    # This is one situation where I do not know how to avoid inverting
    # a matrix explicitely...
    Minv = jnp.linalg.solve(M, jnp.eye(r))
    Sigma_eta_inv = jnp.linalg.solve(Sigma_eta, jnp.eye(r))

    def step(carry, scan_elt):
        nu_tt, Q_tt, ll = carry

        i_tp = scan_elt[0, :]
        I_tp = scan_elt[1:, :]

        S_t = Minv.T @ Q_tt @ Minv
        J_t = jnp.linalg.solve((S_t + Sigma_eta_inv).T, S_t.T).T

        nu_pred = (jnp.eye(r) - J_t) @ Minv.T @ nu_tt
        Q_pred = (jnp.eye(r) - J_t) @ S_t

        nu_up = nu_pred + i_tp
        Q_up = Q_pred + I_tp

        chol_Sigma_t = jnp.linalg.cholesky(
            I_tp @ jnp.linalg.solve(Q_pred, I_tp.T) + I_tp)
        iota = i_tp - I_tp @ jnp.linalg.solve(Q_up, nu_up)

        z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, iota, lower=True)

        if full_likelihood:
            ll_new = ll - jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - \
                0.5 * jnp.dot(z, z) - 0.5 * r * jnp.log(2*jnp.pi)
        else:
            ll_new = ll - \
                jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)

        return (nu_up, Q_up, ll_new), (nu_up, Q_up, ll_new)

    carry, seq = jl.scan(
        step,
        (nu_0, Q_0, 0),
        scan_elts,
    )

    return carry[2], seq[0], seq[1]


@partial(jax.jit, static_argnames=["full_likelihood"])
def information_filter_indep(
    nu_0: ArrayLike,  # initial information vector
    Q_0: ArrayLike,  # initial information matrix
    M: ArrayLike,
    PHI_obs_tuple: tuple,
    sigma2_eta: float,
    sigma2_eps: float,
    ztildes: tuple,
    full_likelihood: bool = True
) -> tuple:

    mapping_elts = jax.tree.map(
        lambda t: (ztildes[t], PHI_obs_tuple[t],
                   ), tuple(range(len(ztildes)))
    )

    r = nu_0.shape[0]

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_obs_k = tup[1]
        i_k = PHI_obs_k.T @ z_k / sigma2_eps
        I_k = PHI_obs_k.T @ PHI_obs_k / sigma2_eps
        return jnp.vstack((i_k, I_k))

    def is_leaf(node):
        # return len(node)==2 # IMPORTANT NOTE: what if T=2?
        return jax.tree.structure(node).num_leaves == 2
        # This works better, but could still be a problem if T=1. But
        # then, why would you even be filtering?
        # Added a raise to filter_information for this case

    scan_elts = jnp.array(jax.tree.map(
        informationify, mapping_elts, is_leaf=is_leaf))

    # This is one situation where I do not know how to avoid inverting
    # a matrix explicitely...
    # With the adjusted information filter, you should be able to fix this now
    Minv = jnp.linalg.solve(M, jnp.eye(r))
    sigma2_eta_inv = 1/sigma2_eta

    def step(carry, scan_elt):
        nu_tt, Q_tt, ll = carry

        i_tp = scan_elt[0, :]
        I_tp = scan_elt[1:, :]

        S_t = Minv.T @ Q_tt @ Minv
        R_t = jnp.fill_diagonal(S_t, jnp.diag(
            S_t) + sigma2_eta_inv, inplace=False)
        J_t = S_t @ jnp.linalg.solve(R_t, jnp.eye(r))

        J_tmin = jnp.fill_diagonal(J_t, jnp.diag(
            J_t) + 1, inplace=False)

        nu_pred = J_tmin @ Minv.T @ nu_tt
        Q_pred = J_tmin @ S_t

        nu_up = nu_pred + i_tp
        Q_up = Q_pred + I_tp

        chol_Sigma_t = jnp.linalg.cholesky(
            I_tp @ jnp.linalg.solve(Q_pred, I_tp.T) + I_tp)
        iota = i_tp - I_tp @ jnp.linalg.solve(Q_up, nu_up)

        z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, iota, lower=True)

        if full_likelihood:
            ll_new = ll - jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - \
                0.5 * jnp.dot(z, z) - 0.5 * r * jnp.log(2*jnp.pi)
        else:
            ll_new = ll - \
                jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)
        return (nu_up, Q_up, ll_new), (nu_up, Q_up, ll_new)

    carry, seq = jl.scan(
        step,
        (nu_0, Q_0, 0),
        scan_elts,
    )

    return carry[2], seq[0], seq[1]


@jax.jit
def kalman_smoother(ms, Ps, mpreds, Ppreds, M):
    # not implemented
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
def lag1_smoother(Ps, Js, K_T, PHI_obs: ArrayLike, M: ArrayLike):
    nbasis = Ps[0].shape[0]
    P_TTmT = (jnp.eye(nbasis) - K_T @ PHI_obs) @ M @ Ps[-2]

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

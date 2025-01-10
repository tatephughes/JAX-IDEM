#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import jax.lax as jl
from jax.typing import ArrayLike

from typing import Callable, NamedTuple

from utilities import *


# ONLY SUPPORTS FIXED OBSERVATION LOCATIONS
@jax.jit
def kalman_filter(
    m_0: ArrayLike,
    P_0: ArrayLike,
    M: ArrayLike,
    PHI_obs: ArrayLike,
    Sigma_eta: ArrayLike,
    Sigma_eps: ArrayLike,
    beta: ArrayLike,
    zs: ArrayLike,  # data matrix, with time across columns
    X_obs: ArrayLike,
) -> tuple:
    """
    Applies the Kalman Filter to a wide-format matrix of data.
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
    Sigma_eta: Arraylike (r,r)
      The Covariance matrix of the process noise
    Sigma_eps: ArrayLike (n,n)
      The Covariance matrix of the observation noise
    beta: ArrayLike (p,)
      The covariate coefficients for the data
    zs: ArrayLike
      The observed data to be filtered, in matrix format
    X_obs: ArrayLike (n,p)
      The matrix of covariate values
    Returns
    ----------
    A tuple containing:
      ll: The log (data) likelihood of the data
      ms: (T,r) The posterior means $m_{t mid t}$ of the process given
    the data 1:t
      Ps: (T,r,r) The posterior covariance matrices $P_{t mid t}$ of
    the process given the data 1:t
      mpreds: (T-1,r) The predicted next-step means $m_{t mid t-1}$
    of the process given the data 1:t-1
      Ppreds: (T-1,r,r) The predicted next-step covariances
    $P_{t mid t-1}$ of the process given the data 1:t-1
      Ks: (n,r) The Kalman Gains at each time step
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
        eps_t = z_t - PHI_obs @ m_pred - X_obs @ beta
        # K_t = (
        #    P_pred
        #    @ PHI_obs.T
        #    @ solve(PHI_obs @ P_pred @ PHI_obs.T + Sigma_eps, jnp.eye(nobs))
        # )

        K_t = (
            jnp.linalg.solve(PHI_obs @ P_pred @ PHI_obs.T + Sigma_eps, PHI_obs)
            @ P_pred.T
        ).T

        m_up = m_pred + K_t @ eps_t

        P_up = (jnp.eye(nbasis) - K_t @ PHI_obs) @ P_pred

        # likelihood of epsilon, using cholesky decomposition
        chol_Sigma_t = jnp.linalg.cholesky(PHI_obs @ P_pred @ PHI_obs.T + Sigma_eps)
        z = jax.scipy.linalg.solve_triangular(chol_Sigma_t, eps_t)
        # ll_new = ll + jnp.linalg.slogdet(chol_Sigma_t)[1] - 0.5 * jnp.dot(z, z)
        ll_new = ll - jnp.sum(jnp.log(jnp.diag(chol_Sigma_t))) - 0.5 * jnp.dot(z, z)

        return (m_up, P_up, m_pred, P_pred, ll_new, K_t), (
            m_up,
            P_up,
            m_pred,
            P_pred,
            ll_new,
            K_t,
        )

    result = jl.scan(
        step,
        (m_0, P_0, m_0, P_0, 0, jnp.zeros((nbasis, nobs))),
        zs.T,
    )

    return result


def information_filter(
    nu_0: ArrayLike,  # initial information vector
    Q_0: ArrayLike,  # initial information matrix
    M: ArrayLike,
    PHI_obs_tuple: tuple,
    sigma2_eta: tuple,
    sigma2_eps: float,
    beta: ArrayLike,
    z: tuple,
    X_obs: tuple,
) -> tuple:

    sigma2_eps_inv = 1 / sigma2_eps
    print(len(z))

    mapping_elts = jax.tree.map(
        lambda t: (z[t], PHI_obs_tuple[t]), tuple(range(len(z)))
    )

    nbasis = nu_0.shape[0]

    def informationify(tup: tuple):
        z_k = tup[0]
        PHI_obs_k = tup[1]
        i_k = PHI_obs_k.T @ z_k * sigma2_eps_inv
        I_k = PHI_obs_k.T @ PHI_obs_k * sigma2_eps_inv
        return jnp.vstack((i_k, I_k))

    def is_leaf(node):
        # return len(node)==2 # IMPORTANT NOTE: what if T=2?
        return jax.tree.structure(node).num_leaves == 2
        # This works better, but could still be a problem if T=1. But
        # then, why would you even be filtering?
        # Added an raise to filter_information for this case

    scan_elts = jnp.array(jax.tree.map(informationify, mapping_elts, is_leaf=is_leaf))

    # This is one situation where I do not know how to avoid inverting
    # a matrix explicitely...
    Minv = jnp.linalg.solve(M, jnp.eye(nbasis))

    def step(carry, scan_elt):

        nu_tt, Q_tt = carry

        i_tp = scan_elts[0, :][0, :]
        I_tp = scan_elts[0, :][1:, :]

        # predict step
        R_t = Minv.T @ Q_tt @ Minv
        C_t = jnp.linalg.solve(R_t.T + sigma2_eps_inv * jnp.eye(nbasis), R_t.T).T
        L_t = jnp.fill_diagonal(R_t, 1 - jnp.diag(R_t), inplace=False)

        Q_pred = L_t @ R_t @ C_t @ C_t.T * sigma2_eps_inv
        nu_pred = L_t @ Minv.T @ nu_tt

        # update step
        nu_up = nu_tt + i_tp
        Q_up = Q_tt + I_tp

        return (nu_up, Q_up), (nu_up, Q_up)

    result = jl.scan(
        step,
        (nu_0, Q_0),
        scan_elts,
    )

    return result

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

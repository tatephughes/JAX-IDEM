#!/usr/bin/env python3

"""
Test functions, implementing 'naiive' versions of the functions to test
against.
"""

import jax.numpy as jnp


def kalman_filter_naive(m_0, P_0, M, PHI_obs, Sigma_eta, Sigma_eps, ztildes):

    r = m_0.shape[0]
    n = ztildes.shape[1]
    T = ztildes.shape[0]

    ll = 0

    ms = [m_0]
    Ps = [P_0]
    mpreds = []
    Ppreds = []
    Ks = []

    for i in range(T):

        mpred = M @ ms[i]
        Ppred = M @ Ps[i] @ M.T + Sigma_eta

        K = Ppred @ PHI_obs.T @ jnp.linalg.solve(PHI_obs @
                                                 Ppred @ PHI_obs.T + Sigma_eps, jnp.eye(n))
        e = ztildes[i] - PHI_obs @ mpred

        mup = mpred + K @ e
        Pup = (jnp.eye(r) - K @ PHI_obs) @ Ppred

        Sigma_t = PHI_obs @ Ppred @ PHI_obs.T + Sigma_eps

        ll = ll - 0.5 * n * jnp.log(2*jnp.pi) - \
            0.5 * jnp.log(jnp.linalg.det(Sigma_t)) -\
            0.5 * e.T @ jnp.linalg.solve(Sigma_t, e)

        ms.append(mup)
        Ps.append(Pup)
        mpreds.append(mpred)
        Ppreds.append(Ppred)
        Ks.append(K)

    ms = jnp.array(ms)
    Ps = jnp.array(Ps)
    mpreds = jnp.array(mpreds)
    Ppreds = jnp.array(Ppreds)
    Ks = jnp.array(Ks)

    return (ll, ms[1:], Ps[1:], mpreds, Ppreds, Ks)


def information_filter_naive(nu_0, Q_0, M, PHI_obs_tuple, Sigma_eta, Sigma_eps_tuple, zs):

    r = nu_0.shape[0]
    T = len(zs)

    ll = 0

    nus = [nu_0]
    Qs = [Q_0]

    Minv = jnp.linalg.solve(M, jnp.eye(r))
    Sigma_eta_inv = jnp.linalg.solve(Sigma_eta, jnp.eye(r))

    for i in range(T):

        i_t = PHI_obs_tuple[i].T @ jnp.linalg.solve(Sigma_eps_tuple[i], zs[i])
        I_t = PHI_obs_tuple[i].T @ jnp.linalg.solve(
            Sigma_eps_tuple[i], PHI_obs_tuple[i])

        S_t = Minv.T @ Qs[-1] @ Minv
        J_t = S_t @ jnp.linalg.solve((S_t + Sigma_eta_inv), jnp.eye(r))

        nu_pred = (jnp.eye(r) - J_t) @ Minv.T @ nus[-1]
        Q_pred = (jnp.eye(r) - J_t) @ S_t

        nu_up = nu_pred + i_t
        Q_up = Q_pred + I_t

        iota = i_t - I_t @ jnp.linalg.solve(Q_pred, nu_pred)

        Sigma_iota = I_t @ jnp.linalg.solve(Q_pred, I_t.T) + I_t

        ll = ll - 0.5 * r * jnp.log(2*jnp.pi) - \
            0.5 * jnp.log(jnp.linalg.det(Sigma_iota)) -\
            0.5 * iota.T @ jnp.linalg.solve(Sigma_iota, iota)

        nus.append(nu_up)
        Qs.append(Q_up)

    nus = jnp.array(nus)
    Qs = jnp.array(Qs)

    return (ll, nus[1:], Qs[1:])

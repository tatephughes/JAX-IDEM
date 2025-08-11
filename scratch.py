def information_filter_nojit(
        nu_0,
        Q_0,
        M,
        PHI,
        S2_eta,
        S2_eps,
        zs
):

    r = nu_0.shape[0]
    nobs = zs.shape[1]

    i = jax.vmap(lambda z: PHI.T @ jnp.linalg.solve(S2_eps, z), in_axes=0)(zs)
    I = PHI.T @ jnp.linalg.solve(S2_eps, PHI)
    
    Minv = jnp.linalg.inv(M)
    S2_eta_inv = jnp.linalg.inv(S2_eta)

    def step(carry, i):
        nu_tt, Q_tt, _, _ = carry

        S_t = Minv.T @ Q_tt @ Minv
        J_t = jnp.linalg.solve((S_t + S2_eta_inv).T, S_t.T).T
        
        nu_pred = (jnp.eye(r) - J_t) @ Minv.T @ nu_tt
        Q_pred = (jnp.eye(r) - J_t) @ S_t

        nu_up = nu_pred + i
        Q_up = Q_pred + I

        new_carry = (nu_up, Q_up, nu_pred, Q_pred)
        
        return new_carry, new_carry
    
    carry, seq = jl.scan(
        step,
        (nu_0, Q_0, jnp.zeros(r), jnp.eye(r)),
        i,
    )
    
    nus, Qs, nupreds, Qpreds = (seq[0], seq[1], seq[2], seq[3])

    def likelihood_func(z, nu_pred, Q_pred):
        e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)
        chol_Sigma_t = jnp.linalg.cholesky(
            PHI @ jnp.linalg.solve(Q_pred, PHI.T) + S2_eps
        )
        z = st(chol_Sigma_t, e, lower=True)

        ll = (
            -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
            - 0.5 * jnp.dot(z, z)
            - 0.5 * nobs * jnp.log(2 * jnp.pi)
        )
        return ll

    lls = jax.vmap(likelihood_func)(zs, nupreds, Qpreds)
    ll = jnp.sum(lls)
    
    filt_results = {"ll": ll,
                    "nus": nus,
                    "Qs": Qs,
                    "nupreds": nupreds,
                    "Qpreds": Qpreds}

    return filt_results


def information_filter_nojit(
        nu_0,
        Q_0,
        M,
        PHI,
        S2_eta,
        S2_eps,
        zs
):

    r = nu_0.shape[0]
    nobs = zs.shape[1]

    i = jax.vmap(lambda z: PHI.T @ jnp.linalg.solve(S2_eps, z), in_axes=0)(zs)
    I = PHI.T @ jnp.linalg.solve(S2_eps, PHI)
    
    Minv = jnp.linalg.inv(M)
    S2_eta_inv = jnp.linalg.inv(S2_eta)

    @jax.jit
    def step(carry, i):
        nu_tt, Q_tt, _, _ = carry

        S_t = Minv.T @ Q_tt @ Minv
        J_t = jnp.linalg.solve((S_t + S2_eta_inv).T, S_t.T).T
        
        nu_pred = (jnp.eye(r) - J_t) @ Minv.T @ nu_tt
        Q_pred = (jnp.eye(r) - J_t) @ S_t

        nu_up = nu_pred + i
        Q_up = Q_pred + I

        new_carry = (nu_up, Q_up, nu_pred, Q_pred)
        
        return new_carry, new_carry
    
    carry, seq = jl.scan(
        step,
        (nu_0, Q_0, jnp.zeros(r), jnp.eye(r)),
        i,
    )
    
    nus, Qs, nupreds, Qpreds = (seq[0], seq[1], seq[2], seq[3])

    @jax.jit
    def likelihood_func(z, nu_pred, Q_pred):
        e = z - PHI @ jnp.linalg.solve(Q_pred, nu_pred)
        chol_Sigma_t = jnp.linalg.cholesky(
            PHI @ jnp.linalg.solve(Q_pred, PHI.T) + S2_eps
        )
        z = st(chol_Sigma_t, e, lower=True)

        ll = (
            -jnp.sum(jnp.log(jnp.diag(chol_Sigma_t)))
            - 0.5 * jnp.dot(z, z)
            - 0.5 * nobs * jnp.log(2 * jnp.pi)
        )
        return ll

    lls = jax.vmap(likelihood_func)(zs, nupreds, Qpreds)
    ll = jnp.sum(lls)
    
    filt_results = {"ll": ll,
                    "nus": nus,
                    "Qs": Qs,
                    "nupreds": nupreds,
                    "Qpreds": Qpreds}

    return filt_results


jit_inf_n   = lambda tup: information_filter_nojit(nu_0, Q_0, tup[0], tup[1], tup[2], tup[3], tup[4])['ll']
jit_inf_y   = lambda tup: information_filter(nu_0, Q_0, tup[0], tup[1], tup[2], tup[3], tup[4])['ll']

time_inf_n   = time_jit(key, jit_inf_n, inp_tree, n=100)
time_inf_y   = time_jit(key, jit_inf_y, inp_tree, n=100)

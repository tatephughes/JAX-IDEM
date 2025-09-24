import jax

jax.config.update("jax_enable_x64", True)
import os
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import jaxidem.idem as idem
import jaxidem.utils as utils
import matplotlib.pyplot as plt
from jax.scipy.linalg import solve

from quadax import quadgk


key = jr.PRNGKey(1)  # PRNG key

radar_df = pd.read_csv("./data/radar_df.csv")
radar_data = utils.pd_to_st(radar_df, "s2", "s1", "time", "z")
model = idem.init_model(
    data=radar_data,
    basis_type="cosine",
    basis_args=[20],
    n_int_grid=2000,
    n_process_grid=100,
)


self = model

ks = self.kernel.params


def con_M_new(ks):
    def k(s, r):
        theta = (
            ks[0] @ self.kernel.basis[0].vfun(s),
            ks[1] @ self.kernel.basis[1].vfun(s),
            jnp.array(
                [
                    ks[2] @ self.kernel.basis[2].vfun(s),
                    ks[3] @ self.kernel.basis[3].vfun(s),
                ]
            ),
        )
        return theta[0] * jnp.exp(-(jnp.sum((r - s - theta[2]) ** 2)) / theta[1])

    # vec_ker = jax.vmap(jax.vmap(kernel_func, in_axes=(None, 0)), in_axes=(0, None))
    # K = vec_ker(self.process_grid.coords, self.process_grid.coords)
    # TODO: Investigate better, faster, more accurate ways to ocmpute this?

    xmin = jnp.min(self.process_grid.coords[:, 0])
    xmax = jnp.max(self.process_grid.coords[:, 0])
    ymin = jnp.min(self.process_grid.coords[:, 1])
    ymax = jnp.max(self.process_grid.coords[:, 1])
    
    phi = self.process_basis.vfun

    def integrand(s, r):
        return k(s, r) * jnp.outer(phi(s), phi(r))

   return quadgk(lambda s: quadgk(lambda r: f(s, r), -1.0, 1.0)[0], 0.0, jnp.pi)[0]

    
    M = solve(self.GRAM, quadgk(integrand))
    
    return (
        solve(self.GRAM, self.PHI_proc.T @ K @ self.PHI_proc)
        * self.process_grid.area**2
    )


# Outer integral over s
def double_integral():
    def outer_integral(s):
        # Inner integral over r
        def inner_integral(r):
            return integrand(s, r)  # shape (400, 400)

        # Integrate over r and get matrix
        result_r, _ = quadgk(lambda x: quadgk(lambda y: inner_integral(r[0], r[1]), , 1.0)[0], 0.0, jnp.pi)[0]
        return result_r  # shape (400, 400)

    # Integrate over s and get final matrix
    result_s, _ = quadgk(outer_integral, s_min, s_max)
    return result_s  # shape (400, 400)


def inner_integral(s):
    def integrate_r1(r1):
        def integrate_r2(r2):
            r = jnp.array([r1, r2])
            return integrand(s, r)  # shape (400, 400)
        return quadgk(integrate_r2, [ymin, ymax])[0]  # shape (400, 400)
    return quadgk(integrate_r1, [xmin, xmax])[0]  # shape (400, 400)


con_M_old = self.con_M


time_old = utils.time_jit(key, con_M_old, ks, n=200)
time_new = utils.time_jit(key, con_M_new, ks, n=200)


print("New compile/compute", time_new.compile_time, time_new.average_time)
print("Old compile/compute", time_old.compile_time, time_old.average_time)

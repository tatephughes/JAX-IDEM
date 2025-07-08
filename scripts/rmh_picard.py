import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as rand
from jax_tqdm import scan_tqdm

from jax.scipy.stats.multivariate_normal import logpdf as logmvn
from jax.random import multivariate_normal as smvn

target_mean = jnp.array([0,0])
target_cov = jnp.array([[1,0.5],[0.5,1]])


def log_target(state):
    return logmvn(state, target_mean, target_cov)

init_state = jnp.zeros(2)
prop_cov = jnp.eye(2)


n = 1000

@scan_tqdm(n)
def rmh_step(state, key):
    prop_key, acc_key = jax.random.split(key, 2)

    proposal = smvn(prop_key, state, prop_cov)
    r = log_target(proposal) - log_target(state)

    log_prob = jnp.minimum(0.0, r)

    u = jax.random.uniform(acc_key)

    new_state = jl.cond((jnp.log(u) < log_prob),
                        0, lambda _: proposal,
                        0, lambda _: state, 0)

    return new_state, new_state


key = jax.random.PRNGKey(42)
keys = jax.random.split(key, n)

result = jax.lax.scan(rmh_step, init_state, keys)

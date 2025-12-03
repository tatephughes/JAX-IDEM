import jax
# Some filters, particularily the sqrt filters, may work in 32-bit, but expect instabilities.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Module imports
import jaxidem.utils as utils
import jaxidem.idem as idem
import jaxidem.filters as filts

import jax.lax as jl
import jax.random as jr
from jax.random import multivariate_normal as smvn
from jax.random import normal as snorm
from functools import partial
from jaxidem.utils import add_variance
from jax.scipy.linalg import solve
from jax.scipy.linalg import solve_triangular as st


from jax.scipy.stats.multivariate_normal import logpdf as lpmvn
from jax.scipy.stats.norm import logpdf as lpnorm


seed = 4
key = jax.random.PRNGKey(seed)
keys = jax.random.split(key, 10)


T = 20
nobs = 400
process_basis = utils.place_cosine_basis(N=10) 

# This puts a point of mass at 0.1,0.9 and a point of 0 at 0.9, 0.1.
# Using the least squares estimator, this creates a point at the corner.
inp_data = jnp.array([[0.1, 0.9, 100],
                      [0.9, 0.1, 0]])


# create a 'ball' at the top left using least squares
PHI = process_basis.mfun(inp_data[:,0:2])
alpha_0 = jnp.linalg.pinv(PHI.T@PHI)@PHI.T @ inp_data[:,2]



# this function creates models like in this example. for practical uses, use `init_model`.

# Fudged numbers to look at least a little interesting.
K_basis = (
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
)
# These values create reasonable drift and diffusion at
s = 0.0001
k = (
    jnp.array([1/(2*jnp.pi*s)]), # Scale parameter    (θ_1)
    jnp.array([2*s]), # Shape paramter  (θ_2)
    jnp.array([-0.02]), # X-axis drift    (θ_3) 
    jnp.array([0.02]), # Y-axis drift     (θ_4)
)
kernel = idem.param_exp_kernel(K_basis, k)

model = idem.gen_example_idem(keys[0],
                              process_basis = process_basis,
                              beta = jnp.array([1.0]),
                              kernel=kernel)




coords = jax.random.uniform(
                keys[0],
                shape=(nobs, 2),
                minval=0,
                maxval=1,
            )
times = jnp.repeat(jnp.arange(1, T + 1), coords.shape[0])
rep_coords = jnp.tile(coords, (T, 1))
x = rep_coords[:,0]
y = rep_coords[:,1]


process_data, obs_data = model.simulate(keys[1], x, y, times, alpha_0 = alpha_0)

# For normalising axes
zmin = float(jnp.nanmin(obs_data.z))
zmax = float(jnp.nanmax(obs_data.z))
zbounds = [zmin, zmax]






# Model to fit to the data

K_basis = (
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
    utils.place_basis(nres=1, min_knot_num=1, basis_fun=lambda s, r: 1),
)
s = 0.002
k = (
    jnp.array([1/(2*jnp.pi*s)]), # Scale parameter    (θ_1)
    jnp.array([2*s]), # Shape paramter  (θ_2)
    jnp.array([-0.00]), # X-axis drift    (θ_3) 
    jnp.array([0.00]), # Y-axis drift     (θ_4)
)
kernel = idem.param_exp_kernel(K_basis, k)

model_0 = idem.gen_example_idem(keys[0],
                                process_basis = process_basis,
                                beta = jnp.array([0.0]),
                                kernel=kernel,
                                S2_eta = 0.01**2,
                                S2_eps = 0.01**2)



filterer = model_0.get_filter(obs_data,
                              method="sikalman",
                              likelihood="none")
M = model_0.M
S2_eta = model_0.S2_eta
S2_eta_shape = model_0.S2_eta_shape

def sample_alphas(key, params):

    filt_results = filterer(params)

    ms = filt_results['ms']
    Ps = filt_results['Ps'] # Will need adjustment depending on the filter.

    def backsample(carry, x):

        alpha_tp = carry
        m_tt, P_tt, key = x

        mpred = M @ m_tt
        Ppred = add_variance(M @ P_tt @ M.T, S2_eta, S2_eta_shape)
        
        C = solve(Ppred, M @ P_tt).T

        alpha_t = smvn(key, m_tt + C @ (alpha_tp - mpred), P_tt - C @ Ppred @ C.T)
        
        return alpha_t, alpha_t

    keys = jr.split(key, ms.shape[0])
    
    xs = (jnp.flip(ms[:-1], axis=0),
          jnp.flip(Ps[:-1], axis=0),
          keys[1:])

    alpha_T = smvn(keys[0], ms[-1], Ps[-1])
    
    carry, seq = jl.scan(
        backsample,
        (alpha_T),
        xs,
    )

    return jnp.vstack([jnp.flip(seq, axis=0), alpha_T[None,:]])


def sample_invgamma(key, alpha, beta):
    g = jr.gamma(key, alpha)/beta
    return 1/g




init_tker = model_0.params["trans_kernel_params"]


# Kernel prior and general posterior pdf
@jax.jit
def kern_prior(tker_params):

    return (lpnorm(tker_params[0], init_tker[0], 1) 
            + lpnorm(tker_params[1], init_tker[1], 10)
            + lpnorm(tker_params[2], init_tker[2], 0.1)
            + lpnorm(tker_params[3], init_tker[3], 0.1))[0]

@jax.jit
def kern_post(tker_params, alphas, S2_eta):

    prior = kern_prior(tker_params)

    means = M @ alphas[1:].T

    proc_like = jnp.sum(jl.map(lambda x: lpmvn(x[0], x[1], S2_eta), (alphas[1:], means.T)))
    
    return prior + proc_like

# initialisers for the hmc
import blackjax
mwg_init_theta = blackjax.hmc.init
mwg_step_fn_theta = blackjax.hmc.build_kernel()

zs_tree = obs_data.zs_tree
X_obs_tree = obs_data.X_obs_tree
PHI_obs_tree = jax.tree.map(model_0.process_basis.mfun, obs_data.coords_tree)
sn = sum([z.size for z in zs_tree])


def square(vec): return jnp.inner(vec, vec)

T = obs_data.T

# S2_eta prior
alp_eta_0 = 2
lam_eta_0 = model_0.S2_eta
V = jnp.eye(model_0.nbasis)
# variance of process noise is S2_eta * V

# S2_eps prior
alp_eps_0 = 2
lam_eps_0 = model_0.S2_eps

# beta prior
p = model_0.beta.size
s2_beta_0 = 1
mu_beta_0 = model_0.beta


parameters = {"trans_kernel_params": {
        "inverse_mass_matrix": jnp.array([1., 1., 1., 1.]),
        "num_integration_steps": 100,
        "step_size": 1e-2
    }}

key = jr.PRNGKey(seed=2)
alphas = sample_alphas(key, model_0.params)
alphas_0 = alphas

initial_state = {
    "trans_kernel_params": mwg_init_theta(
        position=init_tker,
        logdensity_fn=lambda theta: kern_post(tker_params = theta,
                                              alphas=alphas_0,
                                              S2_eta=S2_eta)),
    "S2_eta": model_0.S2_eta,
    "S2_eps": model_0.S2_eps,
    "beta": model_0.beta,
    "alphas": alphas_0
}



state = initial_state






def mwg_kernel(key, state, pars):

    trans_theta = state['trans_kernel_params']
    S2_eta = state['S2_eta']
    S2_eps = state['S2_eps']
    beta = state['beta']
    alphas = state['alphas']

    key_theta, key_eta, key_eps, key_beta, key_alphas= jax.random.split(key, num=5)

    # Sample from theta using a HMC step
    
    def logdensity_theta(theta): return kern_post(tker_params = theta,
                                                  alphas=alphas,
                                                  S2_eta=S2_eta)
    # Doesn't this get recompiled at each step?
    
    state["trans_kernel_params"] = mwg_init_theta(
        position=trans_theta.position,
        logdensity_fn=logdensity_theta
        )
    # Not so sure about this code; isn't this mutation?
    state["trans_kernel_params"], _ = mwg_step_fn_theta(
        rng_key=key_theta,
        state=trans_theta,
        logdensity_fn=logdensity_theta,
        **parameters["trans_kernel_params"]
    )
    
    trans_theta = state['trans_kernel_params']


    # Sample from the variances with the conditional posteriors

    # process variance
    A = alphas[1:] - alphas[:-1] @ M.T
    #S = Psi_0 + A.T@A
    #nu = nu_0 + T
    #S2_eta = sample_invwishart(key_eta, S, nu)

    # Actually keep S2_eta scalar!

    alp = alp_eta_0 + 0.5*model_0.nbasis*T
    lam = lam_eta_0 + 0.5*jnp.sum(jl.map(lambda a: a.T @ solve(V, a, assume_a="pos"), A))
    S2_eta = sample_invgamma(key_eta, alp, lam)
    state['S2_eta'] = S2_eta
    

    # observation variance
    alphas_tree = list(alphas[1:])
    
    square_diffs = jnp.array(jax.tree.map(lambda z,x,p,a: square(z - x@beta - p@a), zs_tree, X_obs_tree, PHI_obs_tree, alphas_tree))
    
    alp = alp_eps_0 + 0.5*sn
    lam = lam_eps_0 + 0.5 * jnp.sum(square_diffs)
    S2_eps = sample_invgamma(key_eps, alp, lam)
    state['S2_eps'] = S2_eps

    # Linear coeffients
    B = jnp.eye(p)/s2_beta_0 + jnp.sum(jnp.array(jax.tree.map(lambda x: x.T@x, X_obs_tree)))/S2_eps
    diffs = jax.tree.map(lambda z,x,p,a: x.T@(z - p @ a), zs_tree, X_obs_tree, PHI_obs_tree, alphas_tree)
    mu = solve(B, jnp.sum(jnp.array(diffs), axis=0), assume_a='pos')/S2_eps

    beta = solve(B, snorm(key_beta, shape = (p,)), assume_a='pos') + mu
    state['beta'] = beta
    
    # Sample from alphas using FFBS
    params = {
            "log_S2_eps": jnp.log(S2_eps),
            "log_S2_eta": jnp.log(S2_eta),
            "trans_kernel_params": trans_theta.position,
            "beta": beta,
        }
    state["alphas"] = sample_alphas(key_alphas, params)

    return state





    
@jax.jit
def one_step(state, rng_key):
    state = mwg_kernel(
        key=rng_key,
        state=state,
        pars=parameters
    )
    return state, state
states_list = [initial_state]

key = jr.PRNGKey(seed=1)

num_samples = 1000
keys = jax.random.split(key, num_samples)

for i in range(num_samples):
    states_list.append(one_step(states_list[-1], keys[i])[0])

    print(f"\nCurrent offsets: {states_list[-1]['trans_kernel_params'].position[2:]}")

    last10states = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *states_list[-100:] # don't include last point
    )
    
    print(f"Running averages: {jax.tree.map(lambda arr:jnp.mean(arr), last10states['trans_kernel_params'].position[2:])}")
    

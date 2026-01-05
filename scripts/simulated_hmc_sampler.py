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

import csv
from datetime import datetime
import os
import pickle

from jax.scipy.stats.multivariate_normal import logpdf as lmvn
from jax.scipy.stats.norm import logpdf as lpnorm

from tqdm.auto import tqdm

seed = 4
key = jax.random.PRNGKey(seed)
keys = jax.random.split(key, 10)


T = 20
nobs = 400
process_basis = utils.place_cosine_basis(N=10) 

# This puts a point of mass at 0.1,0.9 and a point of 0 at 0.9, 0.1.
# Using the least squares estimator, this creates a point at the corner.
inp_data = jnp.array([[0.1, 0.9, 100],
#                      [0.9, 0.1, 0],
                      [0.3, 0.9, 100],
                      [0.1, 0.7, 100],
                      [0.3, 0.7, 100]])

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
    jnp.array([0.5*s]), # Shape paramter  (θ_2)
    jnp.array([-0.025]), # X-axis drift    (θ_3) 
    jnp.array([0.025]), # Y-axis drift     (θ_4)
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


log_marginal = model_0.get_log_like(obs_data, method="sqinf", likelihood='partial', P_0 = 1000*jnp.eye(process_basis.nbasis))

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

p = model.beta.size

init_tker = model_0.params["trans_kernel_params"]
# Kernel prior and general posterior pdf
@jax.jit
def kern_prior(tker_params):

    return (lpnorm(tker_params[0], init_tker[0], 1) 
            + lpnorm(tker_params[1], init_tker[1], 10)
            + lpnorm(tker_params[2], init_tker[2], 0.1)
            + lpnorm(tker_params[3], init_tker[3], 0.1))[0]


def inv_gamma_logpdf(x, alpha, beta):

    return (-alpha-1)*jnp.log(x) - beta/x

def log_post(params):
    
    log_beta_prior = lmvn(params['beta'], mu_beta_0, s2_beta_0 * jnp.eye(p))

    log_eta_prior = inv_gamma_logpdf(jnp.exp(params['log_S2_eta']), alp_eta_0, lam_eta_0)
    log_eps_prior = inv_gamma_logpdf(jnp.exp(params['log_S2_eps']), alp_eps_0, lam_eps_0)

    log_kern_prior = kern_prior(params['trans_kernel_params'])

    return log_marginal(params) + log_beta_prior + log_eta_prior + log_eps_prior + log_kern_prior

    
print('Log posterior made!')



script_path = os.path.abspath(__file__)
directory = os.path.dirname(script_path)



current_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
csv_file = os.path.join(directory, f'results/{current_datetime}_results_hmc_chain.csv')

# Read adaptive results for tuning
import pickle

with open(os.path.join(directory, "results/results_adaptive.pickle"), 'rb') as file:
    j, x, x_mean, prop_cov = pickle.load(file)

_, unflat = utils.flatten_and_unflatten(model_0.params)

# Build the kernel
#inverse_mass_matrix = 0.01*jnp.ones(model_0.nparams)
imm = prop_cov
num_integration_steps = 10
step_size = 1e-5

import blackjax

hmc = blackjax.hmc(log_post, step_size, imm, num_integration_steps=25)

# Initialize the state
state = hmc.init(unflat(x_mean))

hmc_sample = []

# Iterate
step = jax.jit(hmc.step)


sample_key = jr.PRNGKey(67)

hmc_n = 100
accepted = 0
for i in range(hmc_n):

    hmc_key = jax.random.fold_in(sample_key, i)
    state, info = step(hmc_key, state)

    accepted = accepted + info.is_accepted
    hmc_sample.append(state.position)

    print(f"\nCurrent offsets: {state.position['trans_kernel_params'][2:]}", flush=True)
    print(f"Acc Ratio; {accepted / (i+1)}")

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(jnp.concatenate([jnp.array([info.is_accepted]), utils.flatten(state.position)[0]]))


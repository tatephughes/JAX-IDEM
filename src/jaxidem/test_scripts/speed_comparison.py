from tqdm.auto import tqdm
import csv
import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import idem
import utilities
from utilities import time_jit
import filter_smoother_functions as fsf
import timeit
import importlib
import optax
import time
import jax.random as rand
import jax.numpy as jnp
import jax
platform_name = input("Choose platform (cpu or gpu): ").strip().lower()
jax.config.update('jax_platform_name', platform_name)
jax.config.update('jax_enable_x64', True)

reps = int(input("How many times do you want to run the functions? (integer): "))

print(f"Current working directory: {os.getcwd()}")

# Define CSV filename
csv_filename = f"timing_results_{platform_name}.csv"

# Write the header once (if the file is empty or new)
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["nobs", "kf",
                     "sq", "if", "sqif",
                     "kf_comp", "sq_comp", "if_comp", "sqif_comp"])

key = jax.random.PRNGKey(3)


#def timer(func, inp, n, desc):
#
#    tot_time = 0
#
#    result = func(inp) # one run for jit compilation
#
#    for _ in tqdm(range(n), desc=desc):
#        start_time = time.time()
#        result = func(inp)
#        elapsed = time.time() - start_time
#        tot_time = tot_time + elapsed
#
#    av_time = tot_time / n
#
#    return av_time * 1000



process_basis = utilities.place_cosine_basis(N=5)

keys = rand.split(key, 4)
nbasis = process_basis.nbasis

model = idem.gen_example_idem(
    keys[0], k_spat_inv=True,
    process_basis=process_basis,
    Sigma_eta=0.01**2*jnp.eye(nbasis),
    sigma2_eps=0.01**2,
    beta=jnp.array([.2, .2, .2]),
)

alpha_0 = jax.random.normal(keys[1], shape=(nbasis,))

for n in range(300,302):

    print(f"\n\n\n nobs={n}")

    process_data, obs_data = model.simulate(
        nobs=n, T=50, key=keys[2], alpha_0=alpha_0)

    unique_times = jnp.unique(obs_data.t)

    obs_data_wide = obs_data.as_wide()
    X_obs = jnp.column_stack(
        [jnp.ones(obs_data_wide['x'].shape[0]), obs_data_wide['x'], obs_data_wide['y']])

    obs_locs = jnp.column_stack([obs_data_wide["x"], obs_data_wide["y"]])
    obs_locs_long = jnp.column_stack(
        jnp.column_stack((obs_data.x, obs_data.y))).T

    obs_locs_tuple = [obs_locs_long[obs_data.t == t][:, 0:]
                      for t in unique_times]

    PHI_obs = model.process_basis.mfun(obs_locs)
    PHI_obs_tuple = jax.tree.map(model.process_basis.mfun, obs_locs_tuple)

    X_obs_tuple = jax.tree.map(
        lambda locs: jnp.column_stack((jnp.ones(len(locs
                                                    )), locs)), obs_locs_tuple)

    nbasis = model.process_basis.nbasis

    m_0 = jnp.zeros(nbasis)
    nu_0 = jnp.zeros(nbasis)
    P_0 = 100 * jnp.eye(nbasis)
    U_0 = 10 * jnp.eye(nbasis)
    Q_0 = 0.01 * jnp.eye(nbasis)
    R_0 = 0.1 * jnp.eye(nbasis)

    beta = model.beta
    M = model.M
    sigma2_eps = model.sigma2_eps
    sigma2_eta = model.Sigma_eta[0,0]
    zs_tuple = [obs_data.z[obs_data.t == t] for t in unique_times]

    ztildes = obs_data_wide["z"] - (X_obs @ model.beta)[:, None]

    def is_leaf(node):
        return jax.tree.structure(node).num_leaves == 2
    mapping_elts = tuple(
        [[zs_tuple[i], X_obs_tuple[i]] for i in range(len(zs_tuple))])

    def tildify(z, X_obs_i, beta):
        return z - X_obs_i @ beta

    ztildes_tuple = jax.tree.map(
        lambda tup: tildify(tup[0], tup[1], beta), mapping_elts, is_leaf=is_leaf)

    @jax.jit
    def kf(params):
        (
            log_sigma2_eta,
            log_sigma2_eps,
            ks,
            beta,
        ) = params

        logks1, logks2, ks3, ks4 = ks

        ks1 = jnp.exp(logks1)
        ks2 = jnp.exp(logks2)

        sigma2_eta = jnp.exp(log_sigma2_eta)
        sigma2_eps = jnp.exp(log_sigma2_eps)
        M = model.con_M((ks1, ks2, ks3, ks4))
        
        ztildes = obs_data_wide["z"] - (X_obs @ beta)[:, None]
        
        #ztildes_tuple = jax.tree.map(
        #    lambda tup: tildify(tup[0], tup[1], beta), mapping_elts, is_leaf=is_leaf)
        
        ll, _, _, _, _, _ = fsf.kalman_filter_indep(
            m_0,
            P_0,
            M,
            PHI_obs,
            sigma2_eta,
            sigma2_eps,
            ztildes,
            likelihood="full",
        )
        return ll

    @jax.jit
    def sqf(params):
        (
            log_sigma2_eta,
            log_sigma2_eps,
            ks,
            beta,
        ) = params

        logks1, logks2, ks3, ks4 = ks

        ks1 = jnp.exp(logks1)
        ks2 = jnp.exp(logks2)

        sigma2_eta = jnp.exp(log_sigma2_eta)
        sigma2_eps = jnp.exp(log_sigma2_eps)
        M = model.con_M((ks1, ks2, ks3, ks4))
        
        ztildes = obs_data_wide["z"] - (X_obs @ beta)[:, None]
        
        #ztildes_tuple = jax.tree.map(
        #    lambda tup: tildify(tup[0], tup[1], beta), mapping_elts, is_leaf=is_leaf)
        
        ll, _, _, _, _, _ = fsf.sqrt_filter_indep(
            m_0,
            U_0,
            M,
            PHI_obs,
            sigma2_eta,
            sigma2_eps,
            ztildes,
            likelihood="full",
        )
        return ll

    @jax.jit
    def inf(params):
        (
            log_sigma2_eta,
            log_sigma2_eps,
            ks,
            beta,
        ) = params

        logks1, logks2, ks3, ks4 = ks

        ks1 = jnp.exp(logks1)
        ks2 = jnp.exp(logks2)

        sigma2_eta = jnp.exp(log_sigma2_eta)
        sigma2_eps = jnp.exp(log_sigma2_eps)
        M = model.con_M((ks1, ks2, ks3, ks4))
        
        ztildes_tuple = jax.tree.map(
            lambda tup: tildify(tup[0], tup[1], beta), mapping_elts, is_leaf=is_leaf)
        
        ll, _, _, _, _ = fsf.information_filter_indep(
            nu_0,
            Q_0,
            M,
            PHI_obs_tuple,
            sigma2_eta,
            sigma2_eps,
            ztildes_tuple,
            likelihood="full",
        )
        return ll

    @jax.jit
    def sqinf(params):
        (
            log_sigma2_eta,
            log_sigma2_eps,
            ks,
            beta,
        ) = params

        logks1, logks2, ks3, ks4 = ks

        ks1 = jnp.exp(logks1)
        ks2 = jnp.exp(logks2)

        sigma2_eta = jnp.exp(log_sigma2_eta)
        sigma2_eps = jnp.exp(log_sigma2_eps)
        M = model.con_M((ks1, ks2, ks3, ks4))
        
        ztildes_tuple = jax.tree.map(
            lambda tup: tildify(tup[0], tup[1], beta), mapping_elts, is_leaf=is_leaf)
        
        ll, _, _, _, _ = fsf.sqrt_information_filter_indep(
            nu_0,
            R_0,
            M,
            PHI_obs_tuple,
            sigma2_eta,
            sigma2_eps,
            ztildes_tuple,
            likelihood="full",
        )
        return ll

    kf_vg = jax.value_and_grad(kf)
    sqf_vg = jax.value_and_grad(sqf)
    if_vg = jax.value_and_grad(inf)
    sqif_vg = jax.value_and_grad(sqinf)

    ks0 = (
        jnp.array([jnp.log(100.0)]),
        jnp.array([jnp.log(0.002)]),
        jnp.array([0.0]),
        jnp.array([0.0]),
    )
    params0 = (jnp.log(0.0002), jnp.log(0.0002), ks0, jnp.array([0.0,0.0,0.0]))

    #print(f"\nKalman Filter, ll: {kf_vg(params0)[0]}")
    #print(f"Sqrt Filter, ll: {sqf_vg(params0)[0]}")
    #print(f"Inf Filter, ll: {if_vg(params0)[0]}")
    #print(f"SqInf Filter, ll: {sqif_vg(params0)[0]}")

    time_keys = rand.split(jax.random.fold_in(keys[3], n), 4)
    
    kf_comp_time, kf_run_time = time_jit(time_keys[0], kf_vg, params0, n=reps, desc="Kalman Filter...")
    sq_comp_time, sq_run_time = time_jit(time_keys[0], sqf_vg, params0, n=reps, desc="SQRT Filter...")
    if_comp_time, if_run_time = time_jit(time_keys[0], if_vg, params0, n=reps, desc="Information Filter...")
    sqif_comp_time, sqif_run_time = time_jit(time_keys[0], sqif_vg, params0, n=reps, desc="SQRT Inf Filter...")

    #print(f"\nKalman filter average run time: {kf_run_time}\n",
    #      f"SQRT filter average run time: {sq_run_time}\n",
    #      f"Information filter average run time: {if_run_time}\n",
    #      f"SQ Information filter average run time: {sqif_run_time}\n",
    #      f"Kalman filter compilation: {kf_comp_time}\n",
    #      f"SQRT filter compilation: {sq_comp_time}\n",
    #      f"Information SQ Information filter compilation: {if_comp_time}\n",
    #      f"SQ Information filter compilation: {sqif_comp_time}\n",
    #      )

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([n,
                         kf_run_time,
                         sq_run_time,
                         if_run_time,
                         sqif_run_time,
                         kf_comp_time,
                         sq_comp_time,
                         if_comp_time,
                         sqif_comp_time,
                         ])

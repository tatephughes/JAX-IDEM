from tqdm.auto import tqdm
import csv
import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import idem
import utilities
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
                    "sq", "if", "sqif", "grad_kf",
                    "grad_sq", "grad_if", "grad_sqif"])

key = jax.random.PRNGKey(3)


def timer(func, inp, n, desc):

    tot_time = 0

    result = func(inp) # one run for jit compilation

    for _ in tqdm(range(n), desc=desc):
        start_time = time.time()
        result = func(inp)
        elapsed = time.time() - start_time
        tot_time = tot_time + elapsed

    av_time = tot_time / n

    return av_time * 1000


process_basis = utilities.place_cosine_basis(N=5)

for n in tqdm(range(2,200)):

    #print(f"\n\n\n n={n}")

    keys = rand.split(key, 3)
    nbasis = process_basis.nbasis

    model = idem.gen_example_idem(
        keys[0], k_spat_inv=True,
        process_basis=process_basis,
        sigma2_eta=0.01**2,
        sigma2_eps=0.01**2,
        beta=jnp.array([.2, .2, .2]),
    )

    alpha_0 = jax.random.normal(keys[1], shape=(nbasis,))

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
    sigma2_eta = model.sigma2_eta
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

    def kalman_filter(zs):
        ll_kf, _, _, _, _, _ = fsf.kalman_filter_indep(
            m_0,
            P_0,
            model.M,
            PHI_obs,
            model.sigma2_eta,
            model.sigma2_eps,
            zs,
            likelihood="full",
        )
        return ll_kf

    def sqrt_filter(zs):
        ll_sq, _, _, _, _, _ = fsf.sqrt_filter_indep(
            m_0,
            U_0,
            model.M,
            PHI_obs,
            model.sigma2_eta,
            model.sigma2_eps,
            zs,
            likelihood="full",
        )
        return ll_sq

    def if_filter(zs_tree):
        ll_if, _, _, _, _ = fsf.information_filter_indep(
            nu_0,
            Q_0,
            M,
            PHI_obs_tuple,
            sigma2_eta,
            sigma2_eps,
            zs_tree,
            likelihood="full",
        )
        return ll_if

    def sqif_filter(zs_tree):
        ll_if, _, _, _, _ = fsf.sqrt_information_filter_indep(
            nu_0,
            R_0,
            M,
            PHI_obs_tuple,
            sigma2_eta,
            sigma2_eps,
            zs_tree,
            likelihood="full",
        )
        return ll_if

    kf = jax.jit(kalman_filter)
    sq = jax.jit(sqrt_filter)
    iff = jax.jit(if_filter)
    sqif = jax.jit(sqif_filter)
    gradkf = jax.grad(kalman_filter)
    gradsq = jax.grad(sqrt_filter)
    gradiff = jax.grad(if_filter)
    gradsqif = jax.grad(sqif_filter)

    print(f"\nKalman Filter, ll: {kf(ztildes)}")
    print(f"Sqrt Filter, ll: {sq(ztildes)}")
    print(f"Inf Filter, ll: {iff(ztildes_tuple)}")
    print(f"SqInf Filter, ll: {sqif(ztildes_tuple)}")
    
    kf_time = timer(kf, ztildes, n=reps, desc="Kalman Filter...")
    sq_time = timer(sq, ztildes, n=reps, desc="SQRT Filter...")
    if_time = timer(iff, ztildes_tuple, n=reps, desc="Information Filter...")
    sqif_time = timer(sqif, ztildes_tuple, n=reps, desc="SQRT Inf Filter...")
    grad_kf_time = timer(gradkf, ztildes, n=reps, desc="Kalman Filter Gradient...")
    grad_sq_time = timer(gradsq, ztildes, n=reps, desc="SQRT Filter Gradient...")
    grad_if_time = timer(gradiff, ztildes_tuple, n=reps, desc="Information Filter Gradient...")
    grad_sqif_time = timer(gradsqif, ztildes_tuple, n=reps, desc="SQRT Inf Filter Gradient...")

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([n,
                         kf_time,
                         sq_time,
                         if_time,
                         sqif_time,
                         grad_kf_time,
                         grad_sq_time,
                         grad_if_time,
                         grad_sqif_time,
                         ])

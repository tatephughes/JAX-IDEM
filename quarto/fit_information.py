import IDEM
import utilities
import filter_smoother_functions


from IDEM import *
from utilities import *
from filter_smoother_functions import *

key = jax.random.PRNGKey(12)
keys = rand.split(key, 3)

process_basis = place_basis(nres=2, min_knot_num=5)
nbasis = process_basis.nbasis

m_0 = jnp.zeros(nbasis).at[16].set(1)
sigma2_0 = 0.001

truemodel = gen_example_idem(
    keys[0], k_spat_inv=True,
    process_basis=process_basis,
    m_0=m_0, sigma2_0=sigma2_0
)

# Simulation
T = 10
                                            
process_data, obs_data = truemodel.simulate(nobs=50, T=T + 1, key=keys[1])

plt.show()

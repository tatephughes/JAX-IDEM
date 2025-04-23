import os
import sys
import jax.numpy as jnp
import numpy as np
import jax.random as rand
import time
import optax
import importlib
import jax
sys.path.append(os.path.abspath('../'))
import jaxidem.idem as idem
import jaxidem.utils as utils
import jaxidem.filters as filts
#jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_platform_name', 'cpu')
import blackjax
from tqdm.auto import tqdm
from jax_tqdm import scan_tqdm


print(f"Current working directory: {os.getcwd()}")

# unnecessary unless running interactively
importlib.reload(idem)
importlib.reload(utils)
importlib.reload(filts)

import jax
import jax.numpy as jnp


# Choose a reference time (first entry)
start_time = radar_df['time'].iloc[0]

# Compute time differences in seconds
radar_df['time_float'] = (radar_df['time'] - start_time).dt.total_seconds()


def flatten_and_unflatten(pytree):
    # Flatten the PyTree
    flat_leaves, treedef = jax.tree.flatten(pytree)

    # Convert to a 1D array
    flat_array = jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves])

    # **Ensure sizes and split indices are concrete**
    sizes = [leaf.size for leaf in flat_leaves]  # Extract sizes concretely
    split_indices = list(jnp.cumsum(jnp.array(sizes[:-1])))  # Precompute split indices outside JIT

    # JIT-compatible unflattening
    @jax.jit
    def unflatten(flat_array):
        # Dynamically reshape leaves
        splits = [flat_array[start:end] for start, end in zip([0] + split_indices, split_indices + [len(flat_array)])]
        reshaped_leaves = [split.reshape(leaf.shape) for split, leaf in zip(splits, flat_leaves)]
        return jax.tree.unflatten(treedef, reshaped_leaves)

    return flat_array, unflatten

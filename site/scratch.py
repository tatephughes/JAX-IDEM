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

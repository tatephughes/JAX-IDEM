import jax
import jax.numpy as jnp
import jax.random as rand
import pandas as pd

import jaxidem.idem as idem
import jaxidem.utils as utils


air_df = pd.read_csv('../data/german_air.csv')
df_cleaned = air_df.dropna()

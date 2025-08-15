#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "numpy"]
# ///

import jax
import numpy as np

jax.config.update("jax_platforms", "kelvin")

@jax.jit
def add(x, y):
  return x + y

result = add(np.array([4, 5]), np.array([6, 7]))
print(result)

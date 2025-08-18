#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax", "numpy"]
# ///

import jax
import numpy as np

jax.config.update("jax_platforms", "kelvin")

# @jax.jit
# def jitted_add(x, y):
#   return (x + y)

# result = jitted_add(np.arange(256, dtype=np.int32),
#                     np.arange(256, dtype=np.int32))

@jax.jit
def jitted_xor(x, y):
  return (x ^ y)

result = jitted_xor(np.arange(256, dtype=np.int32),
                    np.arange(256, dtype=np.int32))

# @jax.jit
# def jitted_matmul(x, y):
#   return x @ y

# result = jitted_matmul(np.eye(64), np.eye(64))

print(result)

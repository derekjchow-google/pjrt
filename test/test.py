#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax"]
# ///

import jax

@jax.jit
def add(x, y):
  return x + y

result = add(4, 5)
print(result)

# Kelvin PJRT.

This repository is prototyping building a PJRT plugin target jax 0.7

## Building and Running

```bash
rm -rf jax_plugins/kelvin/pjrt_c_api_kelvin_plugin.so && \
bazel build //kelvin_pjrt:pjrt_c_api_kelvin_plugin.so && \
cp bazel-bin/kelvin_pjrt/pjrt_c_api_kelvin_plugin.so test/jax_plugins/kelvin/pjrt_c_api_kelvin_plugin.so

test/test.py
```

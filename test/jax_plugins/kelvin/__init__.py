import os
import jax._src.xla_bridge as xb

def initialize():
  print("~Tuturu initialize plugin", flush=True)
  path = os.path.join(os.path.dirname(__file__), 'pjrt_c_api_kelvin_plugin.so')
  xb.register_plugin('kelvin', priority=500, library_path=path, options=None)

"""stand-in jax (tests only): the calls the levers, the design script and BindCraft's modules at import make."""
__version__ = "0.6.0"
from jax import numpy  # noqa: F401  (BindCraft's colabdesign_utils: `import jax.numpy as jnp` at module level)


class _Device:
    device_kind = "stub-device"
    platform = "cpu"

    def memory_stats(self):
        return {"bytes_limit": 85_000_000_000, "bytes_in_use": 500_000_000, "peak_bytes_in_use": 1_000_000_000}


def devices():
    return [_Device()]


def default_backend():
    return "cpu"


class _Config:
    """stand-in jax.config: `update(name, value)` records the value (lever compilecache sets the persistent-cache names)."""
    values = {}

    def update(self, name, value):
        self.values[name] = value

    def read(self, name):
        return self.values.get(name)


config = _Config()


def _stub_compile(program: str, seconds: float = 0.0):
    """What the stand-in model does where the real one compiles an executable: jax's persistent-cache events reach the listeners lever
    compilecache registered (jax.monitoring) — a request that consulted the cache and missed, and the backend compile duration."""
    import sys
    mon = sys.modules.get("jax.monitoring")
    if mon is None:
        return
    for f in list(mon._events):
        f("/jax/compilation_cache/compile_requests_use_cache"); f("/jax/compilation_cache/cache_misses")
    for f in list(mon._durations):
        f("/jax/core/compile/backend_compile_duration", seconds)

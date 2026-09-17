"""numstate — global numeric-state invariants for a chained-model JAX process (Boltz2/joltz fold + ProteinMPNN loss + refold in one process).
snapshot() records every process-global knob that could change arithmetic; assert_unchanged(ref) raises if any moved.
JAX: jax_default_matmul_precision (None => joltz's own scoped 'float32' contexts decide), jax_enable_x64, jax_numpy_dtype_promotion,
jax_default_prng_impl, jax_threefry_partitionable, jax_disable_jit, XLA_FLAGS / JAX_* env; torch (CPU-only here: checkpoint conversion + nothing in the loop):
default dtype, matmul allow_tf32, cudnn allow_tf32, float32_matmul_precision, num_threads."""
import os, json
def snapshot():
    import jax
    s = {}
    for k in ["jax_default_matmul_precision", "jax_enable_x64", "jax_numpy_dtype_promotion", "jax_default_prng_impl", "jax_threefry_partitionable", "jax_disable_jit", "jax_debug_nans", "jax_platforms", "jax_compilation_cache_dir", "jax_persistent_cache_min_compile_time_secs"]:
        try: s[k] = str(getattr(jax.config, k))
        except Exception as e: s[k] = f"n/a ({type(e).__name__})"
    s["env_XLA_FLAGS"] = os.environ.get("XLA_FLAGS", ""); s["env_JAX"] = {k: v for k, v in os.environ.items() if k.startswith("JAX_") or k.startswith("XLA_PYTHON_CLIENT")}
    s["jax_backend"] = jax.default_backend(); s["devices"] = [str(d) for d in jax.devices()]
    try:
        import torch
        s["torch_default_dtype"] = str(torch.get_default_dtype()); s["torch_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32); s["torch_cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        s["torch_float32_matmul_precision"] = torch.get_float32_matmul_precision(); s["torch_cuda_available"] = bool(torch.cuda.is_available()); s["torch_version"] = torch.__version__
    except Exception as e:
        s["torch"] = f"not importable ({type(e).__name__})"
    return s
NUMERIC_KEYS = ["jax_default_matmul_precision", "jax_enable_x64", "jax_numpy_dtype_promotion", "jax_default_prng_impl", "jax_threefry_partitionable", "jax_disable_jit", "env_XLA_FLAGS", "torch_default_dtype", "torch_matmul_allow_tf32", "torch_cudnn_allow_tf32", "torch_float32_matmul_precision"]
def diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in NUMERIC_KEYS if a.get(k) != b.get(k)}
def assert_unchanged(ref, label=""):
    d = diff(ref, snapshot())
    if d: raise AssertionError(f"global numeric state changed {label}: {d}")
    return True
EXPECTED_STOCK = {"jax_default_matmul_precision": "None", "jax_enable_x64": "False", "jax_numpy_dtype_promotion": "standard", "jax_default_prng_impl": "threefry2x32", "jax_disable_jit": "False"}
def assert_stock(label=""):
    s = snapshot(); bad = {k: (v, s.get(k)) for k, v in EXPECTED_STOCK.items() if s.get(k) != v}
    if bad: raise AssertionError(f"non-stock global numeric state {label}: {bad}")
    return s

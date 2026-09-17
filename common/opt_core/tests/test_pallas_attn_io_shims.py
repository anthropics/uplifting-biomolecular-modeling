"""The carried `pallas_attn` kernel's `_load` / `_store` shims dispatch to the IO home the running jax carries — CPU, no jax needed: the
kernel module is executed against stand-in `jax` modules. On a jax whose `jax.experimental.pallas` has `load` / `store` (0.5–0.7) the shims
call exactly `pl.load(ref, idx, mask=, other=)` / `pl.store(ref, idx, val, mask=)` (the kit's tested path, unchanged); on a jax
without them (0.8+) they call the Triton backend's `plgpu.load(ref.at[idx], mask=, other=)` / `plgpu.store(ref.at[idx], val, mask=)`."""
import importlib.util
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(os.path.dirname(HERE), "opt_core", "kernels", "pallas_attn", "af2_flash_pallas.py")


class _Any:
    """A permissive stand-in: every attribute is another stand-in, calls return a stand-in, usable as a decorator factory."""

    def __init__(self, name="x"):
        self._n = name

    def __getattr__(self, k):
        v = _Any(f"{self._n}.{k}")
        object.__setattr__(self, k, v)
        return v

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not isinstance(a[0], _Any) and not k:
            return _Any(f"{self._n}(fn)")          # decorator use: wrap
        return _Any(f"{self._n}()")

    def __iter__(self):
        return iter(())


class _Recorder:
    def __init__(self):
        self.calls = []

    def load(self, *a, **k):
        self.calls.append(("load", a, k)); return "loaded"

    def store(self, *a, **k):
        self.calls.append(("store", a, k)); return "stored"


class _Ref:
    class _At:
        def __init__(self, ref):
            self.ref = ref

        def __getitem__(self, idx):
            return ("view", self.ref.name, idx)

    def __init__(self, name):
        self.name = name
        self.at = _Ref._At(self)


def _load_kernel_with(pl_has_io: bool):
    """Execute the kernel source against stand-in jax modules; returns (module, pl_recorder, plgpu_recorder)."""
    jax = types.ModuleType("jax")
    anyj = _Any("jax")
    jax.__getattr__ = lambda k: getattr(anyj, k)            # module-level __getattr__ (PEP 562): jax.custom_vjp, jax.lax, ...
    jnp = types.ModuleType("jax.numpy"); jnp.__getattr__ = lambda k: getattr(_Any("jnp"), k)
    experimental = types.ModuleType("jax.experimental")
    pl = types.ModuleType("jax.experimental.pallas")
    pl_rec, gpu_rec = _Recorder(), _Recorder()
    if pl_has_io:
        pl.load, pl.store = pl_rec.load, pl_rec.store
    pl.__getattr__ = lambda k: (_ for _ in ()).throw(AttributeError(k)) if k in ("load", "store") else getattr(_Any("pl"), k)
    plgpu = types.ModuleType("jax.experimental.pallas.triton")
    plgpu.load, plgpu.store = gpu_rec.load, gpu_rec.store
    plgpu.CompilerParams = _Any("CompilerParams")
    experimental.pallas = pl; pl.triton = plgpu; jax.experimental = experimental; jax.numpy = jnp
    saved = {k: sys.modules.get(k) for k in ("jax", "jax.numpy", "jax.experimental", "jax.experimental.pallas", "jax.experimental.pallas.triton")}
    sys.modules.update({"jax": jax, "jax.numpy": jnp, "jax.experimental": experimental, "jax.experimental.pallas": pl,
                        "jax.experimental.pallas.triton": plgpu})
    try:
        spec = importlib.util.spec_from_file_location(f"af2_flash_pallas_standin_{int(pl_has_io)}", KERNEL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, pl_rec, gpu_rec


@pytest.mark.parametrize("pl_has_io", [True, False])
def test_load_store_shims_dispatch_to_the_io_home_the_jax_carries(pl_has_io):
    K, pl_rec, gpu_rec = _load_kernel_with(pl_has_io)
    ref, idx = _Ref("q"), (slice(None), 3)
    assert K._load(ref, idx, mask="m", other=0.0) == "loaded"
    assert K._store(ref, idx, "val", mask="m") == "stored"
    if pl_has_io:                                    # the tested jax 0.5–0.7 path: pl.load(ref, idx, ...) — plgpu untouched
        assert pl_rec.calls == [("load", (ref, idx), {"mask": "m", "other": 0.0}), ("store", (ref, idx, "val"), {"mask": "m"})]
        assert gpu_rec.calls == []
    else:                                            # jax without pl.load/pl.store: plgpu.load(ref.at[idx], ...) on the ref view
        assert pl_rec.calls == []
        assert gpu_rec.calls == [("load", (("view", "q", idx),), {"mask": "m", "other": 0.0}), ("store", (("view", "q", idx), "val"), {"mask": "m"})]


def test_probe_names_pallas_api_removed_only_when_neither_home_has_the_io_names():
    from opt_core.kernels import pallas_attn_serve as S
    assert S.PALLAS_IO_API == ("load", "store")
    assert all(n not in S.PALLAS_API for n in S.PALLAS_IO_API)


def test_backward_f32_precision_word_is_validated_and_described():
    K, _, _ = _load_kernel_with(False)
    assert K.BWD_F32_PRECISIONS == ("tf32", "ieee") and K.F32_PRECISION_DEFAULT == "ieee"
    with pytest.raises(ValueError):
        K.make_flash_attention(bwd_f32_precision="tf32x3")
    with pytest.raises(ValueError):
        K.make_flash_attention(f32_precision="fp8")
    op = K.make_flash_attention(f32_precision="tf32", bwd_f32_precision="tf32")
    d = K.describe_op(op)
    assert (d["f32_precision"], d["bwd_f32_precision"], d["bq_bwd"], d["bk_bwd"]) == ("tf32", "tf32", 64, 64)
    assert K.describe_op(K.make_flash_attention())["bwd_f32_precision"] == "ieee"        # the default op: the v2 statement


def test_dq_mode_word_is_validated_and_described():
    K, _, _ = _load_kernel_with(False)
    assert K.DQ_MODES == ("kernel", "from_ds") and K.DQ_MODE_DEFAULT == "kernel"
    with pytest.raises(ValueError):
        K.make_flash_attention(dq="fused")
    assert K.describe_op(K.make_flash_attention())["dq"] == "kernel"                        # the default op: the v2 statement
    assert K.describe_op(K.make_flash_attention(dq="from_ds", batch_chunk=256))["batch_chunk"] == 256


def test_dbias_mode_word_is_validated_and_described():
    """dbias='xla' (the default = the v2 statement: per-batch dS partials reduced by XLA) | 'kernel' (summed over the batch inside a third
    kernel); any other word raises naming the set; dq='from_ds' needs the dS buffer the kernel mode does not write and says so."""
    K, _, _ = _load_kernel_with(False)
    assert K.DBIAS_MODES == ("xla", "kernel") and K.DBIAS_MODE_DEFAULT == "xla"
    assert K.describe_op(K.make_flash_attention())["dbias"] == "xla"
    assert K.describe_op(K.make_flash_attention(dbias="kernel"))["dbias"] == "kernel"
    with pytest.raises(ValueError):
        K.make_flash_attention(dbias="atomics")
    with pytest.raises(ValueError) as e:
        K.make_flash_attention(dbias="kernel", dq="from_ds")
    assert "from_ds" in str(e.value) and "dbias='kernel'" in str(e.value)

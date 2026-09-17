"""In-process (the fork's interpreter, loaded by the tree's launchers before the first compile): a persistent-compilation-cache key that is
the same on every box of one GPU model.

jax keys each persistent-cache entry on the program, the jax/jaxlib versions, the backend's platform version, the XLA flags, the compile
options and an ``accelerator_config`` component — ``jax/_src/cache_key.py`` ``_hash_accelerator_config``, which hashes
``xla_client.get_topology_for_devices(devices).fingerprint()``. That fingerprint differs between boxes carrying the same GPU model, driver
and CUDA stack (two ``NVIDIA H100 80GB HBM3`` boxes give different values), so a class (``$CACHE``, stack.cache_dir) built on one box is
recompiled on another although the program, the stack and the GPU model are identical. ``install()`` rebinds that one component to jax's
own fallback for backends without a topology description (``_hash_devices``: the ``device_kind`` of every device) — the key then names the
GPU model, which is what the class directory ``<GPU model>__jax<v>_jaxlib<v>`` already names. Every other key component is untouched;
the compiled programs and their outputs are untouched (the key decides reuse only).

Written against the pinned jax (EXPECTED_JAX, stock/PINS.json check_packages): under any other jax version the rebinding stays off — named on the
CACHEKEY line (``accelerator=topology jax=<v> reason=…``: jax's own box-specific key applies, the class is recompiled on another box), never
patched, never a reason to end the process.
"""
PREFIX = "[af3-jax-opt]"
EXPECTED_JAX = "0.10.2"          # the pinned stack's jax; jax/_src/cache_key.py of this version has _hash_accelerator_config(hash_obj, accelerators) and _hash_devices
WORD = "device_kind"             # the CACHEKEY line's accelerator= word (det.CACHEKEY_RX reads it; the CACHE when=after line carries cache_key=<word>)
MARK = "af3_jax_opt_accelerator"


TOPOLOGY = "topology"            # the CACHEKEY line's word when the rebinding stays off: jax's own accelerator_config component (det.CACHE_KEY_DEFAULT spells the same)


class Refused(RuntimeError):
    """The rebinding does not apply to this interpreter's jax (version or module shape) — raised by name, nothing patched; main_guard names it on the line."""


def install() -> str:
    """Rebind jax._src.cache_key._hash_accelerator_config to the device-kind hash; idempotent; prints ONE CACHEKEY line. Returns WORD."""
    import jax
    if jax.__version__ != EXPECTED_JAX:
        raise Refused(f"cache key portability shim expects jax {EXPECTED_JAX}, found jax {jax.__version__}")
    from jax._src import cache_key as ck
    if not callable(getattr(ck, "_hash_accelerator_config", None)) or not callable(getattr(ck, "_hash_devices", None)):
        raise Refused("cache key portability shim expects jax._src.cache_key._hash_accelerator_config and _hash_devices (jax "
                      f"{EXPECTED_JAX}'s module); this jax has a different cache_key module")
    if getattr(ck._hash_accelerator_config, MARK, None) == WORD:      # already installed in this process
        return WORD

    def _hash_accelerator_config(hash_obj, accelerators):           # the signature jax 0.10.2's cache_key.get calls: (hash_obj, devices)
        ck._hash_devices(hash_obj, accelerators)                      # jax's own fallback: the device_kind of every device, in order

    setattr(_hash_accelerator_config, MARK, WORD)
    ck._hash_accelerator_config = _hash_accelerator_config
    print(f"{PREFIX} CACHEKEY accelerator={WORD} jax={jax.__version__} site=jax._src.cache_key._hash_accelerator_config", flush=True)
    return WORD


def main_guard() -> str:
    """The launchers' call: install; under a jax the rebinding was not written for it stays off, NAMED on the line (accelerator=topology
    jax=<v> reason=<why>: jax's own key applies — the compiled programs are the same, only cross-box reuse of the class is lost) and the
    process runs on. An interpreter without jax (nothing to key) is named likewise (accelerator=absent) and left alone."""
    try:
        return install()
    except Refused as e:
        import jax
        print(f"{PREFIX} CACHEKEY accelerator={TOPOLOGY} jax={jax.__version__} reason={str(e).replace(' ', '_')}", flush=True)
        return TOPOLOGY
    except ModuleNotFoundError as e:
        if e.name != "jax":
            raise
        print(f"{PREFIX} CACHEKEY accelerator=absent reason=jax_not_importable", flush=True)
        return "absent"

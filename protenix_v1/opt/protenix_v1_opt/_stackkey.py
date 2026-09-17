"""``python -m protenix_v1_opt._stackkey`` — the probe ``configs/<card>.env`` keys the persistent JIT caches by when ``MODEL_OPT_JIT_ROOT``
is set: prints this stack's cache key ``torch<version>-cu<cuda>-sm<cc>`` (opt_core.jit_cache.key: the torch version without its local
tag, the CUDA version without the dot, the device's compute-capability digits — resolved without importing torch) or the word
``unknown`` when a part cannot be established (the configuration then names a cold, unshared directory itself). Not a command of the
CLI: the configuration's own helper. Statement one is the package's pre-import core pin gate and producers gate, as in ``__main__`` — an
absent or older shared core exits 3 by name, never a swallowed import error, never a traceback."""
import sys

TAG = "protenix-v1-opt"          # == report.TAG (the gates run before report can import)

if __name__ == "__main__":
    from ._core_gate import gate
    gate(__file__, tag=TAG)
    from ._producers import refuse_if_missing
    refuse_if_missing()
    from opt_core.jit_cache import StackKeyUnknown, key
    try:
        print(key())
    except StackKeyUnknown:
        print("unknown")
    sys.exit(0)

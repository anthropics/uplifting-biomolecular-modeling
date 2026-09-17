"""The package's CPU tests: they are written for, and expected green on, a CPU-only interpreter (no GPU visible; jax, when installed, on its CPU
backend). Inside a GPU environment some cases do not hold by construction (they assert the no-GPU refusals, a clean process environment, or
CPU-reference arithmetic) — a red run there says nothing about the kit; run the suite on a CPU machine. The kit runs on the shared core it pins
(`opt/pyproject.toml` `[tool.opt_core]`): when `opt_core` is not installed in the interpreter, the pinned tree is put on `sys.path` here so the
tests run from a bare checkout too."""
import os
import sys

AMBIENT_KIT_ENV = ("MOSAIC_OPT", "MOSAIC_OPT_FORCE", "MOSAIC_OPT_CACHE_ROOT", "MOSAIC_OPT_CACHE_DIR", "MODEL_OPT_JIT_ROOT", "MODEL_OPT_LEVERS_OFF",
                   "MODEL_OPT_TARGET_GPU", "JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
                   "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "XLA_FLAGS")      # the kit's and P1's variables a GPU box exports for its arms: the suite states its own environment per case (mock.patch.dict / monkeypatch), so an ambient value is removed once here — for this process and every child it starts — and a case that needs one sets it; MOSAIC_CACHE_DIR (the weights cache) stays
for _k in AMBIENT_KIT_ENV:
    os.environ.pop(_k, None)

OPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # mosaic/opt


def core_src() -> str:
    """The pinned core's project directory (common/opt_core), read from the pin table without importing the core."""
    path = None
    with open(os.path.join(OPT_DIR, "pyproject.toml"), "r", encoding="utf-8") as fh:
        inside = False
        for ln in fh:
            s = ln.strip()
            if s.startswith("["):
                inside = s.startswith("[tool.opt_core]")
            elif inside and s.startswith("path") and "=" in s:
                path = s.split("=", 1)[1].split("#", 1)[0].strip().strip('"')
    if not path:
        raise RuntimeError(f"{OPT_DIR}/pyproject.toml: [tool.opt_core] path missing")
    return os.path.normpath(os.path.join(OPT_DIR, path))


try:
    import opt_core  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, core_src())

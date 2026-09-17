"""The kit's JIT caches, keyed under one optional persistent root (the model-opt kits' convention: ``MODEL_OPT_JIT_ROOT``).

``apply()`` runs once at driver / in-process activation, before the first lever compiles anything: when ``MODEL_OPT_JIT_ROOT`` is set it
derives the per-stack key ``torch<version sans local tag>-cu<CUDA sans dot>-<MODEL_OPT_TARGET_GPU word>`` without touching CUDA (``MODEL_OPT_STACK_KEY`` when the caller pre-set
one) and points the two caches the levers fill at ``$MODEL_OPT_JIT_ROOT/<key>/triton`` (``TRITON_CACHE_DIR``: `v4trimul`'s Triton rows, the
`compiled` step's Triton kernels) and ``…/inductor`` (``TORCHINDUCTOR_CACHE_DIR``: the `compiled` step's Inductor products) — each ONLY if the
user has not set it. Unset root: the libraries' own defaults (~/.triton/cache, /tmp/torchinductor_$USER) and a fresh machine recompiles. One NOTE
line names the directories either way. Nothing here changes a fold's numbers."""
from __future__ import annotations

import os
from typing import Dict, Optional

ENV_ROOT, ENV_KEY = "MODEL_OPT_JIT_ROOT", "MODEL_OPT_STACK_KEY"
CACHES = (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"))


def cache_key(torch=None, environ=None) -> str:
    """``torch<version sans local tag>-cu<CUDA sans dot>-<target gpu word>`` WITHOUT touching CUDA (the allocator lever reads its setting at the
    first CUDA initialisation, which must stay the fold's): torch's version from the imported module or the installed distribution, the arch
    from the config's ``MODEL_OPT_TARGET_GPU`` word (h100 / a100; ``gpu`` when unset)."""
    env = os.environ if environ is None else environ
    ver, cu = None, None
    if torch is not None:
        ver, cu = str(torch.__version__), getattr(getattr(torch, "version", None), "cuda", None)
    else:
        import sys
        t = sys.modules.get("torch")
        if t is not None:
            ver, cu = str(t.__version__), getattr(getattr(t, "version", None), "cuda", None)
        else:
            ver, cu = _torch_version_file()                                      # torch/version.py read as text: the version AND the CUDA it was built
            if ver is None:                                                      # for, without importing torch (the distribution metadata drops the
                try:                                                             # +cuXXX local tag on some indexes: the key would read cunone)
                    import importlib.metadata as md
                    ver = md.version("torch")
                except Exception:  # noqa: BLE001
                    ver = "unknown"
    local = ver.split("+")[1] if "+" in ver else ""
    cu = (cu or (local if local.startswith("cu") else "") or "none").replace(".", "").replace("cu", "")
    arch = (env.get("MODEL_OPT_TARGET_GPU") or "gpu").strip().lower()
    return f"torch{ver.split('+')[0]}-cu{cu}-{arch}"


def _torch_version_file():
    """(``__version__``, ``cuda``) read from the installed torch/version.py WITHOUT importing torch (None, None when not found)."""
    try:
        import importlib.util, re
        spec = importlib.util.find_spec("torch")
        loc = os.path.dirname(spec.origin) if spec and spec.origin else None
        src = open(os.path.join(loc, "version.py"), encoding="utf-8").read() if loc else ""
        v = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", src, re.M); c = re.search(r"^cuda\s*[:=][^'\"\n]*['\"]([\d.]+)['\"]", src, re.M)
        return (v.group(1) if v else None), (c.group(1) if c else None)
    except Exception:  # noqa: BLE001
        return None, None


def apply(environ=None, torch=None) -> Dict[str, Optional[str]]:
    """Point the unset cache variables under the root (if any); return ``{root, key, TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR, set: [names set here]}``."""
    env = os.environ if environ is None else environ
    root = env.get(ENV_ROOT) or None
    out: Dict[str, object] = {"root": root, "key": None, "set": []}
    if root:
        key = env.get(ENV_KEY) or cache_key(torch, env)
        out["key"] = key
        for var, leaf in CACHES:
            if not env.get(var):
                env[var] = os.path.join(root, key, leaf)
                out["set"].append(var)
    for var, _ in CACHES:
        out[var] = env.get(var) or None
    return out


def note(facts: Dict[str, object]) -> str:
    """The one NOTE line: where the JIT caches live for this run."""
    if facts.get("root"):
        how = f"{ENV_ROOT}={facts['root']} key={facts['key']}" + (f" (set here: {','.join(facts['set'])})" if facts.get("set") else " (pre-set by the caller)")
    else:
        how = f"{ENV_ROOT} unset: the libraries' defaults"
    dirs = " ".join(f"{v}={facts.get(v) or '(default)'}" for v, _ in CACHES)
    return f"JIT caches: {dirs} — {how}"

"""The compiled-kernel cache key of the running stack, and where the caches go.

Contract. Cache keys follow the running stack, never a configured string: :func:`key` is ``torch<version>-cu<cuda>-sm<cc>`` — the torch
version without its local tag, the CUDA version without the dot, the device's compute-capability digits — all resolved WITHOUT importing torch
(distribution metadata, ``torch/version.py`` found through ``importlib.util.find_spec``, the ``nvidia-cuda-runtime`` wheel, nvidia-smi); a part that
cannot be established raises :class:`StackKeyUnknown` by name (the form of a verb whose purpose IS the cache: warm, cache checks). The run
path of a kit reads :func:`key_facts` instead: the same key, and for a part that cannot be established an ISOLATED part
``unknown<token>`` (:func:`isolation_token`: one per process — never a shared ``unknown`` bucket two different stacks could meet in; the
worst case is a cold compile into a directory nobody reuses) plus the word ``cache_key=unknown(<parts>)`` for the activation line.
A kit whose stack is not torch passes its own three strings. :func:`compose` renders the other key SHAPES a cache may be keyed by from the same parts
(:data:`FORMATS`: ``torch-sm``, ``torch-sm-triton``, or the caller's own format string) so a kit's key is one call, byte-identical to the
directory names its caches already carry. :func:`cache_dirs` places the caches under ``<root>/<key>/`` (:data:`CACHE_DIRS`: Triton,
torch extensions, TorchInductor; or the caller's own ``{VAR: subdir}`` names);
:func:`keep_or_key` is the rule for a pre-set cache directory: kept when it or its parent exists (a mounted cache), replaced by the keyed one
otherwise (an image-baked path under an unmounted tree is not a cache); :func:`cache_env` is the two combined for a process environment.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from . import gates

CACHE_DIRS = {"TRITON_CACHE_DIR": "triton", "TORCH_EXTENSIONS_DIR": "torch_extensions", "TORCHINDUCTOR_CACHE_DIR": "inductor"}

FORMATS = {
    "torch-cu-sm": "torch{version}-cu{cuda}-sm{cc}",            # :func:`key` — the module contract
    "torch-sm": "torch{version}-sm{cc}",                         # a Triton-only cache (the CUDA toolkit is not in the key)
    "torch-sm-triton": "torch{version}-sm{cc}-triton{triton}",   # a Triton cache keyed by the Triton wheel too
}


class StackKeyUnknown(ValueError):
    """A part of the cache key could not be established (named): the caller passes it explicitly, reads :func:`key_facts` (the run path:
    an isolated part and a word), or opts into ``strict=False`` (display). A cannot-run event for a verb whose purpose is the cache
    (``opt_core.gates.is_cannot_run``)."""

    cannot_run = True


_TOKEN: Optional[str] = None


def isolation_token() -> str:
    """This process's isolation token: 8 hex digits from pid, time and urandom, drawn once per process. A key part that cannot be read is
    rendered ``unknown<token>`` on the run path so the cache directory it names is this process's alone."""
    global _TOKEN
    if _TOKEN is None:
        import hashlib  # noqa: PLC0415
        import time     # noqa: PLC0415
        _TOKEN = hashlib.sha256(b"%d|%r|%s" % (os.getpid(), time.time(), os.urandom(8).hex().encode())).hexdigest()[:8]
    return _TOKEN


_VERSION_PY = {"version": re.compile(r"^__version__\s*(?::[^=\n]*)?=\s*['\"]([^'\"]+)['\"]", re.M),
               "cuda": re.compile(r"^cuda\b[^=\n]*=\s*['\"]([0-9][^'\"]*)['\"]", re.M)}


def parse_version_py(text: str) -> dict:
    """``{"version", "cuda"}`` read from the TEXT of a ``<dist>/version.py`` (torch writes ``__version__ = '2.8.0+cu128'`` and ``cuda = '12.8'``);
    missing entries are absent. No import of the distribution happens."""
    out = {}
    for k, rx in _VERSION_PY.items():
        m = rx.search(text or "")
        if m:
            out[k] = m.group(1)
    return out


def version_py_facts(dist: str = "torch") -> dict:
    """:func:`parse_version_py` of ``<dist>/version.py`` located through ``importlib.util.find_spec`` (import-free: the package is found, not loaded)."""
    try:
        import importlib.util  # noqa: PLC0415
        spec = importlib.util.find_spec(dist)
    except (ImportError, ValueError):
        return {}
    if spec is None or not spec.origin:
        return {}
    path = os.path.join(os.path.dirname(spec.origin), "version.py")
    try:
        with open(path, encoding="utf-8") as f:
            return parse_version_py(f.read())
    except OSError:
        return {}


def cuda_runtime_dist_version() -> Optional[str]:
    """``major.minor`` of the installed ``nvidia-cuda-runtime-cuXX`` wheel (distribution metadata, import-free), else None."""
    for name in ("nvidia-cuda-runtime-cu13", "nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime-cu11", "nvidia-cuda-runtime"):
        v = gates.dist_version(name)
        if v:
            parts = str(v).split(".")
            return ".".join(parts[:2]) if len(parts) >= 2 else str(v)
    return None


def resolve_parts(version: Optional[str] = None, cuda: Optional[str] = None, cc: Optional[str] = None, *, dist: str = "torch") -> dict:
    """``{"version": <base version or None>, "cuda": <version or None>, "cc": <"9.0" or None>}`` resolved WITHOUT importing ``dist``:
    ``version`` defaults to the distribution metadata version, then ``<dist>/version.py``; ``cuda`` to the version's local tag (``+cu128``), then
    ``<dist>/version.py``'s ``cuda``, then the ``nvidia-cuda-runtime`` wheel's version; ``cc`` to nvidia-smi. None = could not be established."""
    facts = None
    if version is None:
        version = gates.dist_version(dist)
        if version is None:
            facts = version_py_facts(dist)
            version = facts.get("version")
    base, _, local = str(version or "").partition("+")
    if cuda is None:
        if local.startswith("cu") and local[2:].isdigit():
            cuda = local[2:]
        else:
            facts = facts if facts is not None else version_py_facts(dist)
            cuda = facts.get("cuda") or (cuda_runtime_dist_version() if dist == "torch" else None)
    if cc is None:
        cc = gates.nvidia_smi_probe().get("cc")
    return {"version": base or None, "cuda": cuda or None, "cc": cc or None}


def _render(dist: str, parts: dict, unknown: str) -> str:
    return f"{dist}{parts['version'] or unknown}-cu{digits(parts['cuda']) if parts['cuda'] else unknown}-sm{digits(parts['cc']) if parts['cc'] else unknown}"


def key(version: Optional[str] = None, cuda: Optional[str] = None, cc: Optional[str] = None, *, dist: str = "torch", strict: bool = True) -> str:
    """The cache key (module contract) from :func:`resolve_parts`. A part that cannot be established raises :class:`StackKeyUnknown` naming it;
    ``strict=False`` renders it ``unknown`` instead — the display form (a manifest's box-as-found block), never a cache directory: the run
    path reads :func:`key_facts`."""
    parts = resolve_parts(version, cuda, cc, dist=dist)
    missing = [n for n in ("version", "cuda", "cc") if not parts[n]]
    if missing and strict:
        raise StackKeyUnknown(f"cache key: cannot establish {', '.join(missing)} for dist {dist!r} without importing it "
                              f"(pass {'/'.join(missing)}= explicitly, or strict=False for a display-only key)")
    return _render(dist, parts, "unknown")


def key_facts(version: Optional[str] = None, cuda: Optional[str] = None, cc: Optional[str] = None, *, dist: str = "torch") -> dict:
    """The run path's key (module contract): ``{"key", "parts", "unknown", "word"}``. With every part established ``key`` equals :func:`key`,
    ``unknown`` is ``[]`` and ``word`` None. A part that cannot be established is rendered ``unknown<token>`` (:func:`isolation_token`) INSIDE the
    key — ``torch2.7.0-cu126-smunknown3fa2b1c9``: a directory of this process alone, compiled cold, never a bucket another stack shares — and named:
    ``unknown`` lists the parts, ``word`` is ``cache_key=unknown(<part>,…)`` for the kit's activation line (exit 0)."""
    from .report import word as _word  # noqa: PLC0415
    parts = resolve_parts(version, cuda, cc, dist=dist)
    missing = [n for n in ("version", "cuda", "cc") if not parts[n]]
    return {"key": _render(dist, parts, "unknown" + isolation_token()), "parts": parts, "unknown": missing,
            "word": _word("cache_key", "unknown", *missing) if missing else None}


def strip_local(version: Optional[str]) -> str:
    """A version without its local tag (``2.8.0+cu128`` -> ``2.8.0``); ``unknown`` for None/empty."""
    return str(version).partition("+")[0] if version else "unknown"


def digits(value: Optional[str]) -> str:
    """Dots removed (``12.8`` -> ``128``, ``9.0`` -> ``90``); ``unknown`` for None/empty."""
    return str(value).replace(".", "") if value else "unknown"


def compose(fmt: str, **parts) -> str:
    """Render the key shape ``fmt`` (a :data:`FORMATS` name or a format string) from ``parts``: ``cuda``/``cc`` lose their dots, ``gpu`` is
    kept verbatim, every other part (``version``, ``triton``, a library's version) loses its local tag; a missing or empty part reads ``unknown``."""
    template = FORMATS.get(fmt, fmt)
    norm = {}
    for k, v in parts.items():
        if k in ("cuda", "cc"):
            norm[k] = digits(v)
        elif k == "gpu":
            norm[k] = str(v) if v else "unknown"
        else:
            norm[k] = strip_local(v)

    class _Unknown(dict):
        def __missing__(self, k):
            return "unknown"
    return template.format_map(_Unknown(norm))


def cache_dirs(root: str, cache_key: str, names=None) -> dict:
    """``{VAR: <root>/<key>/<sub>}`` for every :data:`CACHE_DIRS` entry (the torch caches: Triton, torch extensions, TorchInductor), or for ``names``
    — an iterable of :data:`CACHE_DIRS` variable names, or the caller's own ``{VAR: subdir}`` mapping."""
    sel = CACHE_DIRS if names is None else (dict(names) if hasattr(names, "items") else {n: CACHE_DIRS[n] for n in names})
    return {var: os.path.join(root, cache_key, sub) for var, sub in sel.items()}


def cache_env(root: str, cache_key: str, environ=None, names=None) -> dict:
    """``{VAR: (value, "kept" | "keyed")}``: :func:`keep_or_key` applied to every :func:`cache_dirs` entry against ``environ`` (default
    ``os.environ``) — what a kit exports and what its stack line records."""
    env = os.environ if environ is None else environ
    return {var: keep_or_key(env.get(var), keyed) for var, keyed in cache_dirs(root, cache_key, names).items()}


def keep_or_key(preset: Optional[str], keyed: str) -> tuple:
    """``(value, "kept" | "keyed")``: a pre-set directory is kept when it or its parent exists; else the keyed directory."""
    if preset and (os.path.isdir(preset) or os.path.isdir(os.path.dirname(preset.rstrip(os.sep)) or os.sep)):
        return preset, "kept"
    return keyed, "keyed"

"""chrombpnet_k1._te_guard — when TensorFlow is imported BEFORE torch in one process (stock's deterministic recipe seeds TF first), an older
typing_extensions without 'deprecated' can already sit in sys.modules, and torch 2.4.1 then fails to import ('cannot import name deprecated
from typing_extensions'). ensure_typing_extensions() runs BEFORE this package imports torch (the first statement of chrombpnet_k1/__init__.py):
if the loaded (or importable) typing_extensions lacks the names torch 2.4.1 needs, the kit's VENDORED copy (_vendor/typing_extensions.py, 4.13.2,
PSF-2.0; VENDOR.json beside it) is executed and installed as sys.modules['typing_extensions'] — modules that already imported the old one keep their
references (harmless for a pure-Python typing helper); every later `import typing_extensions`, torch's included, gets the new one. The outcome is
recorded in TE_GUARD and printed once; apply() repeats it on the kit's status line. No env var, no fallback to a different program."""
import os, sys, importlib, importlib.util
VENDORED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor", "typing_extensions.py")
REQUIRED = ("deprecated", "TypeAliasType", "override")          # names torch 2.4.1 imports from typing_extensions (torch requires typing-extensions >= 4.8.0; 'deprecated' exists from 4.5) — an older copy lacks it
TE_GUARD = {"action": None, "detail": None, "required": REQUIRED}

def _has_required(mod): return all(hasattr(mod, n) for n in REQUIRED)

def ensure_typing_extensions():
    """Idempotent. Returns TE_GUARD: action in {'kept', 'replaced', 'provided'} (or raises ImportError if the vendored copy cannot load)."""
    if TE_GUARD["action"] is not None: return TE_GUARD
    loaded = sys.modules.get("typing_extensions"); was_loaded = loaded is not None
    if loaded is None:
        try: loaded = importlib.import_module("typing_extensions")                  # the environment's copy on the current sys.path
        except Exception as e: loaded = None; TE_GUARD["import_error"] = repr(e)[:120]
    if loaded is not None and _has_required(loaded):
        TE_GUARD.update(action="kept", detail=f"typing_extensions has {REQUIRED} — from {getattr(loaded, '__file__', '?')}" + (" (already loaded before the kit)" if was_loaded else "")); return TE_GUARD
    old_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if not os.path.isfile(VENDORED): raise ImportError(f"[chrombpnet_k1] typing_extensions lacks {REQUIRED} (from {old_file}) and the kit's vendored copy is missing at {VENDORED}")
    spec = importlib.util.spec_from_file_location("typing_extensions", VENDORED); mod = importlib.util.module_from_spec(spec)
    sys.modules["typing_extensions"] = mod                                            # installed BEFORE exec so any self-reference resolves to it
    try: spec.loader.exec_module(mod)
    except Exception as e:
        if loaded is not None: sys.modules["typing_extensions"] = loaded
        else: sys.modules.pop("typing_extensions", None)
        raise ImportError(f"[chrombpnet_k1] the vendored typing_extensions failed to load: {e!r}") from e
    if not _has_required(mod):
        sys.modules["typing_extensions"] = loaded if loaded is not None else sys.modules.pop("typing_extensions", None); raise ImportError(f"[chrombpnet_k1] the vendored typing_extensions lacks {REQUIRED}")
    TE_GUARD.update(action="replaced" if loaded is not None else "provided",
                    detail=(f"typing_extensions from {old_file} lacks {tuple(n for n in REQUIRED if not hasattr(loaded, n))}" + (" (imported BEFORE the kit — e.g. by TensorFlow under the DET recipe)" if was_loaded else "") if loaded is not None else "no typing_extensions importable") + f" -> the kit's vendored 4.13.2 installed HERE-first ({VENDORED})")
    print(f"[chrombpnet_k1] typing_extensions guard: {TE_GUARD['action']} — {TE_GUARD['detail']}", flush=True)
    return TE_GUARD

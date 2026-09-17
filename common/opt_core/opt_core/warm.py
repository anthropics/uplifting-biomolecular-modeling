"""opt_core.warm — import the stack's heavy model libraries EARLY and SHALLOW, through the large-frame trampoline, so that no model code
imports them lazily mid-item from an unlucky call depth.

Why.  CPython 3.11/3.12 map and unmap a 16 KiB data-stack chunk per Python call when the stack top sits at a chunk boundary; in a
syscall-heavy container runtime that is ~40 us per call instead of ~0.1 us.  A library whose import runs a flat Python loop of millions of
iterations (cuequivariance_ops_torch 0.11 builds a 1.57 M-entry autotune input table in a module body) costs ~1.4 s from most call depths
and 60-75 s from a boundary depth -- and a model that binds this core's rows for its trunk typically still reaches such a library LAZILY,
inside the first item, from whatever depth its confidence head or template stack has (measured: 56-75 s of host time at 0 % GPU on item 1).
Importing the library once, early, through :func:`opt_core._pystack.import_padded` makes that later lazy import a dictionary lookup.

    opt_core.warm_imports()                    # a kit: ONE line at start-up (main process, before the model is built; torch is imported first by rule)
    opt_core.warm_imports(rows=("v4", "tmk3_fast"))  # ... also importing the modules those kernels.trimul rows serve through

Every provider face (kernels.trimul / triattn / apb / ln / transition, fpf_trimul_v4.generic) calls :func:`auto_warm` at its FIRST serving
call, so a kit that binds any row gets the libraries imported inside its first pair-stack call even without the line above (the line only
moves the ~1.4 s from the first call to start-up).  ``OPT_CORE_NO_WARM_IMPORTS`` (or the older ``OPT_CORE_NO_STOCK_PRELOAD``) set to any
value: skipped -- for timing a library's import yourself.  Nothing here raises: a library that is not installed is the word ``absent``, one
whose import fails is ``error:<word>`` (the row that needs it refuses BY NAME at its call, as before)."""
import os
import sys
import time

DEFAULT_LIBRARIES = ("torch", "cuequivariance_ops_torch", "cuequivariance_torch")   # TORCH FIRST (stable rule, also enforced per entry below): a CUDA extension imported before torch
                                                                                # cannot resolve the CUDA runtime libraries torch's own loader provides (cu13: libnvrtc.so.13) and stays
                                                                                # broken for the whole process; then the import-time bodies measured heavy (ops first: the table lives there)
TORCH = "torch"
ENV_OFF = ("OPT_CORE_NO_WARM_IMPORTS", "OPT_CORE_NO_STOCK_PRELOAD")
REPORT = {}                         # library -> seconds (imported here) | 'present' (already imported) | 'absent' | 'skipped:env' | 'error:<Type>:<word>'
ORIGINS = []                        # who asked, in order: 'kit', 'trimul', 'triattn', ... (diagnostic)
_DONE = set()                       # libraries already decided in this process


def _present(name):
    if name in sys.modules:
        return True
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_SAID = set()                       # unavailable words already printed (one stderr line per library per process)
LOADER_RETRY_FAMILY = {             # a shared-object load failure of these is retried ONCE after purging the family's half-loaded modules (a caller that
    "cuequivariance_ops_torch": ("cuequivariance_ops", "cuequivariance_ops_torch"),                        # imported them before torch leaves e.g.
    "cuequivariance_torch": ("cuequivariance_ops", "cuequivariance_ops_torch", "cuequivariance_torch"),    # 'cuequivariance_ops' degraded in sys.modules)
}


def _purge(prefixes):
    gone = [k for k in list(sys.modules) if any(k == p or k.startswith(p + ".") for p in prefixes)]
    for k in gone:
        sys.modules.pop(k, None)
    return gone


def _unavailable(name, exc):
    word = "unavailable:%s: %s" % (type(exc).__name__, str(exc).strip().split("\n")[0][:80])
    REPORT[name] = word
    if name not in _SAID:
        _SAID.add(name)
        try:
            sys.stderr.write("[opt_core.warm] warm_imports:%s=%s\n" % (name, word))
        except Exception:                                            # pragma: no cover - a closed stderr is not a reason to raise here
            pass
    return word


def warm_imports(libraries=None, rows=None, *, origin="kit"):
    """Import each of ``libraries`` (default :data:`DEFAULT_LIBRARIES`) that is installed and not imported yet, through the trampoline; with
    ``rows`` also the modules those ``kernels.trimul`` rows serve through (``kernels.trimul.preload``).  NEVER POISONS THE PROCESS: torch is
    imported first, by rule; when torch cannot be imported nothing else is attempted (``REPORT['warm_imports'] = 'torch_not_loaded'``, the other
    libraries ``declined:torch_not_loaded``); each library's import is guarded -- on failure every ``sys.modules`` key it added is rolled back
    (:func:`opt_core._pystack.guarded_import`), the word ``unavailable:<Exception>: <message head>`` is recorded (and written to stderr once), and
    the next library is attempted.  A later plain ``import`` of an unavailable library in the same process starts clean; a later
    ``warm_imports()`` call re-attempts it (decided libraries -- imported / present / absent / skipped -- are not re-decided).  Never raises.
    Returns :data:`REPORT` (a dict)."""
    ORIGINS.append(str(origin))
    names = tuple(DEFAULT_LIBRARIES if libraries is None else libraries)
    names = (TORCH,) + tuple(n for n in names if n != TORCH)           # torch precedes everything, whatever the caller's order (import_padded enforces it per entry too)
    off = any(os.environ.get(k) for k in ENV_OFF)
    try:
        from opt_core._pystack import guarded_import
    except Exception as e:                                          # pragma: no cover - the trampoline module is part of this package
        REPORT["warm_imports"] = "unavailable:%s" % type(e).__name__
        return REPORT
    torch_loaded = TORCH in sys.modules
    for name in names:
        if name in _DONE:
            if name == TORCH:
                torch_loaded = TORCH in sys.modules
            continue
        if name in sys.modules:
            REPORT[name] = "present"; _DONE.add(name)
        elif off:
            REPORT[name] = "skipped:env"; _DONE.add(name)
        elif not _present(name):
            REPORT[name] = "absent"; _DONE.add(name)
            if name == TORCH:
                REPORT["warm_imports"] = "torch_not_loaded"
        elif name != TORCH and not torch_loaded:                    # torch could not be loaded: import NOTHING else (a CUDA extension without torch's runtime poisons the process)
            REPORT[name] = "declined:torch_not_loaded"
        else:
            t0 = time.perf_counter()
            try:
                guarded_import(name)
                REPORT[name] = round(time.perf_counter() - t0, 3); _DONE.add(name)
                REPORT.pop("warm_imports", None) if name == TORCH else None
            except BaseException as e:                              # never raises; sys.modules already rolled back by guarded_import
                err = e
                if name in LOADER_RETRY_FAMILY and TORCH in sys.modules and isinstance(e, (ImportError, OSError)):
                    purged = _purge(LOADER_RETRY_FAMILY[name])          # a loader failure with torch now present: the family's half-loaded modules from an
                    try:                                                # EARLIER pre-torch import are the usual cause -> purge them, retry ONCE (guarded again)
                        import importlib; importlib.invalidate_caches()
                        guarded_import(name)
                        REPORT[name] = round(time.perf_counter() - t0, 3); _DONE.add(name)
                        REPORT[name + ":recovered"] = "purged %d half-loaded module(s), retried once" % len(purged)
                        err = None
                    except BaseException as e2:
                        err = e2
                if err is not None:
                    _unavailable(name, err)
                    if name == TORCH:
                        REPORT["warm_imports"] = "torch_not_loaded"
        if name == TORCH:
            torch_loaded = TORCH in sys.modules
    if rows and torch_loaded:
        try:
            from opt_core.kernels import trimul as _T
            for m, v in _T.preload(rows=tuple(rows)).items():
                REPORT.setdefault(m, v)
        except Exception as e:                                      # a face that cannot import here is reported, not raised
            REPORT.setdefault("kernels.trimul.preload", "error:%s" % type(e).__name__)
    return REPORT


def auto_warm(origin):
    """The provider faces' shared once-guard: the first serving call of any face in the process warms :data:`DEFAULT_LIBRARIES`; every later
    call from any face is one set lookup (an ``unavailable`` library is re-attempted by the next face's first call, guarded the same way).
    Returns :data:`REPORT`; never raises."""
    if _DONE.issuperset(DEFAULT_LIBRARIES):
        return REPORT
    try:
        return warm_imports(origin=origin)
    except BaseException as e:                                      # pragma: no cover - warm_imports does not raise; belt and braces for a serving path
        REPORT["warm_imports"] = "unavailable:%s" % type(e).__name__
        return REPORT

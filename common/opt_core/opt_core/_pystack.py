"""opt_core._pystack — run a Python-heavy one-off (a library import, a first kernel launch that compiles) off the interpreter's
stack-chunk boundary.

The hazard.  CPython 3.11 and 3.12 keep frames on a per-thread "data stack" made of 16 KiB chunks: a call whose frame does not fit the current
chunk maps a new chunk (an anonymous ``mmap``), and the return that empties a chunk unmaps it at once.  A hot loop that makes a Python-level
call per iteration while the stack top sits exactly at a chunk boundary therefore pays one ``mmap`` + one ``munmap`` per iteration.  On a plain
kernel that is ~1-2 us a call; in a syscall-intercepting container runtime (the gVisor class) it is ~40 us a call, half of it
system time -- a loop that takes 1.4 s at any other depth takes 60 s there.  Which depth is "the boundary" is an accident of the caller's frame
sizes: the same import is fast from one call path and a minute from another (measured: cuequivariance_ops_torch 0.11's import-time autotune
input table, 1.57 M iterations each calling an ``Enum.value`` getter, imported lazily from inside a model's confidence module: 62-80 s of host
time at 0 % GPU; 1.4 s from the trunk's call path; per-call cost at boundary depths 37-62 us vs 0.21 us elsewhere, recurring every ~127
frames of one size).

The remedy here.  :func:`padded_call` runs ``fn`` from a trampoline frame that declares tens of thousands of local slots: the interpreter sizes
that frame's chunk to the frame (a few hundred KiB, one ``mmap`` for the whole call), the trampoline occupies its head, and every callee frame
-- the import machinery, the module bodies, their loops' getter calls, a Triton front end -- is pushed inside the same chunk's free tail, so no
callee is ever the first frame of a chunk and nothing is unmapped per call.  Measured in the same runtime: no slow depth for any caller depth
0-419 nor for callee chains up to 299 frames below the trampoline (plain calls: 37 us/call at every boundary depth).  Cost: building the
trampoline once (~0.2 s, a generated function), one chunk map per padded call.  Use it for one-off host work only (imports, first launches);
steady-state calls go direct.

:func:`call_cost_here` measures the per-call cost at the caller's depth (diagnostic: > ~3 us means the caller sits at a boundary in a runtime
of that class)."""
import sys
import time

PAD_SLOTS = 35000                     # local slots of the trampoline frame: 35 000 x 8 B = 280 KB -> the interpreter maps a 512 KiB chunk, ~230 KB of it free for callees
_TRAMPOLINE = {}


def _trampoline(nslots=PAD_SLOTS):
    f = _TRAMPOLINE.get(nslots)
    if f is None:
        src = "def _padded(fn, args, kwargs):\n    if 0:\n" + "".join("        _p%d = 0\n" % i for i in range(int(nslots))) + "    return fn(*args, **kwargs)\n"
        ns = {}
        exec(compile(src, "<opt_core._pystack:trampoline>", "exec"), ns)      # noqa: S102 - a generated no-op-locals wrapper, no input text
        f = _TRAMPOLINE[nslots] = ns["_padded"]
    return f


def padded_call(fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` run from the large-frame trampoline (see the module text).  Exceptions propagate unchanged.  On interpreters
    without the chunked data stack (< 3.11) or when the trampoline cannot be built the call is made directly."""
    if sys.version_info < (3, 11):
        return fn(*args, **kwargs)
    try:
        t = _trampoline()
    except Exception:                                                # pragma: no cover - a build failure never costs the caller its call
        return fn(*args, **kwargs)
    return t(fn, args, kwargs)


def call_cost_here(n=2000):
    """Microseconds per trivial Python-level call at the CALLER's stack depth (median of 3 bursts of ``n`` calls).  ~0.1-0.3 us normally;
    tens of us when the caller's depth is a chunk boundary in a syscall-heavy runtime."""
    def leaf():
        return None
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _i in range(n):
            leaf()
        best.append((time.perf_counter() - t0) / n * 1e6)
    best.sort()
    return best[1]


def import_padded(name):
    """``importlib.import_module(name)`` through :func:`padded_call` when the module is not imported yet (a module already in ``sys.modules``
    is returned directly: the padding is for the one import that executes module bodies)."""
    m = sys.modules.get(name)
    if m is not None:
        return m
    import importlib
    if name.split(".")[0] != "torch" and "torch" not in sys.modules:      # STABLE RULE: torch precedes any other padded import -- a CUDA extension
        import importlib.util                                              # (cuequivariance_ops_torch, ...) loaded before torch cannot resolve the runtime
        try:                                                               # libraries torch's loader provides (cu13: libnvrtc.so.13) and stays broken for
            has_torch = importlib.util.find_spec("torch") is not None      # the process; no torch on the path = nothing to order
        except (ImportError, ValueError):
            has_torch = False
        if has_torch:
            guarded_import("torch")
    return guarded_import(name)


def sweep_orphans(name):
    """Remove ``name``'s sub-modules left in ``sys.modules`` by an EARLIER failed import of ``name`` (the package itself is gone -- Python drops a
    module whose body raised -- but the sub-modules it had imported stay and poison the next attempt).  Returns the removed keys."""
    if name in sys.modules:
        return []
    orphans = [k for k in list(sys.modules) if k.startswith(name + ".")]
    for k in orphans:
        sys.modules.pop(k, None)
    return orphans


def guarded_import(name):
    """``importlib.import_module(name)`` padded, and NEVER POISONING the process: the orphan sub-modules of an earlier failed attempt are swept
    first; when the import raises, every ``sys.modules`` key the attempt added (the half-initialised module, the sub-modules and dependencies
    it dragged in) is removed again before the exception propagates -- so a later import of the same library, once its cause is fixed (torch
    loaded first, a path set), starts clean.  Raises what the import raised (callers refuse by name or record the word)."""
    import importlib
    m = sys.modules.get(name)
    if m is not None:
        return m
    sweep_orphans(name)
    before = set(sys.modules)
    try:
        return padded_call(importlib.import_module, name)
    except BaseException:
        for k in [k for k in list(sys.modules) if k not in before]:
            sys.modules.pop(k, None)
        importlib.invalidate_caches()
        raise

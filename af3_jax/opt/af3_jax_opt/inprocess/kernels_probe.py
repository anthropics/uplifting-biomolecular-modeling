"""kernels_probe.py — the KERNELS reading of a model process: which accelerator implementation the process ACTUALLY bound at the call
sites the fork uses, read inside the process, printed as ONE line at the first kernel call (before any timed item) and ONE census line at exit.

Rides every model process of a ``pred`` pass — the stock route's too — through the hook directory the wrapper puts LAST on the model
process's ``PYTHONPATH`` (the same directory and ``sitecustomize`` that boot the shared core's peak-memory probe; ``af3_jax_opt.kernels.arm``
copies this file there byte for byte). Inert unless the wrapper names an expectation (``OPT_KERNELS_EXPECT``). Standard
library only at import; it never imports jax, tokamax or alphafold3 itself: it installs ONE ``sys.meta_path`` finder that fires when the
process imports ``tokamax`` and wraps the two public entry points the fork calls — ``tokamax.dot_product_attention`` (pair attention,
``alphafold3/model/network/modules.py:183-190``; the diffusion levers' calls) and ``tokamax.gated_linear_unit`` (``modules.py:79,304``,
``diffusion_transformer.py:109``) — in pass-through counters (same arguments, same return: no numerics of their own). At the FIRST call of
either it reads:

  flag          absl ``FLAGS.flash_attention_implementation`` as the process holds it THEN — after the fork's CPU auto-downgrade
                (``run_alphafold.py:1126-1135``) has or has not rewritten it: the value ``make_model_config`` receives
  dpa_site      what ``alphafold3.model.network.modules.tokamax`` IS: ``tokamax`` (the module: stock) | ``attncfg`` (the Pallas add-on's
                ``_TokamaxProxy`` whose ``dot_product_attention`` pins the Pallas-Triton tile config, ``af3_pallas_levers.py:83-95``) | other:<name>
  triatt_site   ``modules.GridSelfAttention``: ``stock`` | ``fpf`` (the FlashPairformer add-on's ``FlashGridSelfAttention``) | other:<module.qualname>
  trimul_site   ``modules.TriangleMultiplication``: ``stock`` | ``glut`` (the Pallas add-on's GLUT subclass) | ``fpf`` | other:<…>
  transition_site ``modules.TransitionBlock``: ``stock`` | other:<module.qualname>
  attn_impls    tokamax's attention ``IMPLEMENTATIONS`` keys (the accelerator's own capability table) and the class bound for ``flag``
  attn_supported ``IMPLEMENTATIONS[flag].supported_on(<the process's default device>)`` (Pallas-Triton: compute capability >= 8.0)
  glu_impls / glu_chain  tokamax's GLU ``IMPLEMENTATIONS`` and the chain ``implementation=None`` resolves to, filtered the way tokamax's own
                api filters it (``triton`` dropped without Triton support, ``gated_linear_unit/api.py:105-107``); ``glu_head`` = the
                implementation that runs when nothing falls back (a fallback is tokamax's own logged event ``Failed to run implementation``,
                ``api.py:116`` — read from the transcript by the wrapper, never silently)
  tokamax / jax versions, backend platform, device kind, compute capability, ``XLA_FLAGS`` / the two client-memory variables as the process sees them

and prints ``[af3-jax-opt] KERNELS-PROBE k=v …`` (stderr). When the expectation names a value the reading contradicts (:func:`verdict` — the
ONE comparison, imported by the wrapper too), the line ends ``verdict=REFUSED:<reasons>`` and the process exits 5 there: before the first
timed item completes. At exit: ``[af3-jax-opt] KERNELS-CENSUS pid=… dpa_calls=<impl:n,…> glu_calls=<impl:n,…>`` (trace-time call counts per
``implementation=`` argument: the corroboration of the first-call reading). A process that imports tokamax and never calls it prints the census
only; one that never imports tokamax prints nothing (the wrapper names a missing probe line and refuses the pass).

"""
import os
import sys
import threading

PREFIX = "[af3-jax-opt]"
ENV_EXPECT = "OPT_KERNELS_EXPECT"            # `k=v;k=v` — the route's expected reading, written by the wrapper from its mode table (af3_jax_opt.kernels.expected)
EXIT_REFUSED = 5
REFUSE_GRACE_S = 60.0                          # after a refusal is raised: the hard exit that follows if the interpreter has not ended by then
STOCK_MODULES = "alphafold3.model.network.modules"
SITE_CLASSES = {("af3_pallas_levers", "_TokamaxProxy"): "attncfg", ("af3_pallas_levers", "TriangleMultiplication"): "glut",     # (module basename, class name) -> the owner's word (kernels.site_owner's words: attncfg | glut | fpf)
                ("patch", "FlashGridSelfAttention"): "fpf", ("patch", "FlashTriangleMultiplication"): "fpf"}                      # af3_flashpairformer/patch.py, imported as `af3_flashpairformer.patch` or by path as `patch`
READING_KEYS = ("flag", "dpa_site", "triatt_site", "trimul_site", "transition_site", "attn_impls", "attn_class", "attn_supported", "glu_impls", "glu_chain",
                "glu_head", "tokamax", "jax", "backend", "device", "cc", "xla_flags", "prealloc", "mem_fraction", "first", "pid")

_LOCK = threading.Lock()
_STATE = {"armed": False, "fired": False, "dpa": {}, "glu": {}, "lines": 0}


# ---------------------------------------------------------------- the ONE comparison (the wrapper imports this function; the probe calls it in-process)
def parse_expect(text):
    """``k=v;k=v`` -> {k: v} (empty text -> {})."""
    out = {}
    for tok in (text or "").split(";"):
        if "=" in tok:
            k, v = tok.split("=", 1); out[k.strip()] = v.strip()
    return out


def verdict(reading, expect):
    """The refusal reasons of ``reading`` against ``expect`` ({} or [] = conforms). Keys compared: ``flag`` (the implementation the process
    resolved == the one the route requests), ``dpa_site`` / ``triatt_site`` / ``trimul_site`` (who owns each call site == what the route's lever
    set installs), ``glu_head`` (the GLU implementation tokamax resolves first == the route's), ``backend`` (gpu), ``attn_supported`` (the bound
    attention class supports the device: ``1``), ``attn_class`` present (the requested implementation is in tokamax's table)."""
    reasons = []
    for k, want in expect.items():
        got = str(reading.get(k, "unread"))
        if k == "attn_class":
            if got in ("absent", "unread"):
                reasons.append(f"flash_impl absent: tokamax has no implementation {reading.get('flag')!r} (IMPLEMENTATIONS={reading.get('attn_impls')})")
            continue
        if k.endswith("_site"):
            got = got.split("+", 1)[0]                      # a wrapped site: the base owns the kernel (the wrappers are worded on the line, not compared)
        if got != str(want):
            reasons.append(f"{k}: expected {want} got {got}")
    return reasons


# ---------------------------------------------------------------- the reading
def _site_word(obj, stock_name):
    """``stock`` when ``obj`` is the fork's own class of ``alphafold3.model.network.modules``; a lever's short name when it is one of SITE_NAMES;
    else other:<module>.<qualname>. A class that wraps another (``__wrapped__``: the memory levers' subclasses, big_levers.py) is worded
    ``<base word>+w:<wrapper name>[+…]`` — the base owns the kernel, the wrappers are named."""
    wrappers, seen = [], set()
    while "__wrapped__" in getattr(obj, "__dict__", {}) and obj.__dict__["__wrapped__"] is not obj and id(obj) not in seen and len(wrappers) < 6:   # an object's OWN __wrapped__ (a subclass inheriting its base's is not itself a wrapper)
        seen.add(id(obj))
        wrappers.append("w:" + str(getattr(obj, "__qualname__", type(obj).__qualname__)).split(".")[-1])
        obj = obj.__dict__["__wrapped__"]
    base = _base_site_word(obj, stock_name)
    return base + ("+" + "+".join(wrappers) if wrappers else "")


def _base_site_word(obj, stock_name):
    mod = getattr(obj, "__module__", None) or type(obj).__module__
    qn = getattr(obj, "__qualname__", None) or type(obj).__qualname__
    if mod == STOCK_MODULES and qn == stock_name:
        return "stock"
    word = SITE_CLASSES.get((str(mod).split(".")[-1], str(qn).split(".")[-1]))          # module basename + class name (a class made inside a function keeps its own name last in __qualname__)
    if word:
        return word
    return f"other:{mod}.{qn}".replace(" ", "")


def _dpa_site_word(M):
    tk = getattr(M, "tokamax", None)
    if tk is None:
        return "unread"
    if type(tk).__name__ == "module" and getattr(tk, "__name__", "") == "tokamax":
        return "tokamax"
    t = type(tk)
    if t.__module__ == "af3_pallas_levers" and t.__name__ == "_TokamaxProxy":
        return "attncfg"
    return f"other:{t.__module__}.{t.__qualname__}".replace(" ", "")


def impl_name(impl):
    """The word for one ``implementation=`` argument: a name as given (``triton``), ``None`` (tokamax's default chain), or ``obj:<Class>[block_q=…]`` for an instance (the pinned Pallas-Triton kernel)."""
    if impl is None:
        return "None"
    if isinstance(impl, str):
        return impl
    if isinstance(impl, (list, tuple)):
        return "seq:" + "/".join(impl_name(i) for i in impl)
    cfg = getattr(impl, "config", None)
    extra = ""
    if cfg is not None:
        fields = [f for f in ("block_q", "block_k", "num_warps", "num_stages") if hasattr(cfg, f)]
        if fields:
            extra = "(" + "x".join(str(getattr(cfg, f)) for f in fields) + ")"   # bq x bk x warps x stages (no commas: the census line is comma-separated)
    return f"obj:{type(impl).__name__}{extra}"


def read(first, tk):
    """The reading (a dict of READING_KEYS) at the first kernel call; every field read defensively (a field that cannot be read says ``unread:<why>``, never raises)."""
    r = {"first": first, "pid": os.getpid()}
    def safe(key, fn):
        try:
            r[key] = fn()
        except Exception as e:  # noqa: BLE001 - a reader: name the failure in the field, never raise into the model
            r[key] = f"unread:{type(e).__name__}"
    def flag():
        from absl import flags
        try:
            return str(flags.FLAGS["flash_attention_implementation"].value)
        except Exception as e:  # noqa: BLE001
            return f"unread:{type(e).__name__}"
    safe("flag", flag)
    M = sys.modules.get(STOCK_MODULES)
    if M is None:
        for k in ("dpa_site", "triatt_site", "trimul_site", "transition_site"):
            r[k] = "unread:modules_not_imported"
    else:
        safe("dpa_site", lambda: _dpa_site_word(M))
        safe("triatt_site", lambda: _site_word(getattr(M, "GridSelfAttention"), "GridSelfAttention"))
        safe("trimul_site", lambda: _site_word(getattr(M, "TriangleMultiplication"), "TriangleMultiplication"))
        safe("transition_site", lambda: _site_word(getattr(M, "TransitionBlock"), "TransitionBlock"))
    def attn():
        from tokamax._src.ops.attention import api as A
        impls = dict(A.IMPLEMENTATIONS)
        r["attn_impls"] = ",".join(sorted(impls))
        cls = impls.get(str(r.get("flag")))
        r["attn_class"] = type(cls).__name__ if cls is not None else "absent"
        if cls is None:
            r["attn_supported"] = "0"
        else:
            import jax
            r["attn_supported"] = "1" if cls.supported_on(jax.devices()[0]) else "0"
        return None
    safe("_attn", attn); r.pop("_attn", None)
    def glu():
        from tokamax._src.ops.gated_linear_unit import api as G
        from tokamax._src import gpu_utils
        r["glu_impls"] = ",".join(sorted(G.IMPLEMENTATIONS))
        chain = [i for i in G._DEFAULT_IMPLEMENTATION if not (i == "triton" and not gpu_utils.has_triton_support())]   # tokamax gated_linear_unit/api.py:105-107, verbatim rule
        r["glu_chain"] = ",".join(chain) or "none"
        r["glu_head"] = chain[0] if chain else "none"
        return None
    safe("_glu", glu); r.pop("_glu", None)
    safe("tokamax", lambda: getattr(tk, "__version__", None) or __import__("importlib.metadata").metadata.version("tokamax"))
    def jaxinfo():
        import jax
        r["jax"] = jax.__version__
        r["backend"] = jax.default_backend()
        d = jax.devices()[0]
        r["device"] = str(getattr(d, "device_kind", "?")).replace(" ", "_")
        r["cc"] = str(getattr(d, "compute_capability", "n/a"))
        return None
    safe("_jax", jaxinfo); r.pop("_jax", None)
    r["xla_flags"] = (os.environ.get("XLA_FLAGS") or "unset").replace(" ", "|")
    r["prealloc"] = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "unset")
    r["mem_fraction"] = os.environ.get("XLA_CLIENT_MEM_FRACTION", os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "unset"))
    for k in READING_KEYS:
        r.setdefault(k, "unread")
    return r


def format_line(tag, fields):
    return " ".join([PREFIX, tag] + [f"{k}={v}" for k, v in fields.items() if v is not None])


def _emit(text):
    try:
        sys.stderr.write(text + "\n"); sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _first_call(first, tk):
    """Print the PROBE line once; exit EXIT_REFUSED when the expectation is contradicted (before the first timed item completes)."""
    with _LOCK:
        if _STATE["fired"]:
            return
        _STATE["fired"] = True
    reading = read(first, tk)
    expect = parse_expect(os.environ.get(ENV_EXPECT, ""))
    reasons = verdict(reading, expect) if expect else []
    fields = {k: reading[k] for k in READING_KEYS}
    fields["expect"] = ";".join(f"{k}={v}" for k, v in expect.items()) or "none"
    fields["verdict"] = ("REFUSED:" + "|".join(x.replace(" ", "_") for x in reasons)) if reasons else "ok"
    _STATE["lines"] += 1
    _emit(format_line("KERNELS-PROBE", fields))
    if reasons:
        _emit(f"{PREFIX} KERNELS REFUSED in pid {os.getpid()}: " + "; ".join(reasons) + f" — exit {EXIT_REFUSED} before the first timed item")
        refuse_exit()


class KernelsRefused(SystemExit):
    """Raised out of the first kernel call on a refusal: unwinds the script's ``with`` blocks (its prefetch workers are joined) and ends the
    interpreter with code EXIT_REFUSED. A hard ``os._exit`` there would orphan the script's worker processes, which hold the transcript pipe
    open — the wrapper would wait on it forever."""


def refuse_exit():
    """End this model process with EXIT_REFUSED: terminate its own multiprocessing children, arm a last-resort hard exit (should anything
    swallow the SystemExit), raise KernelsRefused."""
    try:
        import multiprocessing
        for ch in multiprocessing.active_children():
            try:
                ch.terminate()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    t = threading.Timer(REFUSE_GRACE_S, lambda: os._exit(EXIT_REFUSED)); t.daemon = True; t.start()
    raise KernelsRefused(EXIT_REFUSED)


def _wrap(tk, name, key):
    orig = getattr(tk, name)
    if getattr(orig, "_kernels_probe_wrapped", False):
        return
    def counted(*a, **kw):
        word = impl_name(kw.get("implementation"))
        with _LOCK:
            _STATE[key][word] = _STATE[key].get(word, 0) + 1
        if not _STATE["fired"]:
            _first_call(key, tk)
        return orig(*a, **kw)
    counted._kernels_probe_wrapped = True
    counted.__wrapped__ = orig
    counted.__name__ = getattr(orig, "__name__", name); counted.__doc__ = getattr(orig, "__doc__", None)
    setattr(tk, name, counted)


def arm(tk):
    """Wrap tokamax's two public entry points in the pass-through counters (idempotent)."""
    with _LOCK:
        if _STATE["armed"]:
            return
        _STATE["armed"] = True
    for name, key in (("dot_product_attention", "dpa"), ("gated_linear_unit", "glu")):
        try:
            _wrap(tk, name, key)
        except Exception as e:  # noqa: BLE001
            _emit(f"{PREFIX} KERNELS-PROBE NOTE: could not wrap tokamax.{name}: {e!r} (calls through it go uncounted)")


def _census():
    if not _STATE["armed"]:
        return
    fmt = lambda d: ",".join(f"{k}:{v}" for k, v in sorted(d.items())) or "none"
    _emit(format_line("KERNELS-CENSUS", {"pid": os.getpid(), "dpa_calls": fmt(_STATE["dpa"]), "glu_calls": fmt(_STATE["glu"]), "probe_lines": _STATE["lines"]}))


HOOKS = {"tokamax": lambda m: arm(m)}                                         # module name -> what to do right after it executes


class _TokamaxHook:
    """A meta-path finder for the names in HOOKS (``tokamax``): defers to the rest of ``sys.meta_path`` for the spec and runs the
    hook right after the module executes (a post-import hook; the module object, its loader and its code are the package's own)."""
    _busy = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in HOOKS or _TokamaxHook._busy:
            return None
        _TokamaxHook._busy = True
        try:
            import importlib.util
            spec = importlib.util.find_spec(fullname)
        finally:
            _TokamaxHook._busy = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        loader = spec.loader
        exec_module = loader.exec_module
        def exec_then_arm(module):
            exec_module(module)
            if module.__name__ in HOOKS:
                try:
                    HOOKS[module.__name__](module)
                except Exception as e:  # noqa: BLE001
                    _emit(f"{PREFIX} KERNELS-PROBE NOTE: arm failed for {module.__name__}: {e!r}")
        try:
            loader.exec_module = exec_then_arm          # instance attribute on this loader object only (tokamax's own SourceFileLoader instance)
        except Exception as e:  # noqa: BLE001
            _emit(f"{PREFIX} KERNELS-PROBE NOTE: cannot hook tokamax's loader: {e!r}")
        return spec


def boot(loader="sitecustomize"):
    """The hook's entry: inert without OPT_KERNELS_EXPECT; else install the tokamax post-import hook (or arm at once if tokamax is somehow
    already imported) and the exit census. Never raises into the process."""
    try:
        if ENV_EXPECT not in os.environ:
            return False
        for name, fn in HOOKS.items():
            if name in sys.modules:
                fn(sys.modules[name])
        if any(name not in sys.modules for name in HOOKS):
            sys.meta_path.insert(0, _TokamaxHook())
        import atexit
        atexit.register(_census)
        return True
    except Exception as e:  # noqa: BLE001
        _emit(f"{PREFIX} KERNELS-PROBE REFUSED: boot ({loader}): {e!r}; no reading")
        return False

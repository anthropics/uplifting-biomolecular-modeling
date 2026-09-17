"""boltz2_opt.exactln — the engine adapter of the carried ``exactln`` kernel (the shared core's ``kernels.ln`` row ``exactln``,
``opt_core/kernels/ln/exactln``: a bit-for-bit replica of ATen's CUDA layer_norm forward, torch 2.12.0, on a one-warp-per-row schedule; the kit's own
copy under ``opt/forward/exactln`` is not carried; its word ``on`` is refused by name) onto every ``torch.nn.functional.layer_norm`` call of the worker process:
``nn.LayerNorm`` (the Pairformer stacks' tri-attention / transition / pair-bias LayerNorms, the MSA module's, the template module's, the
confidence module's), boltz's own ``triangular_attention.primitives.LayerNorm``, and the kit adapters that call those modules
(boltz2_opt.pairblock / boltz2_opt.transition feed the module's own LayerNorm output to their cells: that output is now this kernel's).
Attached by ``boltz2_opt.worker_launch --attach exactln`` right after ``boltz.model.models.boltz2`` imports (the worker's own model import).

Switch ``BOLTZ_EXACTLN`` (the mode table sets it, modes.py): ``core`` = ``torch.nn.functional.layer_norm`` is replaced process-wide by
``_layer_norm``: a CUDA call over the last dimension whose ATen dispatch is the vectorized kernel (fp32 or bf16 x, C % 4 == 0, 4 <= C <= 1024,
16-byte aligned rows, weight and bias both present or both absent) and whose input holds at least ``MIN_NUMEL`` elements is served by the replica
— under bf16 autocast exactly as autocast serves it (fp32 compute on the widened input, fp32 output; the widening is fused into the kernel's
load), outside autocast in the input's dtype (fp32; or bf16 storage with fp32 arithmetic, ATen's ``<BFloat16, float>`` instantiation) — and
every other call takes torch's own ``layer_norm`` by a census word: ``below_min_numel`` (the sequence-track and other small LayerNorms: the
replica's Python launch costs more than ATen's C++ one and the tensor is too small to win it back), ``capturing_uncompiled`` (a call inside a CUDA-graph capture whose
kernel is not built yet — no module load inside a capture; the captured calls that ARE built replay the replica's launch, bitwise), ``grad`` (autograd on: inference only), ``cpu``, ``fp16_autocast``,
``dtype:*`` / ``width:*`` / ``normalized_shape`` / ``aten_rowwise_path`` / ``weight_form`` … (``exactln.plan``'s words).  Unset / empty =
nothing is replaced (stock statements).  The kernels for C = 64 and 128 (fp32, affine) are compiled at apply (NVRTC through cuda.bindings,
~0.3 s each); other widths compile at first use, counted ``compiled``.

First-call bit comparison: the first eager call of every shape class (kind, C, affine, dtypes, layout) runs torch's statement too and compares
bit for bit; a class is served only after it matched (``report()['selftest']`` lists the verdicts; a mismatch refuses the class by name and the
gate).  The replica's arithmetic was read off the sm_90 binary of the pinned torch (2.12.0+cu130); ``apply()`` refuses by name on another torch
(``torch_version:...``, the carried package's pin) or a compute capability the core LayerNorm provider does not admit (its word) — the lever steps aside BY NAME (dispositions()); a missing NVRTC route exits the worker 3 through
worker_launch, never a silent stock run.  ``EXPECTED`` lists the census words a healthy run may show; anything else, or a kernel error (the call
is then served by torch's layer_norm and counted), refuses the fail-closed gate (``verdict()``).
"""
import os
import sys
import time
from typing import Any, Dict, List, Optional

TAG = "boltz2-opt"
NAME = "E5.exactln"
SWITCH = "BOLTZ_EXACTLN"
VARIANTS = ("core",)                                    # `core` = the package as opt_core carries it (kernels.ln row `exactln`, opt_core.kernels.ln.carried_module("exactln"):
                                                        # exactln 1.3.3 — one source of the replica for every engine; this adapter's call routing, first-call bit comparison and
                                                        # census around it). The kit's own copy (word `on`, opt/forward/exactln: byte-identical .cu / .py) is not carried:
RETIRED_VARIANTS = ("on",)                              # words whose kit copy left this tree — still named here, refused BY NAME at apply (exactln:on:kit_copy_retired), nothing
                                                        # substituted, never a stock run under the word (as the PAIRFUSE driver's RETIRED_PICKS)
CORE_ROW = "exactln"                                    # the opt_core kernels.ln row the `core` variant binds (its DEFAULT RULE: the row word = exactly that row)
LEVERS = ("exactln",)                                   # the registry lever this adapter installs (worker_launch reads it)
PACKAGE = "exactln"                                     # the name the carried package is registered under (sys.modules["exactln"]): the kit modules that name it see the core's module
# No torch pin or architecture table lives here: the torch whose ATen kernel the replica reproduces is the carried package's own
# ``TORCH_PIN``, and the compute capabilities its bits are proven on are the core LayerNorm provider's (``opt_core.kernels.ln.admits``:
# EXACTLN_PROVEN_CC) — the adapter asks the face and steps aside BY NAME with the face's own word when it declines (dispositions()).
MIN_NUMEL = 1 << 20                                     # inputs below 2**20 elements (4 MiB fp32) take torch's layer_norm (below_min_numel)
RESID_MIN_NUMEL = 1 << 26                                # fused residual passes engage from 2**26 elements of z (C=128: >= 725 tokens): below it the pass saves ~0.02 ms of GPU time per site but costs ~35 us more host issue time than add + LayerNorm, and the small-N trunk is host-bound (measured: +0.2 s/item with fusion at 400 tokens, -0.4 s at 1000, -0.7 s at 1400) — word resid_below_min_numel
PRECOMPILE = {"widths": (64, 128, 256, 384, 512, 768), "affine": (3, 0)}   # fp32 kernels built at apply (a CUBIN disk cache under the kit's JIT root makes it a read on a warm box): every width Boltz-2 builds, affine and not (the DiT's AdaLN) — a call inside a CUDA-graph capture never compiles (capturing_uncompiled)
FUSE_RESID = False                                      # set by apply() from the row word BOLTZ_EXACTLN_RESID=on (lever exactln_resid): the pair residual add fused with the next block's LayerNorm (boltz_trunk_levers.RESID_FUSER)
EXPECTED = ("below_min_numel", "capturing_uncompiled", "capturing_unverified", "grad", "cpu", "copied_input", "dtype",
            "resid_probe_unused", "resid_probe_dtype", "resid_site_off", "resid_below_min_numel",
            "resid_u_layout", "resid_u_align", "resid_z_layout")   # "dtype" declares the dtype:<t> family: a float64 / float16 layer_norm call (never the model's own — e.g. another provider's fp64 reference computation) is torch's, exactly   # bitcmp_mismatch:/bitcmp_refused refuse the gate by construction; the resid probe words are bounded (<= 2 per site class per process); resid_ln_unused resid words that refuse: resid_form / resid_*_layout / resid_ln_unused (a fused LayerNorm nobody took)   # declared census words of a healthy run (cast_site_form: a pair-track cast site outside bf16 autocast — not expected in the modes' rows, refuses)
                                                          # resid_u_layout / resid_u_align / resid_z_layout: the residual fuser takes a contiguous 16-byte-aligned z and an update whose leading
                                                          # dims collapse onto z's rows (the exactln package's resid_layer_norm operand forms); an update it cannot take — the stock
                                                          # TriangleAttentionEndingNode's transposed output on the template Pairformer's C=64 stack, which `cueq` hands back to the stock
                                                          # module on templated passes (pairblock kept_out:64x4x32) — runs the stock statements (z + u, then the LayerNorm), counted by that name
_STATE: Dict[str, Any] = {"disabled": {}, "applied": [], "variant": None, "orig": None, "E": None, "served": {}, "fallback": {}, "fallback_classes": {}, "errors": {}, "calls": 0,
                          "t_apply": None, "facts": {}, "first_error": None, "class_ok": {}, "selftest": {}}
# First-call bit comparison (the exact tier's rule for replica cells): at the first EAGER encounter of each shape class — (kind, C, affine, dtypes,
# layout) — the stock statement and the replica run on the same real operands and are compared bit for bit (NaN-strict); the class is served
# for the rest of the process only if identical, else refused BY NAME (bitcmp_mismatch:<class>, the stock result returned).  A class first met
# inside a CUDA-graph capture cannot be checked there (no stock/replica double run inside a capture): capturing_unverified, stock statement.
SWITCH_RESID = "BOLTZ_EXACTLN_RESID"                    # lever exactln_resid (separate word): the pair residual add fused with the next block's LayerNorm; requires exactln


def variant(environ=None) -> Optional[str]:
    env = os.environ if environ is None else environ
    v = env.get(SWITCH, "").strip().lower()
    return v if v in VARIANTS else None


def requested(environ=None) -> bool:
    return variant(environ) is not None


def active() -> bool:
    """True once apply() has replaced torch.nn.functional.layer_norm (other adapters may ask)."""
    return bool(_STATE["applied"])


def _load_package(v: Optional[str] = None):
    """Import ``exactln``: opt_core's carried copy of the package through the kernels.ln face (``carried_module("exactln")``), registered as
    ``sys.modules["exactln"]`` so the kit modules that name the package see that one module (idempotent). ``v`` is the row word (`core`, the
    one live variant; kept in the signature for the callers)."""
    if PACKAGE not in sys.modules or not getattr(sys.modules[PACKAGE], "__name__", "").startswith("opt_core."):
        from opt_core.kernels import ln as _LN                              # standard library at import; the carried package imports torch / cuda.bindings at its own import
        sys.modules[PACKAGE] = _LN.carried_module(CORE_ROW)
    return sys.modules[PACKAGE]


def _provider_word(v: Optional[str] = None, E=None) -> str:
    """`core:opt_core-<version>` — the copy of the package that serves (LEVER line / report): the shared core's."""
    try:
        import opt_core
        return f"core:opt_core-{opt_core.__version__}"
    except Exception:  # noqa: BLE001
        return "core"


def _cnt(d: Dict[str, int], k: str, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def resid_variant(environ=None) -> Optional[str]:
    env = os.environ if environ is None else environ
    v = env.get(SWITCH_RESID, "").strip().lower()
    return v if v in ("on",) else None


def _bits_equal(a, b) -> bool:
    """NaN-strict bitwise equality of two tensors (same dtype/shape; layouts may differ)."""
    import torch
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    it = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}.get(a.dtype)
    if it is None:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.contiguous().view(it), b.contiguous().view(it)))


def _class_key(kind: str, pl: dict, extra: str = "") -> str:
    lay = "copy" if pl.get("copy") else ("strided" if pl.get("stride_outer") else "contig")
    return f"{kind}:C{pl['C']}:a{pl['affine']}:{pl['tin']}/{pl['tpar']}>{pl['tout']}:{lay}{extra}"


def _selftest(key: str, replica_out, stock_fn) -> bool:
    """Record the verdict of class `key`: replica_out vs stock_fn() bit for bit.  True = serve the class from now on."""
    ref = stock_fn()
    outs = replica_out if isinstance(replica_out, tuple) else (replica_out,)
    refs = ref if isinstance(ref, tuple) else (ref,)
    ok = len(outs) == len(refs) and all(_bits_equal(o, r) for o, r in zip(outs, refs))
    _STATE["class_ok"][key] = ok
    _STATE["selftest"][key] = "pass" if ok else "MISMATCH"
    if not ok:
        _cnt(_STATE["fallback"], f"bitcmp_mismatch:{key}")
        sys.stderr.write(f"[{TAG}] exactln BIT-COMPARE MISMATCH class={key}: the class takes torch's statement for this process\n")
    return ok


def face_word(cc, torch_version) -> Optional[str]:
    """Ask the core LayerNorm provider whether its `exactln` row serves compute capability ``cc`` ((major, minor)) on this torch: None when
    it does, else the word the lever steps aside under — the face's own refusal (``cc:<cc>_not_proven...``: opt_core.kernels.ln.admits, its
    EXACTLN_PROVEN_CC) or ``torch_version:<found>!=<pin>`` (the carried package's TORCH_PIN: the ATen kernel the replica reproduces)."""
    from opt_core.kernels import ln as LN
    try:
        LN.admits(CORE_ROW, tuple(cc), "fp32")
    except LN.Refusal as e:
        return _word(str(getattr(e, "kind", e)))
    pin = getattr(LN.carried_module(CORE_ROW), "TORCH_PIN", None)
    if pin and not str(torch_version).startswith(str(pin)):
        return _word(f"torch_version:{torch_version}!={pin}")
    return None


def _word(text: str) -> str:
    """A census word: no blanks (the LEVER / ACTIVE lines are blank-separated key=value tokens)."""
    return "_".join(str(text).split())[:80]


def dispositions() -> Dict[str, str]:
    """Levers this adapter names as disposed of rather than installed ({lever: word}: the face declined this card / torch) — the launcher's
    attach hook accepts an empty apply() only against these words (worker_launch._attach); the run's census carries them."""
    return dict(_STATE.get("disabled") or {})


def apply(spec: Optional[str] = None) -> List[str]:
    """Replace torch.nn.functional.layer_norm process-wide (idempotent). ``spec`` overrides the env switch. Refuses by name (raises) on a torch or
    card the replica is not proven on."""
    if _STATE["applied"]:
        return list(_STATE["applied"])
    raw = (os.environ.get(SWITCH, "") if spec is None else str(spec)).strip().lower()
    if raw in RETIRED_VARIANTS:                            # the kit copy's word: refused BY NAME before anything is replaced (worker_launch: REFUSED, exit 3) — never a silent re-binding or stock run
        raise RuntimeError(f"{SWITCH}={raw} refused by name — exactln:{raw}:kit_copy_retired (the forward/exactln kit copy left this tree at boltz2 0.3.16; "
                           f"the same package is the shared core's kernels.ln row: {SWITCH}=core)")
    v = variant() if spec is None else (raw if raw in VARIANTS else None)
    if v is None:
        return []
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError("no_cuda: exactln serves CUDA layer_norm calls; no CUDA device in this process")
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    cc = f"{p.major}{p.minor}"
    word = face_word((p.major, p.minor), torch.__version__)          # the core LayerNorm provider's answer for this card and torch: None = admitted, else its refusal word
    if word is not None:                                             # the face declines: the lever steps aside BY NAME (torch's own layer_norm keeps every call; the exact bytes are stock's either way)
        _STATE["disabled"] = {lever: word for lever in LEVERS}
        sys.stderr.write(f"[{TAG}] {NAME}: steps aside at install — {word} (the core LayerNorm provider's word for sm_{cc} / torch {torch.__version__}); torch.nn.functional.layer_norm keeps every call\n")
        return []
    try:
        import cuda.bindings  # noqa: F401 — the NVRTC / driver route (pinned in the kit's lock)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"no_cuda_bindings: {type(e).__name__}: {e}")
    E = _load_package(v)
    t0 = time.time()
    E.precompile(PRECOMPILE["widths"], dtypes=("float32",), affine=PRECOMPILE["affine"])
    _STATE["facts"] = {"precompile_s": round(time.time() - t0, 2), "cc": cc, "device": p.name,
                       "cc_proven": True, "torch_pin": getattr(E, "TORCH_PIN", None), "provider": _provider_word(v, E)}          # informational (LEVER line): on any card the first-call bit comparison per shape class is the gate — a class that does not reproduce torch's bits is refused by name at run time and the run stays stock-exact
    orig = F.layer_norm
    served, fallback, errors = _STATE["served"], _STATE["fallback"], _STATE["errors"]
    bf16, fp16, fp32 = torch.bfloat16, torch.float16, torch.float32

    def _layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        _STATE["calls"] += 1
        if not input.is_cuda:
            _cnt(fallback, "cpu"); return orig(input, normalized_shape, weight, bias, eps)
        if input.numel() < MIN_NUMEL:
            _cnt(fallback, "below_min_numel"); _cnt(_STATE["fallback_classes"], f"C{input.shape[-1]}:{str(input.dtype).replace('torch.', '')}"); return orig(input, normalized_shape, weight, bias, eps)
        if torch.is_grad_enabled() and (input.requires_grad or (weight is not None and weight.requires_grad)):
            _cnt(fallback, "grad"); return orig(input, normalized_shape, weight, bias, eps)
        capturing = torch.cuda.is_current_stream_capturing()      # a captured call replays the replica's launch (bitwise, so free); only a kernel not built yet steps aside (no module load inside a capture)
        x, w, b, widen = input, weight, bias, False
        if torch.is_autocast_enabled("cuda"):
            # autocast's fp32 policy for layer_norm: every lower-precision floating CUDA argument is widened to fp32, the op runs in fp32, returns fp32
            if torch.get_autocast_dtype("cuda") != bf16 or x.dtype == fp16 or (w is not None and w.dtype == fp16) or (b is not None and b.dtype == fp16):
                _cnt(fallback, "fp16_autocast"); return orig(input, normalized_shape, weight, bias, eps)
            if x.dtype == bf16:
                widen = True                                       # bf16 storage read and widened in the kernel's load (exact), fp32 arithmetic and output: autocast's cast, fused
                if w is not None and w.dtype == bf16:
                    w = w.float()
                if b is not None and b.dtype == bf16:
                    b = b.float()
            elif x.dtype != fp32:
                _cnt(fallback, f"dtype:{x.dtype}".replace("torch.", "")); return orig(input, normalized_shape, weight, bias, eps)
            elif (w is not None and w.dtype == bf16) or (b is not None and b.dtype == bf16):
                w = w.float() if w is not None else None; b = b.float() if b is not None else None
        if _SLOT and not widen and x.dtype == fp32:                       # a parked fp32 LayerNorm of this very tensor (the sequence attention's proj_z, autocast disabled)
            ln = _take_slot(w, b, eps, x, fp32)
            if ln is not None:
                return ln
        refused = None
        try:
            sig = E.call_signature(x, normalized_shape, w, b, widen, None)
            hot = _HOT.get(sig)
            if hot is None:
                pl = E.plan(x, normalized_shape, w, b, widen=widen)
                hot = (pl, _class_key("ln", pl))
                if len(_HOT) < 4096:
                    _HOT[sig] = hot
            pl, ck = hot
            v_ = _STATE["class_ok"].get(ck)
            if v_ is False:
                raise E.Unsupported(f"bitcmp_refused:{ck}")
            if capturing and not E.is_compiled(pl["tin"], pl["tpar"], pl["tout"], pl["C"], pl["affine"]):
                raise E.Unsupported("capturing_uncompiled")
            if capturing and v_ is None:
                raise E.Unsupported("capturing_unverified")
            y = E.layer_norm(x, normalized_shape, w, b, eps, widen=widen, _plan=pl)
            if v_ is None and not _selftest(ck, y, lambda: orig(input, normalized_shape, weight, bias, eps)):
                return orig(input, normalized_shape, weight, bias, eps)
        except E.Unsupported as u:
            refused = u.reason
        except Exception as e:  # noqa: BLE001 — a kernel/launch error: counted (the gate refuses), this call served by torch's layer_norm
            k = type(e).__name__
            _cnt(errors, k)
            if _STATE["first_error"] is None:
                _STATE["first_error"] = f"{k}: {e}"
            if errors[k] <= 2:
                sys.stderr.write(f"[{TAG}] exactln: {k}: {e} — this call served by torch.nn.functional.layer_norm\n")
            return orig(input, normalized_shape, weight, bias, eps)
        if refused is not None:
            _cnt(fallback, refused); return orig(input, normalized_shape, weight, bias, eps)
        key = f"C{pl['C']}:{str(x.dtype).replace('torch.', '')}{'>float32' if widen else ''}:{'copy' if pl['copy'] else ('strided' if pl['stride_outer'] else 'contig')}"
        _cnt(served, key)
        return y

    _layer_norm._exactln_orig = orig
    F.layer_norm = _layer_norm
    torch.nn.functional.layer_norm = _layer_norm                   # the same module object; both names for clarity
    _STATE.update(orig=orig, E=E, variant=v, applied=list(LEVERS), t_apply=time.time())
    global FUSE_RESID
    FUSE_RESID = resid_variant() is not None
    if FUSE_RESID:
        E.precompile_resid(widths=(128,), u_dtypes=("bfloat16",), ln_dtypes=("bfloat16", "float32"), affine=(3,))
        _STATE["applied"] = list(LEVERS) + ["exactln_resid"]
    _STATE["patched"] = ["torch.nn.functional.layer_norm"] + _fuse_pair_bias_ln(torch) + _register_resid_fuser()
    if os.environ.get("BOLTZ_LEVERS_VERBOSE", "1") == "1":
        print(f"[{TAG}] exactln APPLIED variant={v} precompiled={len(E.facts()['kernels'])} kernels in {_STATE['facts']['precompile_s']}s (disk_hits={E.facts()['disk_hits']}) on {p.name} (sm_{cc}) torch {torch.__version__}", flush=True)
    return list(_STATE["applied"])


def _register_resid_fuser() -> List[str]:
    """Set boltz_trunk_levers.RESID_FUSER = resid_fuser — now if that module is imported, else the moment the worker imports it (a one-shot
    sys.meta_path hook, the worker_launch pattern)."""
    if not FUSE_RESID:
        return []
    name = "boltz_trunk_levers"

    def _set(mod):
        if hasattr(mod, "RESID_FUSER"):
            mod.RESID_FUSER = resid_fuser
            _STATE["fuser_registered"] = True

    if name in sys.modules:
        _set(sys.modules[name])
        return [f"{name}.RESID_FUSER"]
    import importlib.abc
    import importlib.machinery

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != name:
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _set(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
    return [f"{name}.RESID_FUSER (on import)"]


def remove() -> None:
    if not _STATE["applied"]:
        return
    import torch.nn.functional as F
    F.layer_norm = _STATE["orig"]
    if _STATE.get("orig_apb_init") is not None:
        import boltz.model.layers.attentionv2 as AV2
        AV2.AttentionPairBias.__init__ = _STATE["orig_apb_init"]
    _STATE["applied"] = []


# ----------------------------------------------------------------------------------------------------------------- the residual add fused with the next block's LayerNorm
_SLOT: Dict[str, Any] = {}                              # the one pending fused LayerNorm output: taken by the next block's ln_to_bf16 call on the sum
_HOT: Dict[tuple, Any] = {}                              # call signature -> (plan, class key): plan() runs once per distinct call form


_PREF: Dict[Any, Any] = {}                              # per fusion site class (C, rows, transposed, ln_autocast): the parked dtype its consumer takes — learned: a kit adapter takes bf16 (or an fp32 park through one cast), a stock module's LayerNorm (the F.layer_norm hook) takes fp32 only; "off" after both went untaken


def _site_key(z, transpose_ln, ln_autocast, site=None):
    """The park-preference class of a fused call: the caller's site name (boltz_trunk_levers: tri_att_start / tri_att_end / transition_z / proj_z —
    each has ONE consumer kind, so a site whose consumer wants fp32 never flips the preference of one whose consumer takes bf16) × the call's
    shape class."""
    return (str(site or ""), int(z.shape[-1]), int(z.numel() // z.shape[-1]), bool(transpose_ln), bool(ln_autocast))


def _version(t):
    """The tensor's version counter, or None for an inference-mode tensor (no counter: torch.inference_mode(), Boltz's predict path)."""
    try:
        return t._version
    except Exception:  # noqa: BLE001
        return None


def resid_fuser(z, u, next_ln, transpose_ln: bool = False, ln_autocast: bool = True, site=None):
    """boltz_trunk_levers.RESID_FUSER (never raises into the model: any error is counted — the gate refuses — and the caller's statement adds).
    ``site``: the caller's residual site name (the park preference is learned per site; the served word carries it)."""
    try:
        return _resid_fuser_impl(z, u, next_ln, transpose_ln, ln_autocast, site)
    except Exception as e:  # noqa: BLE001
        k = type(e).__name__; _cnt(_STATE["errors"], k)
        if _STATE["first_error"] is None:
            _STATE["first_error"] = f"resid_fuser {k}: {e}"
            sys.stderr.write(f"[{TAG}] exactln resid_fuser: {k}: {e} — the caller's statement adds\n")
        _SLOT.clear()
        return None


def _resid_fuser_impl(z, u, next_ln, transpose_ln: bool = False, ln_autocast: bool = True, site=None):
    """boltz_trunk_levers.RESID_FUSER: ``z + u`` (torch's promote-add of the pair update, bit for bit) computed in one pass with the NEXT block's
    ``LayerNorm(sum[.transpose(-2,-3)]).to(bf16)`` (the replica's bits), which is parked in ``_SLOT`` for that block's ``ln_to_bf16`` call.
    None = not served (the caller's own statement adds; counted by word).  ``ln_autocast``: the consumer's LayerNorm runs under bf16 autocast
    (the parked value is its bf16 cast) or with autocast disabled (PairformerLayer's sequence attention proj_z: fp32 LayerNorm, fp32 Linear —
    the parked value is fp32)."""
    import torch
    E = _STATE["E"]
    if E is None or not _STATE["applied"] or not FUSE_RESID:
        return None
    fb = _STATE["fallback"]
    if _SLOT:                                                  # the previous fused LayerNorm was never taken: its consumer ran another path (correct — the sum was served — but the LayerNorm half was wasted)
        k_ = _SLOT.get("site"); was = _SLOT.get("dtype"); _SLOT.clear()
        nxt = (torch.float32 if was == torch.bfloat16 else "off")
        if _PREF.get(k_) != "off":
            _PREF[k_] = nxt
            _cnt(fb, "resid_probe_unused")                     # declared: at most two per site class per process (bf16 park untaken -> try fp32 -> off)
        else:
            _cnt(fb, "resid_ln_unused")                        # not expected: a site already learned 'off' parked again
    if z.numel() < RESID_MIN_NUMEL:
        _cnt(fb, "resid_below_min_numel"); _STATE["calls"] -= 1
        return None
    site = _site_key(z, transpose_ln, ln_autocast, site)
    if _PREF.get(site) == "off":
        _cnt(fb, "resid_site_off"); return None
    if not (z.is_cuda and z.dtype == torch.float32 and u.dtype in (torch.bfloat16, torch.float32) and tuple(u.shape) == tuple(z.shape)):
        _cnt(fb, "resid_form"); return None
    if z.numel() < MIN_NUMEL:
        _cnt(fb, "below_min_numel"); _cnt(_STATE["fallback_classes"], f"resid:C{z.shape[-1]}:{str(z.dtype).replace('torch.', '')}"); return None
    if torch.is_grad_enabled() and (z.requires_grad or u.requires_grad):
        _cnt(fb, "grad"); return None
    w, b, eps = getattr(next_ln, "weight", None), getattr(next_ln, "bias", None), float(getattr(next_ln, "eps", 1e-5))
    _STATE["calls"] += 1
    pref = _PREF.get(site)
    ln_dtype = pref if pref in (torch.bfloat16, torch.float32) else ((torch.bfloat16 if int(z.shape[-1]) == 128 else torch.float32) if ln_autocast else torch.float32)
    try:
        ck = (f"resid:C{z.shape[-1]}:a{(1 if w is not None else 0) | (2 if b is not None else 0)}:u={str(u.dtype).replace('torch.', '')}"
              f"{'T' if not u.is_contiguous() else 'N'}:ln{'T' if transpose_ln else 'N'}>{str(ln_dtype).replace('torch.', '')}")
        v_ = _STATE["class_ok"].get(ck)
        if v_ is False:
            raise E.Unsupported(f"bitcmp_refused:{ck}")
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and v_ is None:
            raise E.Unsupported("capturing_unverified")
        z_new, ln = E.resid_layer_norm(z if z.is_contiguous() else z.contiguous(), u, w, b, eps, ln_transposed=bool(transpose_ln), ln_dtype=ln_dtype,
                                       precompiled_only=capturing)
        if v_ is None:
            orig = _STATE["orig"]
            C_ = int(z.shape[-1])

            def _stock():
                zs = z + u
                xs = zs.transpose(-2, -3) if transpose_ln else zs
                return zs, orig(xs, (C_,), w, b, eps).to(ln_dtype)
            if not _selftest(ck, (z_new, ln), _stock):
                _STATE["calls"] -= 1
                return None
    except E.Unsupported as ex:
        _cnt(fb, ex.reason); _STATE["calls"] -= 1
        return None
    except Exception as e:  # noqa: BLE001 — counted (the gate refuses); the caller's statement adds
        k = type(e).__name__; _cnt(_STATE["errors"], k)
        if _STATE["first_error"] is None:
            _STATE["first_error"] = f"{k}: {e}"
        return None
    _SLOT.update(ptr=z_new.data_ptr(), shape=tuple(z_new.shape), transposed=bool(transpose_ln), w=(w.data_ptr() if w is not None else 0),
                 b=(b.data_ptr() if b is not None else 0), eps=eps, ln=ln, version=_version(z_new), dtype=ln_dtype, site=site)
    _cnt(_STATE["served"], f"resid:C{z.shape[-1]}:{'lnT' if transpose_ln else 'lnN'}:{'uT' if not u.is_contiguous() else 'uN'}>{str(ln_dtype).replace('torch.', '')}@{site[0] or '-'}")
    return z_new


def _take_slot(w, b, eps, x, want_dtype):
    """The parked LayerNorm output for this call — (weight, bias, eps) on tensor x, stored as want_dtype — if the pending fused pass produced exactly
    this call's value, else None (never raises)."""
    if not _SLOT:
        return None
    try:
        return _take_slot_impl(w, b, eps, x, want_dtype)
    except Exception:  # noqa: BLE001
        _SLOT.clear()
        return None


def _take_slot_impl(w, b, eps, x, want_dtype):
    import torch
    transposed = x.dim() >= 3 and x.stride(-3) < x.stride(-2)               # the ending node's x.transpose(-2, -3) view of the sum
    shape = tuple(x.shape) if not transposed else tuple(x.shape[:-3]) + (x.shape[-2], x.shape[-3], x.shape[-1])
    hit = (_SLOT["ptr"] == x.data_ptr() and _SLOT["shape"] == shape and _SLOT["transposed"] == transposed and x.dtype == torch.float32
           and _SLOT["w"] == (w.data_ptr() if w is not None else 0) and _SLOT["b"] == (b.data_ptr() if b is not None else 0)
           and _SLOT["eps"] == float(eps) and _version(x) == _SLOT["version"]
           and (x.is_contiguous() if not transposed else x.transpose(-2, -3).is_contiguous()))
    if not hit:
        return None
    parked = _SLOT["dtype"]; site = _SLOT.get("site")
    if parked == want_dtype:
        ln = _SLOT["ln"]; _SLOT.clear()
        _cnt(_STATE["served"], "resid_ln_taken")
        return ln
    if parked == torch.float32 and want_dtype == torch.bfloat16:      # a kit adapter met an fp32 park: one cast now (the stock cast's bits), bf16 parks for this site from now on
        ln = _SLOT["ln"].to(torch.bfloat16); _SLOT.clear()
        _PREF[site] = torch.bfloat16
        _cnt(_STATE["served"], "resid_ln_taken_cast")
        return ln
    _SLOT.clear()                                                      # a stock LayerNorm (fp32) met a bf16 park: not usable; fp32 parks for this site from now on
    _PREF[site] = torch.float32
    _cnt(_STATE["fallback"], "resid_probe_dtype")
    return None


# ----------------------------------------------------------------------------------------------------------------- the consumer's bf16 cast, fused
def ln_to_bf16(module, x):
    """``module(x).to(torch.bfloat16)`` for a LayerNorm ``module`` (nn.LayerNorm or boltz's primitives.LayerNorm) on an fp32/bf16 CUDA ``x`` under
    bf16 autocast — with the cast fused into the replica's store (round-to-nearest-even, the cast's own rounding: the same bf16 bits, one fp32
    [rows, C] write + read fewer) when the lever is on and the call is in the served set; the stock statement otherwise (which, lever on, still
    runs the replica through the layer_norm hook, unfused).  The kit's pair-track adapters call this where stock casts the LayerNorm output to bf16
    for their cells (boltz2_opt.pairblock, boltz2_opt.transition)."""
    import torch
    E = _STATE["E"]
    if E is None or not _STATE["applied"]:
        return module(x).to(torch.bfloat16)
    ns = getattr(module, "normalized_shape", None) or getattr(module, "c_in", None)
    w, b, eps = getattr(module, "weight", None), getattr(module, "bias", None), float(getattr(module, "eps", 1e-5))
    if _SLOT and x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16:
        ln = _take_slot(w, b, eps, x, torch.bfloat16)                # the parked bf16 LayerNorm of this very tensor
        if ln is not None:
            return ln
    _STATE["calls"] += 1
    if not (x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16 and x.dtype in (torch.float32, torch.bfloat16)
            and (w is None or w.dtype == torch.float32) and (b is None or b.dtype == torch.float32)):
        _cnt(_STATE["fallback"], "cast_site_form"); _STATE["calls"] -= 1
        return module(x).to(torch.bfloat16)
    if x.numel() < MIN_NUMEL or (torch.is_grad_enabled() and x.requires_grad):
        _STATE["calls"] -= 1                                          # the hook counts this call's word
        return module(x).to(torch.bfloat16)
    widen = x.dtype == torch.bfloat16
    try:
        sig = E.call_signature(x, ns, w, b, widen, torch.bfloat16)
        hot = _HOT.get(sig)
        if hot is None:
            pl = E.plan(x, ns, w, b, widen=widen)
            hot = (pl, _class_key("ln", dict(pl, tout="bfloat16"), ":cast"))
            if len(_HOT) < 4096:
                _HOT[sig] = hot
        pl, ck = hot
        v_ = _STATE["class_ok"].get(ck)
        if v_ is False:
            raise E.Unsupported(f"bitcmp_refused:{ck}")
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not E.is_compiled(pl["tin"], pl["tpar"], "bfloat16", pl["C"], pl["affine"]):
            raise E.Unsupported("capturing_uncompiled")
        if capturing and v_ is None:
            raise E.Unsupported("capturing_unverified")
        y = E.layer_norm(x, ns, w, b, eps, widen=widen, out_dtype=torch.bfloat16, _plan=pl)
        if v_ is None:
            orig = _STATE["orig"]
            nst = (int(ns),) if isinstance(ns, int) else tuple(int(v) for v in ns)
            if not _selftest(ck, y, lambda: orig(x, nst, w, b, eps).to(torch.bfloat16)):
                _STATE["calls"] -= 1
                return module(x).to(torch.bfloat16)
    except E.Unsupported as u:
        _cnt(_STATE["fallback"], u.reason); _STATE["calls"] -= 1
        return module(x).to(torch.bfloat16)
    except Exception as e:  # noqa: BLE001 — counted (the gate refuses); the stock statement serves the call
        k = type(e).__name__; _cnt(_STATE["errors"], k)
        if _STATE["first_error"] is None:
            _STATE["first_error"] = f"{k}: {e}"
        return module(x).to(torch.bfloat16)
    _cnt(_STATE["served"], f"C{pl['C']}:{str(x.dtype).replace('torch.', '')}>bfloat16:{'copy' if pl['copy'] else ('strided' if pl['stride_outer'] else 'contig')}")
    return y


def _fuse_pair_bias_ln(torch):
    """AttentionPairBias.proj_z = Sequential(LayerNorm(c_z), Linear(c_z, heads), Rearrange): under bf16 autocast the Linear casts the fp32
    LayerNorm output to bf16 — the replica stores bf16 directly (the same bits the cast produces), so the [N, N, c_z] fp32 intermediate is never
    written.  Installed by re-classing each instance's proj_z[0] at construction (parameters, state-dict keys and every other statement unchanged;
    composes with the trunk levers' AttentionPairBias.forward patch)."""
    import boltz.model.layers.attentionv2 as AV2
    nn = torch.nn

    class _PairBiasLayerNorm(nn.LayerNorm):
        def forward(self, x):                                         # noqa: D401 — nn.LayerNorm.forward with the following autocast cast fused
            if x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16:   # bf16 autocast only (REVIEW A14.1); PairformerLayer's sequence attention runs autocast-DISABLED: the plain statement below
                return ln_to_bf16(_Plain(self), x)
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    class _Plain:                                                     # module(x) for ln_to_bf16's stock statement: the plain nn.LayerNorm forward of `m` (no recursion into the re-classed forward)
        __slots__ = ("m",)

        def __init__(self, m):
            self.m = m

        def __call__(self, x):
            m = self.m
            return nn.functional.layer_norm(x, m.normalized_shape, m.weight, m.bias, m.eps)

        def __getattr__(self, k):
            return getattr(self.m, k)

    orig_init = AV2.AttentionPairBias.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        pz = getattr(self, "proj_z", None)
        if isinstance(pz, nn.Sequential) and len(pz) >= 2 and type(pz[0]) is nn.LayerNorm and isinstance(pz[1], nn.Linear):
            pz[0].__class__ = _PairBiasLayerNorm
            _STATE["reclassed"] = _STATE.get("reclassed", 0) + 1

    AV2.AttentionPairBias.__init__ = __init__
    _STATE["orig_apb_init"] = orig_init
    return ["AttentionPairBias.__init__(proj_z[0] -> bf16-storing LayerNorm)"]


# ----------------------------------------------------------------------------------------------------------------- evidence
def census() -> Dict[str, Any]:
    E = _STATE["E"]
    kstats = dict(E.facts()["stats"]) if E is not None else {}
    fb = dict(_STATE["fallback"])
    if kstats.get("copied_input"):
        fb_extra = {"copied_input": kstats["copied_input"]}           # served, after a contiguous copy (what ATen does too): a fact, not a fallback — reported beside
    else:
        fb_extra = {}
    return {"calls": _STATE["calls"], "served": dict(_STATE["served"]), "served_total": sum(_STATE["served"].values()), "fallback": fb,
            "fallback_classes": dict(_STATE["fallback_classes"]),      # the width x dtype classes of the below_min_numel calls (which LayerNorm sites take torch's kernel by size)
            "errors": dict(_STATE["errors"]), "facts": fb_extra, "kernels": (E.facts()["kernels"] if E is not None else []), "kernel_stats": kstats,
            "compiles": (E.facts()["compiles"] if E is not None else 0), "compile_s": (E.facts()["compile_s"] if E is not None else 0.0)}


def unexpected() -> List[str]:
    return sorted(k for k in _STATE["fallback"] if not (k in EXPECTED or k.split(":")[0] in EXPECTED))


def verdict() -> Dict[str, Any]:
    """The fail-closed gate: ok iff applied, no kernel error, no undeclared census word; idle iff applied and no call was served."""
    if not _STATE["applied"]:
        return {"ok": False, "idle": False, "reason": "not applied"}
    if _STATE["errors"]:
        return {"ok": False, "idle": False, "reason": f"errors:{dict(_STATE['errors'])} first={_STATE['first_error']}"}
    un = unexpected()
    if un:
        return {"ok": False, "idle": False, "reason": f"undeclared_fallback:{','.join(un)}"}
    idle = _STATE["calls"] > 0 and sum(_STATE["served"].values()) == 0
    return {"ok": True, "idle": bool(idle), "reason": "no call served (every call below MIN_NUMEL / declared stock paths)" if idle else None}


def line() -> str:
    st = "on" if _STATE["applied"] else "off"
    c = census()
    sc = _STATE.get("selftest") or {}
    served_ln = sum(n for k, n in _STATE["served"].items() if not k.startswith("resid"))
    return (f"LEVER name=exactln state={st} variant={_STATE['variant']} served={served_ln} fallback={sum(c['fallback'].values())} "
            f"errors={sum(c['errors'].values())} selftest={sum(1 for v in sc.values() if v == 'pass')}/{len(sc)}pass min_numel={MIN_NUMEL} torch_pin={_STATE['facts'].get('torch_pin')} "
            f"cc={_STATE['facts'].get('cc')} cc_proven={_STATE['facts'].get('cc_proven')} provider={_STATE['facts'].get('provider')}")


def resid_line() -> str:
    """The exactln_resid lever's own line: fused residual passes served / LayerNorms taken by the next block / parked-but-unused."""
    st = "on" if (FUSE_RESID and _STATE["applied"]) else "off"
    sv = _STATE["served"]
    fused = sum(n for k, n in sv.items() if k.startswith("resid:"))
    taken = sv.get("resid_ln_taken", 0)
    unused = _STATE["fallback"].get("resid_ln_unused", 0)
    reason = "" if st == "on" else (" reason=requires_exactln" if resid_variant() and not _STATE["applied"] else (" reason=word_absent" if not resid_variant() else ""))
    below = _STATE["fallback"].get("resid_below_min_numel", 0)
    return f"LEVER name=exactln_resid state={st} fused={fused} ln_taken={taken} ln_unused={unused} below_min_numel={below} min_numel={RESID_MIN_NUMEL} registered={bool(_STATE.get('fuser_registered'))}{reason}"


def lines() -> List[str]:
    return [line(), resid_line()]


def report() -> Dict[str, Any]:
    if not _STATE["applied"]:
        return {"applied": [], "line": None, "census": None, "gate": None, "variant": None, "disabled": dispositions()}
    return {"applied": list(_STATE["applied"]), "variant": _STATE["variant"], "line": line(), "census": census(), "gate": verdict(),
            "min_numel": MIN_NUMEL, "torch_pin": _STATE["facts"].get("torch_pin"), "expected": list(EXPECTED), "facts": dict(_STATE["facts"]),
            "patched": list(_STATE.get("patched") or []), "reclassed_pair_bias_ln": int(_STATE.get("reclassed", 0)), "fuse_resid": bool(FUSE_RESID), "fuser_registered": bool(_STATE.get("fuser_registered")),
            "selftest": dict(_STATE.get("selftest") or {}), "lines": lines(),
            "impl": "opt_core.kernels.ln:exactln/exactln_fwd.cu", "provider": _STATE["facts"].get("provider")}

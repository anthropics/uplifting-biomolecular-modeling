"""thin_launch.py — thin launchers for the ESM C eager path (pinned stack: TE 2.15 + flash-attn 2.7.4.post1 + Triton).

NO NEW CUDA. Every launch is the stock's own kernel with the stock's own arguments: each item RECORDS the exact
C++-binding / launcher call the stock Python wrapper makes (the same callable object, the same argument objects)
and REPLAYS it with only the activation tensors (and the host-known lengths) substituted. What is skipped is the
Python above the binding: the TE module forward (prepare_forward, dtype/layout checks, fp8 branch, general_gemm
marshalling), flash-attn's torch.library custom-op dispatch + autograd Function + padding logic, and Triton's
per-call JIT launcher (arg binding, specialization key, cache lookup).

Contract (esmc_opt.kits.fused._patch imports this module):
    rec = engage(model)          -> dict (what is engaged, per item; no knobs)
    disengage(model)             -> restores every module forward / symbol touched
Items: 'te' (LayerNormLinear / LayerNormMLP / Linear thin forwards), 'flash' (flash_attn_varlen_func /
_qkvpacked_func -> flash_attn_2_cuda.varlen_fwd directly), 'rotary' (flash_attn.ops.triton.rotary.apply_rotary ->
the cached CompiledKernel launcher directly). Every thin path FALLS BACK to the stock callable whenever a call is
outside the recorded class (grad enabled, other dtype/strides/kwargs, a stream mismatch), so it is never slower
than the stock on any call and bitwise by construction.
"""
import sys; from esmc_opt._oom import is_oom
import types
import threading

import torch

THIN_LAUNCH_VERSION = "0.6"
TESTED_ITEMS = ("te", "flash", "rotary")   # te: boxes 6a + 6b; flash/rotary: box 4 — bitwise on 11/11 units x both configs, S->K->S2
_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------------------------------
# recorder: wraps builtin functions of extension modules (pybind) / a launcher class __call__ and records every call
# ----------------------------------------------------------------------------------------------------------------------
def _meta(t):
    """snapshot of a tensor's equality at call time (TE clears intermediates after use: the live object may be empty later)"""
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t.numel(), t.is_contiguous())


class _Rec:
    __slots__ = ("fn", "args", "kwargs", "out", "name", "owner", "arg_meta", "kw_meta", "out_meta")

    def __init__(self, fn, args, kwargs, out, name, owner=None):
        self.fn, self.args, self.kwargs, self.out, self.name, self.owner = fn, args, kwargs, out, name, owner
        self.arg_meta = [(_meta(x) if isinstance(x, torch.Tensor) else None) for x in args]
        self.kw_meta = {k: (_meta(v) if isinstance(v, torch.Tensor) else None) for k, v in kwargs.items()}
        self.out_meta = [_meta(t) for t in _flat_tensors(out)]


class _Recorder:
    """with _Recorder(modules=[tex], classes=[(LauncherCls, '__call__')]) as r: ...; r.calls"""

    def __init__(self, modules=(), classes=()):
        self.modules, self.classes, self.calls, self._saved = list(modules), list(classes), [], []

    def _wrap_fn(self, name, fn):
        calls = self.calls

        def w(*a, **k):
            out = fn(*a, **k)
            calls.append(_Rec(fn, a, k, out, name))
            return out
        w.__name__ = getattr(fn, "__name__", name)
        return w

    def __enter__(self):
        wrapped = {}
        for m in self.modules:
            for name in dir(m):
                if name.startswith("_"):
                    continue
                fn = getattr(m, name)
                if type(fn).__name__ in ("builtin_function_or_method", "function"):
                    w = self._wrap_fn(name, fn)
                    self._saved.append((m, name, fn))
                    setattr(m, name, w)
                    wrapped[id(fn)] = w
        if wrapped:
            # direct references (`from transformer_engine_torch import X`) held by other loaded modules
            mine = {id(m) for m in self.modules}
            for mname, mod in list(sys.modules.items()):
                if not isinstance(mod, types.ModuleType) or id(mod) in mine or not mname.startswith(("transformer_engine", "flash_attn")):
                    continue
                for attr, val in list(vars(mod).items()):
                    if id(val) in wrapped:
                        self._saved.append((mod, attr, val))
                        setattr(mod, attr, wrapped[id(val)])
        for cls, meth in self.classes:
            orig = getattr(cls, meth)
            calls = self.calls

            def w(self_, *a, __orig=orig, __name=cls.__name__ + "." + meth, **k):
                out = __orig(self_, *a, **k)
                calls.append(_Rec(__orig, a, k, out, __name, owner=self_))
                return out
            self._saved.append((cls, meth, orig))
            setattr(cls, meth, w)
        return self

    def __exit__(self, *exc):
        for m, name, fn in reversed(self._saved):
            setattr(m, name, fn)
        self._saved = []
        return False


def _flat_tensors(o):
    """flatten an output structure into the list of tensors it contains (depth <= 2)"""
    if isinstance(o, torch.Tensor):
        return [o]
    res = []
    if isinstance(o, (tuple, list)):
        for x in o:
            if isinstance(x, torch.Tensor):
                res.append(x)
            elif isinstance(x, (tuple, list)):
                res.extend(y for y in x if isinstance(y, torch.Tensor))
    return res


# ----------------------------------------------------------------------------------------------------------------------
# (h) TE modules: LayerNormLinear / LayerNormMLP / Linear -> the recorded tex.* calls, replayed
# ----------------------------------------------------------------------------------------------------------------------
_TE_CLASSES = ("LayerNormLinear", "LayerNormMLP", "Linear", "LayerNorm", "RMSNorm")
_N_PROBE = 96


def _te_in_features(m):
    for attr in ("weight", "fc1_weight", "layer_norm_weight"):
        w = getattr(m, attr, None)
        if isinstance(w, torch.Tensor):
            return int(w.shape[-1]) if w.dim() == 2 else int(w.shape[0])
    return None


def _tex_refs(tex):
    """ONE scan (cached): every reference to a transformer_engine_torch builtin — the tex module attributes and the direct
    references (`from transformer_engine_torch import X`) held by loaded transformer_engine / flash_attn modules."""
    refs = _STATE.get("tex_refs")
    if refs is not None:
        return refs
    refs, fns = [], {}
    for name in dir(tex):
        if name.startswith("_"):
            continue
        fn = getattr(tex, name)
        if type(fn).__name__ in ("builtin_function_or_method", "function"):
            refs.append((tex, name, fn))
            fns[id(fn)] = name
    for mname, mod in list(sys.modules.items()):
        if not isinstance(mod, types.ModuleType) or mod is tex or not mname.startswith(("transformer_engine", "flash_attn")):
            continue
        for attr, val in list(vars(mod).items()):
            if id(val) in fns:
                refs.append((mod, attr, val))
    _STATE["tex_refs"] = refs
    return refs


class _RefRecorder:
    """with _RefRecorder(refs) as r: ... — swaps the cached references for recording wrappers (microseconds, no module scan)"""

    def __init__(self, refs):
        self.refs, self.calls = refs, []

    def __enter__(self):
        calls, wrapped = self.calls, {}
        for mod, attr, fn in self.refs:
            w = wrapped.get(id(fn))
            if w is None:
                def w(*a, __fn=fn, __name=getattr(fn, "__name__", attr), **k):
                    out = __fn(*a, **k)
                    calls.append(_Rec(__fn, a, k, out, __name))
                    return out
                wrapped[id(fn)] = w
            setattr(mod, attr, w)
        return self

    def __exit__(self, *exc):
        for mod, attr, fn in reversed(self.refs):
            setattr(mod, attr, fn)
        return False


def _ctx_key():
    """the autocast context of a call: (enabled, dtype) — the SDK wrapper runs autocast(bf16), the fused seam autocast off"""
    try:
        en = torch.is_autocast_enabled("cuda")
        return (True, str(torch.get_autocast_dtype("cuda"))) if en else (False, None)
    except TypeError:  # older signature
        en = torch.is_autocast_enabled()
        return (True, str(torch.get_autocast_gpu_dtype())) if en else (False, None)


class _TETemplate:
    """the recorded call list of one TE module forward on an input of n_rows rows, with every argument classified:
    LIT (kept by reference), IN (the input view), PROD (an earlier call's output tensor), OUTBUF (a buffer the Python side
    allocated and the call returned = re-allocated per call with the row count substituted). Built LAZILY from the module's
    first real call per autocast context (build) — or from a probe forward (capture, the debug form)."""

    def __init__(self, module, note):
        self.module, self.note, self.ok, self.reason, self.ambiguous = module, note, False, None, False
        self.calls = []          # list of (fn, name, template_args(list), subs(list of (pos, kind, payload)), kw_template(dict), kw_subs)
        self.final = None        # (call_idx, out_idx)
        self.out_features = None
        self.in_dtype = None
        self.device = None
        self.d = None
        self.n_rows = None
        self.ctx = None
        self.warnings = []

    # ---- the debug form: a probe forward under the recorder ----
    def capture(self, tex, ctx_autocast=True):
        m = self.module
        d = _te_in_features(m)
        if d is None:
            self.reason = "in_features unknown"
            return self
        p = next(m.parameters())
        g = torch.Generator(device="cpu").manual_seed(1234)
        x = (torch.randn(_N_PROBE, d, generator=g) * 0.5).to(device=p.device, dtype=torch.bfloat16)
        orig_forward = type(m).forward
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx_autocast):
            with _RefRecorder(_tex_refs(tex)) as rec:
                y = orig_forward(m, x)
            return self.build(rec.calls, x, y, _ctx_key())

    # ---- build from a recorded real call ----
    def build(self, rec_calls, x, y, ctx):
        m = self.module
        d = _te_in_features(m)
        if d is None:
            self.reason = "in_features unknown"
            return self
        self.d, self.ctx = d, ctx
        self.device, self.in_dtype = x.device, x.dtype
        if not rec_calls:
            self.reason = "no tex call recorded"
            return self
        if not isinstance(y, torch.Tensor):
            self.reason = f"module returned {type(y).__name__}, not a tensor"
            return self
        if x.shape[-1] != d or not x.is_contiguous():
            self.reason = f"input outside the class (last dim {x.shape[-1]} vs {d}, contiguous {x.is_contiguous()})"
            return self
        n_rows = x.numel() // d
        self.n_rows = n_rows
        # the row count must not collide with any fixed dimension of the module (else the substitution is ambiguous): retry on another call
        fixed = {1}
        for p in list(m.parameters()) + list(m.buffers()):
            fixed.update(int(s) for s in p.shape)
        if n_rows in fixed:
            self.reason, self.ambiguous = f"row count {n_rows} equals a fixed dimension of the module", True
            return self
        persist = {t.data_ptr() for t in list(m.parameters()) + list(m.buffers())}
        produced = {}   # data_ptr -> (call_idx, out_idx, meta)
        xptr, xnumel = x.data_ptr(), x.numel()
        n_probe = n_rows
        self.lit_tensors = []
        for ci, c in enumerate(rec_calls):
            out_ptrs = {mm[0]: oi for oi, mm in enumerate(c.out_meta) if mm[5] > 0}
            targs, subs = [], []
            for pos, a in enumerate(c.args):
                kind, payload = self._classify(a, c.arg_meta[pos], xptr, xnumel, produced, out_ptrs, persist, n_probe, ci, pos)
                if kind in ("BAD", "AMBIG"):
                    self.reason, self.ambiguous = payload, kind == "AMBIG"
                    return self
                if kind == "LIT":
                    targs.append(payload)
                else:
                    targs.append(None)
                    subs.append((pos, kind, payload))
            tkw, ksubs = {}, []
            for key, a in c.kwargs.items():
                kind, payload = self._classify(a, c.kw_meta[key], xptr, xnumel, produced, out_ptrs, persist, n_probe, ci, key)
                if kind in ("BAD", "AMBIG"):
                    self.reason, self.ambiguous = payload, kind == "AMBIG"
                    return self
                if kind == "LIT":
                    tkw[key] = payload
                else:
                    tkw[key] = None
                    ksubs.append((key, kind, payload))
            for oi, mm in enumerate(c.out_meta):
                if mm[5] > 0:
                    produced[mm[0]] = (ci, oi, mm)      # the LATEST producer owns the pointer (TE clears intermediates; the allocator re-uses their blocks)
            self.calls.append((c.fn, c.name, targs, subs, tkw, ksubs))
        # every tensor kept by reference must still be what it was at call time (TE clears its intermediates)
        for t, mm in self.lit_tensors:
            if _meta(t) != mm:
                self.reason = f"a tensor kept by reference changed after the call (was {mm[1]} {mm[3]}, now {tuple(t.shape)} {t.dtype}): a per-call temporary"
                return self
        if y.data_ptr() not in produced:
            self.reason = "module output aliases no recorded output"
            return self
        ci, oi, mm = produced[y.data_ptr()]
        if mm[5] != y.numel() or mm[3] != y.dtype or mm[1][-1] != y.shape[-1] or len(mm[1]) != 2 or mm[1][0] != n_rows:
            self.reason = f"module output does not match its producer: call {ci} out {oi} {mm[1]} {mm[3]} vs {tuple(y.shape)} {y.dtype}"
            return self
        if ci != len(self.calls) - 1:
            self.warnings.append(f"module output produced by call {ci} of {len(self.calls)}")
        self.final = (ci, oi)
        self.out_features = int(y.shape[-1])
        if y.numel() != n_rows * self.out_features:
            self.reason = f"unexpected output shape {tuple(y.shape)}"
            return self
        self.ok = True
        return self

    def _classify(self, a, meta, xptr, xnumel, produced, out_ptrs, persist, n_probe, ci, pos):
        if isinstance(a, torch.Tensor):
            ptr, shape, stride, dtype, device, numel, contig = meta
            if numel == 0:
                self.warnings.append(f"call {ci} arg {pos}: empty tensor kept by reference")
                self.lit_tensors.append((a, meta))
                return "LIT", a
            if ptr == xptr and numel == xnumel:
                if not contig:
                    return "BAD", f"call {ci} arg {pos}: non-contiguous input view"
                return "IN", tuple(("N" if s == n_probe else s) for s in shape)
            if ptr in produced:
                pci, poi, pm = produced[ptr]
                if pm[5] != numel:
                    return "BAD", f"call {ci} arg {pos}: partial view of a produced tensor"
                same = pm[1] == shape
                return "PROD", (pci, poi, None if same else tuple(("N" if s == n_probe else s) for s in shape))
            if ptr in out_ptrs:
                if not contig:
                    return "BAD", f"call {ci} arg {pos}: non-contiguous output buffer"
                return "OUTBUF", (tuple(("N" if s == n_probe else s) for s in shape), dtype, device)
            if ptr in persist:
                return "LIT", a
            if numel >= (1 << 20) and dtype == torch.uint8:
                self.lit_tensors.append((a, meta))
                return "LIT", a        # the cuBLAS workspace (TE's global get_workspace())
            if any(s == n_probe for s in shape):
                return "AMBIG", f"call {ci} arg {pos}: unknown tensor with a dimension equal to the row count (shape {shape}, dtype {dtype})"
            self.warnings.append(f"call {ci} arg {pos}: unknown constant tensor shape {shape} dtype {dtype} kept by reference")
            self.lit_tensors.append((a, meta))
            return "LIT", a
        if isinstance(a, bool) or a is None or isinstance(a, (float, str)):
            return "LIT", a
        if isinstance(a, int):
            if a == n_probe:
                return "AMBIG", f"call {ci} arg {pos}: int argument equals the row count ({n_probe})"
            return "LIT", a
        if isinstance(a, (list, tuple)):
            if any(isinstance(e, torch.Tensor) for e in a):
                return "BAD", f"call {ci} arg {pos}: tensor inside a {type(a).__name__} argument"
            return "LIT", a
        return "LIT", a     # enum / quantizer / other opaque object kept by reference

    # ---- replay ----
    def make_replay(self):
        """the replay of this template: replay(inp) for an input already checked to be in the class (dtype/device/last dim/contiguous)"""
        calls, final, d, out_features = self.calls, self.final, self.d, self.out_features
        empty = torch.empty

        def replay(inp):
            N = inp.numel() // d
            outs = []
            for fn, name, targs, subs, tkw, ksubs in calls:
                args = list(targs)
                for pos, kind, payload in subs:
                    if kind == "IN":
                        args[pos] = inp.view(*[N if s == "N" else s for s in payload])
                    elif kind == "PROD":
                        pci, poi, shp = payload
                        t = outs[pci][poi]
                        args[pos] = t if shp is None else t.view(*[N if s == "N" else s for s in shp])
                    else:  # OUTBUF
                        shp, dt, dv = payload
                        args[pos] = empty([N if s == "N" else s for s in shp], dtype=dt, device=dv)
                if ksubs:
                    kw = dict(tkw)
                    for key, kind, payload in ksubs:
                        if kind == "IN":
                            kw[key] = inp.view(*[N if s == "N" else s for s in payload])
                        elif kind == "PROD":
                            pci, poi, shp = payload
                            t = outs[pci][poi]
                            kw[key] = t if shp is None else t.view(*[N if s == "N" else s for s in shp])
                        else:
                            shp, dt, dv = payload
                            kw[key] = empty([N if s == "N" else s for s in shp], dtype=dt, device=dv)
                    out = fn(*args, **kw)
                else:
                    out = fn(*args, **tkw) if tkw else fn(*args)
                outs.append(_flat_tensors(out))
            return outs[final[0]][final[1]].view(*inp.shape[:-1], out_features)
        return replay

    def summary(self):
        rows = []
        for fn, name, targs, subs, tkw, ksubs in self.calls:
            rows.append({"fn": name, "n_args": len(targs), "subs": [(p, k, (str(v)[:60] if k != "OUTBUF" else str(v[0]))) for p, k, v in subs],
                         "kw_subs": [(p, k) for p, k, v in ksubs], "kw": sorted(tkw)})
        return {"ok": self.ok, "reason": self.reason, "module": type(self.module).__name__, "d": self.d, "out_features": self.out_features, "n_rows": self.n_rows,
                "ctx": self.ctx, "final": self.final, "calls": rows, "warnings": self.warnings}


def _find_te_modules(model):
    res = []
    for name, m in model.named_modules():
        cm = type(m).__module__ or ""
        if cm.startswith("transformer_engine") and type(m).__name__ in _TE_CLASSES:
            res.append((name, m))
    return res


def _make_lazy_forward(m, name, orig_forward, tex):
    """the module's forward after engage: the FIRST real call per autocast context runs the stock forward under the recorder and
    builds the template from it (returns the stock's output); later calls in that context replay; a context whose template could
    not be built (or a call outside the class / grad enabled / extra args) runs the stock forward — never a refusal."""
    templates = {}     # ctx -> _TETemplate (ok) | None (this context stays stock)
    st = {"thin": 0, "fallback": 0, "captured": {}, "ambiguous": 0, "capture_failed": {}}
    m._thin_launch_templates, m._thin_launch_stats = templates, st
    refs = _tex_refs(tex)

    def lazy_forward(inp, *a, **k):
        if a or k or torch.is_grad_enabled() or not isinstance(inp, torch.Tensor):
            st["fallback"] += 1
            return orig_forward(m, inp, *a, **k)
        ctx = _ctx_key()
        tpl = templates.get(ctx, False)
        if tpl is False:
            with _RefRecorder(refs) as rec:
                y = orig_forward(m, inp)
            t = _TETemplate(m, name).build(rec.calls, inp, y, ctx)
            if t.ok:
                t.replay = t.make_replay()
                templates[ctx] = t
                st["captured"][str(ctx)] = st["captured"].get(str(ctx), 0) + 1
            elif t.ambiguous:
                st["ambiguous"] += 1                      # try again on the next call (another row count)
            else:
                templates[ctx] = None
                st["capture_failed"][str(ctx)] = t.reason
            return y
        if tpl is None or inp.dtype is not tpl.in_dtype or inp.device != tpl.device or inp.shape[-1] != tpl.d or not inp.is_contiguous():
            st["fallback"] += 1
            return orig_forward(m, inp)
        st["thin"] += 1
        return tpl.replay(inp)
    return lazy_forward


def engage_te(model):
    import transformer_engine_torch as tex
    rec = {"item": "te", "engaged": 0, "skipped": [], "by_class": {}, "form": "lazy: the template of each (module, autocast context) is built from its first real call (that call runs the stock)"}
    n_refs = len(_tex_refs(tex))
    for name, m in _find_te_modules(model):
        if hasattr(m, "_thin_launch_orig_forward"):
            continue
        cls = type(m).__name__
        if _te_in_features(m) is None:
            rec["skipped"].append((name, cls, "in_features unknown"))
            continue
        orig_forward = type(m).forward
        m._thin_launch_orig_forward = orig_forward
        m.forward = _make_lazy_forward(m, name, orig_forward, tex)
        rec["engaged"] += 1
        rec["by_class"][cls] = rec["by_class"].get(cls, 0) + 1
    rec["tex_refs"] = n_refs
    return rec


def disengage_te(model):
    n = 0
    for name, m in _find_te_modules(model):
        if hasattr(m, "_thin_launch_orig_forward"):
            if "forward" in m.__dict__:
                del m.__dict__["forward"]
            del m._thin_launch_orig_forward
            for attr in ("_thin_launch_template", "_thin_launch_templates", "_thin_launch_stats"):
                if hasattr(m, attr):
                    delattr(m, attr)
            n += 1
    return n


def te_counters(model):
    tot = {"thin": 0, "fallback": 0, "ambiguous": 0, "captured": {}, "capture_failed": {}, "by_class_ctx": {}}
    for name, m in _find_te_modules(model):
        st = getattr(m, "_thin_launch_stats", None)
        if st is None:
            continue
        tot["thin"] += st["thin"]
        tot["fallback"] += st["fallback"]
        tot["ambiguous"] += st["ambiguous"]
        cls = type(m).__name__
        for ctx, n in st["captured"].items():
            tot["captured"][ctx] = tot["captured"].get(ctx, 0) + n
            key = f"{cls}@{ctx}"
            tot["by_class_ctx"][key] = tot["by_class_ctx"].get(key, 0) + n
        for ctx, why in st["capture_failed"].items():
            tot["capture_failed"][f"{cls}@{ctx}"] = why
    return tot


# ----------------------------------------------------------------------------------------------------------------------
# (f) flash-attn varlen: flash_attn_varlen_func / flash_attn_varlen_qkvpacked_func -> flash_attn_2_cuda.varlen_fwd directly
# ----------------------------------------------------------------------------------------------------------------------
class _FlashTemplate:
    __slots__ = ("args", "kwargs", "iq", "ik", "iv", "icq", "ick", "imq", "imk", "fn", "ok", "reason")


class ThinFlashVarlen:
    """same signature as flash_attn.flash_attn_interface.flash_attn_varlen_func (2.7.4.post1); lazily records the raw
    varlen_fwd call per (dtype/shape-class/kwargs) class on the first call and replays it after (max_seqlen and the
    tensors substituted). Anything outside the class -> the stock function."""

    def __init__(self, orig, raw_module, packed=False):
        self.orig, self.raw, self.packed = orig, raw_module, packed
        self.templates = {}
        self.counters = {"thin": 0, "fallback": 0, "captured": 0, "unsupported": 0}

    def _capture(self, key, q, k, v, cu_q, cu_k, mq, mk, call_args, call_kwargs, stock=None):
        with _Recorder(modules=[self.raw]) as rec:
            out = stock() if stock is not None else self.orig(*call_args, **call_kwargs)
        t = _FlashTemplate()
        t.ok, t.reason = False, None
        raw = [c for c in rec.calls if c.name == "varlen_fwd"]
        if len(rec.calls) != 1 or len(raw) != 1:
            t.reason = f"expected exactly one raw varlen_fwd call, saw {[c.name for c in rec.calls]}"
            self.templates[key] = t
            self.counters["unsupported"] += 1
            return out
        c = raw[0]
        if c.kwargs:
            t.reason = "raw call uses kwargs"
            self.templates[key] = t
            self.counters["unsupported"] += 1
            return out

        def pos_of(x):
            hits = [i for i, a in enumerate(c.args) if isinstance(a, torch.Tensor) and a.data_ptr() == x.data_ptr() and a.numel() == x.numel() and a.shape == x.shape and a.stride() == x.stride()]
            return hits[0] if len(hits) == 1 else None
        t.iq, t.ik, t.iv = pos_of(q), pos_of(k), pos_of(v)
        hq = [i for i, a in enumerate(c.args) if isinstance(a, torch.Tensor) and a.data_ptr() == cu_q.data_ptr() and a.shape == cu_q.shape and a.stride() == cu_q.stride()]
        hk = [i for i, a in enumerate(c.args) if isinstance(a, torch.Tensor) and a.data_ptr() == cu_k.data_ptr() and a.shape == cu_k.shape and a.stride() == cu_k.stride()]
        if hq == hk and len(hq) == 2:
            t.icq, t.ick = hq[0], hq[1]          # the same tensor at two positions (the qkvpacked form)
        else:
            t.icq = hq[0] if len(hq) == 1 else None
            t.ick = hk[0] if len(hk) == 1 else None
        ints_q = [i for i, a in enumerate(c.args) if type(a) is int and a == mq]
        ints_k = [i for i, a in enumerate(c.args) if type(a) is int and a == mk]
        outs = _flat_tensors(c.out)
        res_ok = isinstance(out, torch.Tensor) and outs and outs[0].data_ptr() == out.data_ptr() and outs[0].numel() == out.numel()
        if None in (t.iq, t.ik, t.iv, t.icq, t.ick) or not res_ok:
            t.reason = f"could not locate tensors in the raw call (q,k,v,cu_q,cu_k)={(t.iq, t.ik, t.iv, t.icq, t.ick)} res_ok={res_ok}"
            self.templates[key] = t
            self.counters["unsupported"] += 1
            return out
        if mq == mk:
            if len(ints_q) != 2:
                t.reason = f"max_seqlen positions ambiguous: {ints_q}"
                self.templates[key] = t
                self.counters["unsupported"] += 1
                return out
            t.imq, t.imk = [ints_q[0]], [ints_q[1]]
        else:
            if len(ints_q) != 1 or len(ints_k) != 1:
                t.reason = f"max_seqlen positions ambiguous: {ints_q} {ints_k}"
                self.templates[key] = t
                self.counters["unsupported"] += 1
                return out
            t.imq, t.imk = ints_q, ints_k
        other_tensor_pos = [i for i, a in enumerate(c.args) if isinstance(a, torch.Tensor) and i not in (t.iq, t.ik, t.iv, t.icq, t.ick)]
        if other_tensor_pos:
            t.reason = f"unexpected extra tensor args at {other_tensor_pos}"
            self.templates[key] = t
            self.counters["unsupported"] += 1
            return out
        t.args = list(c.args)
        for i in (t.iq, t.ik, t.iv, t.icq, t.ick):
            t.args[i] = None
        t.fn, t.ok = c.fn, True
        self.templates[key] = t
        self.counters["captured"] += 1
        return out

    def __call__(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                 causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False,
                 return_attn_probs=False, block_table=None, _stock=None, **extra):
        call_args = (q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
                     window_size, softcap, alibi_slopes, deterministic, return_attn_probs, block_table)
        if (extra or return_attn_probs or dropout_p != 0.0 or alibi_slopes is not None or block_table is not None
                or torch.is_grad_enabled() or type(max_seqlen_q) is not int or type(max_seqlen_k) is not int
                or q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1):
            self.counters["fallback"] += 1
            return _stock() if _stock is not None else self.orig(*call_args, **extra)
        key = (q.dtype, k.dtype, v.dtype, q.shape[1:], k.shape[1:], v.shape[1:], q.stride()[1:], k.stride()[1:], v.stride()[1:],
               q.dim(), cu_seqlens_q.dtype, cu_seqlens_k.dtype, softmax_scale, causal, tuple(window_size), softcap, deterministic,
               max_seqlen_q == max_seqlen_k, q.device.index, _stock is not None)
        t = self.templates.get(key)
        if t is None:
            return self._capture(key, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, call_args, {}, stock=_stock)
        if not t.ok:
            self.counters["fallback"] += 1
            return _stock() if _stock is not None else self.orig(*call_args)
        self.counters["thin"] += 1
        args = list(t.args)
        args[t.iq], args[t.ik], args[t.iv], args[t.icq], args[t.ick] = q, k, v, cu_seqlens_q, cu_seqlens_k
        for i in t.imq:
            args[i] = max_seqlen_q
        for i in t.imk:
            args[i] = max_seqlen_k
        return t.fn(*args)[0]

    def packed_call(self, qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                    softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False, **extra):
        """flash_attn_varlen_qkvpacked_func form: qkv [total, 3, H, D]"""
        stock = lambda: self.orig_packed(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, deterministic, return_attn_probs, **extra)  # noqa: E731
        if extra or return_attn_probs or dropout_p != 0.0 or alibi_slopes is not None or torch.is_grad_enabled() or type(max_seqlen) is not int or qkv.dim() != 4:
            self.counters["fallback"] += 1
            return stock()
        return self(qkv[:, 0], qkv[:, 1], qkv[:, 2], cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, deterministic, return_attn_probs, _stock=stock)

    def summary(self):
        return {"counters": dict(self.counters), "classes": {str(k)[:200]: ({"ok": t.ok, "reason": t.reason, "n_raw_args": len(t.args) if t.ok else None,
                                                                          "pos": (t.iq, t.ik, t.iv, t.icq, t.ick, t.imq, t.imk) if t.ok else None}) for k, t in self.templates.items()}}


# ----------------------------------------------------------------------------------------------------------------------
# (g) Triton rotary: flash_attn.ops.triton.rotary.apply_rotary -> the compiled kernel's launcher directly
# ----------------------------------------------------------------------------------------------------------------------
class _RotTemplate:
    __slots__ = ("launcher", "args", "io", "ix", "ic", "is_", "icu", "ok", "reason", "stream_pos")


class ThinRotary:
    """same signature as flash_attn.ops.triton.rotary.apply_rotary; per (shape, strides, dtype, alignment, ints, flags)
    class: the first call runs the stock (Triton JIT launcher) under a recorder that captures the launcher instance +
    its full argument tuple; later calls launch through the launcher directly with the tensors substituted."""

    def __init__(self, orig, launcher_cls):
        self.orig, self.launcher_cls = orig, launcher_cls
        self.templates = {}
        self.counters = {"thin": 0, "fallback": 0, "captured": 0, "unsupported": 0}

    def _key(self, x, cos, sin, seqlen_offsets, cu_seqlens, max_seqlen, interleaved, inplace, conjugate):
        cu = None if cu_seqlens is None else (tuple(cu_seqlens.shape), cu_seqlens.dtype, cu_seqlens.data_ptr() & 15)
        return (tuple(x.shape), tuple(x.stride()), x.dtype, x.data_ptr() & 15, tuple(cos.shape), tuple(cos.stride()), cos.dtype, cos.data_ptr() & 15,
                tuple(sin.shape), tuple(sin.stride()), sin.dtype, sin.data_ptr() & 15, seqlen_offsets, cu, max_seqlen, bool(interleaved), bool(inplace), bool(conjugate), x.device.index)

    def _capture(self, key, x, cos, sin, seqlen_offsets, cu_seqlens, max_seqlen, interleaved, inplace, conjugate):
        stream = torch.cuda.current_stream(x.device).cuda_stream
        with _Recorder(classes=[(self.launcher_cls, "__call__")]) as rec:
            out = self.orig(x, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=interleaved, inplace=inplace, conjugate=conjugate)
        t = _RotTemplate()
        t.ok, t.reason = False, None
        if len(rec.calls) != 1:
            t.reason = f"expected exactly one launcher call, saw {len(rec.calls)}"
            self.templates[key] = t
            self.counters["unsupported"] += 1
            return out
        c = rec.calls[0]
        if c.kwargs:
            t.reason = "launcher called with kwargs"
            self.templates[key] = t
            self.counters["unsupported"] += 1
            return out
        args = list(c.args)

        def hits_of(z):
            return [i for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.data_ptr() == z.data_ptr() and a.shape == z.shape and a.stride() == z.stride()]

        def pos_of(z):
            h = hits_of(z)
            return h[0] if len(h) == 1 else None
        t.ic, t.is_ = pos_of(cos), pos_of(sin)
        t.icu = pos_of(cu_seqlens) if cu_seqlens is not None else -1
        if inplace:
            hx = hits_of(x)          # OUT and X are the same tensor -> two positions
            t.io, t.ix = (hx[0], hx[1]) if len(hx) == 2 else (None, None)
        else:
            t.io, t.ix = pos_of(out), pos_of(x)
        ten_pos = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
        if None in (t.io, t.ix, t.ic, t.is_, t.icu) or set(ten_pos) - {t.io, t.ix, t.ic, t.is_, t.icu}:
            t.reason = f"could not map launcher tensor args: out/x/cos/sin/cu={(t.io, t.ix, t.ic, t.is_, t.icu)} tensor positions={ten_pos}"
            self.templates[key] = t
            self.counters["unsupported"] += 1
            return out
        if not (len(args) > 4 and all(type(a) is int for a in args[:4]) and args[3] == stream):
            t.reason = f"launcher arg layout not (gridX, gridY, gridZ, stream, ...): head={[type(a).__name__ for a in args[:5]]} stream={stream}"
            self.templates[key] = t
            self.counters["unsupported"] += 1
            return out
        if not inplace and (out.shape != x.shape or cos.shape[-1] < x.shape[-1]):
            t.reason = "partial rotary (rotary_dim < headdim) or unexpected output shape"
            self.templates[key] = t
            self.counters["unsupported"] += 1
            return out
        for i in {t.io, t.ix, t.ic, t.is_} | ({t.icu} if t.icu >= 0 else set()):
            args[i] = None
        t.launcher, t.args, t.stream_pos, t.ok = c.owner, args, 3, True
        self.templates[key] = t
        self.counters["captured"] += 1
        return out

    def __call__(self, x, cos, sin, seqlen_offsets=0, cu_seqlens=None, max_seqlen=None, interleaved=False, inplace=False, conjugate=False):
        if torch.is_grad_enabled() or type(seqlen_offsets) is not int:
            self.counters["fallback"] += 1
            return self.orig(x, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=interleaved, inplace=inplace, conjugate=conjugate)
        key = self._key(x, cos, sin, seqlen_offsets, cu_seqlens, max_seqlen, interleaved, inplace, conjugate)
        t = self.templates.get(key)
        if t is None:
            return self._capture(key, x, cos, sin, seqlen_offsets, cu_seqlens, max_seqlen, interleaved, inplace, conjugate)
        if not t.ok:
            self.counters["fallback"] += 1
            return self.orig(x, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=interleaved, inplace=inplace, conjugate=conjugate)
        self.counters["thin"] += 1
        out = x if inplace else torch.empty_like(x)
        args = list(t.args)
        args[t.io], args[t.ix], args[t.ic], args[t.is_] = out, x, cos, sin
        if t.icu >= 0:
            args[t.icu] = cu_seqlens
        args[3] = torch.cuda.current_stream(x.device).cuda_stream
        t.launcher(*args)
        return out

    def summary(self):
        return {"counters": dict(self.counters), "n_classes": len(self.templates),
                "classes": [{"key": str(k)[:160], "ok": t.ok, "reason": t.reason, "n_args": len(t.args) if t.ok else None,
                             "grid": t.args[:3] if t.ok else None} for k, t in list(self.templates.items())[:12]]}


# ----------------------------------------------------------------------------------------------------------------------
# symbol replacement across loaded modules (the ESM SDK imports these names into its own namespaces)
# ----------------------------------------------------------------------------------------------------------------------
def _replace_symbol(orig, new, state, label):
    hits = []
    for mname, mod in list(sys.modules.items()):
        if not isinstance(mod, types.ModuleType):
            continue
        try:
            d = mod.__dict__
        except Exception:
            continue
        for attr, val in list(d.items()):
            if val is orig:
                setattr(mod, attr, new)
                state.append((mod, attr, orig))
                hits.append(f"{mname}.{attr}")
    return hits


_STATE = {"symbols": [], "flash": None, "rotary": None}


def engage_flash():
    import flash_attn.flash_attn_interface as FI
    raw = getattr(FI, "flash_attn_gpu", None)
    if raw is None:
        import flash_attn_2_cuda as raw
    orig = FI.flash_attn_varlen_func
    thin = ThinFlashVarlen(orig, raw)
    thin.orig_packed = getattr(FI, "flash_attn_varlen_qkvpacked_func", None)
    hits = _replace_symbol(orig, thin, _STATE["symbols"], "flash_attn_varlen_func")
    hits_p = []
    if thin.orig_packed is not None:
        hits_p = _replace_symbol(thin.orig_packed, thin.packed_call, _STATE["symbols"], "flash_attn_varlen_qkvpacked_func")
    _STATE["flash"] = thin
    return {"item": "flash", "raw_module": getattr(raw, "__name__", str(raw)), "raw_fn": "varlen_fwd", "replaced": hits, "replaced_packed": hits_p, "thin": thin}


def engage_rotary():
    import flash_attn.ops.triton.rotary as R
    from triton.runtime.driver import driver
    launcher_cls = driver.active.launcher_cls
    orig = R.apply_rotary
    thin = ThinRotary(orig, launcher_cls)
    hits = _replace_symbol(orig, thin, _STATE["symbols"], "apply_rotary")
    _STATE["rotary"] = thin
    return {"item": "rotary", "launcher_cls": f"{launcher_cls.__module__}.{launcher_cls.__name__}", "replaced": hits, "thin": thin}


def engage(model, items=TESTED_ITEMS):
    """engage the thin launchers on `model` (the SDK's client.model). Returns the engagement record. No knobs.
    Default = every tested item (te, flash, rotary); pass a subset by name to engage fewer."""
    rec = {"thin_launch_version": THIN_LAUNCH_VERSION, "items": {}}
    with _LOCK:
        for it in items:
            try:
                if it == "te":
                    r = engage_te(model)
                elif it == "flash":
                    r = engage_flash()
                elif it == "rotary":
                    r = engage_rotary()
                else:
                    r = {"error": f"unknown item {it}"}
                rec["items"][it] = {k: v for k, v in r.items() if k != "thin"}
            except Exception as ex:  # noqa: BLE001 — an item that cannot engage is reported, never silently skipped
                if is_oom(ex): raise                                  # an out-of-memory propagates: no fallback applied
                import traceback; rec["items"][it] = {"error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()[-2000:]}
    return rec


def disengage(model):
    with _LOCK:
        n_te = disengage_te(model)
        for mod, attr, orig in reversed(_STATE["symbols"]):
            setattr(mod, attr, orig)
        _STATE["symbols"] = []
        _STATE["flash"] = None
        _STATE["rotary"] = None
    return {"te_restored": n_te}


def counters(model=None):
    out = {}
    if model is not None:
        out["te"] = te_counters(model)
    if _STATE["flash"] is not None:
        out["flash"] = _STATE["flash"].summary()
    if _STATE["rotary"] is not None:
        out["rotary"] = _STATE["rotary"].summary()
    return out


# ----------------------------------------------------------------------------------------------------------------------
# the fused seam: esmc_opt.kits.fused._patch._OPS entries (batch 1, T tokens, activations bf16 [T, ...])
#   _OPS.update(thin_launch.ops(model)) after kit.apply(model) — only the entries engage() could build are present
# ----------------------------------------------------------------------------------------------------------------------
def ops(model, items=TESTED_ITEMS):
    """returns the dict of seam entries this module establishes: 'ln_qkv'(mod, x), 'out_proj'(mod, ctx2d), 'ffn'(mod, x) [te],
    'attn'(qkv_packed, cu_seqlens, max_seqlen, softmax_scale) [flash], 'rotary'(rot, qkv_packed, cu_seqlens, max_seqlen) [rotary]."""
    res = {}
    if "te" in items:
        n_te = len([1 for _n, m in _find_te_modules(model) if hasattr(m, "_thin_launch_orig_forward")])
        if n_te == 0:
            engage_te(model)

        def _te_call(mod, x):
            return mod.forward(x)           # the instance's thin forward (falls back to the stock forward outside the class)
        if any(hasattr(m, "_thin_launch_orig_forward") for _n, m in _find_te_modules(model)):
            res["ln_qkv"] = _te_call
            res["out_proj"] = _te_call
            res["ffn"] = _te_call
    if "flash" in items:
        thin = _STATE["flash"]
        if thin is None:
            engage_flash()
            thin = _STATE["flash"]
        if thin is not None and thin.orig_packed is not None:
            def _attn(qkv_packed, cu_seqlens, max_seqlen, softmax_scale):
                return thin.packed_call(qkv_packed, cu_seqlens, max_seqlen, softmax_scale=softmax_scale)
            res["attn"] = _attn
    if "rotary" in items:
        rthin = _STATE["rotary"]
        if rthin is None:
            engage_rotary()
            rthin = _STATE["rotary"]
        if rthin is not None:
            def _rotary(rot, qkv_packed, cu_seqlens, max_seqlen):
                rot._update_cos_sin_cache(max_seqlen, device=qkv_packed.device, dtype=qkv_packed.dtype)
                T = qkv_packed.shape[0]
                x = qkv_packed[:, :2].view(T, -1, qkv_packed.shape[-1])
                rthin(x, rot._cos_cached, rot._sin_cached, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inplace=True)
            res["rotary"] = _rotary
    return res

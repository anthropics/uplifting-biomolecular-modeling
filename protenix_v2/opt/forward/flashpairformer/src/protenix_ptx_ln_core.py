"""protenix_ptx_ln_core.py — the model's standalone LayerNorm calls bound to the shared core's LayerNorm provider by TIER WORD (lever ln_core).

Protenix 2.0.0 builds every LayerNorm module as ``protenix.model.layer_norm.layer_norm.FusedLayerNorm`` (LAYERNORM_TYPE=fast_layernorm, the CLI's
own default = stock): ``forward(x) = FusedLayerNormAffineFunction.apply(x, weight, bias, normalized_shape, eps)`` -> the compiled extension
``fast_layer_norm_cuda_v2`` (LayerNormForwardV2; under bf16 autocast: bf16 x, the fp32 parameters cast to bf16, fp32 statistics, bf16 out — the
provider's ``bf16o`` form, whose cells carry ``fast_layernorm`` as their measured STOCK arm).

This unit wraps ``FusedLayerNorm.forward``: for a call in the bf16o form (bf16 x, fp32 weight) it asks ``opt_core.kernels.ln.select(cc, dtype,
cell, n, word=$PTX_LN_TIER, widen=True, out='bf16', ...)`` ONCE per call class (memoised on (C, rows, affine)) and

  * when the tier word resolves to a CARRIED row (fastln:lp, exactln, ln_rows, rfd, dtk_ln) the provider EXECUTES it
    (``opt_core.kernels.ln.layer_norm(x, ..., selection=sel, out_dtype=torch.bfloat16)``) — counted ``core``, per row;
  * when it resolves to a STOCK row (``fast_layernorm`` — the provider names it and refuses to execute it: not carried — or any other stock word),
    when no cell family lists the width (``uncovered``), when the provider refuses by name, and for every call NOT in the bf16o form (fp32 x: the
    diffusion sampler and the fp32 heads; parameter-free LayerNorms), the module's own statement runs BY NAME: the kit's stream-correct build of the
    library op fast_layernorm (third_party/fastln_prebuilt*, else the per-process source rebuild) — counted ``byname`` with the reason.

Tier words: ``fast`` under --mode fast, ``big`` under --mode big (modes.PACKAGE_POST).  --mode exact does NOT install this unit: no provider row is
byte-identical to fast_layernorm (exactln is bitwise to ATen's kernel, a different summation), so the exact word IS the library op by name = today's
bytes; nothing to bind.  ``capture=False`` is passed on purpose: the by-name arm is the stream-correct (capture-safe) fast_layernorm build and the carried
Triton rows launch on the current stream, so the selection is the same inside and outside the trunk's CUDA graphs.

Steps aside BY NAME (install() returns the reason; the trunk levers record ``LNCORE:unavailable(<reason>)`` and the mode refuses by name, as every
lever of this kit): PTX_LN_TIER not a tier word this unit binds, LAYERNORM_TYPE != fast_layernorm (the stock module is then torch.nn.LayerNorm: nothing
to bind), opt_core without kernels.ln's select/layer_norm surface, no CUDA device.
MODEL_OPT_LEVERS_OFF=ln_core removes PTX_LN_TIER: the unit is not installed and every LayerNorm runs the module's statement (ablation).
"""
import os
import sys

WORDS = ("fast", "big")                     # the tier words this unit binds (exact = the library op by name: not installed)
ENV = "PTX_LN_TIER"
MARK = "LNCORE"

_STATE = {"installed": False, "word": None, "reason": None, "opt_core": None, "stack": None, "cc": None}
_STATS = {"calls": 0, "core": 0, "byname": 0, "byname_form": 0, "byname_stock": 0, "byname_uncovered": 0, "byname_refused": 0, "classes": 0}
_ROWS = {}                                    # served arm word -> calls executed by the provider
_BYNAME = {}                                  # reason word -> calls left on the module's statement
_UNCOVERED = {}                               # "bf16o|C<c>|rows<r>" -> calls (no cell family lists the width: CORE C5 candidates)
_MEMO = {}                                    # (C, rows, has_w, has_b) -> ("core", Selection) | ("byname", reason)
_ORIG = {}


class Unavailable(Exception):
    pass


def word():
    return (os.environ.get(ENV) or "").strip()


def enabled():
    return bool(word())


def _decide(FACE, cc, x, C, rows, has_w, has_b):
    """One provider decision for a bf16o call class: ("core", Selection) when the tier word resolves to a carried row the provider executes,
    ("byname", <reason>) otherwise.  Pure bookkeeping around opt_core.kernels.ln.select (which records the core's cell census once per class)."""
    import torch
    n = int(round(rows ** 0.5))
    cw = FACE.cell_for_rows(rows, n, C)
    if cw is None:
        return ("byname", "uncovered")
    try:
        import triton  # noqa: F401
        has_triton = True
    except ImportError:
        has_triton = False
    try:
        sel = FACE.select(cc, torch.bfloat16, cw, n, word=_STATE["word"], widen=True, out="bf16", capture=False, C=C, rows=rows,
                          has_triton=has_triton, aligned=bool(FACE._rows_aligned(x, C)) if hasattr(FACE, "_rows_aligned") else True)
    except FACE.Refusal as e:
        return ("byname", "refused:%s" % (getattr(e, "kind", None) or str(e))[:48].replace(" ", "_"))
    row = getattr(sel, "row", None)
    if row in getattr(FACE, "STOCK_ROWS", ()) or row is None:
        return ("byname", "stock:%s" % row)
    return ("core", sel)


def _make_forward(FACE, LNM, cc):
    import torch
    orig = LNM.FusedLayerNorm.forward

    def forward(self, input):
        _STATS["calls"] += 1
        w, b = self.weight, self.bias
        if input.dtype != torch.bfloat16 or w is None or w.dtype != torch.float32 or not input.is_cuda:
            _STATS["byname"] += 1; _STATS["byname_form"] += 1
            return orig(self, input)
        C = int(self.normalized_shape[-1])
        if len(self.normalized_shape) != 1 or input.shape[-1] != C:
            _STATS["byname"] += 1; _STATS["byname_form"] += 1
            return orig(self, input)
        rows = input.numel() // C if C else 0
        key = (C, rows, True, b is not None)
        dec = _MEMO.get(key)
        if dec is None:
            _STATS["classes"] += 1
            try:
                dec = _decide(FACE, cc, input, C, rows, True, b is not None)
            except Exception as e:                                  # a provider bookkeeping error is named, never fatal: the module's statement serves the class
                dec = ("byname", "error:%s" % type(e).__name__)
            _MEMO[key] = dec
            if dec[0] == "byname" and dec[1] == "uncovered":
                _UNCOVERED["bf16o|C%d|rows%d" % (C, rows)] = 0
        if dec[0] == "core":
            sel = dec[1]
            try:
                y, _ = FACE.layer_norm(input, (C,), w, b, self.eps, selection=sel, out_dtype=torch.bfloat16, capture=False)
            except FACE.Refusal as e:                                # the row refused this operand at launch: by name for the class from here on (named once)
                _MEMO[key] = ("byname", "refused:%s" % (getattr(e, "kind", None) or str(e))[:48].replace(" ", "_"))
                print(f"[FPF] LNCORE: row {getattr(sel, 'row', '?')} refused C={C} rows={rows} ({e}); the module's fast_layernorm serves this class by name",
                      file=sys.stderr, flush=True)
                _STATS["byname"] += 1; _STATS["byname_refused"] += 1
                return orig(self, input)
            _STATS["core"] += 1
            arm = "%s%s" % (sel.row, (":" + sel.variant) if getattr(sel, "variant", None) else "")
            _ROWS[arm] = _ROWS.get(arm, 0) + 1
            return y
        why = dec[1]
        _STATS["byname"] += 1
        if why == "uncovered":
            _STATS["byname_uncovered"] += 1; _UNCOVERED["bf16o|C%d|rows%d" % (C, rows)] = _UNCOVERED.get("bf16o|C%d|rows%d" % (C, rows), 0) + 1
        elif why.startswith("stock:"):
            _STATS["byname_stock"] += 1
        else:
            _STATS["byname_refused"] += 1
        _BYNAME[why] = _BYNAME.get(why, 0) + 1
        return orig(self, input)

    return orig, forward


def install():
    """Bind FusedLayerNorm.forward to the provider by the tier word in $PTX_LN_TIER.  Returns the state dict; raises Unavailable (named) when the
    unit cannot engage — the caller records LNCORE:unavailable(<reason>)."""
    if _STATE["installed"]:
        return dict(_STATE)
    w = word()
    if w not in WORDS:
        raise Unavailable(f"{ENV}={w!r} is not a tier word this unit binds ({'|'.join(WORDS)}; exact = the library op fast_layernorm by name, not installed)")
    if os.environ.get("LAYERNORM_TYPE", "fast_layernorm") != "fast_layernorm":
        raise Unavailable(f"LAYERNORM_TYPE={os.environ.get('LAYERNORM_TYPE')!r}: the model's LayerNorm is torch.nn.LayerNorm, not the fast_layernorm module this unit binds")
    import torch
    if not torch.cuda.is_available():
        raise Unavailable("no CUDA device")
    try:
        import opt_core
        from opt_core.kernels import ln as FACE
    except Exception as e:
        raise Unavailable(f"opt_core.kernels.ln not importable: {e!r}"[:160])
    for name in ("select", "layer_norm", "cell_for_rows", "Refusal", "STOCK_ROWS"):
        if not hasattr(FACE, name):
            raise Unavailable(f"opt_core {getattr(opt_core, '__version__', '?')} kernels.ln has no {name} (provider surface older than this binding)")
    try:
        import protenix.model.layer_norm.layer_norm as LNM
    except Exception as e:
        raise Unavailable(f"protenix.model.layer_norm.layer_norm not importable: {e!r}"[:160])
    dev = torch.cuda.current_device()
    cc = tuple(torch.cuda.get_device_capability(dev))
    orig, fwd = _make_forward(FACE, LNM, cc)
    _ORIG["forward"] = orig
    LNM.FusedLayerNorm.forward = fwd
    try:
        stack = FACE.stack_word(dev)
    except Exception:
        stack = "?"
    _STATE.update(installed=True, word=w, reason=None, opt_core=getattr(opt_core, "__version__", "?"), stack=stack, cc="%d.%d" % cc)
    print(f"[FPF] LNCORE:on word={w} — FusedLayerNorm.forward bound to opt_core {_STATE['opt_core']} kernels.ln by tier word (bf16o calls: a carried row the word "
          f"resolves to is executed by the provider; the stock row fast_layernorm, uncovered widths, refusals and every non-bf16o call run the module's "
          f"stream-correct fast_layernorm BY NAME); cc {_STATE['cc']}, stack {stack}", file=sys.stderr, flush=True)
    return dict(_STATE)


def uninstall():
    if _STATE["installed"] and "forward" in _ORIG:
        import protenix.model.layer_norm.layer_norm as LNM
        LNM.FusedLayerNorm.forward = _ORIG.pop("forward")
        _STATE["installed"] = False


def report():
    """Live counters for the LEVER line / the trunk levers' report(): installed, word, calls, core (executed by a provider row), byname (+ split),
    rows (arm -> calls), byname reasons, uncovered classes (width x rows with no cell family: CORE cell candidates), decision classes."""
    return {"installed": bool(_STATE["installed"]), "word": _STATE["word"], "opt_core": _STATE["opt_core"], "stack": _STATE["stack"], "cc": _STATE["cc"],
            **{k: int(v) for k, v in _STATS.items()}, "rows": dict(_ROWS), "byname_reasons": dict(_BYNAME), "uncovered": dict(_UNCOVERED)}


def summary_line():
    r = report()
    rows = ",".join(f"{k}:{v}" for k, v in sorted(r["rows"].items())) or "none"
    unc = ",".join(sorted(r["uncovered"])) or "none"
    return (f"[FPF] LNCORE summary: word={r['word']} calls={r['calls']} core={r['core']} rows={rows} byname={r['byname']} "
            f"(form={r['byname_form']} stock={r['byname_stock']} uncovered={r['byname_uncovered']} refused={r['byname_refused']}) classes={r['classes']} uncovered_cells={unc}")

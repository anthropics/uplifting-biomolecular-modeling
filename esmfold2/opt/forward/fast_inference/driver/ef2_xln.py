"""ef2_xln — the `xln` lever: the shared core's LayerNorm provider row **exactln** (``opt_core.kernels.ln``: an
NVRTC-compiled CUDA replica of ATen's ``vectorized_layer_norm_kernel`` forward, bit for bit, launched through the CUDA driver API) serves the
PAIR-SIZED ``nn.LayerNorm`` sites of the ESMFold2 forward that every other lever of the kit leaves on ATen.

    import ef2_xln
    ef2_xln.install(model)          # in ef2_server.configure(), after the pair / msa2 / hoist levers, before ef2_dit; instance patches only
    ef2_xln.levers_on() -> {"xln": True|False}; ef2_xln.refusal() -> None | "<word>"; ef2_xln.stats(); ef2_xln.describe(); ef2_xln.uninstall(model)

WHAT CHANGES.  Nothing numerically: exactln's statement IS ``torch.nn.functional.layer_norm`` (same reduction tree, same rsqrt, same fma order;
``torch.equal`` on every served class below, and whole folds byte-identical at ``--det 1``).  What goes away is ATen's memory traffic: under the
model's bf16 autocast ``F.layer_norm`` is an fp32-policy op, so ATen first materialises an fp32 copy of the bf16 pair tensor (``direct_copy_kernel``)
and then runs the fp32 LayerNorm over it; exactln's *widen* form reads the bf16 rows, widens in registers and writes the fp32 result once (the
autocast result, bit for bit), and its fp32 form is the same kernel arithmetic with one pass fewer over global memory than ATen's two-pass launch.

SITES (an ``nn.LayerNorm`` instance patch; the module decides per call).  Every ``nn.LayerNorm`` of the model whose width exactln serves
(``C % 4 == 0``, ``4 <= C <= 1024``: the pair / MSA / atom / confidence norms, C 128 / 256 / 384 / 512) EXCEPT the ESMC language model
(``language_model.model``: C 2560, wider than the row) and the diffusion token transformer (``structure_head.diffusion_module.token_transformer``:
rows = tokens, see FLOOR) is patched; a patched module serves a call through exactln when the call has at least ``FLOOR_ROWS`` rows on a CUDA tensor in
fp32 or bf16 (the ``.float()``-under-autocast *widen* form included) and runs its verbatim ``F.layer_norm`` statement otherwise, counted by reason.
Which sites that leaves live depends on the mode (python-level count over one Full-model fold, 21 trunk passes):
    exact  (opt7x)      714 launches / fold: the MSA module's reference-path norms — TriangleMultiplicativeUpdate ``norm_start`` (bf16 -> fp32 widen,
                        N^2 x 256, 168), ``norm_mix`` (fp32 on ef2_hoist's contiguous product, 168), ``pair_transition.norm`` (84), pair-weighted
                        averaging ``compute_bias.0`` (63) + ``norm_single`` / outer-product-mean ``norm`` / ``msa_transition.norm`` ((N*depth) x 128, 210)
                        — plus ``parcae_input_norm`` (the injected pair, once per pass inside the tg / ls graphs: 21), ``language_model.base_z_mlp.1``,
                        the confidence head's ``z_norm`` / ``pae_ln`` / ``pde_ln`` and the conditioning ``z_input_norm`` (c 512) / ``z_transitions.*.norm``.
    fast / big        28 launches / fold: the MSA-module sites belong to m15 / m16 / m17 / t15msa|t16 (their fused kernels own those LayerNorms);
    (opt14_msa)         parcae + base_z + confidence + conditioning remain.  The Fast model (no MSA module) has the same 28 (22 in the trunk).
FLOOR.  ``FLOOR_ROWS = 4096``: below it the call is launch-bound and ATen's runtime launch is cheaper than the driver-API launch (rows = 400..1200,
C 384..1024: ATen 0.022 ms vs exactln 0.034 ms on H100, 0.046 vs 0.070 ms on A100 — x0.65 on both cards); the diffusion transformer's 52 LayerNorms
per step (rows = tokens) therefore stay on ATen by construction (the token transformer is not patched) and every other site decides per call.

MEASURED (H100 80GB HBM3 + A100-SXM4-80GB, torch 2.13.0+cu130, CUDA events 10 warm-up + 50 timed, 3 random
inputs ``torch.equal`` each):  per call @ 1200 tokens (rows 1.44 M) H100: fp32 c256 2.507 -> 1.099 ms (x2.28), widen c256 3.633 -> 1.000 ms (x3.63),
fp32 c128 2.160 -> 0.682 (x3.17), widen c128 bitwise; A100: fp32 c256 3.923 -> 2.059 (x1.91), widen c256 5.800 -> 1.937 (x2.99), widen c128 4.368 -> 1.187
(x3.68), fp32 c128 3.400 -> 1.162 (x2.93) — bitwise at every class on both cards.  Per Full-model fold, exact tier, ABBA vs the mode as shipped:
H100 trunk -0.698 s (-5.4 %) @ 800 tokens, -1.537 s (-5.3 %) @ 1200; A100 -1.05 s (-3.9 %) @ 800, -2.1 s (-3.4 %) @ 1200; Fast model -0.109 s @ 1200;
fast tier (28 sites) -0.015 / -0.047 s @ 800 / 1200 (>= 0, byte-identical to fast as shipped).  ``--det 1 --seeds 0``: exact + xln == exact == ``--mode off
--backend fused`` byte for byte (cif + npz) on 400-token ladder items and the identity canaries, both cards.

REFUSALS (by name; the lever steps aside, the mode runs — ``levers_on()['xln'] is False`` and ``refusal()`` says why; the package prints it on the
LEVER line as ``state=skipped reason=guard:off:<word>``):  ``core_below_0.5.28.0:<version>`` (the importable opt_core has no kernels.ln),
``no_cuda`` (no device at install: the kernels are compiled for the visible card), ``cc:<X.Y>_not_proven`` (exactln's bitwise cells are 9.0 and 8.0),
``import:cuda.bindings(nvrtc)`` (the NVRTC / driver bindings are not importable), ``compile:<ExceptionType>`` (NVRTC refused a kernel), ``no_sites``.
Per call (counted in ``stats()``, never a refusal): ``below_floor``, ``not_cuda``, ``dtype:<dtype>``, and exactln's own ``Unsupported.reason`` words
(``width``, ``rows_int32``, ...) -> the module's verbatim ``F.layer_norm``.

GRAPHS.  The MSA encoder runs inside ef2_opt's generic encoder graph (eg) and ``parcae_input_norm`` inside the trunk-pass graphs (tg / ls / rg): every
(dtype form, C) the captured regions reach is NVRTC-compiled and its module loaded at install (``PRECOMPILE``), so nothing compiles or loads inside a
stream capture; a driver-API launch on the capturing stream is recorded like any kernel (21 passes replayed per fold in the measurements above).
"""
import collections
import sys
import time
import types

import torch
import torch.nn.functional as F

LEVERS = ("xln",)
MIN_CORE = (0, 5, 28, 0)                     # opt_core's first version with kernels/ln (exactln 1.3.3)
FLOOR_ROWS = 4096                            # calls with fewer rows run the module's own F.layer_norm (measured slower through the driver-API launch, both cards)
MIN_C, MAX_C = 4, 1024                       # exactln's served widths (C % 4 == 0); ESMC's C 2560 rows are never patched
EXCLUDE_PREFIXES = ("_esmc.", "language_model.model.",                                  # the ESMC language model (C 2560: wider than the row; named so a narrower LM norm never joins unmeasured)
                    "structure_head.diffusion_module.token_transformer.",                # the DiT token path: 52 norms per diffusion step, rows = tokens x samples — launch-bound, ATen by name
                    "structure_head.diffusion_module.s_step_norm", "structure_head.diffusion_module.token_norm")   # the sampler's per-step single-track norms (rows = tokens), same reason
PRECOMPILE = {                               # (x dtype, param dtype, out dtype) -> widths: every (form, C) a graph-captured region reaches, + the eager confidence / conditioning sites
    ("float32", "float32", "float32"): (128, 256, 384, 512),      # norm_mix, confidence z_norm / pae_ln / pde_ln (256), plddt_ln / resolved_ln (384), z_input_norm (512), atom norms (128)
    ("bfloat16", "float32", "float32"): (128, 256),               # the autocast widen form: norm_start / pair_transition / compute_bias / parcae / base_z (256), the MSA-row norms (128)
}
AFFINE_BOTH = 3                              # exactln's affine bitmask for weight + bias (every patched site is elementwise_affine with bias)

STATS = collections.Counter()                # served / below_floor / fallback:<reason> / not_cuda / dtype:<dtype> — per call, this process
_STATE = {"on": False, "refusal": None, "sites": {}, "patched": [], "facts": {}, "install_s": None, "E": None}


class XlnUnavailable(RuntimeError):
    """Raised by install(strict=True) when the lever cannot engage; ``.reason`` is the blank-free refusal word."""

    def __init__(self, reason):
        super().__init__(f"ef2_xln: xln cannot engage on this stack: {reason}")
        self.reason = reason


def _slug(s):
    return str(s).strip().replace(" ", "_").replace("=", ":")[:80]


def _core_version():
    import opt_core
    parts = []
    for p in str(getattr(opt_core, "__version__", "0")).split(".")[:4]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts), str(getattr(opt_core, "__version__", "?"))


def _autocast_state():
    """(enabled, dtype) of CUDA autocast in this thread (torch >= 2.4 device-generic API, else the legacy one)."""
    try:
        en = torch.is_autocast_enabled("cuda")
        return en, (torch.get_autocast_dtype("cuda") if en else None)
    except TypeError:
        en = torch.is_autocast_enabled()
        return en, (torch.get_autocast_gpu_dtype() if en else None)


def _site_family(name):
    for fam, key in (("msa_encoder", "msa_encoder."), ("parcae", "parcae_input_norm"), ("base_z", "language_model.base_z"), ("confidence", "confidence_head."),
                     ("conditioning", "structure_head.diffusion_module.conditioning."), ("atom", "atom")):
        if key in name:
            return fam
    return "other"


def eligible_sites(model):
    """``[(qualified name, module)]``: the nn.LayerNorm instances this lever patches on ``model`` (see SITES in the module doc)."""
    out = []
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.LayerNorm):
            continue
        if any(name.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        shape = tuple(m.normalized_shape)
        if len(shape) != 1:
            continue
        C = int(shape[-1])
        if C % 4 or C < MIN_C or C > MAX_C:
            continue
        if m.weight is None or m.bias is None:            # exactln serves affine none / weight / bias / both; the kit vouches for the model's form (weight + bias) only
            continue
        out.append((name, m))
    return out


def _xln_forward(self, input):
    """The patched nn.LayerNorm.forward: exactln at >= FLOOR_ROWS rows on CUDA fp32 / bf16 (autocast widen included), else the module's own statement."""
    C = int(self.normalized_shape[-1])
    if not input.is_cuda:
        STATS["not_cuda"] += 1
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
    rows = input.numel() // C if C else 0
    if rows < FLOOR_ROWS:
        STATS["below_floor"] += 1
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
    w, b = self.weight, self.bias
    ac, ac_dtype = _autocast_state()
    x = input
    if x.dtype == torch.bfloat16:
        if ac or w.dtype == torch.float32:              # ATen under autocast: layer_norm is an fp32-policy op — bf16 x is widened to fp32, fp32 params stay, fp32 out
            widen = True
            if w.dtype != torch.float32:
                w, b = w.float(), b.float()                # (a bf16 -> fp32 parameter cast is exact; the model's LayerNorm parameters are fp32)
        else:
            widen = False                                  # bf16 x, bf16 params, no autocast: exactln's bf16 form (bitwise vs ATen's bf16 kernel)
    elif x.dtype == torch.float32:
        widen = False
        if w.dtype != torch.float32:
            if not ac:
                STATS["dtype:mixed"] += 1
                return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
            w, b = w.float(), b.float()
    else:
        STATS["dtype:" + str(x.dtype).replace("torch.", "")] += 1
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
    E = _STATE["E"]
    try:
        y = E.layer_norm(x, (C,), w, b, self.eps, widen=widen)
    except E.Unsupported as e:                             # exactln's own per-call words (width, rows_int32, strides ...): the verbatim statement serves, counted
        STATS["fallback:" + _slug(getattr(e, "reason", type(e).__name__))] += 1
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
    STATS["served"] += 1
    return y


def _refuse(reason, strict):
    _STATE["on"] = False
    _STATE["refusal"] = reason
    print(f"[ef2_xln] xln steps aside by name: {reason}", file=sys.stderr, flush=True)
    if strict:
        raise XlnUnavailable(reason)
    return describe()


def install(model, xln=True, strict=False, precompile=True):
    """Patch the eligible nn.LayerNorm instances of ``model`` (see SITES) to serve through exactln and compile every PRECOMPILE instantiation now.
    Returns describe().  A stack the row does not serve makes the lever step aside BY NAME (``refusal()``; strict=True raises XlnUnavailable)."""
    if not xln:
        return describe()
    if _STATE["patched"]:
        uninstall(model)
    t0 = time.time()
    ver, ver_s = _core_version()
    if ver < MIN_CORE:
        return _refuse(f"core_below_0.5.28.0:{ver_s}", strict)
    try:
        from opt_core.kernels import ln as LN                 # noqa: F401 — the provider face (its LN_CELLS name exactln the exact-class row on 9.0 and 8.0)
        from opt_core.kernels.ln import exactln as E
    except Exception as e:  # noqa: BLE001
        return _refuse("import:opt_core.kernels.ln:" + type(e).__name__, strict)
    if not torch.cuda.is_available():
        return _refuse("no_cuda", strict)
    try:
        dev = next((p.device for p in model.parameters() if p.is_cuda), torch.device("cuda", torch.cuda.current_device()))
    except Exception:  # noqa: BLE001
        dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    ccw = f"{cc[0]}.{cc[1]}"
    proven = tuple(str(c) for c in getattr(LN, "EXACTLN_PROVEN_CC", ("9.0", "8.0")))
    if ccw not in proven:
        return _refuse(f"cc:{ccw}_not_proven", strict)
    try:
        import cuda.bindings.nvrtc  # noqa: F401
        import cuda.bindings.driver  # noqa: F401
    except Exception:  # noqa: BLE001
        return _refuse("import:cuda.bindings(nvrtc)", strict)
    sites = eligible_sites(model)
    if not sites:
        return _refuse("no_sites", strict)
    _STATE["E"] = E
    if precompile:
        try:
            with torch.cuda.device(dev):
                for (tin, tpar, tout), widths in PRECOMPILE.items():
                    for C in widths:
                        E._compile(tin, tpar, tout, int(C), AFFINE_BOTH, E._rows_per_warp(int(C)))
        except Exception as e:  # noqa: BLE001
            return _refuse("compile:" + type(e).__name__ + ":" + _slug(str(e))[:60], strict)
    fams = collections.Counter()
    for name, m in sites:
        m.forward = types.MethodType(_xln_forward, m)
        _STATE["patched"].append(m)
        fams[_site_family(name)] += 1
    _STATE["sites"] = dict(fams)
    try:
        _STATE["facts"] = dict(E.facts())
    except Exception:  # noqa: BLE001
        _STATE["facts"] = {}
    _STATE["on"] = True
    _STATE["refusal"] = None
    _STATE["install_s"] = round(time.time() - t0, 2)
    _STATE["cc"] = ccw
    _STATE["core"] = ver_s
    return describe()


def uninstall(model=None):
    """Restore every patched module's class forward; keeps the compiled kernels (process-wide cache in exactln)."""
    for m in _STATE["patched"]:
        try:
            del m.forward                                   # drop the instance attribute: nn.LayerNorm.forward serves again
        except AttributeError:
            pass
    _STATE["patched"] = []
    _STATE["on"] = False
    return describe()


def levers_on():
    return {"xln": bool(_STATE["on"])}


def refusal():
    return _STATE["refusal"]


def stats():
    """Flat per-process counters for the package's EXIT tally: xln_served / xln_below_floor / xln_fallback (+ per-reason words) / xln_sites."""
    out = {"xln_sites": int(sum(_STATE["sites"].values())) if _STATE["sites"] else 0}
    fb = 0
    for k, v in STATS.items():
        if k == "served":
            out["xln_served"] = int(v)
        elif k == "below_floor":
            out["xln_below_floor"] = int(v)
        else:
            fb += int(v)
            out["xln_" + k.replace(":", "_").replace(".", "_")] = int(v)
    out.setdefault("xln_served", 0)
    out.setdefault("xln_below_floor", 0)
    out["xln_fallback"] = fb
    return out


def evidence():
    """Blank-free words for the LEVER line: sites=<n> floor_rows=4096 core=<version> kernels=<n> cc=<X.Y>."""
    f = _STATE.get("facts") or {}
    return {"sites": int(sum((_STATE["sites"] or {}).values())), "floor_rows": FLOOR_ROWS, "core": str(_STATE.get("core") or "?"),
            "kernels": len(f.get("kernels") or ()), "cc": str(_STATE.get("cc") or "?"), "provider": "opt_core.kernels.ln.exactln"}


def describe():
    return {"levers": levers_on(), "refusal": _STATE["refusal"], "sites": dict(_STATE["sites"] or {}), "floor_rows": FLOOR_ROWS, "install_s": _STATE["install_s"],
            "kernels": list((_STATE.get("facts") or {}).get("kernels") or ()), "torch_pin": (_STATE.get("facts") or {}).get("torch_pin"), "stats": stats()}

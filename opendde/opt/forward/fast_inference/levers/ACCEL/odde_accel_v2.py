"""odde_accel_v2.py -- OPENDDE_ACCEL_V2 add-on lever for OpenDDE, installed by the served-levers hook after the runner exists.  Nothing under
site-packages is edited; the lever is a default-OFF monkeypatch selected by its env flag.

  lever (env)                          class     what
  ODDE_DIT_ATTN=bf16                   TIER-2    sampler token-attention recast: the 24 DiffusionTransformer
                                                 blocks' AttentionPairBias core runs F.scaled_dot_product_attention with q/k/v AND the pair bias in bf16
                                                 (8-element-pitched bias copy so the fused mem-efficient kernel is eligible) instead of the stock fp32 upcast.
                                                 Numerics-changing -> never called exact.  Trunk / confidence
                                                 head / atom transformer untouched (the flag is set only inside those 24 modules' forward).
  ODDE_DIT_ATTN=<exact|fast|big|row> |       the SAMPLER unit's levers (levers/SAMPLER/odde_sampler.py: the two attention sites through the core pair-bias-attention provider by word,
  ODDE_ATOM_ATTN=<word> | ODDE_COND_DEDUPE |      the fused sampler schedules opendde_fpf_ditfast): this file is their ROUTING
  ODDE_DIT_FUSED (+ODDE_DIT_LOWP) |              site -- install_dit_attn() hands the model to odde_sampler.install() after the bf16 route is decided and publishes the
  ODDE_ATOM_FUSED                              unit's live counters in STATS (opendde_opt/ran.py reads them); every install error raises by name.
"""
from opt_core.oom import is_oom                      # an out-of-memory error is re-raised before any reroute below (the core's one classifier)
import os, sys, json, time, threading
import torch
import torch.nn.functional as F

__version__ = "0.3.0"
STATS = {"dit_attn": {"installed": False, "mode": None, "n_modules": 0, "bf16_calls": 0, "fp32_calls": 0, "mask_aligned_copies": 0, "errors": 0}}

def _log(msg):
    print(f"[odde_accel_v2] {msg}", file=sys.stderr, flush=True)

def _flag(k):
    return os.environ.get(k, "0") not in ("", "0")

# =====================================================================================================================================================
# Tier-2: DiT token attention in bf16
# =====================================================================================================================================================
_TLS = threading.local()
_ATT = {"orig": None}

def _aligned_bf16_bias(bias):
    """bf16 copy of the (possibly broadcast) fp32 pair bias with the last dim pitched to a multiple of 8 elements (16 B) so the fused kernel accepts it."""
    idx = tuple(slice(0, 1) if (st == 0 and sz > 1) else slice(None) for sz, st in zip(bias.shape, bias.stride()))
    core = bias[idx]
    n = core.shape[-1]; pitch = (n + 7) // 8 * 8
    if pitch == n and core.is_contiguous():
        cb = core.to(torch.bfloat16)
    else:
        buf = torch.empty(*core.shape[:-1], pitch, dtype=torch.bfloat16, device=core.device)
        buf[..., :n].copy_(core); cb = buf[..., :n]; STATS["dit_attn"]["mask_aligned_copies"] += 1
    return cb.expand(bias.shape) if cb.shape != bias.shape else cb

def _attention_v2(q, k, v, attn_bias=None, use_efficient_implementation=True, inplace_safe=False):
    S = STATS["dit_attn"]
    if getattr(_TLS, "bf16", False) and q.is_cuda:
        try:
            input_dtype = q.dtype
            qb = q.to(torch.bfloat16); kb = k.to(torch.bfloat16); vb = v.to(torch.bfloat16)
            bias = _aligned_bf16_bias(attn_bias) if attn_bias is not None else None
            out = F.scaled_dot_product_attention(query=qb, key=kb, value=vb, attn_mask=bias, scale=1.0)
            S["bf16_calls"] += 1
            return out.to(dtype=input_dtype)
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise
            S["errors"] += 1
            if S["errors"] <= 3: _log(f"bf16 SDPA failed ({e!r}) -> stock fp32 path for this call")
    S["fp32_calls"] += 1
    return _ATT["orig"](q, k, v, attn_bias=attn_bias, use_efficient_implementation=use_efficient_implementation, inplace_safe=inplace_safe)

def install_dit_attn(model):
    """The DiT / atom attention ROUTING site: ODDE_DIT_ATTN=bf16 -> this file's bf16 SDPA recast (below); every SAMPLER-unit switch (ODDE_DIT_ATTN=<any other word>,
    ODDE_ATOM_ATTN, ODDE_COND_DEDUPE, ODDE_DIT_FUSED/ODDE_DIT_LOWP, ODDE_ATOM_FUSED; the legacy alias ODDE_DIT_ATTN_EXACT=1) -> levers/SAMPLER/odde_sampler.install
    (raises by name on any failure; the hook's STRICT switch ends the process). Returns this file's dit_attn STATS section (the hook records the bf16 lever from it)."""
    S = STATS["dit_attn"]
    mode = os.environ.get("ODDE_DIT_ATTN", "").strip().lower()
    _route_sampler_unit(model)
    if mode not in ("bf16",):
        return S
    if S["installed"]:
        return S
    import opendde.model.modules.primitives as PR
    dm = getattr(model, "diffusion_module", None)
    if dm is None:
        raise RuntimeError("model.diffusion_module not found")
    blocks = list(dm.diffusion_transformer.blocks)
    mods = []
    for blk in blocks:
        apb = blk.attention_pair_bias; att = getattr(apb, "attention", None)
        if att is None: raise RuntimeError("AttentionPairBias.attention not found")
        mods.append(att)
    if _ATT["orig"] is None:
        _ATT["orig"] = PR._attention; PR._attention = _attention_v2
    for att in mods:
        orig_fwd = att.forward
        def make(orig_fwd):
            def fwd(*a, **k):
                prev = getattr(_TLS, "bf16", False); _TLS.bf16 = True
                try:
                    return orig_fwd(*a, **k)
                finally:
                    _TLS.bf16 = prev
            return fwd
        att.forward = make(orig_fwd)
    S.update(installed=True, mode="bf16", n_modules=len(mods), n_heads=getattr(blocks[0].attention_pair_bias, "n_heads", None))
    _log(f"TIER-2 lever ODDE_DIT_ATTN=bf16 installed on {len(mods)} DiffusionTransformer token-attention modules (numerics-changing; never 'exact')")
    return S

def _route_sampler_unit(model):
    """Hand the model to the SAMPLER unit when one of its switches (or its probe test hook) is set; publish its live counter sections in STATS."""
    keys = ("ODDE_ATOM_ATTN", "ODDE_COND_DEDUPE", "ODDE_DIT_FUSED", "ODDE_DIT_LOWP", "ODDE_ATOM_FUSED", "ODDE_DIT_ATTN_EXACT", "ODDE_SAMPLER_PROBE")
    wanted = os.environ.get("ODDE_DIT_ATTN", "").strip().lower() not in ("", "0", "off", "bf16") or any(os.environ.get(k, "").strip() not in ("", "0", "off") for k in keys)
    if not wanted or STATS.get("sampler", {}).get("routed"):
        return
    unit = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SAMPLER")     # levers/SAMPLER (on the line's PYTHONPATH; the sibling directory otherwise)
    if unit not in sys.path and not any(os.path.realpath(p) == os.path.realpath(unit) for p in sys.path if p):
        sys.path.append(unit)
    import odde_sampler
    rep = odde_sampler.install(model)
    STATS.update(odde_sampler.SECTIONS)
    STATS["sampler"] = {"routed": True, "installed": list(rep.get("installed", [])), "version": odde_sampler.__version__}
    _log(f"SAMPLER unit routed: levers {','.join(rep.get('installed', [])) or '-'} (odde_sampler {odde_sampler.__version__})")


def stats():
    return json.loads(json.dumps(STATS, default=str))

def _at_exit():
    try:
        if STATS["dit_attn"]["installed"] or STATS.get("sampler", {}).get("routed"):
            slim = stats()
            _log("STATS@exit " + json.dumps(slim, default=str))
            rp = os.environ.get("ODDE_SERVED_LEVERS_REPORT")
            if rp:
                with open(rp, "a") as f: f.write(json.dumps({"event": "accel_v2_exit", "leg": os.environ.get("ODDE_SERVED_LEG"), "pid": os.getpid(), "t": time.time(), "levers": slim}, default=str) + "\n")
    except Exception: pass  # noqa: BLE001
import atexit as _atexit
_atexit.register(_at_exit)

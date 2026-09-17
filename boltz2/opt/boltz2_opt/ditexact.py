"""boltz2_opt.ditexact — the engine adapter of the DITEXACT levers (``opt/forward/ditexact/src/dx_token.py``) onto the diffusion module's TOKEN
transformer (``DiffusionModule.token_transformer``: boltz.model.modules.diffusionv2 / transformersv2, 24 DiffusionTransformerLayers, fp32,
autocast off).  Attached by ``boltz2_opt.worker_launch --attach ditexact`` right after ``boltz.model.modules.diffusionv2`` imports: it wraps
``DiffusionModule.__init__`` so that every DiffusionModule built in the worker (``Boltz2.load_from_checkpoint``) gets the schedule installed as the
INSTANCE forward of its token transformer — the class-level patches of the CUDA-graph sampler's hoist (boltz_dit_hoist, applied later by the worker)
stay in force for everything else and supply, through their cache, the 24 expanded per-layer pair biases and the additive token-mask term this
schedule reads (``terms``).  Without the hoist's level-2 cache (BOLTZ_DIT_HOIST unset or < 2, or a prediction the memory-headroom gate sent to the
stock sampler) a call takes the class forward, counted ``prev_by=no_hoist_cache`` — the stock arithmetic either way.

Switch ``BOLTZ_DIT_EXACT`` (the mode table sets it, modes.py; the ``BOLTZ_DIT_`` family is stripped from a caller's environment, stock/PINS.json):
a comma list of lever words —
  ``par``  (registry lever ``dit_par``): the layer's independent GEMMs / LayerNorms / sigmoids issued on forked CUDA streams inside the captured
           step (s-conditioned path prefetched one layer ahead, k|v|g beside q, a_to_b beside swish_gate); same kernels, same operands: bitwise.
  ``mask`` (registry lever ``dit_mask``): the additive token mask skipped when the cached term is -0.0 in every element (token_pad_mask all ones,
           AttentionPairBias.inf finite): x + (-0.0) == x; a mask with a zero keeps the stock add (``mask_applied``).
  ``sba``  (registry lever ``dit_sba``): `attn / head_dim**0.5 + bias [+ maskterm]` as ONE in-place Triton pass with ATen's roundings
           (x * fp32(1/s), then + bias, then + mask; FMA contraction off) instead of two / three full passes over the fp32 [B*m,16,N,N] scores;
           refuses by name per call (``sba_torch_by``: no_triton / cpu / dtype / layout) and the torch statements serve.
  ``smx``  (registry lever ``dit_smx``): the whole `softmax(attn / sqrt(d) + bias [+ mask], -1)` as one in-place NVRTC-compiled CUDA kernel —
           the sba arithmetic followed by a line-by-line replica of ATen's softmax_warp_forward (128 < N <= 2048); BIT-COMPARED AT RUN TIME per
           (N, masked) class at its first eager encounter (torch statements vs kernel on the same scores, torch.equal) — a class that does not
           compare equal is refused by name (``smx_bitcmp``) and sba / torch serve it; other declines by name: no_nvrtc / n_range / layout / dtype /
           recip_probe (``smx_declined_by``).
  ``glue`` (registry lever ``dit_glue``): the layer's main-chain elementwise glue as three NVRTC CUDA kernels with ATen's per-element op order
           and roundings (affine sc*x+sb, gated residual a+g*o, (silu(gates)*x)*ab) — 6 graph nodes per layer fewer; each kernel BIT-COMPARED AT
           RUN TIME per shape class at first eager use, declining by name for a class that does not compare equal (``glue_bitcmp``).
  Compute-capability gate: smx / glue replicate ATen math sequences and serve only on the capabilities they were proven on (dx_token.PROVEN_CC,
  today sm_90); on another card they are refused BY NAME at install (``cc_refused`` = unproven_cc:sm_XY: that lever reports state=off with the
  reason, the other words serve, the run stays active).
Schedule constants (source, printed on the LEVER line): ``TUNE`` = n_spath 1 (s-path side streams; + 3 a-path side streams), prefetch 1
(layers of s-path look-ahead).

Evidence: ``report()`` -> ``ditexact_report`` in the worker log (applied levers, words, the schedule's census: calls / calls_par / layers /
mask_skipped / mask_applied / prev_by / capturing_calls, install count, errors) and the fail-closed ``verdict()``: refused when a lever of the row was
not installed, when the token transformer never reached the schedule although denoiser calls happened, on any error, or on a ``prev_by`` reason
outside ``EXPECTED``.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

TAG = "boltz2-opt"
SWITCH = "BOLTZ_DIT_EXACT"
WORDS = ("par", "mask", "sba", "smx", "glue")             # lever words of the switch, in dx_token's vocabulary
LEVER_OF = {"par": "dit_par", "mask": "dit_mask", "sba": "dit_sba", "smx": "dit_smx", "glue": "dit_glue"}   # word -> registry lever
LEVERS = tuple(LEVER_OF[w] for w in WORDS)                # the registry levers this adapter installs (worker_launch reads it)
TUNE = {"n_spath": 1, "prefetch": 1}                    # source constants of the schedule (printed on the LEVER line): s-path side streams, layers of s-path look-ahead (1/1 measured fastest of {1,3,6} x {0,1,3} at 400-1400 tokens)
EXPECTED = ("no_hoist_cache",)                            # declared class-forward paths: a denoiser call without the hoist's level-2 cache (hoist off / headroom-gated prediction)
SMX_DECLARED = ("n_range",)                               # declared size range of the softmax replica: rows of N <= 128 or N > 2048 tokens take torch's statements, counted by name (not a refusal)
KIT_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "forward", "ditexact", "src"))   # opt/forward/ditexact/src beside this package (the kit tree; nothing staged)
_STATE: Dict[str, Any] = {"applied": [], "words": (), "installed": 0, "install_refused": {}, "errors": {}, "tune": {}, "module": None, "hoist": None}


def words(environ=None) -> tuple:
    """The lever words of ``BOLTZ_DIT_EXACT`` (order of WORDS); unknown words raise ValueError (a refusal by name at apply)."""
    env = os.environ if environ is None else environ
    raw = [w.strip().lower() for w in (env.get(SWITCH) or "").split(",") if w.strip()]
    bad = [w for w in raw if w not in WORDS]
    if bad:
        raise ValueError(f"{SWITCH}: unknown word(s) {bad}; the words are {list(WORDS)}")
    return tuple(w for w in WORDS if w in raw)


def requested(environ=None) -> bool:
    try:
        return bool(words(environ))
    except ValueError:
        return True


def _tune(environ=None) -> Dict[str, int]:
    return dict(TUNE)


def _dx():
    """The lever module (opt/forward/ditexact/src/dx_token.py), imported from the kit tree by path."""
    if _STATE["module"] is None:
        if KIT_SRC not in sys.path:
            sys.path.insert(0, KIT_SRC)
        import dx_token                                   # noqa: E402
        if os.path.dirname(os.path.realpath(dx_token.__file__)) != os.path.realpath(KIT_SRC):
            raise RuntimeError(f"dx_token imported from {dx_token.__file__}, not the kit tree {KIT_SRC}")
        _STATE["module"] = dx_token
    return _STATE["module"]


def _hoist():
    """boltz_dit_hoist as the worker imported it (sys.modules), or None (hoist not applied in this process)."""
    return sys.modules.get("boltz_dit_hoist")


def terms(dt, bias, mask, multiplicity):
    """(bias_layers, maskterm, mask_all_ones) from the hoist's level-2 cache for THIS call, or None (-> the class forward, counted no_hoist_cache).
    The all-ones predicate of the cached float mask is evaluated once per cache generation OUTSIDE capture (one host sync at the eager first step)."""
    H = _hoist()
    if H is None:
        return None
    C = H._active()
    if C is None or int(H.STATS.get("level") or 0) < 2:
        return None
    if bias is not C.t.get(("dt_bias_src", id(dt))):
        return None
    bl = []
    for layer in dt.layers:
        ent = C.t.get(("apb_bias", id(layer.pair_bias_attn)))
        if ent is None:
            return None
        bl.append(ent)
    for _ in bl:
        H._hit("apb_bias")                                 # the hoist's own census: its cached biases were read (by this schedule instead of its _apb_forward)
    maskf = C.t.get(("tok_maskf",))
    if mask is maskf and maskf is not None:
        mterm = C.t.get(("tok_maskterm",)); H._hit("tok_maskterm")
        key = ("dx_mask_all_ones",)                        # (cache generation, flag): the hoist refreshes the cached mask's VALUES in place per sample() (new gen) -> re-evaluated per prediction
        ent = C.t.get(key)
        flag = ent[1] if (ent is not None and ent[0] == C.gen) else None
        if flag is None and not _capturing():
            flag = bool((maskf == 1).all().item())
            C.t[key] = (C.gen, flag)
    else:
        mterm, flag = None, None
    return bl, mterm, flag


def _capturing() -> bool:
    import torch
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def apply(spec: Optional[str] = None) -> List[str]:
    """Wrap DiffusionModule.__init__ so every instance's token transformer carries the schedule (idempotent). ``spec`` overrides the env switch."""
    if _STATE["applied"]:
        return list(_STATE["applied"])
    ws = words() if spec is None else tuple(w for w in WORDS if w in [x.strip().lower() for x in str(spec).split(",")])
    if not ws:
        return []
    DX = _dx()
    tune = _tune()
    import boltz.model.modules.diffusionv2 as DM

    orig_init = DM.DiffusionModule.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        why = DX.qualifies(self.token_transformer)
        if why is not None:                                # a token transformer of another structure: the class forward, named (install_refused)
            _STATE["install_refused"][why] = _STATE["install_refused"].get(why, 0) + 1
            sys.stderr.write(f"[{TAG}] ditexact: token transformer not served ({why}); the class forward stands\n")
            return
        DX.install(self.token_transformer, terms, words=ws, n_spath=tune["n_spath"], prefetch=tune["prefetch"])
        _STATE["installed"] += 1

    DM.DiffusionModule.__init__ = __init__
    _STATE.update(applied=[LEVER_OF[w] for w in ws], words=ws, tune=tune)
    sys.stderr.write((line() or "") + "\n")
    return list(_STATE["applied"])


def refused_levers() -> Dict[str, str]:
    """Registry levers of the row refused by name at install on this card (unproven compute capability) -> reason word."""
    DX = _STATE["module"]
    cc = dict((DX.STATS.get("cc_refused") or {}) if DX is not None else {})
    return {LEVER_OF[w]: why for w, why in cc.items() if w in LEVER_OF}


def census() -> Dict[str, Any]:
    DX = _STATE["module"]
    st = dict(DX.STATS) if DX is not None else {}
    st.pop("last", None)
    return {"installed": _STATE["installed"], "install_refused": dict(_STATE["install_refused"]), **st}


IDLE = "idle: no denoiser call reached the token transformer's schedule (no prediction sampled in this process)"


def verdict() -> Dict[str, Any]:
    """Fail-closed: refused when the schedule was never installed although applied, on any install refusal, on a class-forward reason outside
    EXPECTED, or when calls arrived and none was served by the schedule while the hoist cache stood (calls_prev == calls with reasons outside EXPECTED)."""
    c = census()
    if not _STATE["applied"]:
        return {"ok": False, "idle": False, "reason": "not_applied"}
    if c.get("install_refused"):
        return {"ok": False, "idle": False, "reason": "install_refused:" + ",".join(sorted(c["install_refused"]))}
    if not c.get("installed"):
        return {"ok": False, "idle": False, "reason": "no_token_transformer_built"}
    unexpected = sorted(k for k in (c.get("prev_by") or {}) if k not in EXPECTED)
    if unexpected:
        return {"ok": False, "idle": False, "reason": "class_forward_by:" + ",".join(unexpected)}
    glue_unexpected = {k: v for k, v in (c.get("glue_declined_by") or {}).items()}
    if "glue" in [w for w in _STATE["words"] if w not in (c.get("cc_refused") or {})] and (glue_unexpected or any(v != "served" for v in (c.get("glue_bitcmp") or {}).values())):
        return {"ok": False, "idle": False, "reason": "glue:" + ",".join(sorted(set(list(glue_unexpected.keys()) + [k + "=" + v for k, v in (c.get("glue_bitcmp") or {}).items() if v != "served"])))}
    smx_unexpected = {k: v for k, v in (c.get("smx_declined_by") or {}).items() if k not in SMX_DECLARED}
    served_words = [w for w in _STATE["words"] if w not in (c.get("cc_refused") or {})]
    if "smx" in served_words and (smx_unexpected or any(v != "served" for v in (c.get("smx_bitcmp") or {}).values())):   # the fused softmax declined or failed its bit-comparison on a class it was asked to serve: named, gate refuses (torch's statements served — stock arithmetic)
        return {"ok": False, "idle": False, "reason": "smx:" + ",".join(sorted(set(list(smx_unexpected.keys()) + [k + "=" + v for k, v in (c.get("smx_bitcmp") or {}).items() if v != "served"])))}
    if "sba" in _STATE["words"] and (c.get("sba_torch_by") or {}):     # the fused pass refused a call it was asked to serve: named, and the gate refuses (the torch statements served it — stock arithmetic — but the row's lever did not act)
        return {"ok": False, "idle": False, "reason": "sba_torch_by:" + ",".join(sorted(c["sba_torch_by"]))}
    calls = int(c.get("calls") or 0)
    if calls == 0:
        return {"ok": True, "idle": True, "reason": IDLE}
    served = calls - int(c.get("calls_prev") or 0)
    if served == 0:
        return {"ok": False, "idle": False, "reason": f"served_none_of:{calls}"}
    if "par" in _STATE["words"] and int(c.get("calls_par") or 0) == 0:
        return {"ok": False, "idle": False, "reason": "par_never_ran"}
    return {"ok": True, "idle": False, "reason": None}


def line() -> Optional[str]:
    """The adapter's own activation line (stderr at apply and in the report); the kit's LEVER lines proper are rendered by boltz2_opt.report from
    ditexact_report."""
    if not _STATE["applied"]:
        return None
    c = census()
    return (f"[{TAG}] DITEXACT applied={','.join(_STATE['applied'])} words={','.join(_STATE['words'])} n_spath={_STATE['tune'].get('n_spath')} "
            f"prefetch={_STATE['tune'].get('prefetch')} installed={c.get('installed')} calls={c.get('calls', 0)} calls_par={c.get('calls_par', 0)} "
            f"mask_skipped={c.get('mask_skipped', 0)} mask_applied={c.get('mask_applied', 0)} sba_fused={c.get('sba_fused', 0)} sba_torch_by={c.get('sba_torch_by') or '-'} "
            f"smx_fused={c.get('smx_fused', 0)} smx_bitcmp={c.get('smx_bitcmp') or '-'} smx_declined_by={c.get('smx_declined_by') or '-'} recip={ {k: v.get('served') for k, v in (c.get('recip') or {}).items()} or '-'} "
            f"glue_fused={c.get('glue_fused', 0)} glue_bitcmp={c.get('glue_bitcmp') or '-'} glue_declined_by={c.get('glue_declined_by') or '-'} cc_refused={c.get('cc_refused') or '-'} nvrtc_compile_s={c.get('nvrtc_compile_s') or '-'} "
            f"calls_prev={c.get('calls_prev', 0)} prev_by={c.get('prev_by') or '-'} capturing_calls={c.get('capturing_calls', 0)}")


def report() -> Dict[str, Any]:
    refused = refused_levers()                             # levers refused by name at install on this card: not in `applied`, named under `refused`
    return {"applied": [l for l in _STATE["applied"] if l not in refused], "refused": refused, "words": list(_STATE["words"]), "variant": ",".join(_STATE["words"]),
            "tune": dict(_STATE["tune"]), "census": census(), "gate": verdict(), "line": line(), "expected": list(EXPECTED)}

"""ptx_native_core -- lever ``triattn_native`` (TOLERANCE class; fast + big, every card the shared core's table carries cells for): the block
core's Tier-2 triangle attention bound to the shared core's triangle-attention provider ``opt_core.kernels.triattn`` BY TIER WORD ALONE.  Every
call asks ``select(cc, bf16, 32, H, S, word='fast', form=<bias_only|mask_bias>)`` -- no row preference, no kit floor, no kit cell table: the
provider's measured cell decides the row per (cc, heads, size, call form) and the LEVER line names the rows that SERVED (``rows=<row>:<calls>``).
For this kit's D32 / H8 statement in the bias_only form the cells name, on cc 9.0, triattn_native (its Gluon member at small sizes,
cuda_sm90a and its sm_90a extensions above) and, on cc 8.0, triattn_native (its sm_80 member) at every
size.  The word is the MODE's tier word (PTX_T_ATT=fast | big), handed to the face literally (a face older than MIN_CORE that does not carry the
word refuses it by name, unknown_word, at the install probes -> CoreAttnUnavailable: the mode refuses by name).  Nothing here is a kernel:
this module is the binding; the kernels live in the shared core and are selected by its cells.

Route (env.sh, ARM=T): ``PTX_T_ATT=fast|big`` (the mode's own tier word; no alias words) -> ``PTX_BLK_ATT=ptx_native_core:attn`` + ``PTX_BLK_ATT_PROVIDER_MIN_TOKENS=0`` (ptx_trunk2_levers' generic
Tier-2 provider slot; min 0 = the provider owns every size the statement hands it; the ARM T size gate PTX_T_MIN_TOKENS around the whole statement
is fpf_smalln's, unchanged).  The kit README rows of cc 9.0 and cc 8.0 pre-export PTX_T_ATT=fast and big exports PTX_T_ATT=big (protenix_opt.modes README_ROWS /
extras); a card the provider has no cell for keeps env.sh's default word (k2b) by its row.  ``MODEL_OPT_LEVERS_OFF=triattn_native`` re-sources env.sh
with the displaced lever's word PTX_T_ATT=k2b (registry words / replaced_by).

Import == install (ptx_trunk2_levers._apply_blk2 imports this module at sitecustomize time, before any model object / forward / capture):
  1. ``cuda``   a CUDA device                                                  -> else CoreAttnUnavailable BY NAME (the mode refuses)
  2. ``core``   opt_core >= MIN_CORE with kernels.triattn importable           -> else CoreAttnUnavailable BY NAME
  3. ``select`` the provider's select() for H=8 at 256 / 512 / 1536 tokens (bias_only) names the rows it will serve (describe() printed); a tier
              word every candidate refuses at install (no cell for this cc, no prebuilt for this ABI ...) is CoreAttnUnavailable BY NAME.
An exception out of the import is recorded by ptx_trunk2_levers as ``BLK_ATT:provider ptx_native_core:attn unavailable(<exc>) -> cueq``: the BLK
line then names cueq, lever triattn_native has no marker and protenix_opt refuses the mode BY NAME.
Per call (``attn``; cuEquivariance convention: q / k / v [B,N,H,S,D] or [N,H,S,D] bf16, bias [B,1,H,S,S] or [1,H,S,S] fp32 / 16-bit, mask None or
[B,N,1,1,S]; returns [B,N,H,S,D] / [N,H,S,D] as given):
  * the provider: select(...) once per (H, S, dtype, form) (cached), then triangle_attention(..., selection=sel, form=<form>, stock=<the statement's
    library call>); the call FORM is the kit's own: ``bias_only`` when the statement passes no mask (the nomask lever: every BLK2 call) else
    ``mask_bias``; served rows and forms are counted (``rows``, ``forms``, ``sizes``);
  * a ``Refusal`` BY NAME for THIS call (or the package's typed ``Unavailable``): the row the provider names instead (``Refusal.fallback``; the
    library op ``cueq`` when it names none) serves the call through the same face by its row word -- counted under ``fallback_rows``, the refusal
    kind tallied and named once on stderr; never a silent substitute; any other exception propagates (fail loud).
``report()`` is what protenix_opt's LEVER line and the [FPF] REPORT read.
"""
import math
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch

WORD = os.environ.get("PTX_T_ATT", "fast") if os.environ.get("PTX_T_ATT", "fast") in ("fast", "big") else "fast"   # the MODE's tier word (env.sh case PTX_T_ATT fast | big), handed to the face literally: what the LEVER line's word= / bind=tier:<word> name
PREFER = None                                                   # no row preference: the provider's cell decides (kept as a name for the LEVER line: prefer=none)
MIN_CORE = (0, 5, 114, 0)                                       # oldest opt_core whose triattn face carries the big tier word (fast | exact | big), select(form=) and the per-form D32 cells of cc 9.0 AND cc 8.0 (triattn_native sm_80 member)
HEAD_DIM, HEADS = 32, 8                                         # Protenix v2 pairformer triangle attention (c_hidden 32 x 8 heads); the MSA-module / confidence pair stacks share it
PROBE_SIZES = (256, 512, 1536)                                  # install-time select() probes (named on the [FPF] line; a refusal of every candidate there is the install refusal)

PROVIDER = {"name": "core", "label": "TIER2", "kv_pad": 1, "wants_kv_len": False}     # ptx_trunk2_levers provider meta (unpadded layout); no `Refused` attribute on purpose: refused_exc=()


class CoreAttnUnavailable(RuntimeError):
    """Install-time refusal BY NAME: ``kind`` in cuda | core | select; str() = '<kind>: <detail>'."""

    def __init__(self, kind: str, detail: str):
        self.kind, self.detail = kind, detail
        super().__init__(f"{kind}: {detail}")


NativeUnavailable = CoreAttnUnavailable                          # alias kept for callers that import the older name


COUNTS: Dict[str, int] = {"calls": 0, "native": 0, "other_row": 0, "refused": 0}
ROWS: Dict[str, int] = {}                                       # provider-served calls by the row that served them (triattn_native | cuda_sm90a | k2b | ...)
FALLBACK_ROWS: Dict[str, int] = {}                              # calls a row refused by name, by the fallback row the provider named (served through the face by that row word)
REFUSED_KINDS: Dict[str, int] = {}                              # per-call refusal kind -> count (each named once on stderr)
SIZES: Dict[str, int] = {"lt512": 0, "512_1023": 0, "1024_2047": 0, "2048_3072": 0, "gt3072": 0}   # provider-served calls by key count
FORMS: Dict[str, int] = {}                                      # provider-served calls by call form (bias_only | mask_bias)
_INSTALL: Dict[str, Any] = {}
_SEL: Dict[Tuple[int, int, str, str], Any] = {}                 # (H, S, dtype, form) -> the provider's Selection
_SAID = set()
_FACE = None                                                    # opt_core.kernels.triattn (module)
_STACK: Optional[str] = None
_CC = (0, 0)
_CC_UNAVAILABLE: Tuple[type, ...] = ()
_STOCK = None                                                   # the statement's library call (protenix.model.triangular.layers.cuequivariance_triangular_attn), bound at first need


def _say_once(key: str, msg: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        print(msg, file=sys.stderr, flush=True)


def _vt(s: str):
    out = []
    for p in str(s).split("+")[0].split("."):
        try:
            out.append(int(p))
        except ValueError:
            break
    return tuple(out)


def _install() -> Dict[str, Any]:
    global _FACE, _STACK, _CC, _CC_UNAVAILABLE
    t0 = time.time()
    rep: Dict[str, Any] = {"installed": False, "word": WORD, "prefer": None, "bind": f"tier:{WORD}", "min_tokens": 0, "aside": None, "native_pkg": "none"}
    # 1. cuda
    if not torch.cuda.is_available():
        raise CoreAttnUnavailable("cuda", "CUDA is not available")
    if os.environ.get("LOCAL_RANK"):                             # the multi-GPU line's ranks: this rank's device explicit before the kernels first touch CUDA
        try:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        except Exception:
            pass
    _CC = tuple(torch.cuda.get_device_capability())
    rep["cc"] = f"{_CC[0]}.{_CC[1]}"
    # 2. core
    try:
        import opt_core
        rep["opt_core"] = getattr(opt_core, "__version__", "?")
        if _vt(rep["opt_core"]) < MIN_CORE:
            raise ImportError(f"opt_core {rep['opt_core']} < {'.'.join(map(str, MIN_CORE))}")
        from opt_core.kernels import triattn as FACE
        if _CC == (9, 0):                                         # the ABI key the provider checks a prebuilt against at select time (the face's own rule in triangle_attention)
            from opt_core.kernels.triattn import cuda_sm90a as _CS
            _STACK = _CS.stack_key()
        elif _CC == (8, 0):
            from opt_core.kernels.triattn import triattn_native as _CK
            _STACK = _CK.stack_key()
    except Exception as e:
        raise CoreAttnUnavailable("core", f"opt_core.kernels.triattn not usable: {str(e)[:200]}") from None
    _FACE = FACE
    rep["key"] = _STACK
    try:
        from opt_core.kernels.triattn import triattn_native as CC
        _CC_UNAVAILABLE = (CC.Unavailable,) if hasattr(CC, "Unavailable") else ()
        rep["pkg_active"] = getattr(CC, "ACTIVE_PKG", None)
        rep["native_pkg"] = f"v{rep['pkg_active']}" if rep["pkg_active"] and not str(rep["pkg_active"]).startswith("v") else str(rep["pkg_active"] or "none")
    except Exception as e:                                       # the package module is the provider's: a missing one only means the tier word never names triattn_native
        rep["native_pkg"] = f"unavailable({type(e).__name__})"
    # 3. select at the probe sizes (the row each will serve first; a refusal of every candidate = this card / stack has no row for the tier word)
    rep["select"] = {}
    refused = []
    for S in PROBE_SIZES:
        try:
            sel = _select(HEADS, S, "bf16", "bias_only")
            rep["select"][S] = sel.row
            rep.setdefault("describe", FACE.describe(sel))
        except Exception as e:
            rep["select"][S] = f"refused({getattr(e, 'kind', type(e).__name__)})"
            refused.append(f"S={S}: {getattr(e, 'kind', type(e).__name__)}: {str(e)[:160]}")
    if len(refused) == len(PROBE_SIZES):
        raise CoreAttnUnavailable("select", f"the provider's tier word {WORD!r} has no row for cc {rep['cc']} / key {_STACK}: " + " | ".join(refused))
    rep["installed"] = True
    rep["load_s"] = round(time.time() - t0, 2)
    return rep


def _select(H: int, S: int, dt: str, form: str):
    key = (int(H), int(S), dt, form)
    sel = _SEL.get(key)
    if sel is None:
        sel = _SEL[key] = _FACE.select(_CC, dt, HEAD_DIM, int(H), int(S), word=WORD, prefer=None, stack=_STACK, form=form)
    return sel


def _stock():
    """The statement's own library call in the provider's 5-D convention (rows cueq / exact_headsplit / a stock fallback need it)."""
    global _STOCK
    if _STOCK is None:
        import protenix.model.triangular.layers as TL
        lib = TL.cuequivariance_triangular_attn

        def stock5(Q, K, V, Bi, mask=None, scale=None, _lib=lib):
            o_ = _lib(Q, K, V, Bi, mask, scale)
            o_ = o_[0] if isinstance(o_, (tuple, list)) else o_
            return o_ if o_.dim() == 5 else o_.unsqueeze(0)

        _STOCK = stock5
    return _STOCK


def _as5(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() == 5 else t.unsqueeze(0)


_DT = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: Optional[torch.Tensor] = None,
         scale: Optional[float] = None) -> torch.Tensor:
    """Triangle attention forward in the cuEquivariance convention (module docstring). Returns the layout it was given ([B,N,H,S,D] or [N,H,S,D])."""
    S, H, D = int(q.shape[-2]), int(q.shape[-3]), int(q.shape[-1])
    q5, k5, v5 = _as5(q), _as5(k), _as5(v)
    b5 = bias if bias.dim() == 5 else (bias.unsqueeze(0) if bias.dim() == 4 else bias.reshape(1, 1, *bias.shape[-3:]))
    m5 = None if mask is None else (mask if mask.dim() == 5 else mask.unsqueeze(0))
    sc = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    form = "bias_only" if m5 is None else "mask_bias"                        # the kit's statement passes no mask under the nomask lever; protenix's own mask is per pair row
    COUNTS["calls"] += 1
    try:
        sel = _select(H, S, _DT.get(q.dtype, str(q.dtype)), form)
        out = _FACE.triangle_attention(q5, k5, v5, b5, m5, sc, word=WORD, prefer=None, selection=sel, form=form,
                                       stock=(_stock() if sel.row in _FACE.NEEDS_STOCK else None))
        row = sel.row
    except (_FACE.Refusal, NotImplementedError) + _CC_UNAVAILABLE as e:      # by name, this call only: the row the provider names instead serves it (the package's Unsupported is a NotImplementedError)
        kind = str(getattr(e, "kind", None) or type(e).__name__).split(":")[0][:60]
        fb = getattr(e, "fallback", None) or "cueq"
        REFUSED_KINDS[kind] = REFUSED_KINDS.get(kind, 0) + 1
        COUNTS["refused"] += 1
        _say_once("refused:" + kind, f"[ptx_native_core] the core's triangle-attention provider refused a call by name ({kind}: {str(e)[:160]}); q {tuple(q.shape)} {q.dtype} "
                                      f"strides {tuple(q.stride())}, bias {tuple(bias.shape)} {bias.dtype}, mask {None if mask is None else tuple(mask.shape)} -> row {fb!r} serves it by name (counted fallback_rows)")
        out = _FACE.triangle_attention(q5, k5, v5, b5, m5, sc, word=fb, stock=_stock(), form=form)
        row = fb
        FALLBACK_ROWS[fb] = FALLBACK_ROWS.get(fb, 0) + 1
    ROWS[row] = ROWS.get(row, 0) + 1
    FORMS[form] = FORMS.get(form, 0) + 1
    COUNTS["native" if row == "triattn_native" else "other_row"] += 1
    SIZES["lt512" if S < 512 else "512_1023" if S < 1024 else "1024_2047" if S < 2048 else "2048_3072" if S <= 3072 else "gt3072"] += 1
    out = out[0] if isinstance(out, (tuple, list)) else out
    if q.dim() == 4 and out.dim() == 5:
        out = out[0]
    elif q.dim() == 5 and out.dim() == 4:
        out = out.unsqueeze(0)
    return out


def report() -> Dict[str, Any]:
    """Live census for protenix_opt's LEVER line / the kit REPORT: install record + per-route call counts."""
    r = dict(_INSTALL)
    r["counts"] = dict(COUNTS)
    r["rows"] = dict(ROWS)
    r["fallback_rows"] = dict(FALLBACK_ROWS)
    r["forms"] = dict(FORMS)
    r["refused_kinds"] = dict(REFUSED_KINDS)
    r["sizes"] = {k: n for k, n in SIZES.items() if n}
    r["selections"] = {f"H{h}S{s}:{fm}": getattr(sel, "row", "?") for (h, s, _dt, fm), sel in sorted(_SEL.items())[:24]}
    fb: Dict[str, int] = {}                                      # the package's own operand-copy tallies (its CUDA members copy a q / k / v whose strides its TMA descriptors cannot take, counted)
    for name, mod in list(sys.modules.items()):
        if name.endswith(("cuda_b.triattn_m1", "cuda_c.triattn_mw", "cuda.triattn_sm90", "cuda_sm80.triattn_sm80")) and isinstance(getattr(mod, "FALLBACKS", None), dict):
            for k2, n in mod.FALLBACKS.items():
                fb[k2] = fb.get(k2, 0) + int(n or 0)
    r["pkg_fallbacks"] = fb
    return r


_INSTALL.update(_install())                                     # import == install (raises CoreAttnUnavailable by name; see module docstring)
print(f"[FPF] ARM=T attention: core = opt_core {_INSTALL.get('opt_core')} kernels.triattn word={WORD} prefer=none bind={_INSTALL.get('bind')} (cc {_INSTALL.get('cc')}, key {_INSTALL.get('key')}; "
      f"package triattn_native {_INSTALL.get('native_pkg')}) at every size the statement hands it; rows at {'/'.join(str(s) for s in PROBE_SIZES)} tokens: "
      f"{'/'.join(str(_INSTALL['select'].get(s)) for s in PROBE_SIZES)}; a per-call refusal is served by the row the provider names (fallback_rows); install {_INSTALL.get('load_s')} s", flush=True)
print(f"[FPF] ARM=T attention: core SELECT {_INSTALL.get('describe')}", flush=True)

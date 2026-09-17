"""ef2_trimul_nosave — the no-save forward of the checkpointed pair blocks on a forward-only triangle-multiplication kernel of the
shared core (numerics class: fast; modes fast / big).

Under the memory plan (ef2_bwd_ckpt) a checkpointed PairUpdateBlock runs its forward TWICE per grad-mode trunk pass: once in the
step's forward, where torch.utils.checkpoint keeps nothing but the block input, and once more inside the backward as the recompute.
Without this lever both passes run the grad-path kernels under grad (ef2_trimul's fused Function saves its (a|b), contraction and mask planes only
for checkpoint to drop them).  The first pass needs no backward at all, so it can run a FORWARD-ONLY triangle multiplication
the shared core's provider serves for this card / dtype / pair width — ``opt_core.kernels.trimul`` rows asked BY NAME in the order
``ROWS``
(``tx_sm90a``: an sm_90a TMA / wgmma kernel served from a prebuilt per torch / CUDA ABI, REFUSED BY NAME on a stack without one >
``esm_v61``: a prebuilt sm_90a cubin loaded through the CUDA driver > ``esm_v5_fwd``: a Triton forward, torch + triton only > ``v4``: the
generic Triton row), the first one this stack serves — and build no autograd graph:

    import ef2_trimul_nosave
    h = ef2_trimul_nosave.enable(model)          # after agk.enable(...) and ef2_trimul.enable(model, variant="fused")
    ...                                           # design steps
    print(ef2_trimul_nosave.describe(h))
    ef2_trimul_nosave.disable(model)

What enable() installs (instance level, reversible):
  * on the main FoldingTrunk: agk's patched trunk forward checkpoints a block through ``_NoSaveCheckpoint`` instead of
    torch.utils.checkpoint — first pass = the block's own (kit) forward under enable_grad on a DETACHED input (frozen weights: no
    graph is recorded and nothing is saved; the block still counts as a grad-mode bf16 call and takes agk's kit path, the transition
    its no-grad form), backward = the block forward once more on an input that requires grad (the same recompute as without the lever: the fused
    Function, K-D3) and torch.autograd.grad for d(pair) — the design loop forms no parameter gradients;
  * on every TriangleMultiplicativeUpdate of that trunk, outermost: a forward that serves ``pair + TriMul(pair)`` through the
    provider row when the call needs no backward (grad disabled, or the pair does not require grad) on a bf16 CUDA pair of >= 101
    tokens, and hands every other call to the forward that was there (ef2_trimul's fused Function under grad).  The residual is
    included; the block's row_drop is the identity for such outputs (ef2_trimul's flag, eval-time dropout being the identity).

Steps aside BY NAME (state=stepped_aside reason=...; nothing installed): ``core_absent`` / ``core_too_old:<v>`` (opt_core < 0.5.26
has no trimul provider), ``cc_no_gain:sm_NN`` (only the capabilities in
``ENGAGE_CC`` engage: sm_90), ``no_row:<refusals>`` (every candidate
row refused this stack — e.g. tx_sm90a no_prebuilt:<abi> with v4 unimportable), ``canary:<row>:<maxabs>`` (the selected row disagrees
with the module's own forward beyond bf16 noise on a closed-form input), ``agk_absent`` (the trunk is not agk-patched: no checkpoint
seam), ``params_require_grad`` (a trainable trunk would record a graph in the first pass), ``no_trunk``.  Refusals of individual rows
are words on the LEVER line, each row passed over BY NAME with the provider's own reason (``row=esm_v61
refused=tx_sm90a:no_prebuilt:torch211_cu128_sm90``; ``not_in_core:<v>`` = a core older than the row) — the next row serves.

The tier word.  The rows are asked BY NAME in the module's order (``ROWS``), not through the provider's tier word ``fast``: the
provider's per-stack table can resolve ``fast`` to another row than the head of ``ROWS`` on a given stack.  The LEVER line prints what
the tier word resolves to on the running stack at the design's size (``tier_fast=<row>``; ``tier_row``: the provider's pure ``select``,
nothing launched) next to the row that serves, so the two can be compared; when it resolves to the head of ``ROWS`` they agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import types

import torch
from torch import Tensor

import transformers.models.esmfold2.modeling_esmfold2_common as C

NAME = "ef2_trimul_nosave"
__all__ = ["enable", "disable", "describe", "stats", "extra_bytes", "tier_row", "Handle", "NAME", "ROWS", "TIER_WORD"]

TIER_WORD = "fast"                   # the provider's tier word this lever's class would ask; resolved for the LEVER line (tier_fast=<row>), not bound (module docstring: The tier word)
ROWS = ("tx_sm90a", "esm_v61", "esm_v5_fwd", "v4")   # candidate rows in preference order, asked BY NAME; each refusal steps to the next by name
ENGAGE_CC = ((9, 0),)                # sm_80: cc_no_gain (module docstring); other capabilities are not engaged either
MIN_TOKENS = 101                     # the rows' own n_min (they refuse n<101 by name); shorter pairs take the forward that was there
CORE_FLOOR = (0, 5, 26)              # the provider's first release; rows a given core does not carry are passed over by name (not_in_core:<v>) — esm_v61 needs >= 0.5.45, the kit's [tool.opt_core] floor
CANARY_TOKENS = 112
CANARY_TOL = 0.03                    # max|row - module forward| / max|module forward| on the closed-form canary (bf16 statements of the same op sit at ~4e-3 of scale; a wrong weight layout or direction at O(1))

try:                                                         # the provider (opt_core >= 0.5.26); absent / too old -> the lever steps aside by name at enable
    import opt_core as _core
    from opt_core.kernels import trimul as PT
    _CORE_VERSION = tuple(int(x) for x in str(_core.__version__).split(".")[:3])
    _CORE_WHY = "" if _CORE_VERSION >= CORE_FLOOR else "core_too_old:%s" % _core.__version__
except Exception as e:                                       # noqa: BLE001 — any import failure is the same word
    PT = None; _CORE_VERSION = None; _CORE_WHY = "core_absent:%s" % type(e).__name__

try:
    import ef2_trimul as _TM                                 # the residual flag + row_drop identity patch are ef2_trimul's (one mechanism for both levers)
except Exception:                                            # noqa: BLE001
    _TM = None


@dataclass
class Handle:
    word: str = "fast"
    rows: tuple = ()                                         # the candidate rows asked, in order (ROWS unless enable(rows=) named others)
    row: Optional[str] = None                                # the row that serves (None = stepped aside)
    reason: str = ""                                         # step-aside word ('' when on)
    refused: dict = field(default_factory=dict)              # row -> refusal kind (rows passed over, by name)
    canary: str = ""
    cc: tuple = ()
    stats: dict = field(default_factory=lambda: {"served": 0, "inner": 0, "first_pass": 0, "recompute": 0, "refused_calls": 0})
    caches: dict = field(default_factory=dict)               # id(engine) -> the provider's per-call-site cache (packed weights, selection)
    wcache: dict = field(default_factory=dict)               # id(engine) -> {"ver", "w10", "engine_ref"}
    n_tmu: int = 0
    ckpt: bool = False
    tier_fast: Optional[str] = None                          # the row the provider's tier word `fast` resolves to on this stack at the design's size (tier_row); None = not asked yet
    n_tokens: Optional[int] = None                           # the design's complex size when enable() was told it (else the first served pair's N)

    @property
    def on(self) -> bool:
        return self.row is not None and not self.reason


# ----------------------------------------------------------------------------------------------- weights / route
def _w10(engine, h: Handle) -> dict:
    """The provider's ten canonical tensors from a fork TriangleMultiplicativeBlock, refreshed when a parameter version changes.
    proj_bundle rows: value [0,2C) = a|b projections, gate [2C,4C) = a|b gates (the fork's split_kernel_weights / chunk order)."""
    key = id(engine)
    ver = tuple(int(p._version) for p in engine.parameters())
    ent = h.wcache.get(key)
    if ent is not None and ent["ver"] == ver:
        return ent["w10"]
    Wb = engine.proj_bundle.weight.detach()
    Cd = int(engine.latent_channels)
    w10 = dict(ln_in_w=engine.norm_start.weight.detach(), ln_in_b=engine.norm_start.bias.detach(),
               w_ap=Wb[0:Cd], w_bp=Wb[Cd:2 * Cd], w_ag=Wb[2 * Cd:3 * Cd], w_bg=Wb[3 * Cd:4 * Cd],
               ln_out_w=engine.norm_mix.weight.detach(), ln_out_b=engine.norm_mix.bias.detach(),
               w_o=engine.proj_emit.weight.detach(), w_og=engine.proj_gate.weight.detach())
    h.wcache[key] = {"ver": ver, "w10": w10, "engine_ref": engine}
    h.caches.pop(key, None)                                  # a new parameter version invalidates the provider's packed copies too
    return w10


def _serve(h: Handle, engine, z: Tensor, mask: Optional[Tensor], row: str) -> Tensor:
    """pair + TriMul(pair, mask) through `row` (raises PT.Refusal by name when the row cannot serve this call)."""
    m = mask
    if m is not None and m.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        m = m.to(torch.float32)
    cache = h.caches.setdefault(id(engine), {})
    return PT.triangle_multiplication(z, m, direction=engine._kernel_flow_direction(), weights=_w10(engine, h), word=row, residual=True,
                                    cache=cache, eps=float(getattr(engine.norm_start, "eps", 1e-5)))


_serve_opaque = torch._dynamo.disable(_serve) if hasattr(torch, "_dynamo") else _serve   # the rows launch on the live CUDA stream through the driver / Triton: a compiled caller (the block under the memory plan's stock-kernel policy) graph-breaks here and the row runs as written, never traced


def _tmu_forward_nosave(self: "C.TriangleMultiplicativeUpdate", z: Tensor, mask: Tensor | None = None) -> Tensor:
    """Bound as the OUTERMOST forward of a TriangleMultiplicativeUpdate: no-backward calls on the provider row, the rest inward."""
    h: Handle = self._ef2_nosave_handle
    if h.tier_fast is None and z.dim() == 4:                 # enable() without n_tokens: the tier word resolved at the first pair's own N (pure; once)
        _resolve_tier(h, self._engine, int(z.shape[1]))
    if (h.row is not None and z.is_cuda and z.dtype == torch.bfloat16 and z.dim() == 4 and z.shape[0] == 1 and z.shape[1] >= MIN_TOKENS
            and not (torch.is_grad_enabled() and z.requires_grad) and not (self.training and getattr(self._ef2_nosave_block.row_drop, "_r", 0.0) > 0)):
        try:
            out = _serve_opaque(h, self._engine, z, mask, h.row)
        except PT.Refusal as e:                              # a call the row cannot take (a shape outside its cells): named, counted, handed inward
            h.stats["refused_calls"] += 1; h.refused.setdefault(h.row + "@call", e.kind)
        else:
            h.stats["served"] += 1
            if _TM is not None:
                _TM._patch_row_drop(self._ef2_nosave_block.row_drop)     # set_kernel_backend() rebuilds row_drop: re-patch (idempotent)
            out._ef2_trimul_residual_included = True
            return out
    h.stats["inner"] += 1
    return self._ef2_nosave_inner_forward(z, mask=mask)


# ----------------------------------------------------------------------------------------------- the no-save checkpoint
class _NoSaveCheckpoint(torch.autograd.Function):
    """checkpoint(block) whose first pass records no graph: forward = block(pair.detach()) under enable_grad (frozen weights => no
    autograd recording, no saved tensors; grad-enabled so the block takes its grad-mode (kit) path and counts as such), backward =
    block(pair') under enable_grad on pair' = pair.detach().requires_grad_() and torch.autograd.grad((out,), (pair',), (d_out,))."""

    @staticmethod
    def forward(ctx, pair: Tensor, block, mask, h):
        ctx.block, ctx.mask, ctx.h = block, mask, h
        ctx.ac = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        ctx.save_for_backward(pair)
        with torch.enable_grad(), torch.autocast("cuda", dtype=ctx.ac[1], enabled=ctx.ac[0]):
            out = block(pair.detach(), pair_attention_mask=mask)
        h.stats["first_pass"] += 1
        if out.requires_grad:                                # cannot happen with frozen weights (enable() checked); never hand a foreign graph out
            out = out.detach()
        return out

    @staticmethod
    def backward(ctx, d_out: Tensor):
        (pair,) = ctx.saved_tensors
        p = pair.detach().requires_grad_(True)
        with torch.enable_grad(), torch.autocast("cuda", dtype=ctx.ac[1], enabled=ctx.ac[0]):
            out = ctx.block(p, pair_attention_mask=ctx.mask)
            (d_pair,) = torch.autograd.grad((out,), (p,), (d_out,), retain_graph=False, allow_unused=False)
        ctx.h.stats["recompute"] += 1
        return d_pair, None, None, None


def _ckpt_fn(h: Handle):
    def nosave_checkpoint(block, pair: Tensor, pair_attention_mask):
        if not pair.requires_grad:                           # nothing upstream wants a gradient: a plain no-graph forward is all checkpoint would do
            with torch.enable_grad():
                return block(pair.detach(), pair_attention_mask=pair_attention_mask)
        return _NoSaveCheckpoint.apply(pair, block, pair_attention_mask, h)
    nosave_checkpoint._ef2_nosave = True
    return nosave_checkpoint


# ----------------------------------------------------------------------------------------------- canary / enable / disable
def _canary_inputs(dev, Cd: int):
    n = CANARY_TOKENS
    i = torch.arange(n, dtype=torch.float32, device=dev); c = torch.arange(Cd, dtype=torch.float32, device=dev)
    z = (torch.sin(0.05 * i[:, None, None] + 0.11 * c[None, None, :]) * 1.5 + torch.cos(0.23 * i[None, :, None] - 0.07 * c[None, None, :])
         + 0.5 * torch.sin(0.013 * (i[:, None, None] * i[None, :, None]) + 0.5 * c[None, None, :]))[None].to(torch.bfloat16).contiguous()
    mask = (((i[:, None] * 7 + i[None, :] * 3) % 11) != 0).to(torch.float32)[None].contiguous()
    return z, mask


def _main_trunk(model):
    for name, m in model.named_modules():
        if isinstance(m, C.FoldingTrunk) and "confidence" not in name:
            return m
    return None


def tier_row(cc, c: int, n_tokens: int, direction: str = "outgoing") -> str:
    """The row the provider's tier word ``fast`` resolves to for (cc, bf16, c_z = c_hidden = c, n_tokens) on THIS stack — the provider's
    per-stack selection (``PT.select`` with ``PT.stack_word()``: pure, no kernel launched; the provider's default selection where the
    running stack has no entry).  ``unresolved:<why>`` when the provider is absent or refuses the question.  Printed on the LEVER line as ``tier_fast=``."""
    if PT is None:
        return "unresolved:%s" % (_CORE_WHY or "core_absent")
    try:
        sel = PT.select(tuple(cc), "bf16", int(c), int(c), int(n_tokens), direction, word=TIER_WORD, stack=PT.stack_word(), has_cueq=None)
        return str(sel.row)
    except Exception as e:                                   # noqa: BLE001 — a table without the entry, an older select signature: a word, never a raise
        return "unresolved:%s" % type(e).__name__


def _resolve_tier(h: Handle, engine, n_tokens) -> None:
    """Record h.tier_fast once (the design's size when known, else the size given)."""
    if h.tier_fast is None and h.cc and n_tokens:
        try:
            c = int(engine.latent_channels)
        except Exception:                                    # noqa: BLE001
            c = 256
        h.tier_fast = tier_row(h.cc, c, int(n_tokens))


def _select_row(h: Handle, tmu, dev) -> None:
    """Pick the first candidate row that serves the canary on this stack and agrees with the module's own forward; record the others by name."""
    eng = tmu._engine
    z, mask = _canary_inputs(dev, int(eng.latent_channels))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):   # the loop's precision state (fp32 master weights, bf16 pair under autocast)
        ref = z + eng(z, visibility=mask)                    # the module's own statement of pair + TriMul(pair) (whatever backend it dispatches)
    known = tuple(getattr(PT, "ROW_NAMES", ()) or ())
    for row in (h.rows or ROWS):
        if known and row not in known:                       # a core older than this row: passed over by name, the next row asked
            h.refused[row] = "not_in_core:%s" % getattr(_core, "__version__", "?")
            continue
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = _serve(h, eng, z, mask, row)
        except PT.Refusal as e:
            h.refused[row] = e.kind.split("(")[0].replace(" ", "_")
            continue
        except Exception as e:                               # noqa: BLE001 — a row whose package fails to import / launch on this stack: named, next row
            h.refused[row] = ("%s:%s" % (type(e).__name__, str(e).splitlines()[0][:60] if str(e) else "")).replace(" ", "_")
            continue
        d = float((out.float() - ref.float()).abs().max()) / max(float(ref.float().abs().max()), 1e-6)
        if not (d <= CANARY_TOL):
            h.refused[row] = "canary:%.3g" % d
            continue
        h.row, h.canary = row, "pass:%.2g" % d
        break
    torch.cuda.synchronize(dev)
    h.caches.clear(); h.wcache.clear()                       # the canary packed the real `_engine` modules' weights; dropped here (one cheap rebuild at the first served call) so enable() leaves no cache state


def enable(model: torch.nn.Module, word: str = "fast", rows: Optional[tuple] = None, n_tokens: Optional[int] = None) -> Handle:
    """Install on `model` (its main FoldingTrunk).  Never raises for a condition the docstring names: the handle says state / reason.
    ``n_tokens``: the design's complex size, for the LEVER line's ``tier_fast=`` (the provider's cells are sized; absent = the first pair's N)."""
    prev = model.__dict__.get("_ef2_nosave_handle") if hasattr(model, "__dict__") else None
    if prev is not None and prev.on:                        # idempotent: a second enable re-installs nothing (the handle's counters keep running)
        return prev
    h = Handle(word=word, rows=tuple(rows) if rows else ROWS, n_tokens=int(n_tokens) if n_tokens else None)   # rows=: a caller's own candidate list (an ablation binds one row by name); the module's order otherwise
    trunk = _main_trunk(model)
    dev = next((p.device for p in model.parameters() if p.is_cuda), None)
    if trunk is not None and len(getattr(trunk, "blocks", [])) and dev is not None and not _CORE_WHY:
        h.cc = tuple(torch.cuda.get_device_capability(dev))  # read on every card (the step-aside line names it; tier_fast is asked for it)
        try:
            _resolve_tier(h, trunk.blocks[0].tri_mul_out._engine, h.n_tokens)
        except Exception:                                    # noqa: BLE001 — a fork without the engine attribute: tier_fast stays unasked
            pass
    if trunk is None or not len(getattr(trunk, "blocks", [])):
        h.reason = "no_trunk"
    elif dev is None:
        h.reason = "cpu"
    elif _CORE_WHY:
        h.reason = _CORE_WHY
    elif word != "fast":
        h.reason = "word:%s(only the fast class routes a forward-only row; exact keeps the compiled block intact)" % word
    else:
        if h.cc not in ENGAGE_CC:
            h.reason = "cc_no_gain:sm_%d%d" % h.cc
        elif getattr(trunk, "_agk_cfg", None) is None or not hasattr(trunk, "_agk_orig_forward"):
            h.reason = "agk_absent"
        elif any(p.requires_grad for p in trunk.parameters()):
            h.reason = "params_require_grad"
    if h.reason:
        model.__dict__["_ef2_nosave_handle"] = h
        return h
    tmus = [(blk, tmu) for blk in trunk.blocks if isinstance(blk, C.PairUpdateBlock) for tmu in (blk.tri_mul_out, blk.tri_mul_in)]
    _select_row(h, tmus[0][1], dev)
    if h.row is None:
        h.reason = "no_row:" + ",".join("%s:%s" % kv for kv in h.refused.items())
        model.__dict__["_ef2_nosave_handle"] = h
        return h
    for blk, tmu in tmus:
        if not hasattr(tmu, "_ef2_nosave_inner_forward"):
            tmu._ef2_nosave_inner_forward = tmu.forward      # ef2_trimul's bound fused forward, or the module's own
            tmu.forward = types.MethodType(_tmu_forward_nosave, tmu)
        tmu.__dict__["_ef2_nosave_handle"] = h
        tmu.__dict__["_ef2_nosave_block"] = blk
        h.n_tmu += 1
    trunk.__dict__["_agk_ckpt_fn"] = _ckpt_fn(h); h.ckpt = True
    model.__dict__["_ef2_nosave_handle"] = h
    return h


def disable(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "_ef2_nosave_inner_forward"):
            m.forward = m._ef2_nosave_inner_forward
            del m._ef2_nosave_inner_forward
            for k in ("_ef2_nosave_handle", "_ef2_nosave_block"):
                m.__dict__.pop(k, None)
        if isinstance(m, C.FoldingTrunk) and getattr(m.__dict__.get("_agk_ckpt_fn"), "_ef2_nosave", False):
            m.__dict__.pop("_agk_ckpt_fn", None)
    model.__dict__.pop("_ef2_nosave_handle", None)


def extra_bytes(model, n_tokens: int) -> float:
    """What the memory plan reserves for this lever: the no-save checkpoint keeps one recomputed block output (a bf16 pair plane) alive
    through that block's backward, which torch's checkpoint does not. 0 when off."""
    h = model.__dict__.get("_ef2_nosave_handle") if hasattr(model, "__dict__") else None
    if h is None or not h.on:
        return 0.0
    d = 256
    trunk = _main_trunk(model)
    try:
        d = int(trunk.blocks[0].tri_mul_out._engine.proj_emit.weight.shape[0])
    except Exception:                                        # noqa: BLE001
        pass
    return float(int(n_tokens) ** 2 * d * 2)


def stats(model) -> dict:
    h = model.__dict__.get("_ef2_nosave_handle") if hasattr(model, "__dict__") else None
    return {} if h is None else dict(h.stats, row=h.row, reason=h.reason, refused=dict(h.refused), canary=h.canary, word=h.word, tier_fast=h.tier_fast or "unasked")


def describe(h: Optional[Handle]) -> str:
    """One status line in the kit's LEVER style (the caller prints it; this module prints nothing)."""
    if h is None:
        return f"LEVER name={NAME} state=off"
    refused = ",".join("%s:%s" % kv for kv in h.refused.items()) or "none"
    tier = f"tier_fast={h.tier_fast or 'unasked'}"                      # what the provider's tier word resolves to here (module docstring: The tier word) — next to the row asked by name
    if not h.on:
        return f"LEVER name={NAME} state=stepped_aside reason={h.reason} refused={refused} {tier}"
    s = h.stats
    return (f"LEVER name={NAME} state=on word={h.word} row={h.row} served={s['served']} first_pass={s['first_pass']} recompute={s['recompute']} "
            f"inner={s['inner']} refused_calls={s['refused_calls']} refused={refused} canary={h.canary} tmu={h.n_tmu} cc=sm_{h.cc[0]}{h.cc[1]} {tier}")

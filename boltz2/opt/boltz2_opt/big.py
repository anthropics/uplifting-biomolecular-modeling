"""boltz2_opt.big — the engine adapter of the memory mode (`big`) on the core's memory-mode library ``opt_core.mem``:
this engine's memory levers by registry name, their hook points on the stock classes (boltz 2.2.1, ``stock/src``), the applied record, the
per-unit census and the exit gate. Applied in the worker process by ``boltz2_opt.worker_launch --attach xl`` right after the worker's own
``boltz.model.models.boltz2`` import; the switches are the mode's env row (modes.py ``_XL_ROW``: BOLTZ_XL / BOLTZ_XL_LEVERS /
BOLTZ_XL_MIN_TOKENS / BOLTZ_XL_TRANS_ROWS / BOLTZ_XL_COND_ROWS / BOLTZ_XL_COND_FP32 / PYTORCH_CUDA_ALLOC_CONF) — the kit's own words for its worker,
set by the caller (stack.child_env) and nothing else: the selection the core applies is explicit (``mem.apply(line, ctx, switches=None,
allow_partial=False)``: every lever of the line on, the row's settings through ``ctx.settings``, no census opt-out), no environment word is read
for it, and a ``BOLTZ2_BIG_*`` word in the caller's environment is refused by name before anything runs (cli.resolve_mode, _autoload.gate).
A partial census fails closed in this process (``exit_gate``) and in the caller's evidence (stack.evidence).

The line (composed through ``opt_core.mem.compose_big`` on this kit's mode table; the base is ``fast`` on one of two pair-stack bases BY NAME
(``attention_base``, modes.BIG_ATTN_BASE — chosen by measurement): ``pairtrack`` = the row carries fast's fused pair track
(BOLTZ_PAIRBLOCK + BOLTZ_TRANSITION + BOLTZ_PAIRFUSE: the layer driver serves every C=128 pair stack on one resident bf16 z; xl_trans serves the stacks
it hands back and is ``skipped`` by name — superseded_by:pairfuse@c128 — on a unit where none was), or ``flash`` = the staged flash triangle-attention
patch (BOLTZ_TRIATTN=flash) inside the stock Pairformer statements (xl_trans over the stock Transition); at n_gpu > 1 off the pair track the
row-sharded trunk owns the pair stacks (``rowpair``). A row with neither is refused by name; the graph sampler and the DiT hoist are the base levers left out:

    big = fast + [expandable_segments, xl_trans, xl_cond, xl_free, relpos_lazy] - [graph_sampler, dit_hoist]

``BOLTZ_XL_LEVERS`` names the unit-scope levers of the line by token (``trans`` / ``cond`` / ``free`` / ``relpos``); ``expandable_segments`` is the
core's process-scope allocator lever (``opt_core.mem.allocator``), in force through the row's ``PYTORCH_CUDA_ALLOC_CONF`` and confirmed by the
allocator read-back.

Levers (registry names; family / exactness label — what each rebinds):
  expandable_segments  allocator / bitwise   ``opt_core.mem.allocator``: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for the worker process (fragmentation
                                             only, no arithmetic); the record confirms ``allocator_settings.expandable_segments`` once CUDA is up.
  xl_trans             chunk / bitwise       ``boltz.model.layers.transition.Transition.forward`` (transition.py:47): a pair-shaped ``[B, N, N, C]`` input at
                                             an unchunked eval-mode call site (pairformer.py transition_z, the confidence stack, the conditioner's transitions)
                                             is evaluated BOLTZ_XL_TRANS_ROWS rows at a time by ``opt_core.mem.chunk.pair_transition_chunked`` over the STOCK
                                             forward per row block (norm -> silu(fc1)*fc2 -> fc3, transition.py:61-65; fp32 inputs refuse by name); any other
                                             call form (upstream's hidden-dim chunk_size, a non-pair input, training) is the stock forward, counted
                                             ``trans_stock_calls`` (a declared guard, not a fallback).
  xl_cond              chunk / bitwise       ``boltz.model.modules.diffusion_conditioning.DiffusionConditioning.forward`` (diffusion_conditioning.py:83-116): the
                                             pairwise conditioner (encodersv2.PairwiseConditioning.forward: cat -> init projection -> transitions with residual)
                                             evaluated per row block (``chunk.chunk_rows``, row-local statements); the atom encoder and the atom biases as
                                             stock; the 24 per-layer token pair biases written into ONE tensor — fp32 under BOLTZ_XL_COND_FP32=1, assembled from
                                             the autocast pieces by the denoiser's own widening cast (diffusionv2.py:159 ``.float()`` then aliases instead of
                                             copying: the one intentional dtype move of the line, exact).
  xl_free              setting / bitwise     ``boltz.model.modules.confidencev2.ConfidenceModule.forward`` entry: the conditioning outputs registered by the
                                             conditioner (q, c, to_keys, the atom / token biases — dead once sampling returned; tensors aliasing the trunk's
                                             s / z / feats are excluded by storage pointer) have their storages released (``untyped_storage().resize_(0)``);
                                             no value changes, an exception propagates.
  relpos_lazy          chunk / bitwise       ``boltz.model.modules.encodersv2.RelativePositionEncoder.forward``: the fp32 ``[B, N, N, c_z]`` rel-pos encoding is
                                             released right after its one consumer ran (a forward hook on the trunk's / the confidence module's ``token_bonds``
                                             Linear: boltz2.py:424-426, confidencev2.py:166-168) and recomputed by the SAME module call from ``feats`` where the
                                             conditioner reads it (a deterministic one_hot + Linear: the same values); 4*N^2*c_z bytes across sampling and
                                             confidence.

Units: one ``Boltz2.predict_step`` = one unit of the census (``opt_core.mem.record``): every unit-scope lever of the line marks ``ran`` on every
unit from its own call counts, ``xl_free`` with nothing registered on a unit is ``skipped`` with the reason; a lever absent on a unit is
``partial`` and the exit gate refuses (``exit.ok`` false in the report; the parent's stack.evidence names it). No lever disables itself: an
exception inside a levered path propagates (boltz's own handler in predict_step turns an out-of-memory into a skipped item, by name).

Contract with worker_launch: ``LEVERS`` (the lever names this adapter derives); ``apply()`` -> the applied registry names (``[]`` when the row's
switch BOLTZ_XL is not set: the launcher refuses by name); ``report()`` -> the ``xl_report`` block of the worker log (stack.evidence /
report.lever_state read ``applied`` / ``stats`` / ``env`` / ``exit``). Worker-side lines on stderr under ``[boltz2-opt big]``: the record's
ACTIVE line and one LEVER line per lever (``opt_core.mem.record.AppliedRecord.lever_lines``: strategy ids F7.chunked_eval /
F7.expandable_segments); the parent's ``[boltz2-opt]`` LEVER lines are report.lever_lines over the evidence.
``BOLTZ2_MEMTRACE=1`` records allocator counters at the model's phase boundaries into the report (a diagnostic, not a lever).
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Optional

PREFIX = "BOLTZ2"                                   # the record's stem (the core's Ctx.prefix; recorded on the ACTIVE line, no environment word derives from it)
TAG = "boltz2-opt big"                            # the worker-side lines of this adapter (the parent's lines are report.TAG)
MEMTRACE_VAR = "BOLTZ2_MEMTRACE"
FLASH = "flash"                                     # the BOLTZ_TRIATTN value of the fast base (the flash triangle attention in the row)

BIG_LEVERS = ("expandable_segments", "xl_trans", "xl_cond", "xl_free", "relpos_lazy")     # the line, in apply order (relpos_lazy after xl_cond: it recomputes inside the conditioner)
CORE_API = ("register", "apply", "undo", "compose_big", "Ctx", "Applied", "Refused", "RefusalError", "refuse")   # the memory-mode library's names this adapter uses (opt_core.mem)
REQUIRED_PRODUCERS = ("opt_core.mem.registry", "opt_core.mem.record", "opt_core.mem.compose", "opt_core.mem.chunk", "opt_core.mem.allocator")   # the core modules the memory modes import


def missing_producers() -> list:
    """The members of ``REQUIRED_PRODUCERS`` the installed core does not provide (``[]`` when the core carries them all; every member when the core is
    absent) — found by module spec, nothing heavy imported. The memory modes' gate and the worker's attach read it: a non-empty list is
    the refusal ``producer_missing:<modules>``."""
    import importlib.util
    out = []
    for m in REQUIRED_PRODUCERS:
        try:
            found = importlib.util.find_spec(m) is not None
        except (ModuleNotFoundError, ValueError):                       # a parent package absent (opt_core / opt_core.mem)
            found = False
        if not found:
            out.append(m)
    if not out:
        from opt_core import mem
        out = [f"opt_core.mem:{n}" for n in CORE_API if not hasattr(mem, n)]
    return out


def core_missing() -> Optional[str]:
    """``None`` when the installed core carries the memory-mode library this adapter is written on (``REQUIRED_PRODUCERS``);
    else the refusal reason in the kits' vocabulary — ``core_missing:opt_core …`` (no core importable) or ``producer_missing:<modules> …``
    (a core without them): the worker's attach refuses the memory modes by name with it (`[boltz2-opt attach] REFUSED xl: ModuleNotFoundError:
    producer_missing:… `, exit 3), never a silent stock run and never a second implementation here."""
    import importlib.util
    try:
        core = importlib.util.find_spec("opt_core")
    except (ModuleNotFoundError, ValueError):
        core = None
    if core is None:
        return "core_missing:opt_core — the shared core is not importable; boltz2_opt.big imports opt_core >= 0.4.0 (the memory-mode library opt_core.mem)"
    missing = missing_producers()
    if missing:
        import opt_core
        return (f"producer_missing:{','.join(missing)} — boltz2_opt.big imports opt_core >= 0.4.0 (the memory-mode library); the installed opt_core is "
                f"{getattr(opt_core, '__version__', '?')}; this package pins opt_core 0.4.1 ([tool.opt_core])")
    return None


LEVERS = BIG_LEVERS                                                                        # worker_launch's / stack.attachment_problems' probe: the lever names this adapter derives
DROP = ("graph_sampler", "dit_hoist")                                                        # base levers left out (resident graph pool / hoist cache) — unless the row carries them:


def drops_for(env) -> tuple:
    """The base levers the composed line names as left out, following the row: when the row carries
    the DiT hoist's word (BOLTZ_DIT_HOIST) the hoist is not a dropped lever of the line."""
    env = env or {}
    return tuple(d for d in DROP if not (d == "dit_hoist" and str(env.get("BOLTZ_DIT_HOIST", "")).strip().lower() not in ("", "0", "off", "false")))
LINES = {"fast": ("big", BIG_LEVERS)}    # base mode -> (memory mode, its full lever line): the memory mode composes on fast
XL_NAMES = {"trans": "xl_trans", "cond": "xl_cond", "free": "xl_free", "relpos": "relpos_lazy"}   # BOLTZ_XL_LEVERS tokens -> registry names (the row selects the unit levers)
DEFAULTS = {"xl_trans": {"rows": 256}, "xl_cond": {"rows": 256, "fp32": 1}}
STATS = {                                           # report()["stats"] keys <- the adapter's per-process call totals (stack.XL_ACTIVITY reads the first of each lever)
    "trans_rowchunked_calls": "xl_trans", "trans_stock_calls": "xl_trans_stock",
    "cond_rowchunked_calls": "xl_cond", "cond_stock_calls": "xl_cond_stock",
    "free_events": "xl_free", "free_tensors": "xl_free_tensors", "free_bytes": "xl_free_bytes", "free_empty": "xl_free_empty",
    "free_registered": "xl_free_registered", "free_skipped_aliased": "xl_free_skipped_aliased",
    "relpos_lazy_released": "relpos_lazy", "relpos_lazy_bytes": "relpos_lazy_bytes", "relpos_lazy_recomputed": "relpos_lazy_recompute",
}

_LOCK = threading.Lock()
_S = {"registered": False, "record": None, "ctx": None, "orig": {}, "settings": {}, "counts": {}, "totals": {}, "pending_free": [], "free_log": [],
      "lazy_relpos": None, "lazy_bytes": 0, "relpos_recompute": False, "trunk_rel_pos": None, "n_items": 0, "unit": None, "min_tokens": 0,
      "memtrace": [], "memtrace_on": False, "mode": None, "base": None, "row_levers": (), "chunk_entries": []}


def _mem():
    """``opt_core.mem`` imported at first use (module level imports the standard library only: modes.py / stack.py import this file on CPU boxes)."""
    from opt_core import mem
    return mem


def _cnt(key: str, n: int = 1) -> None:
    c = _S["counts"]
    c[key] = c.get(key, 0) + n


# ---------------------------------------------------------------------------------------------------------------------------------------
# hook-point helpers
# ---------------------------------------------------------------------------------------------------------------------------------------
def _pair_ok(x) -> bool:
    """A pair-shaped [B, N, N, C] tensor at or above the size gate, outside autograd bookkeeping."""
    import torch
    if not torch.is_tensor(x) or x.dim() != 4 or x.shape[1] != x.shape[2] or int(x.shape[1]) < int(_S.get("min_tokens", 0) or 0):
        return False
    if torch.is_grad_enabled() and x.requires_grad:
        return False
    return True


def _storage_ptr(t) -> Optional[int]:
    try:
        return int(t.untyped_storage().data_ptr())
    except Exception:  # noqa: BLE001
        return None


def _nbytes(t) -> int:
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:  # noqa: BLE001
        return -1


def _chunk_record(lever: str, site: str, exact: str, reason: str, **details) -> None:
    """The chunk module's per-call record sink (``opt_core.mem.chunk`` ``record=``): entries folded per unit by ``chunk.summary`` at unit end."""
    _S["chunk_entries"].append({"lever": lever, "site": site, "exact": exact, "reason": reason, **details})


# ---------------------------------------------------------------------------------------------------------------------------------------
# xl_trans — row-chunked pair Transition
# ---------------------------------------------------------------------------------------------------------------------------------------
def _trans_forward(self, x, chunk_size=None):
    if chunk_size is not None or self.training or not _pair_ok(x):
        _cnt("xl_trans_stock")                                     # the call forms the lever does not serve (upstream's own chunking / non-pair / training)
        return _S["orig"]["Transition.forward"](self, x, chunk_size)
    chunk = _mem().chunk
    orig = _S["orig"]["Transition.forward"]
    out = chunk.pair_transition_chunked(x, lambda blk: orig(self, blk, None), chunk=int(_S["settings"]["xl_trans"]["rows"]), record=_chunk_record,
                                        fp32="refuse", row_dim=1, lever="xl_trans")     # an exception propagates: a defect, never a fallback
    _cnt("xl_trans")
    return out


def _xl_trans_applies(ctx):
    ctx.require("xl_trans", "transition_cls")
    return None


def _xl_trans_apply(ctx):
    mem = _mem()
    cls = ctx.hook("xl_trans", "transition_cls")
    rows = ctx.setting("xl_trans", "rows", DEFAULTS["xl_trans"]["rows"], cast=int)
    if rows < 1:
        raise mem.RefusalError(mem.refuse("xl_trans", "settings.rows", f"rows must be >= 1, got {rows}"))
    _S["settings"]["xl_trans"] = {"rows": rows}
    _S["orig"]["Transition.forward"] = cls.forward
    cls.forward = _trans_forward

    def undo():
        cls.forward = _S["orig"]["Transition.forward"]
    return mem.Applied("xl_trans", settings={"rows": rows}, sites=("boltz.model.layers.transition.Transition.forward",), undo=undo)


# ---------------------------------------------------------------------------------------------------------------------------------------
# xl_cond — row-chunked DiffusionConditioning; xl_free registration; relpos_lazy recompute
# ---------------------------------------------------------------------------------------------------------------------------------------
def _relpos_for_cond(relative_position_encoding, feats):
    """The trunk's rel-pos encoding for the conditioner: the tensor as passed, or — when relpos_lazy released it — recomputed from feats."""
    if _nbytes(relative_position_encoding) != 0:
        return relative_position_encoding
    rp = _S.get("trunk_rel_pos")
    if rp is None:
        raise RuntimeError("relpos_lazy: the trunk's rel_pos module is unknown at the conditioner (predict_step not seen)")
    _S["relpos_recompute"] = True
    try:
        out = rp(feats)
    finally:
        _S["relpos_recompute"] = False
    _cnt("relpos_lazy_recompute")
    return out


def _cond_rows(self, s_trunk, z_trunk, relative_position_encoding, feats, rows: int, fp32: bool):
    import torch
    chunk = _mem().chunk
    pc = self.pairwise_conditioner
    rel = _relpos_for_cond(relative_position_encoding, feats)

    def conditioner_rows(blk, i0, i1):                             # encodersv2.PairwiseConditioning.forward on one row block (:211-217)
        zr = pc.dim_pairwise_init_proj(torch.cat((blk, rel[:, i0:i1]), dim=-1))
        for transition in pc.transitions:
            zr = transition(zr) + zr
        return zr
    z = chunk.chunk_rows(conditioner_rows, z_trunk, 1, rows, exact="bitwise", reason="row-local: cat + LayerNorm + Linear + transitions per position",
                         record=_chunk_record, with_offsets=True, lever="xl_cond", site="pairwise_conditioner")
    del rel
    q, c, p, to_keys = self.atom_encoder(feats=feats, s_trunk=s_trunk, z=z)     # diffusion_conditioning.py:95-109 as stock
    atom_enc_bias = torch.cat([layer(p) for layer in self.atom_enc_proj_z], dim=-1)
    atom_dec_bias = torch.cat([layer(p) for layer in self.atom_dec_proj_z], dim=-1)
    L = len(self.token_trans_proj_z)
    tb = None
    dt = src_dt = None
    Hh = 0
    for li, layer in enumerate(self.token_trans_proj_z):           # :111-114 — the per-layer pair biases into ONE tensor (no list + cat)
        if tb is None:
            probe = layer(z[:, :1])
            Hh = int(probe.shape[-1]); src_dt = probe.dtype
            dt = torch.float32 if (fp32 and src_dt in (torch.bfloat16, torch.float16)) else src_dt
            tb = torch.empty(tuple(z.shape[:3]) + (L * Hh,), dtype=dt, device=probe.device)
            _S["counts"]["xl_cond_bias_dtype"] = f"{str(dt).replace('torch.', '')} from {str(src_dt).replace('torch.', '')}"
            del probe
        fn = layer if dt == src_dt else (lambda blk, layer=layer: layer(blk).to(dt))   # the widening cast per block = the denoiser's own .float() of the same values
        chunk.chunk_rows(fn, z, 1, rows, exact="bitwise", reason="row-local: LayerNorm + Linear per position (+ the exact widening cast to the storage dtype)", record=_chunk_record,
                         out=tb[..., li * Hh:(li + 1) * Hh], lever="xl_cond", site="token_trans_bias")
    _cnt("xl_cond")
    return q, c, to_keys, atom_enc_bias, atom_dec_bias, tb


def _register_free(outs, protect, extra) -> None:
    import torch
    prot = set()

    def walk(o):
        if torch.is_tensor(o):
            p = _storage_ptr(o)
            if p is not None:
                prot.add(p)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(list(protect))
    cands = []
    for t in list(outs) + list(extra):
        if torch.is_tensor(t) and t.is_cuda:
            p = _storage_ptr(t)
            if p is None or p in prot:
                _cnt("xl_free_skipped_aliased")
                continue
            cands.append(t)
    with _LOCK:
        _S["pending_free"] = cands
    _cnt("xl_free_registered", len(cands))


def _cond_forward(self, s_trunk, z_trunk, relative_position_encoding, feats):
    import torch
    st = _S["settings"].get("xl_cond")
    if st is not None and _pair_ok(z_trunk) and not self.training:
        outs = _cond_rows(self, s_trunk, z_trunk, relative_position_encoding, feats, int(st["rows"]), bool(st["fp32"]))
    else:
        _cnt("xl_cond_stock")                                            # the stock conditioner (xl_cond off / a call form it does not serve); a rel-pos encoding relpos_lazy released is recomputed first
        outs = _S["orig"]["DiffusionConditioning.forward"](self, s_trunk, z_trunk, _relpos_for_cond(relative_position_encoding, feats), feats)
    if "xl_free" in _S["settings"] and torch.is_tensor(z_trunk) and z_trunk.is_cuda and not self.training and not torch.is_grad_enabled():
        _register_free(outs, protect=(s_trunk, z_trunk, feats), extra=(relative_position_encoding,))
    return outs


def _xl_cond_applies(ctx):
    ctx.require("xl_cond", "cond_cls")
    return None


def _xl_cond_apply(ctx):
    mem = _mem()
    cls = ctx.hook("xl_cond", "cond_cls")
    rows = ctx.setting("xl_cond", "rows", DEFAULTS["xl_cond"]["rows"], cast=int)
    fp32 = ctx.setting("xl_cond", "fp32", DEFAULTS["xl_cond"]["fp32"], cast=int)
    if rows < 1 or fp32 not in (0, 1):
        raise mem.RefusalError(mem.refuse("xl_cond", "settings", f"rows must be >= 1 and fp32 0|1, got rows={rows} fp32={fp32}"))
    _S["settings"]["xl_cond"] = {"rows": rows, "fp32": fp32}
    _S["orig"]["DiffusionConditioning.forward"] = cls.forward
    cls.forward = _cond_forward

    def undo():
        cls.forward = _S["orig"]["DiffusionConditioning.forward"]
    return mem.Applied("xl_cond", settings={"rows": rows, "fp32": fp32}, sites=("boltz.model.modules.diffusion_conditioning.DiffusionConditioning.forward",), undo=undo)


# ---------------------------------------------------------------------------------------------------------------------------------------
# xl_free — release the dead conditioning outputs at the confidence module's entry
# ---------------------------------------------------------------------------------------------------------------------------------------
def _do_free(where: str) -> None:
    with _LOCK:
        pend = list(_S["pending_free"])
        _S["pending_free"] = []
    if not pend:
        _cnt("xl_free_empty")
        return
    nbytes = n = 0
    for t in pend:
        st = t.untyped_storage()                                     # an exception propagates: a defect, never a quiet miss
        nb = int(st.nbytes())
        if nb > 0:
            st.resize_(0)
            nbytes += nb
            n += 1
    _cnt("xl_free")
    _cnt("xl_free_tensors", n)
    _cnt("xl_free_bytes", nbytes)
    _S["free_log"].append({"where": where, "tensors": n, "GiB": round(nbytes / 2**30, 3), "unit": _S["unit"]})


def _conf_forward(self, *a, **k):
    _do_free("ConfidenceModule.forward")
    return _S["orig"]["ConfidenceModule.forward"](self, *a, **k)


def _xl_free_applies(ctx):
    ctx.require("xl_free", "confidence_cls", "cond_cls")
    return None


def _xl_free_apply(ctx):
    mem = _mem()
    cls = ctx.hook("xl_free", "confidence_cls")
    _S["settings"]["xl_free"] = {}
    _S["orig"]["ConfidenceModule.forward"] = cls.forward
    cls.forward = _conf_forward
    notes = []
    if "xl_cond" not in _S["settings"]:                                   # the registration point is the conditioner hook: install it over the stock forward
        cond = ctx.hook("xl_free", "cond_cls")
        _S["orig"].setdefault("DiffusionConditioning.forward", cond.forward)
        cond.forward = _cond_forward
        notes.append("registration hook installed on DiffusionConditioning.forward (xl_cond not in the line: the conditioner itself runs stock)")

    def undo():
        cls.forward = _S["orig"]["ConfidenceModule.forward"]
        if notes:
            ctx.hook("xl_free", "cond_cls").forward = _S["orig"]["DiffusionConditioning.forward"]
    return mem.Applied("xl_free", sites=("boltz.model.modules.confidencev2.ConfidenceModule.forward", "boltz.model.modules.diffusion_conditioning.DiffusionConditioning.forward (registration)"),
                       notes=notes, undo=undo)


# ---------------------------------------------------------------------------------------------------------------------------------------
# relpos_lazy — the rel-pos encoding released after its one consumer, recomputed at the conditioner
# ---------------------------------------------------------------------------------------------------------------------------------------
def _relpos_forward(self, feats):
    out = _S["orig"]["RelativePositionEncoder.forward"](self, feats)
    if not _S["relpos_recompute"] and _S["unit"] is not None:
        _S["lazy_relpos"] = out                                      # registered; released by the next token_bonds call (its one consumer ran)
    return out


def _after_token_bonds(module, inputs, output):
    t = _S.get("lazy_relpos")
    if t is None:
        return
    _S["lazy_relpos"] = None
    nb = _nbytes(t)
    if nb > 0:
        t.untyped_storage().resize_(0)
        _S["lazy_bytes"] += nb
        _cnt("relpos_lazy")
        _cnt("relpos_lazy_bytes", nb)


def _install_instance_hooks(model) -> None:
    """Once per model instance: the token_bonds Linear of the trunk and of the confidence module release the registered rel-pos encoding."""
    for mod in (getattr(model, "token_bonds", None), getattr(getattr(model, "confidence_module", None), "token_bonds", None)):
        if mod is not None and not getattr(mod, "_big_relpos_hook", False):
            mod.register_forward_hook(_after_token_bonds)
            mod._big_relpos_hook = True


def _relpos_lazy_applies(ctx):
    ctx.require("relpos_lazy", "relpos_cls", "model_cls", "cond_cls")
    if "xl_cond" not in _S["settings"] and "xl_free" not in _S["settings"]:
        return _mem().refuse("relpos_lazy", "lever.xl_cond", "relpos_lazy recomputes the encoding inside the conditioner hook: xl_cond (or xl_free's registration hook) must be applied first")
    return None


def _relpos_lazy_apply(ctx):
    mem = _mem()
    cls = ctx.hook("relpos_lazy", "relpos_cls")
    _S["settings"]["relpos_lazy"] = {}
    _S["orig"]["RelativePositionEncoder.forward"] = cls.forward
    cls.forward = _relpos_forward

    def undo():
        cls.forward = _S["orig"]["RelativePositionEncoder.forward"]
    return mem.Applied("relpos_lazy", sites=("boltz.model.modules.encodersv2.RelativePositionEncoder.forward", "Boltz2.token_bonds (forward hook)",
                                             "ConfidenceModule.token_bonds (forward hook)", "DiffusionConditioning.forward (recompute)"), undo=undo)


# ---------------------------------------------------------------------------------------------------------------------------------------
# units — one predict_step = one unit of the census; the memtrace diagnostic
# ---------------------------------------------------------------------------------------------------------------------------------------
def _trace(phase: str) -> None:
    if not _S["memtrace_on"]:
        return
    import torch
    if not torch.cuda.is_available():
        return
    _S["memtrace"].append({"unit": _S["unit"], "phase": phase, "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                           "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                           "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
                           "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)})
    torch.cuda.reset_peak_memory_stats()


def _install_memtrace(model) -> None:
    if getattr(model, "_big_memtrace", False):
        return
    model._big_memtrace = True

    def pre(name):
        return lambda m, i: _trace(f"{name}:in")

    def post(name):
        return lambda m, i, o: _trace(f"{name}:out")
    for name in ("input_embedder", "rel_pos", "template_module", "msa_module", "pairformer_module", "distogram_module", "diffusion_conditioning",
                 "confidence_module"):
        m = getattr(model, name, None)
        if m is not None:
            m.register_forward_pre_hook(pre(name)); m.register_forward_hook(post(name))
    pf = getattr(model, "pairformer_module", None)
    layers = getattr(pf, "layers", None)
    if layers is not None and len(layers):
        for i in (0, 1, len(layers) - 1):
            layers[i].register_forward_pre_hook(pre(f"pairformer.layer{i}")); layers[i].register_forward_hook(post(f"pairformer.layer{i}"))
    sm = getattr(model, "structure_module", None)
    if sm is not None and hasattr(sm, "sample") and not getattr(sm, "_big_memtrace", False):
        orig = sm.sample

        def sample(*a, **k):
            _trace("structure_module.sample:in")
            try:
                return orig(*a, **k)
            finally:
                _trace("structure_module.sample:out")
        sm.sample = sample; sm._big_memtrace = True


def _close_unit(uid: str) -> None:
    rec = _S["record"]
    counts = _S["counts"]
    for lv in rec.expected:
        if rec.scopes.get(lv) == "process":
            continue                                                 # a process-scope lever (the allocator setting) is marked on the process unit by the library
        n = counts.get(lv, 0)
        if n > 0:
            rec.mark(lv, unit=uid, detail=f"calls={n}")
        elif lv == "xl_free" and counts.get("xl_free_registered", 0) == 0 and counts.get("xl_cond", 0) == 0 and counts.get("xl_cond_stock", 0) == 0:
            rec.skip(lv, "the conditioner did not run on this unit (nothing registered to release)", unit=uid)
        elif lv == "xl_trans" and _S.get("pairfuse"):
            rec.skip(lv, "superseded_by:pairfuse@c128 (the layer driver served every C=128 pair stack of this unit on its resident z; no stack was handed back to the module forward)", unit=uid)
        # else: absent on this unit — the census names it and the exit gate refuses
    summary = _mem().chunk.summary(_S["chunk_entries"]) if _S["chunk_entries"] else None
    rec.units[uid].events.append({"counts": dict(counts), "relpos_lazy_bytes_released": _S["lazy_bytes"], "chunk_summary": summary})
    rec.unit_end(uid)
    tot = _S["totals"]
    for k, v in counts.items():
        if isinstance(v, int):
            tot[k] = tot.get(k, 0) + v
    _S["chunk_entries"] = []
    _S["counts"] = {}
    _S["lazy_bytes"] = 0
    _S["unit"] = None


def _predict_step(self, batch, batch_idx, dataloader_idx=0):
    rec = _S["record"]
    uid = f"predict_step{_S['n_items']}"
    _S["n_items"] += 1
    with _LOCK:
        _S["pending_free"] = []
    _S["lazy_relpos"] = None
    _S["trunk_rel_pos"] = getattr(self, "rel_pos", None)
    if "relpos_lazy" in _S["settings"]:
        _install_instance_hooks(self)
    if _S["memtrace_on"]:
        _install_memtrace(self)
    _S["unit"] = uid
    rec.unit_begin(uid)
    _trace("predict_step:in")
    try:
        return _S["orig"]["Boltz2.predict_step"](self, batch, batch_idx, dataloader_idx)
    finally:
        _trace("predict_step:out")
        _close_unit(uid)


# ---------------------------------------------------------------------------------------------------------------------------------------
# registration, composition, attach
# ---------------------------------------------------------------------------------------------------------------------------------------
def register_levers() -> None:
    """Register this engine's levers with the library (idempotent); the allocator lever is the library's own (opt_core.mem.allocator)."""
    if _S["registered"]:
        return
    reg = _mem().register
    reg("xl_trans", family="chunk", exact="bitwise", applies=_xl_trans_applies, preconditions=("hooks.transition_cls",), settings=("rows",), strategy="F7.chunked_eval",
        exact_reason="row blocks of the stock statement (per-position LayerNorm, M=rows*N GEMMs) through opt_core.mem.chunk.pair_transition_chunked; fp32 inputs refuse by name",
        description="row-chunked pair Transition (opt_core.mem.chunk.pair_transition_chunked over the stock forward per row block)")(_xl_trans_apply)
    reg("xl_cond", family="chunk", exact="bitwise", applies=_xl_cond_applies, preconditions=("hooks.cond_cls",), settings=("rows", "fp32"), strategy="F7.chunked_eval",
        exact_reason="row-local statements per row block; the 24 pair biases written into one tensor (FP32=1: fp32 storage of the bf16 values by the denoiser's own widening cast)",
        description="row-chunked diffusion conditioner; one token pair-bias tensor")(_xl_cond_apply)
    reg("xl_free", family="setting", exact="bitwise", applies=_xl_free_applies, preconditions=("hooks.confidence_cls", "hooks.cond_cls"), strategy="F7.chunked_eval",
        exact_reason="lifetime only: dead tensors released before the confidence module; no value changes",
        description="release the conditioning outputs dead after sampling at the confidence module's entry")(_xl_free_apply)
    reg("relpos_lazy", family="chunk", exact="bitwise", applies=_relpos_lazy_applies, preconditions=("hooks.relpos_cls", "hooks.model_cls", "hooks.cond_cls", "lever.xl_cond"),
        strategy="F7.chunked_eval",
        exact_reason="lifetime + recompute of a deterministic function of feats (one_hot + Linear) by the same module: the same values",
        description="release the rel-pos encoding after its consumer; recompute it at the conditioner")(_relpos_lazy_apply)
    _S["registered"] = True


def compose(base: str, levers=None, drop=None):
    """``(mode, base, BigLine)`` — the line for a base mode composed on this kit's mode table through the library (``levers`` = the subset
    the row selects, in the line's order; default the full line)."""
    mem = _mem()
    from opt_core import modes as core_modes
    from . import modes as kit_modes
    if base not in LINES:
        raise ValueError(f"unknown big base {base!r}: {sorted(LINES)}")
    mode, line_levers = LINES[base]
    levers = tuple(line_levers) if levers is None else tuple(levers)
    stray = [lv for lv in levers if lv not in line_levers]
    if stray:
        raise ValueError(f"levers {stray} are not in the {mode} line {line_levers}")
    table = core_modes.ModeTable(tuple(m for m in kit_modes.MODE_NAMES if m not in {v[0] for v in LINES.values()}), kit_modes.DEFAULT_MODE)
    _, line = mem.compose_big(table, base, levers, name=mode, drop=(DROP if drop is None else tuple(drop)))
    return mode, base, line


def row_settings(env) -> Optional[dict]:
    """The mode's row (modes._XL_ROW) read into the adapter's base, lever subset and settings; ``None`` when BOLTZ_XL is not 1 (the
    mode's attach switch absent: nothing to apply). BOLTZ_XL_LEVERS selects the unit levers by token (XL_NAMES); the base is ``fast`` (the row
    carries one of the two pair-stack bases, modes.BIG_ATTN_BASES; a row with neither is refused by name); the settings are the row's (DEFAULTS with the row's
    BOLTZ_XL_* values), read by the levers through ctx.setting. Refuses by name on a token this adapter does not derive."""
    if env.get("BOLTZ_XL", "") != "1":
        return None
    tokens = [n for n in env.get("BOLTZ_XL_LEVERS", "").split(",") if n]
    unknown = [n for n in tokens if n not in XL_NAMES]
    if unknown:
        raise RuntimeError(f"big: BOLTZ_XL_LEVERS names {unknown}: not derived by this adapter ({sorted(XL_NAMES)})")
    attn_base = attention_base(env)                           # by name: the flash patch in the stock Pairformer statements, or fast's fused pair track; anything else refuses
    base = "fast"
    wanted = {XL_NAMES[t] for t in tokens}
    levers = tuple(lv for lv in LINES[base][1] if lv == "expandable_segments" or lv in wanted)
    settings = {k: dict(v) for k, v in DEFAULTS.items()}
    settings["xl_trans"]["rows"] = int(env.get("BOLTZ_XL_TRANS_ROWS", settings["xl_trans"]["rows"]))
    settings["xl_cond"]["rows"] = int(env.get("BOLTZ_XL_COND_ROWS", settings["xl_cond"]["rows"]))
    settings["xl_cond"]["fp32"] = int(env.get("BOLTZ_XL_COND_FP32", settings["xl_cond"]["fp32"]))
    return {"base": base, "attn_base": attn_base, "pairfuse": env.get("BOLTZ_PAIRFUSE", "") not in ("", "off"), "tokens": tokens, "levers": levers, "settings": settings,
            "min_tokens": int(env.get("BOLTZ_XL_MIN_TOKENS", "0") or 0)}


ATTN_BASES = ("flash", "pairtrack", "rowpair")     # modes.BIG_ATTN_BASES + the n_gpu > 1 line: the pair-stack base the memory line composes on, read off the row's words
ROWPAIR_ENV_P = "BOLTZ_TP"                         # rowpair.ENV_P: the worker's n_gpu word (> 1 = the row-sharded trunk owns the pair stacks, its own flash core per row block)


def attention_base(env) -> str:
    """``flash`` when the row carries the staged flash triangle-attention patch (BOLTZ_TRIATTN=flash: the stock Pairformer statements, xl_trans over
    the stock Transition); ``pairtrack`` when it carries fast's fused pair track (BOLTZ_PAIRBLOCK + BOLTZ_TRANSITION + BOLTZ_PAIRFUSE: the layer
    driver serves the C=128 pair stacks on one resident z, xl_trans serves the stacks it hands back and is superseded by name elsewhere);
    ``RuntimeError`` naming the problem for a row with neither or both (no kernels-off memory line exists)."""
    flash = env.get("BOLTZ_TRIATTN", "") == FLASH
    track = all(env.get(k, "") not in ("", "off") for k in ("BOLTZ_PAIRBLOCK", "BOLTZ_TRANSITION", "BOLTZ_PAIRFUSE"))
    try:
        P = int(str(env.get(ROWPAIR_ENV_P, "1")).strip() or "1")
    except ValueError:
        P = 1
    if P > 1 and not flash and not track:
        return "rowpair"                                        # n_gpu > 1 off the pair-track base: the fused block / driver left by name (modes.TP_DROPS); the sharded trunk's row statements carry the flash core
    if flash and track:
        raise RuntimeError("big: the row names both the flash triangle-attention patch (BOLTZ_TRIATTN=flash) and fast's fused pair track (BOLTZ_PAIRBLOCK/BOLTZ_TRANSITION/BOLTZ_PAIRFUSE): one pair-stack base by name")
    if flash:
        return "flash"
    if track:
        return "pairtrack"
    raise RuntimeError("big: the memory mode composes on fast — the row carries neither fast's flash triangle attention (BOLTZ_TRIATTN=flash) nor fast's fused pair track "
                       "(BOLTZ_PAIRBLOCK + BOLTZ_TRANSITION + BOLTZ_PAIRFUSE); no kernels-off memory line exists")


def hooks_from_boltz() -> dict:
    """The hook points on the imported stock classes (the trigger module ``boltz.model.models.boltz2`` has executed)."""
    from boltz.model.layers.transition import Transition
    from boltz.model.models.boltz2 import Boltz2
    from boltz.model.modules.confidencev2 import ConfidenceModule
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
    from boltz.model.modules.encodersv2 import RelativePositionEncoder
    return {"xl_trans": {"transition_cls": Transition},
            "xl_cond": {"cond_cls": DiffusionConditioning},
            "xl_free": {"confidence_cls": ConfidenceModule, "cond_cls": DiffusionConditioning, "model_cls": Boltz2},
            "relpos_lazy": {"relpos_cls": RelativePositionEncoder, "model_cls": Boltz2, "cond_cls": DiffusionConditioning}}


def attach(row: dict, hooks: Optional[dict] = None, environ=None):
    """Apply the row's line in this process (strict: a refusal raises ``opt_core.mem.Refused`` carrying the record), patch
    ``Boltz2.predict_step`` for the census, print the ACTIVE line and the LEVER lines. Returns the record."""
    mem = _mem()
    register_levers()
    env = os.environ if environ is None else environ
    mode, base, line = compose(row["base"], row["levers"], drop=drops_for(os.environ if environ is None else environ))
    hooks = hooks_from_boltz() if hooks is None else hooks
    ctx = mem.Ctx(prefix=PREFIX, tag=TAG, framework="torch", hooks=hooks, settings=row["settings"], environ=env, graphs=False,
                  extra={"mode": mode, "base": base, "attn_base": row.get("attn_base", "flash"), "line": line.describe(), "row_tokens": list(row["tokens"]), "min_tokens": row["min_tokens"]})
    _S.update({"mode": mode, "base": base, "attn_base": row.get("attn_base", "flash"), "pairfuse": bool(row.get("pairfuse")), "row_levers": tuple(row["levers"]),
               "memtrace_on": str(env.get(MEMTRACE_VAR, "")).strip() in ("1", "true", "on"), "settings": {}, "counts": {}, "totals": {}, "min_tokens": row["min_tokens"]})
    try:
        rec = mem.apply(line, ctx, strict=True, switches=None, allow_partial=False)   # the selection, explicit: every lever of the line on, the row's settings (ctx.settings), no census opt-out — nothing read from the environment
    except mem.Refused as e:
        e.record.mode = mode
        sys.stderr.write(e.record.not_active_line(TAG) + "\n")
        mem.undo(e.record)
        raise
    rec.mode = mode                                                       # the record names the memory mode of this kit's table (big)
    _S["record"] = rec; _S["ctx"] = ctx
    model_cls = hooks["xl_free"]["model_cls"]
    _S["orig"]["Boltz2.predict_step"] = model_cls.predict_step
    model_cls.predict_step = _predict_step
    sys.stderr.write(rec.active_line(TAG, min_tokens=row["min_tokens"]) + "\n")
    for ln in rec.lever_lines(TAG):
        sys.stderr.write(ln + "\n")
    return rec


def apply() -> list:
    """The attach-hook entry (worker_launch.ATTACH['xl'], right after boltz.model.models.boltz2 executed): the line from the worker's environment
    row, applied strictly; returns the applied lever names (registry names) — ``[]`` when BOLTZ_XL is not set (the launcher refuses by name),
    never from a partial line (a refusal raises)."""
    if os.environ.get("BOLTZ_XL", "") != "1":
        return []
    missing = core_missing()
    if missing is not None:
        raise ModuleNotFoundError(missing)                                # worker_launch prints `[boltz2-opt attach] REFUSED xl: ModuleNotFoundError: producer_missing:… | core_missing:…`, exit 3
    row = row_settings(os.environ)
    rec = attach(row)
    return [a.lever for a in rec.applied]


def report(rc: int = 0) -> dict:
    """The exit-time ``xl_report`` block of the worker log (stack.evidence / report.lever_state read it): ``applied`` (registry names) /
    ``patched`` / ``stats`` / ``env`` / ``settings``, the library's ``record`` (manifest block), ``census`` and ``exit`` gate, the diagnostics."""
    rec = _S["record"]
    if rec is None:
        return {"applied": [], "error": "big not attached in this process", "env": {"PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}}
    environ = _S["ctx"].environ if _S["ctx"] is not None else os.environ
    verdict = rec.exit_gate(rc, allow_partial=False)                 # no census opt-out (= the apply-time value): a partial census fails closed here; the caller's evidence refuses it too (stack.evidence)
    tot = _S["totals"]
    stats = {k: tot.get(src, 0) for k, src in STATS.items()}
    es = next((a for a in rec.applied if a.lever == "expandable_segments"), None)
    effective = None if es is None else ("pending" if es.verified is None else ("true" if es.verified.get("ok") else "false"))
    return {"applied": [a.lever for a in rec.applied], "mode": _S["mode"], "base": _S["base"], "attn_base": _S.get("attn_base", "flash"), "row_levers": list(_S["row_levers"]),
            "patched": [s for a in rec.applied for s in a.sites], "stats": stats, "settings": dict(_S["settings"]),
            "env": {"PYTORCH_CUDA_ALLOC_CONF": environ.get("PYTORCH_CUDA_ALLOC_CONF"), "alloc_effective": effective, MEMTRACE_VAR: environ.get(MEMTRACE_VAR)},
            "record": rec.manifest_block(), "census": verdict.get("census"), "exit": {k: v for k, v in verdict.items() if k != "census"},
            "free_log": list(_S["free_log"]), "memtrace": list(_S["memtrace"]) if _S["memtrace_on"] else None, "n_units": _S["n_items"]}


def reset_for_tests() -> None:
    """Restore every patched attribute and clear the process state (the tests' fixture)."""
    rec = _S["record"]
    if rec is not None:
        _mem().undo(rec)
    orig = _S["orig"]
    if "Boltz2.predict_step" in orig and _S["ctx"] is not None:
        _S["ctx"].hooks["xl_free"]["model_cls"].predict_step = orig["Boltz2.predict_step"]
    _S.update({"record": None, "ctx": None, "orig": {}, "settings": {}, "counts": {}, "totals": {}, "pending_free": [], "free_log": [], "lazy_relpos": None,
               "lazy_bytes": 0, "relpos_recompute": False, "trunk_rel_pos": None, "n_items": 0, "unit": None, "min_tokens": 0, "memtrace": [],
               "memtrace_on": False, "mode": None, "base": None, "row_levers": (), "chunk_entries": []})

"""rowpair_xfold — the af3_torch ADAPTER of the shared core's row-sharded pair representation (``opt_core.mem.rowpair`` 0.4.3): what
``--mode big --n_gpu P`` (P > 1) runs in every rank of the model process (forward.py spawns the ranks through
``opt_core.mem.rowpair.launch.run_sharded``; users never type torchrun). Loaded BY PATH inside the model process (torch venv), never
imported by the wrapper package: it imports torch and xfold. At ``--n_gpu 1`` nothing here runs — every entry point refuses a P = 1 /
replicated layout by name (``opt_core.mem.rowpair.dist.require_sharded``) and the engine's own statements run unchanged.

Layout. xfold is unbatched: the pair representation is ``z[N, N, C]``, rows first. Rank q holds rows ``[r0_q, r1_q)`` of z — the core's
aligned row partition ``dist.Layout`` (``align``, default ``ALIGN``, handed to the core as ``ROWPAIR_CHUNK_ALIGN``): the trunk pair ``[R, N, 128]`` (bf16 under
the kit's autocast, fp32 between recycling iterations exactly as the engine casts it), the confidence head's per-sample pair ``[R, N, 128]``
(fp32, the engine's dtype), the conditioned diffusion pair ``z_cond [R, N, 128]`` and every ``[N, N]``-shaped head output as ``[R, N]`` rows.
Nothing ``N x N x c`` is ever whole on a rank. REPLICATED BY DESIGN (named in the schedule census ``xfold_replicated``): the single
representation ``[N, 384]``, ``target_feat [N, 447]``, the MSA representation ``[S_msa, N, 64]``, every per-token feature, every atom
tensor (atom cross-attention windows are sequence-local), the diffusion positions, the gathered triangle-attention bias ``[N, N, 4]`` per
attention call, and on rank 0 only the four ``[N, N]`` fp32 confidence matrices + the distogram contact map per sample
(``conf_full_matrices_rank0``: AlphaFold 3 writes full PAE / PDE / contact matrices and computes pTM / ipTM from them in its CPU
post-processing, so they are model OUTPUTS: assembled on rank 0's HOST column block by column block — never whole on any device).

What binds where (engine statement -> core seam; every callable below is the xfold module's OWN layers evaluated on a row block — a
per-(i, j) map or a GEMM whose M is the row count — so each element equals the dense statement's; the schedules, the communication and
every block size are the core's):
  PairformerBlock (trunk 48, MSA-stack 4, template 2, confidence 4)  -> pairstack.pair_block_ via PairBlockFns:
      TriangleMultiplication      trimul.TriMulFns(proj = LN -> projection * mask * sigmoid(gate), interleaved a|b channels; out = center_norm ->
                                  output_projection; gate = sigmoid(gating_linear(LN)))  [the kit's fused FPF kernel assumes a square pair and is
                                  REPLACED by the core's row schedule under P > 1 (gate word ``trimul=replaced:rowpair_rows``): with the trimul lever
                                  on, ``opt_core.mem.rowpair.trimul_fused`` runs its fused row kernels (K1 projection / K3 epilogue per tile) over
                                  these torch callables — their named fallback per unit, and the statements below the core's min_tokens pair size]
      GridSelfAttention           triatt.TriAttFns(ln = act_norm, bias = pair_bias_projection, attend = the kit's flash triangle attention on the
                                  row batch with the WHOLE gathered bias (its own eager statement when the flash lever is off / its gate says no));
                                  the ending node (OpenFold3 weight layout: bias from the transposed pair) = the core's starting form on rows of z^T
      Transition                  row-local (the kit's fused LN+SwiGLU kernel when its lever is on)
      attention-pair-bias (single) transition.apb_local_queries: query rows local, pair-logit rows from the local pair rows, keys/values all rows,
                                  one all_gather of the updated single rows per block
  DistogramHead                   -> heads.sym_logit_rows(half_logits) -> softmax . is_contact_bin * pair-mask rows -> per_row_outputs_to_rank0
  ConfidenceHead (per sample)     -> heads.embed_rows(pair.to(dtype) + target-feat outer sum + distogram_feat_project(dgram rows * mask rows)) —
                                  the trunk rows read from the item's trunk-shard plan (heads.ZTrunkPlan: parked on the leased pinned host from the
                                  roll-out entry, retired at the last pass's embed; the distogram head ran before the roll-out: forward_impl);
                                  4 x pair_block_; PAE = heads.logit_rows -> softmax -> expected-error rows + TM-adjusted rows (global / interface);
                                  PDE = heads.sym_logit_rows -> expected-error rows; the [N, N] matrices via confidence.per_row_outputs_to_rank0;
                                  pLDDT / experimentally-resolved = the dense statements on the replicated single (every rank)
  DiffusionHead                   -> the trunk shard parked (ZTrunkPlan.park_now); diffusion.pair_cond_rows (conditioning + 2 transitions, once per
                                  roll-out, rows from the park) -> z_cond rows; the pair-bias cache on iff it fits the agreed free bytes (_bias_cache_rule);
                                  diffusion.PairBiasCache x 24 of pair_bias_rows; diffusion.diffusion_transformer_sharded (24 DiTBlockFns: AdaLN,
                                  k|v of all rows, attention of local query rows with local bias rows, AdaLN-zero epilogue + conditioned transition
                                  on local rows, one all_gather per block); the atom-cross-attention encoder's trunk-pair term via diffusion.band_plan
                                  + pair_band_rows + band_lookup (once per roll-out); single conditioning, atom encoder/decoder, positions replicated;
                                  the sampler loop, the noise schedule and every random draw are the engine's (torch RNG seeded identically on every
                                  rank; diffusion.sync_replicated proves the positions agree)
  Evoformer (trunk entry, recycling, relative encoding, bonds, template embedding, MSA module, Pairformer stack) -> trunk.run_trunk_sharded with
                                  the row statements of trunk / template / msa: ``run_trunk_sharded_xfold``.
Kit levers under P > 1: bf16w / triattn (flash) / transition (fused) / apb (SDPA on local query rows) / opm (its kernels per output row block of
the rank: opm_rows_kernel_fn) COMPOSE; trimul (FPF) is REPLACED by the core contraction; stepgraph is OFF (collectives inside a captured graph); hoist is REPLACED by the
per-roll-out z_cond rows + bias cache + banded encoder statics; DTK FusedDiT is SKIPPED by name (a whole-transformer fusion cannot serve local
query rows); compile composes (row-local modules). forward.py prints one LEVER line per name with these states.

Guarantee: tier 2 (``big`` never claims exact): the per-element arithmetic is the engine's, GEMM launch shapes differ in M, reductions inside
the fused kernels may reorder. The R1 guards (features, RNG state, diffusion positions bitwise identical across ranks) are always on.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

try:                                                                     # instrumentation only (TPCENSUS lines under OPT_CORE_TP_CENSUS=1); absent in a core without the census module
    from opt_core.mem.rowpair import census as _census
except ImportError:                                                      # noqa: E722 — gates instrumentation, never work
    _census = None

ALIGN = 32                     # row-partition alignment (every rank boundary a multiple of it): the triangle-attention row batches and the a|b tile grid stay 4-aligned
ENV_ALIGN = "ROWPAIR_CHUNK_ALIGN"   # the shared core's variable: install() sets it to this engine's alignment for the core's dist.chunk_align() (the row-block unit); never read here
QBLOCK = 256                        # rows per triangle-attention / transition launch inside a shard (a positive multiple of 4)
TRIMUL_EPS = 1e-5              # both LayerNorms of xfold's TriangleMultiplication (nn.LayerNorm default): the eps the core's fused TriMul rows are handed
                               # (trimul_fns: opt_core.mem.rowpair.trimul_fused serves the seam under P > 1 whenever the trimul lever is on; the core's own
                               # ROWPAIR_TRIMUL_KERNELS=torch declines every unit by name on its F2.trimul_rows line, and pairs below its min_tokens run the
                               # eager callables, reason below_gate; every decline reason of that line is one of the core's documented gates — outputs
                               # those of the eager statements — and where the fused rows cannot run at all the core raises RowpairRefused naming
                               # `--mode off` / ROWPAIR_TRIMUL_KERNELS=torch: the item fails by name, the pred exits non-zero)

def _fresh_state() -> dict:
    return {"P": 1, "rank": 0, "installed": False, "layout": None, "K": None, "align": ALIGN, "seq_mask": None, "mask_shard": None,
            "bias_memo": {}, "diff_ctx": None, "seams": "all",
            "stats": {"items": 0, "pair_blocks": 0, "presharded_calls": 0, "trunk_rows": 0, "template_rows": 0, "msa_blocks_rows": 0,
                      "conf_rows": 0, "distogram_rows": 0, "zcond_rows": 0, "dit_blocks": 0, "band_rows": 0, "gathers_in_trunk": 0, "host_matrices": 0, "boundary_sync": None, "boundaries": [],
                      "levers": {}}}


class _PerRankState(threading.local):
    """The adapter's per-rank state. One model process = one rank = one thread in production (``launch.run_sharded`` spawns processes);
    the CPU tests run P ranks as threads of one interpreter (``opt_core.testing.run_ranks``), so the state is thread-local. Dict API."""

    def __init__(self):
        self.d = _fresh_state()

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v

    def __contains__(self, k):
        return k in self.d

    def get(self, k, default=None):
        return self.d.get(k, default)

    def setdefault(self, k, v):
        return self.d.setdefault(k, v)

    def update(self, *a, **kw):
        self.d.update(*a, **kw)


STATE = _PerRankState()


class XfoldTPRefused(RuntimeError):
    """A layout / lever / input this adapter does not serve under n_gpu > 1 — refused by name, never degraded silently."""


def _log(msg):
    print("[rowpair_xfold r%d] %s" % (STATE["rank"], msg), flush=True)


def _mark(stage: str, **kw):
    if _census is not None:
        _census.mark(stage, **kw) if kw else _census.mark(stage)


def _core():
    """The core modules (imported late: the model process puts --opt-core on sys.path first)."""
    from opt_core.mem.rowpair import (dist as D, shard as SH, trunk as TR, pairstack as PS, trimul as TM, triatt as TA, transition as TN,
                                      msa as MS, template as TP, heads as HD, confidence as CF, diffusion as DF, bcast as BC, evidence as EV,
                                      RowpairRefused)
    return dict(D=D, SH=SH, TR=TR, PS=PS, TM=TM, TA=TA, TN=TN, MS=MS, TP=TP, HD=HD, CF=CF, DF=DF, BC=BC, EV=EV, Refused=RowpairRefused)


# ============================================================================================================================ setup
def install(model, P: int, rank: int, align: Optional[int] = None, seams: str = "all"):
    """Record P / rank / the kit's kernel module, the row alignment (``align`` or ``ALIGN``, set for the core as ``ROWPAIR_CHUNK_ALIGN``) and the seam set the
    CALLER drives (``seams``: ``'all'`` = trunk + heads + diffusion, this kit's own path; ``'heads'`` = the heads + diffusion drivers only, for a
    caller whose trunk runs elsewhere — the item exit gate then requires ``SEAMS_REQUIRED_HEADS``; any other value refused by name). Nothing
    is rebound: the sharded drivers below call the xfold modules' own layers. Refused by name: a non-OpenFold3 weight layout (the ending-node
    bias and the bond symmetrisation follow ``xfold.of3.OF3``), a model in training mode (per-rank dropout draws), P < 2."""
    import xfold.of3 as OF3
    C = _core()
    if seams not in SEAM_SETS:
        raise C["Refused"](f"rowpair_xfold.install: seams={seams!r}: one of {sorted(SEAM_SETS)}")
    if int(P) < 2:
        raise C["Refused"](f"rowpair_xfold.install: P={P}: the adapter installs nothing at n_gpu=1 (the engine's single-GPU statements run)")
    if not getattr(OF3, "OF3", False):
        raise C["Refused"]("rowpair_xfold: xfold.of3.OF3 is False: the row-sharded statements follow the OpenFold3 weight layout (ending-node bias "
                           "from the transposed pair, symmetric bond contacts); the AlphaFold3-layout variant is not wired")
    if getattr(model, "training", False):
        raise C["Refused"]("rowpair_xfold: n_gpu>1 requires model.eval() (a module in training mode draws per-rank dropout masks)")
    a = int(align) if align else ALIGN
    os.environ[ENV_ALIGN] = str(a)                                       # the core's chunk_align() reads it (iter_row_blocks unit)
    _placement_defaults()                                                # the kit's placement words under P > 1 (XP_PLACEMENT_DEFAULTS; an exported value wins)
    try:                                                                 # per-rank CPU threads capped to cores / ranks (ROWPAIR_RANK_THREADS; a core without the call: nothing to cap)
        from opt_core.mem.rowpair import dist as _D
        if hasattr(_D, "apply_rank_threads"):
            STATE["stats"]["rank_threads"] = str(_D.apply_rank_threads())
    except Exception as e:                                               # noqa: BLE001 — a resource hint, never a reason to refuse the item
        STATE["stats"]["rank_threads"] = f"skipped:{type(e).__name__}"
    STATE.update(P=int(P), rank=int(rank), align=a, K=getattr(model, "_af3t_kernels", None), installed=True, C=C, seams=seams)
    _log("installed P=%d rank=%d align=%d seams=%s (no class rebinding; kernels module %s)" % (int(P), int(rank), a, seams, "present" if STATE["K"] else "absent"))
    return STATE


XP_PLACEMENT_DEFAULTS = (                                            # the kit's placement of the row shards under P > 1: the support library's words, set unless the environment
    ("ROWPAIR_CONF_PARK_ZTRUNK", "1"),                                   #   exports them — the fp32 trunk shard parked on the pinned host through the roll-out and the confidence passes
    ("ROWPAIR_TRANSPOSE_INPLACE", "1"),                                  #   (heads.ZTrunkPlan: run_diffusion_sharded / run_confidence_sharded); the ending attention's transposes in the
    ("ROWPAIR_PARK_ZINIT", "recompute"),                                 #   shard's own storage (pairstack: one peer block in flight, not a second shard + all-at-once staging); the z_init
    ("ROWPAIR_TRIATT_ROWBLOCK", None),                                   #   shard parked on the pinned host between recycles (trunk.ShardPark); the triangle-bias LayerNorm pass in
    # ROWPAIR_DIFF_BIAS_CACHE(_GB) stay unset: the kit's free-memory rule (_bias_cache_rule) decides per roll-out; an exported value wins  #   QBLOCK-row blocks (triatt.triangle_bias_rows) — None here = str(QBLOCK) at install; the roll-out's pair-bias
    ("ROWPAIR_RANK_THREADS", "auto"),                                    #
    ("ROWPAIR_HOST_SLAB", "lease"),                                      #   cache budget from the free bytes AGREED across ranks at roll-out entry (diffusion.DiffusionSchedule: a fixed-GB
)                                                                        #   cap turns the cache off above some N and multiplies the sampler's time) ; per-rank CPU threads = cores / ranks
                                                                         #   (dist.apply_rank_threads). All are data placement / resource words: identical outputs (the
                                                                         #   library's contract for each word); the SCHEDULE census names what each item did.


_KIT_EXPORTED = {}                                                       # the words THIS process's install() exported (a later install in the same process reports them as the kit's)


def _placement_defaults() -> dict:
    """Export XP_PLACEMENT_DEFAULTS into this rank process's environment where the variable is unset (an exported value — the user's or the
    launcher's — wins and is recorded as such). Returns and records ``stats['placement'] = {word: '<value>:kit' | '<value>:env'}``."""
    out = {}
    for k, v in XP_PLACEMENT_DEFAULTS:
        v = str(QBLOCK) if v is None else v
        cur = os.environ.get(k)
        if cur is None or cur == "" or _KIT_EXPORTED.get(k) == cur:
            os.environ[k] = v
            _KIT_EXPORTED[k] = v
            out[k] = f"{v}:kit"
        else:
            out[k] = f"{cur}:env"
    STATE["stats"]["placement"] = dict(out)
    return out


def _ztrunk_new(C, z_shard, passes: int):
    """The item's trunk-shard placement plan across the roll-out and ``passes`` confidence passes (heads.ZTrunkPlan; the park lever reads
    ``ROWPAIR_CONF_PARK_ZTRUNK`` — 1 under this kit unless exported otherwise). One plan per item (the trunk shard tensor is the item's)."""
    _ztrunk_close(need_device=False)
    plan = C["HD"].ZTrunkPlan(z_shard, passes=max(1, int(passes)), name="z_trunk", log=_log)
    STATE["ztrunk"] = plan
    STATE["ztrunk_pass"] = 0
    return plan


def _ztrunk_source(z_shard):
    """What the diffusion conditioning reads trunk rows from right now: the live park (``.zrows``) or the shard tensor."""
    plan = STATE.get("ztrunk")
    if plan is None or plan.closed or plan.z is not z_shard:
        return z_shard
    return plan.source()


def _ztrunk_close(need_device: bool) -> Optional[str]:
    """Finish the item's live plan, if any: the passes the caller did not run are begun and ended at once (a parked plan moves no data for
    that), the last with ``device_needed_next=need_device`` — True restores the device shard whole (the distogram head reads rows AND column
    slabs), False drops the host copy (item exit). Returns the last pass's word, None when no plan was open. A plan some pass closed already
    is left as it is: the confidence passes restore the shard at their last pass; a shard whose rows are gone is refused by name where read."""
    plan = STATE.get("ztrunk")
    if plan is None or plan.closed:
        return None
    i = int(STATE.get("ztrunk_pass") or 0)
    word = None
    while i < plan.passes:
        plan.begin(i, last_use=False, out_dtype=plan.z.dtype)          # last_use=False: never the in-place form (the distogram reads the shard after the passes)
        plan.end(i, device_needed_next=bool(need_device) and i == plan.passes - 1)
        word = plan.words[i]
        i += 1
    STATE["ztrunk_pass"] = plan.passes
    return word


def begin_item(N: int):
    """The row layout of this item (``N`` = the featurised bucket): the core's aligned partition from the live group (refused by name when a
    rank would own zero rows: ``Layout.checked``)."""
    if os.environ.get("XFOLD_TP_LEAKCHECK"):
        _leakreport("begin_item")
    C = STATE["C"] if "C" in STATE else _core()
    lay = C["D"].layout_here(int(N), B=STATE["align"], checked=True, align=STATE["align"])   # the refusing, ALIGNED form (never a grid-replicated layout)
    if lay.P != STATE["P"]:
        raise C["Refused"](f"rowpair_xfold.begin_item: the group has P={lay.P} ranks, install() recorded P={STATE['P']}")
    STATE["layout"] = lay
    STATE["stats"]["items"] += 1
    STATE["stats_at_entry"] = {k: STATE["stats"].get(k, 0) for k in SEAMS_REQUIRED}   # the exit gate compares against the entry values
    _log("item N=%d %r" % (int(N), lay))
    return lay


SEAMS_REQUIRED = ("pair_blocks", "trunk_rows", "template_rows", "msa_blocks_rows", "distogram_rows", "conf_rows", "zcond_rows", "dit_blocks",
                  "band_rows", "host_matrices")   # every bound seam's call counter: each must have RUN on the item (an ACTIVE line is a claim; the count is the proof)
SEAMS_REQUIRED_HEADS = tuple(k for k in SEAMS_REQUIRED if k not in ("trunk_rows", "template_rows", "msa_blocks_rows"))   # a caller whose trunk runs elsewhere (install(seams='heads'))
SEAM_SETS = {"all": SEAMS_REQUIRED, "heads": SEAMS_REQUIRED_HEADS}


def end_item(gate: bool = True):
    """Item exit gate (P > 1): every bound seam ran at least once on this item — refused by name otherwise (a seam silently bypassed by a
    rebinding that did not take would leave its counter at the item's entry value). Census keys: ``SEAMS_REQUIRED``."""
    entry = STATE.get("stats_at_entry") or {}
    required = SEAM_SETS[STATE.get("seams", "all")]
    idle = [k for k in required if STATE["stats"].get(k, 0) <= entry.get(k, 0)]
    STATE["layout"] = None
    _ztrunk_close(need_device=False)                                     # the item's trunk-shard plan: host copy dropped, nothing restored
    plan = STATE.get("ztrunk")
    if plan is not None and torch.is_tensor(getattr(plan, "z", None)) and plan.z.is_cuda:   # the item is over: the fp32 trunk shard's device storage is returned here, not when the
        from opt_core.mem.rowpair.shard import release_storage_          # last Python reference dies (a frame kept by a compile on this rank can hold it across items: _release_rollout)
        try:
            nb = int(plan.z.untyped_storage().nbytes()); release_storage_(plan.z)
            STATE["stats"]["ztrunk_released_gb"] = round(float(STATE["stats"].get("ztrunk_released_gb") or 0.0) + nb / 1e9, 3)
        except Exception:                                                # noqa: BLE001
            pass
    STATE["ztrunk"] = None
    if gate and idle:
        raise XfoldTPRefused(f"seams bound but never ran on this item: {idle} (seams={STATE.get('seams', 'all')}; census {dict((k, STATE['stats'].get(k, 0)) for k in required)})")


def layout():
    return STATE["layout"]


def _qblock() -> int:
    return QBLOCK


def _act_dtype(like):
    """The dtype an nn.Linear produces here: the autocast dtype when autocast is on for this device type, else ``like.dtype``."""
    dev = like.device.type
    try:
        if torch.is_autocast_enabled(dev):
            return torch.get_autocast_dtype(dev)
    except TypeError:                                                    # torch < 2.4 spelling
        if dev == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
        if dev == "cpu" and torch.is_autocast_cpu_enabled():
            return torch.get_autocast_cpu_dtype()
    return like.dtype


def _lever(name: str, state: str):
    STATE["stats"]["levers"][name] = state


def _kernel_on(name: str) -> bool:
    K = STATE["K"]
    return K is not None and name in getattr(K, "_ON", ()) and name not in getattr(K, "_DEAD", {})


# ================================================================================================================= pair-block callables
def trimul_fns(mod, outgoing: bool):
    """TriMulFns of an xfold ``TriangleMultiplication`` (eager layers). Channel layout: ``projection`` / ``gate`` emit ``2C`` interleaved
    channels, ``a_c = ch[2c]``, ``b_c = ch[2c+1]``; xfold's incoming equation ``'ckj,cki->cij'`` (``x_ij = sum_k a_kj b_ki``) is the core's
    incoming ``sum_k A_ki B_kj`` with ``A = b``, ``B = a``."""
    C = STATE["C"] if "C" in STATE else _core()
    Cz = int(mod.c_pair)
    sel_a = 0 if outgoing else 1

    def proj(z_blk, m_blk, is_a):
        x = mod.left_norm_input(z_blk)
        p = mod.projection(x)
        p.mul_(m_blk)                                                    # in place: keeps the activation dtype like the dense `projection *= mask`
        p.mul_(torch.sigmoid(mod.gate(x)))
        p = p.unflatten(-1, (Cz, 2))[..., sel_a if is_a else 1 - sel_a]
        return p if p.dtype == z_blk.dtype else p.to(dtype=z_blk.dtype)   # the operand travels the ring in the PAIR's dtype on every rank: under
                                                                         # autocast the projection is bf16 while an fp32 pair's (confidence head)
                                                                         # empty ring slab is fp32 — a rank with no sub-block at a ring step would
                                                                         # post a different byte count than its peers (F9: the x4@768 / x8@1536 hang).
                                                                         # bf16 -> fp32 is exact and the matmul runs under the same autocast policy,
                                                                         # so every element equals the previous statement's.

    def out(x):
        return mod.output_projection(mod.center_norm(x))

    def gate(z_blk):
        return torch.sigmoid(mod.gating_linear(mod.left_norm_input(z_blk)))

    stock = C["TM"].TriMulFns(proj, out, gate, Cz)
    if not _kernel_on("trimul"):                                         # `--levers` without trimul (or no kernels module): the eager callables alone, the streamed contraction's hook-less path
        return stock
    RF = trimul_fused_module()
    if RF is None:                                                       # a core without the module: named, whole process (never silently)
        _lever("trimul_rows", "fallback:core_without_trimul_fused")
        return stock
    _lever("trimul_rows", "fpf_v4")
    return RF.fused_trimul_fns(trimul_weights(mod, outgoing), stock, eps=TRIMUL_EPS)   # cells=None: the kit's cells.json via FPF_TRIMUL_V4_CELLS, as at P = 1; the core decides per unit (below_gate / env torch / shape) and counts it


def trimul_fused_module():
    """``opt_core.mem.rowpair.trimul_fused`` when the staged core carries it, else None (the adapter then keeps the torch callables by name)."""
    try:
        from opt_core.mem.rowpair import trimul_fused as RF
    except ImportError:
        return None
    return RF if hasattr(RF, "fused_trimul_fns") else None


def trimul_weights(mod, outgoing: bool) -> dict:
    """The ten canonical TriMul tensors of an xfold ``TriangleMultiplication`` keyed by ``opt_core.trimul.WEIGHT_KEYS`` — the vocabulary the
    P = 1 fpf_v4 pack takes (af3_kernels._trimul_weights packs the same tensors): ``projection`` / ``gate`` rows are interleaved a|b
    (``a = rows[0::2]``, ``b = rows[1::2]``); xfold's incoming equation (``'ckj,cki->cij'``) is the generic incoming form with a and b SWAPPED.
    No biases anywhere (xfold's TriMul linears are bias-free)."""
    Pw, Gw = mod.projection.weight.detach(), mod.gate.weight.detach()
    a_p, b_p, a_g, b_g = Pw[0::2].contiguous(), Pw[1::2].contiguous(), Gw[0::2].contiguous(), Gw[1::2].contiguous()
    if not outgoing:
        a_p, b_p, a_g, b_g = b_p, a_p, b_g, a_g
    return dict(ln_in_w=mod.left_norm_input.weight.detach(), ln_in_b=mod.left_norm_input.bias.detach(), w_ag=a_g, w_ap=a_p, w_bg=b_g, w_bp=b_p,
                ln_out_w=mod.center_norm.weight.detach(), ln_out_b=mod.center_norm.bias.detach(),
                w_o=mod.output_projection.weight.detach(), w_og=mod.gating_linear.weight.detach())


def trimul_rows_record() -> Optional[dict]:
    """The core's plain fields for the run record (``trimul_fused.describe()``: state, served / fallback counts by kind and reason, facts
    cells / settings / k1 / k3 / k1_impl / kernel_copy / min_tokens) when the fused rows were requested and the core carries them; None otherwise."""
    if STATE["stats"]["levers"].get("trimul_rows") != "fpf_v4":
        return None
    RF = trimul_fused_module()
    return RF.describe() if RF is not None else None


def emit_trimul_rows_line(tag: str):
    """The core's ``LEVER name=F2.trimul_rows …`` census line of this process (``trimul_fused.emit_line(tag)``: the core renders it)
    when the fused rows were requested and the core carries them; None and nothing emitted otherwise."""
    if STATE["stats"]["levers"].get("trimul_rows") != "fpf_v4":
        return None
    RF = trimul_fused_module()
    return RF.emit_line(tag) if RF is not None else None


def _bias_hnn(tb_full, dtype):
    """``[N, N, H] -> [1, 1, H, N, N]`` contiguous in ``dtype``, memoised per gathered-bias tensor: one copy per triangle-attention call, shared by
    every row batch of the call. The memo HOLDS the source tensor and keys on its identity (never on an address: a freed
    bias's storage is reused by the next call's identically-shaped bias), so a hit is provably the same values; the next call's bias (a new
    tensor) replaces both entries — at most one extra ``[N, N, H]`` stays alive between calls (named: ``triangle_bias`` in the replicated census)."""
    memo = STATE["bias_memo"]
    if memo.get("src") is tb_full and memo.get("dtype") == dtype:      # identity of a HELD tensor (inference tensors carry no version counter;
        return memo["v"]                                                 # the gathered bias is read-only for the duration of its call by contract)
    memo.clear()
    v = tb_full.permute(2, 0, 1).to(dtype=dtype).contiguous()[None, None]
    memo.update(src=tb_full, dtype=dtype, v=v)
    return v


def triatt_fns(mod, mask_rows_of: Callable[[int, int], object]):
    """TriAttFns of an xfold ``GridSelfAttention``: ``ln = act_norm``; ``bias = pair_bias_projection``; ``attend`` = the kit's fused
    q|k|v|gate GEMM + flash triangle attention + gate + output projection on the LayerNorm'd row batch when the kit's ``triattn`` lever is on
    in this process (its own ``flash_supported`` gate falls back COUNTED in the kit census, as on one GPU), else the module's ``_attention``.
    ``mask_rows_of(i0, i1)`` -> the ``[rows, N]`` pair-mask rows of LOCAL rows ``i0:i1`` in attention order (the driver hands mask^T rows to the
    ending node)."""
    C = STATE["C"] if "C" in STATE else _core()
    K = STATE["K"]
    H, Cz = int(mod.num_head), int(mod.c_pair)
    Dh = Cz // H

    def attend(x, mask_rows, tb_full, blk):
        if mask_rows is None:
            mask_rows = mask_rows_of(*blk)
        if not _kernel_on("triattn"):
            return mod._attention(x, mask_rows, tb_full.permute(2, 0, 1))
        import lnl_fused as RFU, flash_triattn as FT                     # the kit's kernels (opt/forward/af3t/kernels), importable in the model process
        W = K._triattn_weights(mod)
        xx = x if x.is_contiguous() else x.contiguous()
        qkvg = F.linear(xx, W["Wcat"])                                   # [rows, N, 4C]: q | k | v | gate in ONE GEMM (the kit's fast statement)

        def heads(t):
            return t.unflatten(-1, (H, Dh)).permute(0, 2, 1, 3)[None]

        q5, k5, v5 = heads(qkvg[..., 0:Cz]), heads(qkvg[..., Cz:2 * Cz]), heads(qkvg[..., 2 * Cz:3 * Cz])
        bias5 = _bias_hnn(tb_full, torch.bfloat16)
        mb = mask_rows if mask_rows.dtype == torch.bool else (mask_rows != 0)
        mask5 = mb[None, :, None, None, :]
        ok, why = FT.flash_supported(q5, k5, v5, bias5, mask5)
        if not ok:                                                       # the kit's own gate, counted in its census (FALLBACK line) exactly as on one GPU
            K._fallback("triattn", "flash:" + why)
            return mod._attention(x, mask_rows, tb_full.permute(2, 0, 1))
        o = FT.flash_triangle_attention(q5, k5, v5, bias=bias5, mask=mask5, scale=Dh ** -0.5)
        y = RFU.gate_transpose(o, qkvg[None], 3 * Cz, transpose=False)[0]
        K._count("triattn", "served:%s_rowpair" % ("end" if mod.transpose else "start"))
        return F.linear(y, mod.output_projection.weight)

    return C["TA"].TriAttFns(mod.act_norm, mod.pair_bias_projection, attend)


def transition_fn(mod):
    """``fn(x_rows, mask_u_rows) -> delta``: the xfold ``Transition`` (no mask; the kit's fused kernel when its lever is on — row-local)."""
    return lambda x, _m: mod(x)


def _attn_bias_rows(blk, z_rows):
    """The single-attention pair logits of pair rows: ``single_pair_logits_projection(single_pair_logits_norm(z_rows))`` -> ``[H, rows, N]``."""
    return blk.single_pair_logits_projection(blk.single_pair_logits_norm(z_rows)).permute(2, 0, 1)


def _attention_core(sa, xq, k, v, bias_q, mask):
    """The attention of xfold ``SelfAttention`` for normalised QUERY rows ``xq [q, c]`` against precomputed ``k`` / ``v`` heads ``[1, H, N, d]``
    of all tokens: ``q_projection``, logits + ``bias_q [H, q, N]`` + the key mask ``[N]``, softmax over all N keys, PV, the sigmoid query gate.
    Returns the gated weighted average ``[q, c]`` (the input of ``adaptive_zero_init``). The kit's ``apb`` lever selects its SDPA route (bias and
    key mask folded into one additive mask), else the module's ``fastnn.dot_product_attention`` statement."""
    H = sa.num_head
    q = sa.q_projection(xq).unflatten(-1, (H, -1)).movedim(-2, 0).unsqueeze(0)          # [1, H, q, d]
    if _kernel_on("apb"):
        am = bias_q.to(q.dtype).unsqueeze(0) + (1e9 * (mask.to(q.dtype) - 1.0))[None, None, None, :]
        wa = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=am, scale=q.shape[-1] ** -0.5)
    else:
        from xfold import fastnn
        wa = fastnn.dot_product_attention(q, k, v, mask=mask, bias=bias_q)
    wa = wa.squeeze(0).movedim(0, -2).flatten(-2)                        # [q, H*d]  ('h q c -> q (h c)')
    return wa * torch.sigmoid(sa.gating_query(xq))


def _kv_heads(sa, x):
    """k / v heads ``[1, H, N, d]`` of xfold ``SelfAttention`` from the normalised activation of ALL tokens."""
    H = sa.num_head
    k = sa.k_projection(x).unflatten(-1, (H, -1)).movedim(-2, 0).unsqueeze(0)
    v = sa.v_projection(x).unflatten(-1, (H, -1)).movedim(-2, 0).unsqueeze(0)
    return k, v


def self_attention_rows(sa, x_q, x_all, bias_q, mask):
    """xfold ``SelfAttention.forward`` (unconditioned instance: the Pairformer single attention) with QUERY rows ``x_q`` against all tokens
    ``x_all``: LN per token, k / v from all rows, the attention of the query rows with their pair-logit rows ``bias_q [H, q, N]``, the
    AdaLN-zero epilogue. Returns ``[q, c]`` == rows of the dense statement."""
    xq = sa.adaptive_layernorm(x_q, None)
    k, v = _kv_heads(sa, sa.adaptive_layernorm(x_all, None))
    return sa.adaptive_zero_init(_attention_core(sa, xq, k, v, bias_q, mask), None)


def apb_fn(blk):
    """``apb(s, z_shard, layout) -> s``: ``s + single_attention_`` with local query rows (transition.apb_local_queries gathers the rows)."""
    C = STATE["C"] if "C" in STATE else _core()

    def attn(q_rows, s_full, z_shard):
        lay = STATE["layout"]
        bias_q = _attn_bias_rows(blk, z_shard)                           # [H, R, N] from MY pair rows == rows r0:r1 of the dense [H, N, N]
        return self_attention_rows(blk.single_attention_, q_rows, s_full, bias_q, STATE["seq_mask"])

    def apb(s, z_shard, lay):
        upd = C["TN"].apb_local_queries(attn, s, z_shard, lay, gather=True)
        return s + upd

    return apb


def pair_block_fns(blk, layout, *, with_single: bool, stats_key: str):
    """PairBlockFns of one xfold PairformerBlock / EvoformerBlock pair half (its own modules as the core's callables)."""
    C = STATE["C"] if "C" in STATE else _core()
    lay = layout

    def mask_rows(i0, i1):                                               # attention-order mask rows: the driver passes them explicitly; this is the fallback for a None
        pm = STATE.get("mask_shard")
        return None if pm is None else pm[i0:i1]

    trimul_kw = {"inplace_chunk": _qblock()}                             # the engine's in-place triangle-multiplication column chunk (xfold's statement is unchunked: the adapter's row batch is the grid)
    fns = C["PS"].bind(trimul_out=trimul_fns(blk.triangle_multiplication_outgoing, True),
                       trimul_in=trimul_fns(blk.triangle_multiplication_incoming, False),
                       triatt_start=triatt_fns(blk.pair_attention1, mask_rows), triatt_end=triatt_fns(blk.pair_attention2, mask_rows),
                       transition=transition_fn(blk.pair_transition), chunk=_qblock(),
                       apb=apb_fn(blk) if with_single else None,
                       single_transition=(lambda s: s + blk.single_transition(s)) if with_single else None,
                       stats=STATE.setdefault("contract_stats", {}), trimul_kw=trimul_kw)
    STATE["stats"]["presharded_calls"] += 1
    return fns


def run_pair_stack_sharded(blocks, z_shard, mask_shard, layout, *, s=None, seq_mask=None, with_single=False, key="pairformer"):
    """``pairstack.pair_stack_`` over xfold blocks on a PRE-SHARDED ``z_shard [R, N, C]`` (in place; returns ``(z_shard, s)``)."""
    C = STATE["C"] if "C" in STATE else _core()
    STATE["seq_mask"] = seq_mask
    STATE["mask_shard"] = mask_shard
    fns = [pair_block_fns(b, layout, with_single=with_single, stats_key=key) for b in blocks]
    z_shard, s = C["PS"].pair_stack_(fns, z_shard, mask_shard, layout, s=s, transition_mask=False)
    STATE["stats"]["pair_blocks"] += len(blocks)
    return z_shard, s



# ============================================================================================================================ trunk
def _bond_pairs(evo, batch):
    """The (i, j) index pairs ``Evoformer._embed_bonds`` scatters into its ``[N, N]`` contact matrix (a few per input; replicated): the same
    gather-index statements, returned as two int64 vectors instead of a dense scatter target."""
    import xfold.of3 as OF3
    t2plb = batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
    gi_pl = t2plb.gather_idxs
    gm_pl = t2plb.gather_mask.prod(dim=1).to(dtype=gi_pl.dtype)[:, None]
    gi_pl = gi_pl * gm_pl
    t2llb = batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
    gi_ll = t2llb.gather_idxs
    gm_ll = t2llb.gather_mask.prod(dim=1).to(dtype=gi_ll.dtype)[:, None]
    gi_ll = gi_ll * gm_ll
    gather_idxs = torch.concatenate([gi_pl, gi_ll])
    ii, jj = gather_idxs[:, 0].long(), gather_idxs[:, 1].long()
    if getattr(OF3, "OF3", False):
        ii, jj = torch.concatenate([ii, jj]), torch.concatenate([jj, ii])
    return ii, jj


def _contact_rows(ii, jj, g0: int, g1: int, N: int, dtype, device):
    """Rows ``[g0, g1)`` of the bond contact matrix: 1.0 at the scattered pairs, ``[0, 0] = 0``."""
    cm = torch.zeros((g1 - g0, N), dtype=dtype, device=device)
    sel = (ii >= g0) & (ii < g1)
    if bool(sel.any()):
        cm[ii[sel] - g0, jj[sel]] = 1.0
    if g0 == 0:
        cm[0, 0] = 0.0
    return cm


def _template_precursors(ste, templates_t, dtype):
    """The per-token part of ``SingleTemplateEmbedding.construct_input`` for one template slot (replicated; the engine's statements): pseudo-beta
    positions / mask, residue-type one-hot, backbone rigid frames + mask, frame origins."""
    from xfold import scoring, geometry, protein_data_processing
    from xfold.constants import residue_names
    from xfold.nn.template import make_backbone_rigid
    aatype = templates_t.aatype
    dense_atom_mask = templates_t.atom_mask
    dense_atom_positions = templates_t.atom_positions
    dense_atom_positions *= dense_atom_mask[..., None]
    pb_pos, pb_mask = scoring.pseudo_beta_fn(templates_t.aatype, dense_atom_positions, dense_atom_mask)
    from xfold.nn import template as T
    aat_idx = T.template_restype_for_one_hot(aatype, dense_atom_mask)       # the dense statement's OpenFold3 empty-slot GAP substitution — the one-hot operand only (pseudo-beta and the rigid groups read templates.aatype)
    aat = torch.nn.functional.one_hot(aat_idx.to(dtype=torch.int64), residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP).to(dtype=dtype)
    tgi = torch.take_along_dim(protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(device=aatype.device),
                               aatype.to(dtype=torch.int64)[..., None, None], dim=0)
    rigid, backbone_mask = make_backbone_rigid(geometry.Vec3Array.from_array(dense_atom_positions), dense_atom_mask, tgi.to(dtype=torch.int32))
    return dict(pb_pos=pb_pos, pb_mask=pb_mask, aat=aat, rigid=rigid, backbone_mask=backbone_mask, points=rigid.translation)


def _construct_input_rows(ste, z_rows, pre, asym_id, g0: int, g1: int, dtype):
    """Rows ``[g0, g1)`` of ``SingleTemplateEmbedding.construct_input`` (every 2-D operand is a per-(i, j) map of per-token template data:
    the left operand is sliced to the rows, the right operand stays whole)."""
    from xfold import geometry
    TRm = (STATE["C"] if "C" in STATE else _core())["TR"]
    multichain_rows = TRm.same_rows(asym_id, g0, g1).to(dtype=dtype)              # rows of (asym_id[:, None] == asym_id[None, :]).to(dtype)
    pb_mask = pre["pb_mask"]
    pseudo_beta_mask_2d = pb_mask[g0:g1, None] * pb_mask[None, :]
    pseudo_beta_mask_2d *= multichain_rows
    dgram = _dgram_rows(pre["pb_pos"], g0, g1, ste.dgram_features_config)
    dgram *= pseudo_beta_mask_2d[..., None]
    dgram = dgram.to(dtype=dtype)
    pseudo_beta_mask_2d = pseudo_beta_mask_2d.to(dtype=dtype)
    to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]
    aat = pre["aat"]
    to_concat.append((aat[None, :, :], 1))
    to_concat.append((aat[g0:g1, None, :], 1))
    rg = pre["rigid"]
    R_ = rg.rotation
    rot_rows = geometry.Rot3Array(R_.xx[g0:g1, None], R_.xy[g0:g1, None], R_.xz[g0:g1, None],
                                  R_.yx[g0:g1, None], R_.yy[g0:g1, None], R_.yz[g0:g1, None],
                                  R_.zx[g0:g1, None], R_.zy[g0:g1, None], R_.zz[g0:g1, None])
    T_ = rg.translation
    trans_rows = geometry.Vec3Array(T_.x[g0:g1, None], T_.y[g0:g1, None], T_.z[g0:g1, None])
    rigid_rows = geometry.Rigid3Array(rot_rows, trans_rows)
    rigid_vec = rigid_rows.inverse().apply_to_point(pre["points"])
    unit_vector = rigid_vec.normalized()
    unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]
    unit_vector = [x.to(dtype=dtype) for x in unit_vector]
    backbone_mask = pre["backbone_mask"].to(dtype=dtype)
    backbone_mask_2d = backbone_mask[g0:g1, None] * backbone_mask[None, :]
    backbone_mask_2d *= multichain_rows
    unit_vector = [x * backbone_mask_2d for x in unit_vector]
    to_concat.extend([(x, 0) for x in unit_vector])
    to_concat.append((backbone_mask_2d, 0))
    query_embedding = ste.query_embedding_norm(z_rows)
    to_concat.append((query_embedding, 1))
    act = 0
    for i, (x, n_input_dims) in enumerate(to_concat):
        if n_input_dims == 0:
            x = x[..., None]
        act += ste.__getattr__(f'template_pair_embedding_{i}')(x)
    return act


def template_rows_fn(evo, batch, layout, pair_mask_rows):
    """``template_fn(z_loc, cycle)`` for run_trunk_sharded: ``z_loc += TemplateEmbedding(z, templates, pair_mask, multichain_mask)`` on this rank's
    rows via template.template_embed_rows (per slot: construct_input rows -> 2 template pair blocks on the row shard; close: sequential slot
    sum in the pair dtype after the output LayerNorm, / (1e-7 + T), relu, output_linear — the engine's TemplateEmbedding.forward order)."""
    C = STATE["C"] if "C" in STATE else _core()
    te = evo.template_embedding
    ste = te.single_template_embedding
    templates = batch.templates
    asym_id = batch.token_features.asym_id
    T = int(templates.aatype.shape[0])

    def template_fn(z_loc, cycle):
        if T == 0:
            return z_loc
        dtype = z_loc.dtype
        pres = [_template_precursors(ste, templates[t], dtype) for t in range(T)]
        keys = [templates.aatype, templates.atom_positions, templates.atom_mask]
        groups = C["TP"].template_slot_groups(keys, T, layout, device=z_loc.device)

        def unit_rows_fn(z_rows, slot, g):
            return _construct_input_rows(ste, z_rows, pres[slot], asym_id, g[0], g[1], dtype)[None]

        def pair_stack_fn(u, mask_loc):
            zz, _ = run_pair_stack_sharded(ste.template_embedding_iteration, u[0], mask_loc, layout, with_single=False, key="template")
            return zz[None]                                              # the output LayerNorm rides in finish_fn (per slot, before the slot sum)

        def finish_fn(t):                                                # t: [T, rows, N, c_t] slots in original order
            acc = t.new_zeros(tuple(t.shape[1:]), dtype=dtype)          # `summed_template_embeddings = query_embedding.new_zeros(...)`
            for k in range(int(t.shape[0])):
                acc += ste.output_layer_norm(t[k])                       # `summed += single_template_embedding(...)` (its final LayerNorm)
            emb = acc / (1e-7 + T)
            emb = torch.relu(emb)
            return te.output_linear(emb)

        STATE["stats"]["template_rows"] += layout.R
        out = C["TP"].template_embed_rows(z_loc, layout, n_templ=T, c_t=int(te.num_channels), unit_rows_fn=unit_rows_fn,
                                          pair_stack_fn=pair_stack_fn, finish_fn=finish_fn, mask_loc=pair_mask_rows, slot_groups=groups, add=True)
        _mark("after_template")
        return out

    return template_fn


def _opm_outer_fn(opm, mask):
    """outer_fn(a_blk, b, g0, g1) for transition.opm_rows: xfold ``OuterProductMean.forward`` for the output ROWS of the left operand's tokens
    ``[g0, g1)`` (``a_blk = left_act[:, rows]``, ``b = right_act`` all tokens; ``mask [S, N, 1]``)."""
    def outer_fn(a_blk, b, g0, g1):
        left_act = a_blk.permute(0, 2, 1)                                # [S, c, rows]
        act = torch.einsum('acb,ade->dceb', left_act, b)                 # [N, c, e, rows]
        act = torch.einsum('dceb,cef->dbf', act, opm.output_w) + opm.output_b
        act = act.permute(1, 0, 2)                                       # [rows, N, C]
        norm = torch.einsum('abc,adc->bdc', mask[:, g0:g1], mask)        # [rows, N, 1]
        return act / (opm.epsilon + norm)
    return outer_fn


def _opm_statement_bytes_per_row(N: int, opm) -> int:
    """One OUTPUT row's transient on the statement path (_opm_outer_fn), for the core's row budget: the two ``[N, c, c, rows]``-shaped einsum
    intermediates alive together (pair dtype, 2 B) + the ``[rows, N, c_z]`` result and its normalised copy."""
    c = int(opm.num_outer_channel); F = int(opm.num_output_channel)
    return int(N) * (2 * c * c * 2 + 2 * F * 2)


def opm_rows_kernel_fn(KM, opm, msa, msa_mask):
    """``(outer_fn, bytes_per_row)`` for the core's OPM row schedule (msa.opm_rows_budgeted) on the kit's ``opm`` lever kernels
    (kernels/third_party/af3t_opm.py — the kernels ``OuterProductMean.forward`` runs on at P = 1, af3t_msa._opm_forward; ``KM`` = that module):
    ONCE per call, on the replicated MSA representation, ``ln_proj2`` (input LayerNorm + left|right projections + the MSA mask in one pass ->
    ``LT [N, c, S]`` token-major, ``R [S, N, c]``, bf16); then per output row block ``[g0, g1)`` of this rank ONE bf16 GEMM
    ``T[(b,c),(d,e)] = sum_a LT[b,c,a] R[a,d,e]`` over the block's tokens b and all tokens d (``[rows*c, N*c]``, fp32 accumulate) and ``opm_out``
    (the (c,e) -> c_z contraction with output_w, + output_b, / (eps + norm) -> the block's rows ``[rows, N, c_z]`` fp32, which the schedule adds
    into the shard). norm rows = ``mask[:, block]^T mask`` (exact counts accumulated in fp32 -> bf16: the statement's rounding, as
    af3t_msa._mask_norm). The per-(i, j) arithmetic is the P = 1 kernel path's at any block size (the GEMM's M is the block; K = S and the
    epilogue are per element): numerics = the ``opm`` lever's class (registry.LEVERS `opm`), invariant under P and under the row block.
    Transients: LT + R (``4·c·S·N`` B, alive for the call) and per block T (``2·c·c·N`` B per row) + the fp32 rows — never an ``[N, c, c, R]``."""
    import af3t_opm as KO                                                # beside af3t_msa on the model process's sys.path (kernels/third_party)
    cell = KM.OPM_CELLS[torch.cuda.get_device_capability(msa.device)[0]]
    W = KM._opm_weights(opm)
    S, N, _ = (int(x) for x in msa.shape)
    c = int(opm.num_outer_channel); F = int(opm.num_output_channel); eps = float(opm.epsilon)
    msa_c = msa.contiguous(); mask_c = msa_mask.contiguous()
    with torch.autocast("cuda", enabled=False):
        LT, R = KO.ln_proj2(msa_c, W["lnw"], W["lnb"], W["WT"], mask_c, W["eps"], BA=cell["BA"], num_warps=cell["warps"])
    A = LT.view(N * c, S); B = R.view(S, N * c)
    m32 = mask_c.float()                                                 # [S, N]

    def outer_fn(_a_blk, _b, g0, g1):
        g0, g1 = int(g0), int(g1); nb = g1 - g0
        with torch.autocast("cuda", enabled=False):
            T = torch.mm(A[g0 * c:g1 * c], B)                            # [nb*c, N*c] bf16: the outer product summed over sequences, this block's tokens x all tokens
            norm = torch.mm(m32[:, g0:g1].t(), m32).to(torch.bfloat16)   # [nb, N]
            out = torch.empty((nb, N, F), device=msa_c.device, dtype=torch.float32)
            KO.opm_out(T, W["W16"], W["bias"], norm, eps, 0, nb, N, out, False, BD=cell["BD"], num_warps=cell["warps"])
        del T, norm
        return out

    return outer_fn, N * (2 * c * c + 4 * F + 2 * F + 2)               # per output row: T (bf16) + the fp32 rows + the schedule's add operand + norm


def msa_rows_fn(evo, batch, layout, pair_mask_rows, target_feat):
    """``msa_fn(z_loc, cycle)`` for run_trunk_sharded: ``Evoformer._embed_process_msa`` with the pair track sharded — MSA shuffle / truncation /
    features / embedding REPLICATED (the shuffle draws from the torch RNG, proven identical across ranks by the driver's guard), then per
    EvoformerBlock: OPM rows added into the shard per budgeted OUTPUT ROW BLOCK (msa.opm_rows_budgeted; the `opm` lever's kernels through
    opm_rows_kernel_fn where their cell serves, the module's statements per block otherwise), pair-weighted averaging with local pair-logit rows
    (msa.pwa_bias_rows + msa.pwa_rows; m replicated), the MSA transition (replicated statement), the pair block on the shard."""
    C = STATE["C"] if "C" in STATE else _core()
    from xfold.nn import featurization

    def msa_fn(z_loc, cycle):
        dtype = z_loc.dtype
        msa_batch = featurization.shuffle_msa(batch.msa)
        msa_batch = featurization.truncate_msa_batch(msa_batch, evo.num_msa)
        msa_mask = msa_batch.mask.to(dtype=dtype)
        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype=dtype)
        msa = evo.msa_activations(msa_feat)
        msa += evo.extra_msa_target_feat(target_feat)[None]
        for blk in evo.msa_stack:
            # pair += outer_product_mean(msa, msa_mask)            (rows of the left operand's tokens are this rank's rows)
            opm = blk.outer_product_mean
            KM = sys.modules.get("af3t_msa")                             # the kit's MSA-kernel module: present iff this model process built the `opm` lever (af3_torch_api.build_model)
            why = "absent" if KM is None else KM._opm_cell_reason(opm, msa, msa_mask, None)   # None = the lever is on and its kernels serve this cell (CUDA, bf16 autocast, c_m 64 / c 32 / c_z 128, cc 8|9)
            if why is None:                                              # the `opm` lever's kernels on this rank's OUTPUT ROW BLOCKS (opm_rows_kernel_fn): no [N, c, c, rows] intermediate anywhere
                outer_fn, per_row = opm_rows_kernel_fn(KM, opm, msa, msa_mask)
                a_op = b_op = msa                                        # the schedule's shape contract only (token rows on dim 1 of a replicated [S, N, c] operand): outer_fn reads its own LT / R
                KM._count("opm", "served:rows")
                _lever("opm_rows", "af3t_opm")
            else:                                                        # the module's own statements per row block: CPU, lever off, or a cell the kernels do not serve (counted by name, as at P = 1)
                mask = msa_mask.unsqueeze(-1)
                m_ln = opm.layer_norm_input(msa)
                a_op = mask * opm.left_projection(m_ln)                  # left_act [S, N, c]
                b_op = mask * opm.right_projection(m_ln)                 # right_act [S, N, c]
                del m_ln
                outer_fn, per_row = _opm_outer_fn(opm, mask), _opm_statement_bytes_per_row(layout.N, opm)
                if KM is not None and why not in ("off", "cpu"):
                    KM._fallback("opm", why)
                _lever("opm_rows", "statements:" + why)
            C["MS"].opm_rows_budgeted(a_op, b_op, layout, outer_fn, C_z=int(opm.num_output_channel), out=z_loc, add=True, global_rows=True, row_dim=1,
                                      bytes_per_row=per_row)             # the core's BUDGETED row block (ROWPAIR_OPM_ROWS pin | ROWPAIR_ROWBLK_MB target | its share of the agreed free bytes; SCHEDULE opm_rows=): bounded at every N — the whole-shard block was this line's per-rank peak at every measured size
            del a_op, b_op, outer_fn
            # msa += msa_attention1(msa, msa_mask, pair)           (pair-logit rows of this rank; softmax rows complete; m replicated)
            att = blk.msa_attention1
            key_bias = 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)                 # [N]

            def prep_fn(z_rows, g0, g1, att=att, key_bias=key_bias):
                logits = att.pair_logits(att.pair_norm(z_rows)).permute(2, 0, 1)      # [H, rows, N]
                logits += key_bias
                return logits

            bias_shard = C["MS"].pwa_bias_rows(prep_fn, z_loc, layout)

            def values_fn(m_chunk, att=att):
                m_n = att.act_norm(m_chunk)
                v = att.v_projection(m_n).unflatten(-1, (att.num_head, att.value_dim))  # [b, k, h, c]
                g = torch.sigmoid(att.gating_query(m_n))                                # [b, N, c_m] gate per (sequence, token)
                return (v, g)

            def attend_fn(w, state, g0, g1):
                v, g = state
                v_avg = torch.einsum('hqk,bkhc->bqhc', w.to(v.dtype) if w.dtype != v.dtype and not torch.is_autocast_enabled() else w, v)
                v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))
                return v_avg * g[:, g0:g1]

            upd = C["MS"].pwa_rows(msa, bias_shard, layout, values_fn=values_fn, attend_fn=attend_fn, out_fn=att.output_projection)
            msa += upd
            del upd, bias_shard
            msa += blk.msa_transition(msa)                               # replicated statement (m is replicated by design)
            # pair block on the shard
            z_loc, _ = run_pair_stack_sharded([blk], z_loc, pair_mask_rows, layout, with_single=False, key="msa")
        STATE["stats"]["msa_blocks_rows"] += layout.R
        _mark("after_msa")
        return z_loc

    return msa_fn


def run_trunk_sharded_xfold(A, model, batch, layout, *, num_recycles: Optional[int] = None):
    """``af3_torch_api.run_trunk`` (num_recycles + 1 Evoformer passes from zero prev) with the pair representation BORN and CARRIED as this rank's
    row shard (trunk.run_trunk_sharded). Returns ``emb_loc = {'pair': [R, N, 128] fp32 shard, 'single': [N, 384] fp32, 'target_feat': [N, 447]
    fp32}`` on every rank — the heads consume the shard (no gather). Statement map (Evoformer.forward):
      pair = left_single(tf)[:, None] + right_single(tf)[None]      -> init_rows_fn (z_init = the outer sum ONLY, parked per ROWPAIR_PARK_ZINIT)
      pair += prev_embedding(LN(prev['pair']))                       -> recycle_update_fn (cycle 0: LN of a zeros row block, as prev = zeros)
      pair += position_activations(rel_feat); bonds; template      -> template_fn head: relpos rows + bond rows added in the engine's order, then
                                                                        the template embedder rows (template_rows_fn)
      MSA module                                                      -> msa_fn (msa_rows_fn)
      single = single_activations(tf) + prev_single_embedding(LN(prev['single']))  -> single_recycle_fn (replicated)
      48 x PairformerBlock                                            -> pairstack_fn (run_pair_stack_sharded, single track via apb_local_queries)
      prev = {pair: e['pair'].float(), single: e['single'].float()}   -> the fp32 casts ride in the LN callables' inputs; the returned shard /
                                                                        single are cast to fp32 once at the end."""
    C = STATE["C"] if "C" in STATE else _core()
    lay = C["D"].require_sharded(layout, "run_trunk_sharded_xfold")
    evo = model.evoformer
    b = A._B(model, batch)
    n_iter = (model.num_recycles if num_recycles is None else int(num_recycles)) + 1
    tf = A.run_target_feat(model, b)
    N = int(tf.shape[0])
    if N != lay.N:
        raise C["Refused"](f"run_trunk_sharded_xfold: target_feat has N={N}, the item layout N={lay.N}")
    left = evo.left_single(tf)                                           # [N, C] (the activation dtype)
    right = evo.right_single(tf)
    zdtype = left.dtype
    mask = b.token_features.mask
    pair_mask_rows = C["TR"].pair_mask_rows(mask, lay.r0, lay.r1).to(dtype=zdtype)      # rows of (mask[:, None] * mask[None, :]).to(dtype)
    ii, jj = _bond_pairs(evo, b)
    tfeat = b.token_features

    def init_rows_fn(g0, g1):
        return C["TR"].outer_sum_rows(left, right, g0, g1)

    def recycle_update_fn(z_rows):
        return evo.prev_embedding(evo.prev_embedding_layer_norm(z_rows.float()))

    template_fn_inner = template_rows_fn(evo, b, lay, pair_mask_rows)

    def relpos_rows(g0, g1):
        return evo.position_activations(_rel_feat_rows(tfeat, g0, g1, zdtype))

    def bond_rows(g0, g1):
        return evo.bond_embedding(_contact_rows(ii, jj, g0, g1, N, zdtype, tf.device)[:, :, None])

    def template_fn(z_loc, cycle):
        C["SH"].produce_rows_(z_loc, lay, relpos_rows, op="add")        # pair += position_activations(rel_feat)
        C["SH"].produce_rows_(z_loc, lay, bond_rows, op="add")          # pair = pair + bonds_act   (a separate add: the engine's rounding order)
        STATE["stats"]["trunk_rows"] += lay.R
        _mark("trunk_entry") if cycle == 0 else None
        return template_fn_inner(z_loc, cycle)

    msa_fn = msa_rows_fn(evo, b, lay, pair_mask_rows, tf)

    def single_recycle_fn(s, cycle):
        out = evo.single_activations(tf)
        out += evo.prev_single_embedding(evo.prev_single_embedding_layer_norm(s.float()))
        return out

    def pairstack_fn(s, z_loc, cycle):
        z_loc, s = run_pair_stack_sharded(evo.trunk_pairformer, z_loc, pair_mask_rows, lay, s=s, seq_mask=mask, with_single=True, key="trunk")
        _mark("after_pairstack")
        return s, z_loc

    s_init = torch.zeros(N, evo.seq_channel, device=tf.device)
    like = torch.empty((N, int(evo.pair_channel)), dtype=zdtype, device=tf.device)
    g_before = STATE["stats"]["gathers_in_trunk"]
    out = C["TR"].run_trunk_sharded(lay, n_cycles=n_iter, init_rows_fn=init_rows_fn, init_like=like, recycle_update_fn=recycle_update_fn,
                                    s_init=s_init, single_recycle_fn=single_recycle_fn, template_fn=template_fn, msa_fn=msa_fn,
                                    pairstack_fn=pairstack_fn, gather="none", log=_log)
    STATE["stats"]["gathers_in_trunk"] = g_before                        # pair gathers inside the trunk: none by construction (gather='none')
    STATE["trunk_record"] = out.record
    _mark("no_gather")
    return {"pair": out.z.float(), "single": out.s.float(), "target_feat": tf}


# ============================================================================================================================ heads

def rows_to_host_rank0(rows_shard, layout):
    """``[R, W]`` row shards of an ``[N, W]`` OUTPUT matrix (PAE / PDE / contact maps the engine writes) -> the ``[N, W]`` matrix on rank 0's
    HOST (pinned), None elsewhere — the core's one output-assembly primitive ``dist.gather_rows_to_rank0_host``: column blocks, one row gather
    per block, rank 0's device holds at most one ``N x w`` block at a time (census ``host_gather_cols`` / ``host_gather_blocks``). COLLECTIVE."""
    C = STATE["C"] if "C" in STATE else _core()
    STATE["stats"]["host_matrices"] += 1
    return C["D"].gather_rows_to_rank0_host(rows_shard.contiguous(), layout)



def _pair_mask_rows(seq_mask, lay, dtype):
    C = STATE["C"] if "C" in STATE else _core()
    return C["TR"].pair_mask_rows(seq_mask.to(dtype=dtype), lay.r0, lay.r1)


def run_distogram_sharded(model, batch, emb_loc, layout, *, rows: Optional[int] = None):
    """``DistogramHead.forward`` on the trunk pair shard: rank 0 gets ``{'bin_edges': [63], 'contact_probs': [N, N] fp32}``, other ranks None.
    COLLECTIVE (call on every rank). ``rows``: the head row block (``heads.conf_rows`` default)."""
    C = STATE["C"] if "C" in STATE else _core()
    import af3_torch_api as A
    dh = model.distogram_head
    b = A._B(model, batch)
    lay = C["D"].require_sharded(layout, "run_distogram_sharded")
    z = emb_loc["pair"]
    plan = STATE.get("ztrunk")
    if plan is not None and plan.z is z:                                 # the symmetrised logits read rows AND column slabs: the device shard whole (a live park is restored here —
        _ztrunk_close(need_device=True)                                  # the confidence passes restore it at their last pass already; this covers a caller that ran fewer passes)
        if plan.consumed:
            raise XfoldTPRefused("run_distogram_sharded: the trunk shard's rows are gone (its plan dropped or embedded them) — the distogram head reads the device shard whole")
    seq_mask = b.token_features.mask.to(dtype=torch.bool)
    pm_rows = seq_mask[lay.r0:lay.r1, None] * seq_mask[None, :]         # rows r0:r1 of `seq_mask[:, None] * seq_mask[None, :]`
    nb = int(dh.is_contact_bin.shape[0])
    rows, _src = C["HD"].conf_rows(lay.N, nb, rows)
    contact_rows = None
    for i0, i1, logits in C["HD"].sym_logit_rows(dh.half_logits, z, lay, rows, bins=nb):   # left + right^T per row block (one a2a window per step)
        probs = torch.softmax(logits, dim=-1)
        cp = torch.einsum('ijk,k->ij', probs, dh.is_contact_bin)
        cp = pm_rows[i0:i1] * cp
        if contact_rows is None:
            contact_rows = cp.new_empty((lay.R, lay.N))
        contact_rows[i0:i1] = cp
        del probs, logits, cp
    STATE["stats"]["distogram_rows"] += lay.R
    _mark("distogram")
    full = rows_to_host_rank0(contact_rows, lay)                        # [N, N] on rank 0's HOST (an output matrix; never whole on a device)
    del contact_rows
    if full is None:
        return None
    return {"bin_edges": dh.breaks.cpu(), "contact_probs": full}


def _dgram_rows(positions, g0: int, g1: int, config):
    """Rows ``[g0, g1)`` of ``xfold.nn.template.dgram_from_positions(positions, config)`` (a per-(i, j) map of per-token coordinates)."""
    lower_breaks = torch.square(torch.linspace(config.min_bin, config.max_bin, config.num_bins, device=positions.device))
    upper_breaks = torch.concatenate([lower_breaks[1:], torch.ones(1, device=lower_breaks.device) * 1e8], dim=-1)
    dist2 = torch.sum(torch.square(positions[g0:g1, None, :] - positions[None, :, :]), dim=-1, keepdims=True)
    return (dist2 > lower_breaks).to(dtype=torch.float32) * (dist2 < upper_breaks).to(dtype=torch.float32)


def run_confidence_sharded(model, batch, emb_loc, layout, positions, *, rows: Optional[int] = None):
    """``ConfidenceHead.forward`` for ONE sample (``positions``: its dense atom positions ``[N, 24, 3]``, replicated) on the trunk pair shard.
    Rank 0 gets the engine's dict (``predicted_lddt [N, 24]``, ``predicted_experimentally_resolved [N, 24]``, ``average_pde []`` on the device;
    ``full_pde``, ``full_pae``, ``tmscore_adjusted_pae_global``, ``tmscore_adjusted_pae_interface`` ``[N, N]`` fp32 on the HOST — output matrices are
    assembled there column block by column block, never whole on a device; ``average_pde`` comes from the row sums); other ranks None. COLLECTIVE. The trunk shard ``emb_loc['pair']`` is left untouched (the per-sample pair is a fresh ``[R, N, 128]``)."""
    if os.environ.get("XFOLD_TP_LEAKCHECK"):
        _leakreport("confidence_entry")
    C = STATE["C"] if "C" in STATE else _core()
    import af3_torch_api as A
    from xfold.nn import atom_layout
    ch = model.confidence_head
    b = A._B(model, batch)
    lay = C["D"].require_sharded(layout, "run_confidence_sharded")
    r0, r1, N = lay.r0, lay.r1, lay.N
    seq_mask = b.token_features.mask
    asym_id = b.token_features.asym_id
    dtype = positions.dtype
    seq_mask_cast = seq_mask.to(dtype=dtype)
    pm_rows = (seq_mask_cast[r0:r1, None] * seq_mask_cast[None, :]).to(dtype=dtype)     # rows of pair_mask (fp32)
    target_feat = emb_loc["target_feat"].clone().to(dtype=dtype)
    single_act = emb_loc["single"].clone().to(dtype=dtype)
    # ---- pair input: pair.to(dtype) + left/right target-feat outer sum + distogram_feat_project(dgram * mask), per row block (heads.embed_rows)
    left = ch.left_target_feat_project(target_feat)                      # [N, C] replicated (tiny)
    right = ch.right_target_feat_project(target_feat)
    pb_pos = atom_layout.convert(b.pseudo_beta_info.token_atoms_to_pseudo_beta, positions, layout_axes=(-3, -2))   # [N, 3] replicated

    def embed(z_rows, g0, g1):
        out = left[None, :, :] + right[g0:g1, None, :]                   # `left[..., None, :, :] + right[..., None, :]`: out[i, j] = left[j] + right[i]
        dg = _dgram_rows(pb_pos, g0, g1, ch.dgram_features_config)
        dg *= pm_rows[g0 - r0:g1 - r0, :, None]
        out = out + ch.distogram_feat_project(dg)                        # `out += proj(dgram)` (same dtype: the activation dtype)
        return z_rows.to(dtype=dtype) + out                              # `pair_act (fp32 clone) += out`

    hb, _src = C["HD"].conf_rows(N, 64, rows)
    plan = STATE.get("ztrunk")
    ip = None
    if plan is not None and not plan.closed and plan.z is emb_loc["pair"] and int(STATE.get("ztrunk_pass") or 0) < plan.passes:
        ip = int(STATE["ztrunk_pass"])                                   # this pass's index in the item's trunk-shard plan (heads.ZTrunkPlan): a parked shard serves its rows from the
        src, inplace = plan.begin(ip, last_use=False, out_dtype=torch.float32)   # host block by block, its device storage released since the roll-out entry; resident otherwise
    else:
        src, inplace = emb_loc["pair"], False
    pair = C["HD"].embed_rows(embed, src, lay, hb, inplace=inplace, out_dtype=torch.float32)   # [R, N, 128] fp32 (fresh storage unless the plan embeds in place)
    if ip is not None and hasattr(plan, "retire"):                          # the trunk rows' last reader of the LAST pass ran: its host park retires before this pass's pair stack
        _lever("ztrunk_retire", str(plan.retire(ip)))                       # (heads.ZTrunkPlan.retire: `kept:not_last` on other passes, `nothing_parked` when resident)
    _mark("confidence")
    # ---- confidence pairformer (4 blocks, single track with local query rows)
    pair, single_act = run_pair_stack_sharded(ch.confidence_pairformer, pair, pm_rows, lay, s=single_act, seq_mask=seq_mask,
                                              with_single=True, key="confidence")
    # ---- PDE: symmetrised distance logits per row block; PAE: plain logits per row block; TM-adjusted rows (global / interface)
    nb_pde = int(ch.bin_centers.shape[0])
    pde_rows = torch.empty((lay.R, N), dtype=torch.float32, device=pair.device)
    for i0, i1, logits in C["HD"].sym_logit_rows(lambda zz: ch.left_half_distance_logits(ch.logits_ln(zz)), pair, lay, hb, bins=nb_pde):
        probs = torch.softmax(logits, dim=-1)
        pde_rows[i0:i1] = torch.sum(probs * ch.bin_centers, dim=-1) * pm_rows[i0:i1]
        del probs, logits
    nb_pae = int(ch.pae_bin_centers.shape[0])
    pae_rows = torch.empty((lay.R, N), dtype=torch.float32, device=pair.device)
    tmg_rows = torch.empty((lay.R, N), dtype=torch.float32, device=pair.device)
    tmi_rows = torch.empty((lay.R, N), dtype=torch.float32, device=pair.device)
    pm_bool = pm_rows.to(dtype=torch.bool)
    # the context sizes of _get_tmscore_adjusted_pae from replicated per-token data — no [N, N] intermediate: num_chain_tokens[i] =
    # sum_j (asym_i == asym_j) * pair_mask[i, j] = mask_i * #{j : mask_j, asym_j == asym_i} (integer counts: exact in any order)
    seq_mask_b = seq_mask_cast.to(dtype=torch.bool)
    _uniq, inv = torch.unique(asym_id, return_inverse=True)
    per_chain = torch.bincount(inv[seq_mask_b], minlength=int(_uniq.shape[0])).to(dtype=torch.int32)
    num_chain_tokens = per_chain[inv] * seq_mask_b.to(dtype=torch.int32)                  # [N] int32
    n_glob = seq_mask.sum()
    for i0, i1, logits in C["HD"].logit_rows(lambda zz: ch.pae_logits(ch.pae_logits_ln(zz)), pair, lay, hb, bins=nb_pae):
        probs = torch.softmax(logits, dim=-1)
        g0, g1 = r0 + i0, r0 + i1
        pae_rows[i0:i1] = torch.sum(probs * ch.pae_bin_centers, dim=-1) * pm_bool[i0:i1]
        # rows of num_interface_tokens / num_global_tokens (head.py _get_tmscore_adjusted_pae, per row i in [g0, g1)):
        xs = asym_id[g0:g1, None] == asym_id[None, :]                   # rows of `asym_id[None, :] == asym_id[:, None]` (symmetric)
        nit = num_chain_tokens[None, :] + num_chain_tokens[g0:g1, None]
        nit -= xs * (nit // 2)
        nit = nit * pm_bool[i0:i1]
        ngt = torch.ones(size=xs.shape, dtype=torch.int32, device=xs.device)
        ngt *= n_glob
        tmg_rows[i0:i1] = _tm_term(ngt, ch.pae_bin_centers, probs)
        tmi_rows[i0:i1] = _tm_term(nit, ch.pae_bin_centers, probs)
        del probs, logits, xs, nit, ngt
    STATE["stats"]["conf_rows"] += lay.R
    # ---- pLDDT / experimentally resolved: dense statements on the replicated single (every rank; [N, 24 * 50])
    import einops
    plddt_logits = einops.rearrange(ch.plddt_logits(ch.plddt_logits_ln(single_act)), '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=ch.num_plddt_bins)
    predicted_lddt = torch.sum(torch.softmax(plddt_logits, dim=-1) * ch.plddt_bin_centers, dim=-1) * 100.0
    er_logits = einops.rearrange(ch.experimentally_resolved_logits(ch.experimentally_resolved_ln(single_act)), '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=2)
    predicted_experimentally_resolved = torch.softmax(er_logits, dim=-1)[..., 1]
    # ---- average_pde = sum(full_pde) / sum(pair_mask) from ROW sums (one all-reduce of two scalars; every rank)
    sums = torch.stack([pde_rows.sum(dtype=torch.float32), pm_rows.sum(dtype=torch.float32)])
    C["D"].allreduce_(sums, op="sum")
    average_pde = sums[0] / sums[1]
    # ---- the [N, N] OUTPUT matrices to rank 0's HOST, column block by column block (conf_full_matrices_rank0: never whole on a device)
    full_pde, full_pae, tmg, tmi = (rows_to_host_rank0(x, lay) for x in (pde_rows, pae_rows, tmg_rows, tmi_rows))
    del pde_rows, pae_rows, tmg_rows, tmi_rows, pair
    if ip is not None:                                                   # the pass's pair input is dead: close the pass (nothing is restored: the distogram head — the shard's one
        plan.end(ip, device_needed_next=False)                           # dense reader — ran before the roll-out; after the last pass the host copy is gone and the device storage
        STATE["ztrunk_pass"] = ip + 1                                    # stays released)
        _lever("ztrunk", plan.words[ip])
    _mark("confidence")
    if full_pde is None:
        return None
    return {"predicted_lddt": predicted_lddt, "predicted_experimentally_resolved": predicted_experimentally_resolved,
            "full_pde": full_pde, "average_pde": average_pde, "full_pae": full_pae,
            "tmscore_adjusted_pae_global": tmg, "tmscore_adjusted_pae_interface": tmi}


def _tm_term(num_interface_tokens, bin_centers, pae_probs):
    """``ConfidenceHead._get_tmscore_adjusted_pae.get_tmscore_adjusted_pae`` on a row block (per-(i, j) map)."""
    clipped_num_res = torch.clamp(num_interface_tokens, min=19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
    d0 = d0[:, :, None]
    bc = bin_centers[None, None, :]
    tm_per_bin = 1.0 / (1 + torch.square(bc) / torch.square(d0))
    return torch.sum(pae_probs * tm_per_bin, dim=-1)


# ======================================================================================================================== diffusion
def _rel_feat_rows(tf, g0: int, g1: int, dtype):
    """Rows ``[g0, g1)`` of ``xfold.nn.featurization.create_relative_encoding(token_features, 32, 2)`` (``[rows, N, 139]``) from the core's
    lazy relative-position row statements — no ``[N, N, 139]`` one-hot exists."""
    TR = (STATE["C"] if "C" in STATE else _core())["TR"]
    asym_same = TR.same_rows(tf.asym_id, g0, g1)
    rel_pos = TR.relpos_onehot_rows(tf.residue_index, g0, g1, 32, condition=asym_same, dtype=dtype)
    res_same = asym_same & TR.same_rows(tf.residue_index, g0, g1)
    rel_token = TR.relpos_onehot_rows(tf.token_index, g0, g1, 32, condition=res_same, dtype=dtype)
    ent_same = TR.same_rows(tf.entity_id, g0, g1)
    rel_chain = TR.relpos_onehot_rows(tf.sym_id, g0, g1, 2, condition=ent_same, dtype=dtype)
    return torch.concatenate([rel_pos, rel_token, ent_same.to(dtype=dtype)[..., None], rel_chain], dim=-1)


def dit_block_fns(tr, i: int, mask):
    """DiTBlockFns of block ``i`` of an xfold ``DiffusionTransformer`` (OpenFold3 layout: per-block pair LayerNorm + logits projection):
    ``norm`` = the block's AdaLN of all tokens; ``kv`` = k / v heads of all tokens; ``attn`` = q of the query rows, logits + local bias rows + key
    mask, softmax, PV, query gate; ``update`` = ``a[r0:r1] + adaptive_zero_init(o_rows, s[r0:r1])`` then ``+= transition_block(., s[r0:r1])``
    (the conditioned transition on my rows); ``bias`` = ``pair_logits_projection[i](pair_input_layer_norm[i](z_rows))`` -> ``[rows, N, H]``."""
    C = STATE["C"] if "C" in STATE else _core()
    sa, tb = tr.self_attention[i], tr.transition_block[i]
    if not tr.of3:
        raise XfoldTPRefused("dit_block_fns: the AlphaFold3 super-block pair-logit layout is not wired (xfold.of3.OF3 False)")

    def norm(a, s):
        return sa.adaptive_layernorm(a, s)                              # AdaLN per token, all rows (k / v need every row)

    def kv(x):
        return _kv_heads(sa, x)

    def attn(x_q, kvh, bias_q, g):
        return _attention_core(sa, x_q, kvh[0], kvh[1], bias_q, mask)

    def update(a, o_rows, s, rr):
        q0, q1 = rr
        s_rows = s[q0:q1]
        a_rows = a[q0:q1] + sa.adaptive_zero_init(o_rows, s_rows)      # `act += self_attention(act, mask, pair_logits, single_cond)` on my rows
        a_rows += tb(a_rows, s_rows)                                    # `act += transition_block(act, single_cond)` on my rows
        return a_rows

    def bias(z_rows):
        return tr.pair_logits_projection[i](tr.pair_input_layer_norm[i](z_rows))          # [rows, N, H]

    return C["DF"].DiTBlockFns(norm, kv, attn, update, bias)


DIFF_CACHE_RESERVE_GB = 18.0                                             # what the cache decision leaves free besides z_cond: the row-block work budget (8 GB), the banded encoder statics +
                                                                         # the attention block's mask / workspace (≈ 6 GB at 10k tokens on two ranks), 4 GB margin — the card never at its edge


def _bias_cache_rule(C, lay, *, n_blocks: int, H: int, elt: int = 2, c_cond: int = 128) -> Optional[bool]:
    """The kit's pair-bias cache decision for this roll-out, from MEASURED memory: on iff the cache (``n_blocks · H · Rmax · N · elt`` bytes —
    one ``[H, R, N]`` pair-dtype block per DiT block) plus the z_cond rows built right after this decision fit in the free bytes AGREED across
    ranks right now (``dist.agreed_free_bytes``: one collective every rank reaches here; the trunk shard is parked already) minus ``DIFF_CACHE_RESERVE_GB``. ``None`` (= the support library's
    own rule) when the environment pins the decision (``ROWPAIR_DIFF_BIAS_CACHE`` / ``ROWPAIR_DIFF_BIAS_CACHE_GB`` exported) or no rank has
    CUDA. Why not the library default: a fixed cap (24 GB) turns the cache off above the N where it matters most (x2 at 8k tokens: 24
    LayerNorm + projection passes over z_cond per denoiser step instead of one per roll-out) and its ``auto`` share (0.30 of free) does the
    same there; the reach is protected by the reserve, not by a constant. Identical on every rank (agreed bytes). Census ``diff_bias_cache_rule``."""
    for k in ("ROWPAIR_DIFF_BIAS_CACHE", "ROWPAIR_DIFF_BIAS_CACHE_GB"):
        v = (os.environ.get(k) or "").strip().lower()
        if v not in ("", "auto"):
            STATE["stats"]["diff_bias_cache_rule"] = f"env:{k}"
            return None
    try:
        free = C["D"].agreed_free_bytes()
    except Exception:                                                    # noqa: BLE001
        free = None
    if free is None:
        STATE["stats"]["diff_bias_cache_rule"] = "core:no_cuda"
        return None
    R = int(getattr(lay, "Rmax", None) or lay.R)
    need = int(n_blocks) * int(H) * R * int(lay.N) * int(elt)              # the cache: one [H, R, N] block per DiT block
    zc = R * int(lay.N) * int(c_cond) * int(elt)                         # z_cond rows, built right after this decision (diffusion.pair_cond_rows) and alive through the roll-out
    on = need + zc + int(DIFF_CACHE_RESERVE_GB * 1e9) <= int(free)
    STATE["stats"]["diff_bias_cache_rule"] = f"kit:{'on' if on else 'off'}:need={need / 1e9:.1f}+{zc / 1e9:.1f}GB:free={free / 1e9:.1f}GB:reserve={DIFF_CACHE_RESERVE_GB:.0f}GB"
    return bool(on)


class _DiffCtx(object):
    """Per-roll-out sharded conditioning of the DiffusionHead: z_cond rows, the 24 pair-bias row blocks (PairBiasCache), the atom encoder's
    statics with the trunk-pair term read from the band of z_cond, the decoder's pair logits. Built on the first denoiser call."""

    def __init__(self, model, layout):
        self.model, self.lay = model, layout
        self.primed = False


def _encoder_static_banded(enc, batch, trunk_single_cond, zc_shard, lay, sched):
    """``AtomCrossAttEncoder.compute_static`` with the trunk-pair term ``pair_act += convert(trunk_pair_to_atom_pair, embed(LN(trunk_pair_cond)))``
    read from the diagonal band of the SHARDED conditioned pair (diffusion.band_plan / pair_band_rows / band_lookup: exactly
    ``zp[q_idx, k_idx] * gather_mask``); every other statement is the engine's, verbatim, on replicated per-atom tensors."""
    C = STATE["C"] if "C" in STATE else _core()
    from xfold.nn import atom_layout
    token_atoms_single_cond, _ = enc._per_atom_conditioning(batch)
    token_atoms_mask = batch.predicted_structure_info.atom_mask
    queries_single_cond = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, token_atoms_single_cond, layout_axes=(-3, -2))
    queries_mask = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, token_atoms_mask, layout_axes=(-2, -1))
    if trunk_single_cond is not None:
        tsc = enc.embed_trunk_single_cond(enc.lnorm_trunk_single_cond(trunk_single_cond))
        queries_single_cond += atom_layout.convert(batch.atom_cross_att.tokens_to_queries, tsc, layout_axes=(-2,))
    queries_single_cond = queries_single_cond * queries_mask[..., None]
    keys_single_cond = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_single_cond, layout_axes=(-3, -2))
    keys_mask = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_mask, layout_axes=(-2, -1))
    row_act = enc.single_to_pair_cond_row_1(torch.relu(queries_single_cond))
    pair_cond_keys_input = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_single_cond, layout_axes=(-3, -2))
    col_act = enc.single_to_pair_cond_col_1(torch.relu(pair_cond_keys_input))
    pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]
    # ---- trunk pair term from the band of the sharded z_cond
    t2q, t2k = batch.atom_cross_att.tokens_to_queries, batch.atom_cross_att.tokens_to_keys
    q_idx, k_idx = t2q.gather_idxs.long()[None], t2k.gather_idxs.long()[None]            # [1, ns, nq], [1, ns, nk]
    gmask = t2q.gather_mask[:, :, None] & t2k.gather_mask[:, None, :]                     # [ns, nq, nk]
    plan = C["DF"].band_plan(q_idx, k_idx, gmask[None], lay.N, max_w=None, max_extra_rows=None)   # the band as wide as the valid atom pairs need (no row travels whole)
    proj = lambda z_rows: enc.embed_trunk_pair_cond(enc.lnorm_trunk_pair_cond(z_rows))   # noqa: E731 — [rows, N, 16]
    band, extras = C["DF"].pair_band_rows(proj, zc_shard, lay, plan, rows=sched.band_rows if sched else None)
    tp = C["DF"].band_lookup(band, extras, plan)[0]                                       # [ns, nq, nk, 16] == embed(LN(z_cond)).reshape(N*N,16)[gather_idxs]
    tp *= gmask.reshape(gmask.shape + (1,))                                               # atom_layout.convert's mask multiply
    pair_act += tp
    del band, extras, tp
    STATE["stats"]["band_rows"] += lay.R
    # ---- the rest verbatim
    queries_ref_pos = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, batch.ref_structure.positions, layout_axes=(-3, -2))
    queries_ref_space_uid = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, batch.ref_structure.ref_space_uid, layout_axes=(-2, -1))
    keys_ref_pos = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_ref_pos, layout_axes=(-3, -2))
    keys_ref_space_uid = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_ref_space_uid, layout_axes=(-2, -1))   # keys gathered from the QUERIES layout
    return _encoder_static_tail(enc, pair_act, queries_ref_pos, queries_ref_space_uid, keys_ref_pos, keys_ref_space_uid,
                                token_atoms_mask, queries_single_cond, queries_mask, keys_single_cond, keys_mask)


def _encoder_static_tail(enc, pair_act, queries_ref_pos, queries_ref_space_uid, keys_ref_pos, keys_ref_space_uid,
                         token_atoms_mask, queries_single_cond, queries_mask, keys_single_cond, keys_mask):
    import xfold.of3 as OF3
    offsets_valid = queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
    if getattr(OF3, "OF3", False):                                       # the OpenFold3 layout excludes padded key atoms
        offsets_valid = offsets_valid & keys_mask[:, None, :].to(dtype=torch.bool)
    offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
    pair_act += enc.embed_pair_offsets_1(offsets) * offsets_valid[:, :, :, None]
    sq_dists = torch.sum(torch.square(offsets), dim=-1)
    pair_act += enc.embed_pair_distances_1(1.0 / (1 + sq_dists[:, :, :, None])) * offsets_valid[:, :, :, None]
    pair_act += enc.embed_pair_offsets_valid(offsets_valid[:, :, :, None].to(dtype=torch.float32))
    pair_act2 = enc.pair_mlp_1(torch.relu(pair_act))
    pair_act2 = enc.pair_mlp_2(torch.relu(pair_act2))
    pair_act += enc.pair_mlp_3(torch.relu(pair_act2))
    enc_pair_logits = enc.atom_transformer_encoder.compute_pair_logits(pair_act)
    return dict(token_atoms_mask=token_atoms_mask, queries_single_cond=queries_single_cond, queries_mask=queries_mask,
                keys_single_cond=keys_single_cond, keys_mask=keys_mask, pair_cond=pair_act, enc_pair_logits=enc_pair_logits)


def _prime_diffusion(ctx: _DiffCtx, batch, embeddings, use_conditioning: bool):
    """Once per roll-out: the DiffusionSchedule, z_cond rows (pair_cond_rows: conditioning projection of ``cat[use_cond * z_rows, relpos rows]``
    + the two pair transitions), the bias cache, the banded encoder statics, the decoder pair logits."""
    C = STATE["C"] if "C" in STATE else _core()
    DF = C["DF"]
    model, lay = ctx.model, ctx.lay
    dh = model.diffusion_head
    z = embeddings["pair"]                                               # [R, N, 128] fp32 shard
    adt = _act_dtype(z)
    elt = torch.empty((), dtype=adt).element_size()
    ctx.sched = DF.DiffusionSchedule.decide(lay, c_z=int(z.shape[-1]), c_in=int(dh.c_pair_cond_initial), c_cond=int(dh.pair_channel),
                                            H=int(dh.transformer.num_head), S=1, n_blocks=int(dh.transformer.num_blocks),
                                            c_pair=int(dh.atom_cross_att_encoder.per_atom_pair_channels), elt=elt,
                                            bias_cache=_bias_cache_rule(C, lay, n_blocks=int(dh.transformer.num_blocks), H=int(dh.transformer.num_head), elt=elt,
                                                                        c_cond=int(dh.pair_channel)))
    tf = batch.token_features

    def embed(z_rows, g0, g1):
        pe = use_conditioning * z_rows
        rel = _rel_feat_rows(tf, g0, g1, pe.dtype)
        f2d = torch.concatenate([pe, rel], dim=-1)
        return dh.pair_cond_initial_projection(dh.pair_cond_initial_norm(f2d))

    trans = [lambda x, g0, g1: dh.pair_transition_0(x), lambda x, g0, g1: dh.pair_transition_1(x)]
    out = torch.empty((lay.R, lay.N, int(dh.pair_channel)), dtype=adt, device=z.device)   # the activation dtype (bf16 under the kit's autocast), as the dense pair_cond
    ctx.zc = DF.pair_cond_rows(embed, _ztrunk_source(z), lay, c_out=int(dh.pair_channel), rows=ctx.sched.cond_rows or None, transitions=trans, out=out)
    STATE["stats"]["zcond_rows"] += lay.R
    ctx.cache = DF.PairBiasCache(enabled=ctx.sched.bias_cache)
    seq_mask = batch.token_features.mask
    ctx.blocks = [dit_block_fns(dh.transformer, i, seq_mask) for i in range(int(dh.transformer.num_blocks))]
    ctx.enc_static = _encoder_static_banded(dh.atom_cross_att_encoder, batch, embeddings["single"], ctx.zc, lay, ctx.sched)
    ctx.dec_pl = dh.atom_cross_att_decoder.atom_transformer_decoder.compute_pair_logits(ctx.enc_static["pair_cond"])
    ctx.primed = True
    _mark("diffusion_start")


def _dh_forward_sharded(ctx: _DiffCtx, positions_noisy, noise_level, batch, embeddings, use_conditioning: bool):
    """``DiffusionHead.forward`` with the pair conditioning as this rank's rows: single conditioning / atom encoder / decoder replicated (the
    engine's statements), the 24-block transformer = diffusion.diffusion_transformer_sharded."""
    C = STATE["C"] if "C" in STATE else _core()
    # every rank conditions on ONE set of noisy positions per step: the replicated statements of this engine run per rank on kernels each
    # process autotunes (torch.compile, the fused Triton tiles), so replicated tensors agree to kernel-selection rounding, not bitwise —
    # rank 0's positions are broadcast at every denoiser call (GATE positions_sync) so that rounding never compounds across steps
    synced = C["DF"].sync_replicated(positions_noisy.contiguous(), "diffusion.positions_noisy", mode=_positions_sync_mode())
    if synced.data_ptr() != positions_noisy.data_ptr():
        positions_noisy.copy_(synced)                                       # IN PLACE into the sampler's own tensor: the engine's update step after this call
    del synced                                                            # (positions_noisy + d_t * grad) then runs on rank 0's state on EVERY rank, so the ranks'
                                                                          # trajectories are one trajectory by construction (exit spread = the update's rounding only)

    C = STATE["C"] if "C" in STATE else _core()
    from xfold.nn import diffusion_head as DHM
    dh, lay = ctx.model.diffusion_head, ctx.lay
    if not ctx.primed:
        _prime_diffusion(ctx, batch, embeddings, use_conditioning)
    trunk_single_cond = dh._single_conditioning(batch, embeddings, noise_level, use_conditioning)
    sequence_mask = batch.token_features.mask
    atom_mask = batch.predicted_structure_info.atom_mask
    act = positions_noisy * atom_mask[..., None]
    act = act / torch.sqrt(noise_level ** 2 + DHM.SIGMA_DATA ** 2)
    enc = dh.atom_cross_att_encoder(batch=batch, token_atoms_act=act, trunk_single_cond=embeddings['single'], trunk_pair_cond=ctx.zc,
                                    static=ctx.enc_static)                                # trunk_pair_cond is only asserted non-None when static is given
    act = enc.token_act
    act += dh.single_cond_embedding_projection(dh.single_cond_embedding_norm(trunk_single_cond))
    act = C["DF"].diffusion_transformer_sharded(ctx.blocks, act, trunk_single_cond, ctx.zc, lay, schedule=ctx.sched, bias_cache=ctx.cache)
    STATE["stats"]["dit_blocks"] += len(ctx.blocks)
    act = dh.output_norm(act)
    position_update = dh.atom_cross_att_decoder(batch=batch, token_act=act, enc=enc, pair_logits=ctx.dec_pl)
    skip_scaling = DHM.SIGMA_DATA ** 2 / (noise_level ** 2 + DHM.SIGMA_DATA ** 2)
    out_scaling = noise_level * DHM.SIGMA_DATA / torch.sqrt(noise_level ** 2 + DHM.SIGMA_DATA ** 2)
    out = (skip_scaling * positions_noisy + out_scaling * position_update.to(torch.float32)) * atom_mask[..., None]   # DiffusionHead.forward's statement: the denoised coordinates are formed in fp32
    out = C["DF"].sync_replicated(out.contiguous(), "diffusion.denoised", mode=_positions_sync_mode())   # rank 0's denoised on every rank: the sampler's
    return out                                                            # update then yields ONE next state bitwise on every rank (input state + output both rank 0's)


POSITIONS_SYNC = "bcast"      # how the diffusion positions are kept replicated across ranks: rank 0 authoritative (its denoiser input state and denoised output
                              # replace every rank's at each call, its samples after the roll-out)


def det_level() -> int:
    """The determinism level recorded on the schedule census: 1 when deterministic kernels are in force in this process
    (``torch.are_deterministic_algorithms_enabled()`` — this kit ships no such mode), else 0."""
    try:
        return 1 if torch.are_deterministic_algorithms_enabled() else 0
    except Exception:
        return 0


def _positions_sync_mode() -> str:
    """The replicated-tensor sync policy of the diffusion roll-out (POSITIONS_SYNC = ``bcast``): rank 0 authoritative — its denoiser input
    state and denoised output replace every rank's at each call, its samples after the roll-out (the replicated statements run per rank on
    per-process-autotuned kernels: equal to rounding, not bitwise, so nothing is left to compound). Recorded once per process: lever
    ``positions_sync``, census ``det`` / ``noise_sync`` / ``diff_noise_sync``."""
    m = POSITIONS_SYNC
    if not STATE.get("sync_recorded"):
        C = STATE["C"] if "C" in STATE else _core()
        C["EV"].record_schedule(det=det_level(), noise_sync=m, diff_noise="bcast_rank0_state")
        STATE["sync_recorded"] = True
    _lever("positions_sync", m)
    return m


def _dh_dispatch(dh):
    """The instance-level ``forward`` of the DiffusionHead while a sharded roll-out of THIS rank is active (``STATE['diff_ctx']``, per rank);
    with no active context it is the class's own forward (a dense call between items is served unchanged)."""
    def forward(positions_noisy, noise_level, batch, embeddings, use_conditioning=True):
        ctx = STATE.get("diff_ctx")
        if ctx is None:
            return type(dh).forward(dh, positions_noisy, noise_level, batch, embeddings, use_conditioning)
        return _dh_forward_sharded(ctx, positions_noisy, noise_level, batch, embeddings, use_conditioning)
    forward._rowpair_dispatch = True
    return forward


ROLLOUT_PRIVATE_STATICS = ("pair_cond", "enc_pair_logits", "queries_single_cond", "keys_single_cond")   # _encoder_static_banded's own tensors (never the batch's)


def _release_rollout(ctx) -> float:
    """The roll-out is over: return the DEVICE STORAGE of everything the sharded conditioning built for it — z_cond rows, the pair-bias
    cache blocks, the banded encoder statics this adapter created, the decoder pair logits — whoever still references the Python objects.
    Why by storage and not by reference: a denoiser call whose shapes made torch.compile (re)compile on this rank leaves that call's Python
    frames referenced past their return (observed: ``_dh_forward_sharded`` / the DiT and atom-encoder ``forward`` frames alive at the next
    item, on the compiling rank only), and those frames' locals pin the whole context — up to u + 3u + the statics of device memory carried
    into the heads and the NEXT item (rank-asymmetric peaks, an allocator at the card's limit at 8k tokens). ``untyped_storage().resize_(0)``
    (the support library's ``shard.release_storage_``) frees the memory now; the tensor objects die whenever their last referrer does.
    Returns the GB released (census ``rollout_released_gb``)."""
    from opt_core.mem.rowpair.shard import release_storage_
    freed = 0
    seen = set()

    def rel(t):
        nonlocal freed
        if not torch.is_tensor(t) or not t.is_cuda:
            return
        st = t.untyped_storage()
        key = int(st.data_ptr()) if st.nbytes() else id(st)
        if key in seen or st.nbytes() == 0:
            return
        seen.add(key)
        nb = int(st.nbytes())
        try:
            release_storage_(t)
            freed += nb
        except Exception:                                                # noqa: BLE001 — a storage that cannot be resized (shared / external) is left to its referrers
            pass

    rel(getattr(ctx, "zc", None))
    cache = getattr(ctx, "cache", None)
    store = getattr(cache, "store", None)
    if isinstance(store, dict):
        for t in list(store.values()):
            rel(t)
        store.clear()
    rel(getattr(ctx, "dec_pl", None))
    es = getattr(ctx, "enc_static", None)
    if isinstance(es, dict):
        for k in ROLLOUT_PRIVATE_STATICS:
            rel(es.get(k))
    for k in ("zc", "cache", "dec_pl", "enc_static", "blocks", "sched"):
        if hasattr(ctx, k):
            setattr(ctx, k, None)
    gb = freed / 1e9
    st = STATE["stats"]
    st["rollout_released_gb"] = round(float(st.get("rollout_released_gb") or 0.0) + gb, 3)
    return gb


def _leakcheck(ctx) -> None:
    """Diagnostic (``XFOLD_TP_LEAKCHECK=1``): remember WEAK references to the roll-out context's large members; :func:`_leakreport` (called at
    the next confidence pass and the next item's entry) prints, for every member still alive then, the chain of referrers gc can see."""
    import weakref
    refs = []
    for k in ("zc", "dec_pl"):
        v = getattr(ctx, k, None)
        if v is not None:
            try: refs.append((k, weakref.ref(v)))
            except TypeError: pass
    es = getattr(ctx, "enc_static", None)
    if isinstance(es, dict):
        for k, v in es.items():
            if torch.is_tensor(v) and v.is_cuda and v.numel() * v.element_size() > (64 << 20):
                refs.append((f"enc_static[{k}]", weakref.ref(v)))
    cache = getattr(ctx, "cache", None)
    for attr in ("entries", "_entries", "cache", "_cache", "store", "_store", "blocks", "_blocks"):
        v = getattr(cache, attr, None)
        if isinstance(v, (list, dict)) and len(v):
            e0 = (list(v.values()) if isinstance(v, dict) else list(v))[0]
            if torch.is_tensor(e0):
                refs.append((f"cache.{attr}[0]", weakref.ref(e0)))
            break
    try: refs.append(("ctx", weakref.ref(ctx)))
    except TypeError: pass
    STATE["_leak_refs"] = refs


def _leakreport(where: str) -> None:
    """Print the referrer chains (depth 3) of every remembered roll-out member that is still alive (``XFOLD_TP_LEAKCHECK=1``)."""
    refs = STATE.get("_leak_refs") or []
    if not refs:
        return
    import gc
    def desc(r):
        if isinstance(r, dict):
            return "dict[" + ",".join(str(k)[:28] for k in list(r.keys())[:10]) + "]"
        if type(r).__name__ == "frame":
            return f"frame:{r.f_code.co_name}@{os.path.basename(r.f_code.co_filename)}:{r.f_lineno}"
        if isinstance(r, (list, tuple)):
            return f"{type(r).__name__}(len={len(r)})"
        if type(r).__name__ in ("cell", "method", "function", "builtin_function_or_method", "partial"):
            return f"{type(r).__name__}:{getattr(r, '__qualname__', getattr(getattr(r, 'func', None), '__qualname__', '?'))}"
        return f"{type(r).__module__}.{type(r).__qualname__}"
    def chain(obj, depth, seen):
        out = []
        if depth == 0:
            return out
        for r in gc.get_referrers(obj):
            if id(r) in seen or r is refs or type(r).__name__ == "frame" and r.f_code.co_name in ("chain", "_leakreport"):
                continue
            seen.add(id(r))
            out.append(("  " * (4 - depth)) + desc(r))
            if isinstance(r, (dict, list, tuple)) or type(r).__name__ in ("cell", "partial", "traceback", "frame"):
                out.extend(chain(r, depth - 1, seen))
        return out
    alive = [(k, w()) for k, w in refs if w() is not None]
    _log(f"[leakcheck] at {where}: alive={[k for k, _ in alive]} of {[k for k, _ in refs]}")
    for k, obj in alive:
        lines = chain(obj, 3, {id(alive), id(obj)})
        _log(f"[leakcheck] {k} {tuple(obj.shape) if torch.is_tensor(obj) else type(obj).__name__} referrers:\n" + "\n".join(lines[:40]))
    STATE["_leak_refs"] = []


RANK_SPREAD_REFUSE_A = 1.0   # a cross-rank spread of the finished samples above this (Angstrom) is refused by name: the per-call state broadcast keeps
                             # every rank on rank 0's trajectory, so the residual spread is the last step's replicated arithmetic only (~1e-3 A)


def _rank_spread_A(C, xyz) -> float:
    """max over ranks of ``max |xyz_rank - xyz_rank0|`` (Angstrom): rank 0's tensor is broadcast into a scratch copy, the local max-abs
    difference all-reduced with ``max`` (two collectives; every rank gets the same number). Recorded as ``diff_rank_spread_A``."""
    D = C["D"]
    if not D.is_dist():
        return 0.0
    ref = C["BC"].broadcast_tensordict({"xyz": xyz.detach().clone().contiguous()}, src=0)["xyz"].to(xyz.device)   # rank 0's samples, a scratch copy
    d = (xyz.float() - ref.float()).abs().amax().reshape(1)
    D.allreduce_(d, op="max")
    v = float(d.item())
    C["EV"].record_schedule(diff_rank_spread_A=round(v, 6), diff_noise="bcast_rank0_state")
    STATE["stats"]["diff_rank_spread_A"] = v
    return v


def run_diffusion_sharded(model, batch, emb_loc, layout, seed, *, num_samples: Optional[int] = None, steps: Optional[int] = None):
    """The AF3 sampler (``af3_torch_api.run_diffusion``: the engine's loop, schedule, random augmentation and noise draws, torch RNG seeded with
    ``seed`` identically on every rank) with the denoiser's pair conditioning ROW-SHARDED (``_dh_forward_sharded``). Returns the positions
    ``[S, N, 24, 3]`` fp32, REPLICATED on every rank (proven by diffusion.sync_replicated). The kit's hoist / step-graph levers are held off
    for the call (their dense statics cannot be built from a shard; the sharded per-roll-out statics replace the hoist) and restored after."""
    C = STATE["C"] if "C" in STATE else _core()
    import af3_torch_api as A
    lay = C["D"].require_sharded(layout, "run_diffusion_sharded")
    dh = model.diffusion_head
    if not getattr(dh.__dict__.get("forward"), "_rowpair_dispatch", False):
        dh.forward = _dh_dispatch(dh)                                    # the one instance-level hook of the adapter (the sampler loop calls self.diffusion_head(...))
    saved = (getattr(dh, "use_hoist", False), getattr(dh, "use_step_graph", False), getattr(model, "sample_batch", 0))
    dh.use_hoist = False                                                 # GATE (R-TP-3): hoist -> the sharded per-roll-out statics; stepgraph -> off (collectives inside a graph)
    dh.use_step_graph = False
    model.sample_batch = 0                                               # GATE: sbatch -> skipped (the sharded denoiser serves one sample's rows per call: the serial sampler)
    _lever("hoist", "replaced:rowpair_zcond_rows+bias_cache+band_statics")
    _lever("stepgraph", "off:n_gpu>1")
    _lever("sbatch", "skipped:rowpair_sampler")
    if getattr(model.diffusion_head, "use_atom_window", False):   # lever 'atom_window': its operands ride the single-GPU hoist (DiffusionHead.prime_static), which this
        model.diffusion_head.use_atom_window = False                # schedule replaces -> the stock atom blocks on every rank, by name
    _lever("atom_window", "skipped:rowpair_hoist")
    _lever("atom_rows", "skipped:rowpair_hoist")                 # rides atom_window's path (never reached here)
    if getattr(model, "batched_prologue", False):                # lever 'prologue': the batched sampler's; the sharded roll-out is per sample
        model.batched_prologue = False
    _lever("prologue", "skipped:rowpair_sampler")
    plan = _ztrunk_new(C, emb_loc["pair"], passes=int(num_samples or model.num_samples))   # the fp32 trunk shard's placement from here to the distogram (heads.ZTrunkPlan):
    _lever("ztrunk_entry", plan.park_now())                             # parked:<where> under ROWPAIR_CONF_PARK_ZTRUNK (this kit's default at P > 1) — its device storage is released
    STATE["diff_ctx"] = _DiffCtx(model, lay)                             # for the roll-out and the passes; z_cond and each pass's pair input read its rows from the host block by block
    try:
        xyz = A.run_diffusion(model, batch, emb_loc, seed=seed, steps=steps, num_samples=num_samples)
    finally:
        ctx, STATE["diff_ctx"] = STATE["diff_ctx"], None
        dh.use_hoist, dh.use_step_graph, model.sample_batch = saved
        if ctx is not None:
            if os.environ.get("XFOLD_TP_LEAKCHECK"):                      # diagnostic: which roll-out members outlive it and through whom (rank transcripts; _leakreport)
                _leakcheck(ctx)
            _release_rollout(ctx)                                        # the conditioning's device storage back NOW, whoever still references the objects (docstring)
        del ctx
    _mark("diffusion_peak")
    xyz = xyz.contiguous()
    spread = _rank_spread_A(C, xyz)                                       # census diff_rank_spread_A: max |x_rank - x_rank0| over ranks / samples / atoms (all-reduced), BEFORE the replace
    if spread > RANK_SPREAD_REFUSE_A:
        raise XfoldTPRefused(f"diffusion.positions: cross-rank spread {spread:.3g} A > {RANK_SPREAD_REFUSE_A} A before the rank-0 replace "
                             f"(diff_rank_spread_A) — the ranks' roll-outs diverged beyond kernel-selection rounding")
    xyz = C["DF"].sync_replicated(xyz.contiguous(), "diffusion.positions", mode=_positions_sync_mode())   # rank 0's sample set on every rank (the heads read one structure)
    return xyz


# ============================================================================================================================ guards
class ConsumeOnce(dict):
    """The prev embeddings handed to Evoformer.forward under prev_free: a key is read once and its reference dropped here AND in the source
    dict, so the fp32 prev pair is freed as soon as the Evoformer has embedded it; a second read is a named error, never a silent None."""

    def __init__(self, src):
        super().__init__(src)
        self._src = src
        self.freed = 0

    def __getitem__(self, k):
        if not dict.__contains__(self, k):
            raise KeyError(f"prev[{k!r}] read twice under prev_free (the reference was released on the first read)")
        v = dict.pop(self, k)
        self._src.pop(k, None)
        self.freed += 1
        return v


def assert_params_replicated(model) -> str:
    """R1 guard, once per model process: the model's parameters are bitwise identical on every rank — ONE fp64 checksum per rank
    (sum over parameters and buffers of their fp64 sums, + the element count) all-gathered and compared (``bcast.assert_replicated``); a
    per-rank weight desync (a loader or RNG-dependent constructor differing by rank) REFUSES BY NAME (``params@install``) instead of folding
    garbage. Returns ``ok:<n tensors>``."""
    from opt_core.mem.rowpair import bcast
    n, tot, cnt = 0, torch.zeros((), dtype=torch.float64), 0
    for t in list(model.parameters()) + list(model.buffers()):
        tot = tot + t.detach().to(dtype=torch.float64).sum().cpu(); cnt += int(t.numel()); n += 1
    bcast.assert_replicated({"param_checksum": torch.stack([tot, torch.tensor(float(cnt), dtype=torch.float64)])}, "params@install")
    return f"ok:{n}"


def assert_features_replicated(batch) -> str:
    """R1 guard (always on): the featurised batch is bitwise identical on every rank (``bcast.assert_replicated``: checksums all-gathered;
    a differing leaf is refused by name — ``RowpairRefused`` naming ``features@S1.<leaf>``). Returns ``ok:<n leaves>`` for the item record."""
    from opt_core.mem.rowpair import bcast
    leaves = {k: v for k, v in dict(batch).items() if torch.is_tensor(v)}
    bcast.assert_replicated(leaves, "features@S1")
    return "ok:%d" % len(leaves)


def assert_rng_replicated(torch_mod) -> str:
    """R1 guard (always on): the CPU and CUDA torch generator states are bitwise identical on every rank right before the trunk (the MSA
    row shuffle and the diffusion sampler draw from them; forward.py seeds them per item from the item's explicit seed).
    ``dist.allreduce_checksum`` refuses by name (``rng_state@trunk.cpu`` / ``.cuda``) on a mismatch."""
    from opt_core.mem.rowpair import dist as D
    D.allreduce_checksum(torch_mod.get_rng_state(), "rng_state@trunk.cpu")
    if torch_mod.cuda.is_available():
        D.allreduce_checksum(torch_mod.cuda.get_rng_state(), "rng_state@trunk.cuda")
    return "ok"


def barrier():
    from opt_core.mem.rowpair import dist as D
    if D.is_dist():
        D.barrier()


BOUNDARY_SYNC = "barrier"                                          # the stage-boundary sync policy under n_gpu > 1 (census word ``boundary_sync``)


def stage_boundary(name: str):
    """The stage boundary ``name`` (trunk_done / diffusion_done) under n_gpu > 1: every rank's work of the stage is complete (device
    synchronize) and the ranks have MET (``dist.barrier``) before the next stage posts its first point-to-point group. Recorded in the census:
    ``stats['boundary_sync'] = 'barrier'`` and ``stats['boundaries']`` (the boundary names passed on this item), printed on the STAGES line."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    st = STATE["stats"]
    st["boundary_sync"] = BOUNDARY_SYNC
    st["boundaries"] = list(st.get("boundaries") or []) + [str(name)]
    return BOUNDARY_SYNC


def gather_peaks(value: Optional[float] = None) -> dict:
    """{rank: peak GiB} over the group (COLLECTIVE) — the core's ``evidence.gather_peaks`` (each rank's ``max_memory_allocated``), or, when the
    driver passes ``value`` (its max over the stage-local peaks it resets between stages), that value all-gathered (``dist.comm().allgather_obj``)."""
    from opt_core.mem.rowpair import evidence as EV
    if value is None:
        return EV.gather_peaks()
    C = STATE["C"] if "C" in STATE else _core()
    if not C["D"].is_dist():
        return {0: float(value)}
    P, r = C["D"].world()
    return {int(rr): float(g) for rr, g in C["D"].comm().allgather_obj((r, float(value)))}


def census() -> dict:
    """This rank's adapter census (counts + lever states + the replicated-by-design list) for the item record / the schedule census."""
    out = dict(STATE["stats"])
    out["xfold_replicated"] = "single,target_feat,msa,token_features,atoms,diffusion_positions,triangle_bias[N,N,4],conf_full_matrices_rank0"
    return out

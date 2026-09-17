"""boltz2_opt.rowpair — the memory mode's tensor-parallel line at ``--n_gpu P > 1`` (registry: ``rowpair_tp``, strategy F7.tensor_parallel):
boltz 2.2.1's trunk with the pair representation ``z [1, N, N, 128]`` ROW-SHARDED across the P ranks of one node — rank r holds rows
``[r0, r1)`` of z (``z_loc [1, R, N, 128]``, fp32 storage) from the statement that creates it (``boltz2.py:420-429``: the pair init) through
recycling, the template module, the MSA module, the 64-block Pairformer and the distogram head; nothing ``N x N x 128`` is materialised
whole on a rank inside the trunk. The mechanism — layout, every collective (ring / all-to-all TriMul contractions, the triangle-bias
all-gather, the row all-gathers, the shard transpose), the row-block schedules and their census — is the core's (``opt_core.mem.rowpair``);
what is boltz2's and lives here is the STATEMENT of each boltz module over a row shard: the stock module's own sub-modules (their weights,
LayerNorms, projections, gates, the mode's triangle-attention kernel) called on this rank's rows in the stock order, with row-sliced ``i``
operands where a statement pairs token ``i`` with token ``j``.

Seams (of3 numbering; boltz sites ``stock/src/boltz/model/...``):
  1  pair init            z_init rows: ``z_init_1(s)[:, r0:r1, None] + z_init_2(s)[:, None, :]`` + rel_pos rows + token_bonds rows (+ bond types)
                          + contact_conditioning rows (row views of the ``feats`` pair planes) — boltz2.py:420-429
  2  recycling            ``z_loc = z_init_loc + z_recycle(z_norm(z_loc))`` — boltz2.py:455, row-local
  3  template module      TemplateV2Module.forward (trunkv2.py:411-509) with the ``i`` operands of its pair features sliced to the rows; its
                          2-block pair stack presharded (one template at a time); ``u`` rows added to z_loc
  4  MSA representation   ``m [1, S, N|R, 64]`` TOKEN-SHARDED under ``ROWPAIR_MSA_M_LAYOUT=token_sharded`` (the default: rank q embeds
                          and updates its token columns ``r0_q:r1_q`` — the pair shard's rows; ``boltz2_opt.rowpair_msa.msa_input`` one-hots
                          only those columns) or replicated ``[1, S, N, 64]`` under ``=replicated``; S = every MSA row when the module does not
                          subsample (the worker line: up to 8192), ``<= num_subsampled_msa`` when it does; the subsample draw (trunkv2.py:632,
                          CPU generator) proven identical on every rank (core ``trunk.guard_replicated``); the raw MSA features on the device or
                          HOST-RESIDENT under ``ROWPAIR_MSA_HOST`` (``boltz2_opt.rowpair_msa``: only the rows a cycle uses reach the device)
  5  OuterProductMean     output rows local: core ``msa.opm_rows_budgeted`` over boltz's own chunked outer statement (outer_product_mean.py:57-98);
                          token-sharded m: ``a`` = this rank's token rows, ``b`` all-gathered once per call (``[S, N, 32]``)
  6  PairWeightedAvg      bias from local z rows, softmax over j within a row, values from all of m: core ``msa.pwa_rows`` over boltz's per-head
                          statement (pair_averaging.py:73-135); token-sharded m: one S-chunk of m all-gathered at a time (``ROWPAIR_PWA_S_CHUNK``,
                          128 rows) feeds every head's values / gates, the m-update stays on the token shard; replicated m: the token rows of the
                          m-update all-gathered
  7  MSA pair layers      presharded through the layer statement below (no gather between MSA blocks)
  8-12 pair layer         TriMul out = core ring-streamed contraction; TriMul in = core all-to-all; TriAtt start = rows + all-gathered bias;
                          TriAtt end = core shard transpose; Transition row-local (``_pair_layer``)
  13 seq attention        local query rows + all-gather of the ``[R, 384]`` s update (core ``transition.apb_local_queries``)
  14 trunk Pairformer     presharded, 64 blocks, z leaves as rows
  15 distogram            ``(z + zT)`` rows = z_loc + core shard transpose, Linear on rows -> ``pdistogram`` rows
  16-17 confidence         ``boltz2_opt.rowpair_heads``: the confidence pair init on rows, its Pairformer presharded, PAE logits per row block,
                          PDE over ``(z + zT)`` rows, the pTM family as row partial sums, pae / pde ``[N, N]`` assembled on rank 0's HOST only
  18 diffusion cond.      ``rowpair_heads``: z_cond rows once per roll-out, the 24 per-layer token pair biases per row block, the atom
                          encoder's pair term through row bands (``diffusion.pair_band_rows``) — ``token_trans_bias`` never whole
  19 diffusion transf.    ``rowpair_heads``: AtomDiffusion.sample with the token transformer's queries = this rank's rows (one all-gather per
                          block); every denoiser call conditions on rank 0's state, rank 0's coordinates replace every rank's at exit
                          (``diff_rank_spread_A`` recorded, refused above 1.0 A)

Replicated by design (named in ``report()['replicated']`` and the schedule census): s / s_inputs ``[1, N, 384]``, the MSA representation m
only under ``ROWPAIR_MSA_M_LAYOUT=replicated`` (token-sharded by default), the MSA mask ``[1, S, N]``, the per-token feats, the full token pair mask ``[1, N, N]`` fp32, the triangle-attention bias ``[4, N, N]`` per call (all-gathered), the
atom-level tensors and coordinates of the sampler. Host-resident: the ``feats`` token-pair planes (token_bonds, type_bonds,
contact_conditioning, contact_threshold: rows to the device per block — named, ``PAIR_PLANES``; nothing else the featurizer emits is moved:
a per-token feature is shaped like a pair plane whenever its width equals N — res_type / profile / msa / template_restype are one-hot over
``const.num_tokens`` = 33, so at N = 33 they ARE ``[1, N, N]`` — and the model reads those on the device; the distogram loss's target
``disto_target`` [1, N, N, E, 64], read by nothing at inference, leaves the device whole by name, ``PARK_UNREAD``; any other pair-shaped tensor
stays where the featurizer put it and is named in the census, ``pair_shaped_on_device``). Generators: rank 0's CPU and CUDA generator states replace every rank's at each sharded-forward entry.

Placement words the line exports (``modes.TP_EXPORTS``, set by ``stack.child_env`` at P>1 — the row's alone: a caller's copy is stripped with the
other lever words; ``report()['tp_exports']`` = the values in force, the core's schedule census = what they did): ``ROWPAIR_PARK_ZINIT=1`` —
the initial pair shard ``z_init [R, N, 128]`` is PARKED on pinned host between recycles by the trunk driver (core ``trunk.ShardPark`` in
``run_trunk_sharded``; the recycle statement reads it back per row block): the device holds ``z`` + row blocks, not ``z`` + ``z_init``,
through the template / MSA / Pairformer stages (−512·N²/P bytes per rank; census ``park_z_init=host_pinned|host_pageable:<kind>|host|device``);
``ROWPAIR_MSA_HOST=rank0`` — the raw MSA features stay on rank 0's pinned host and only the rows a cycle uses reach the device (seam 4,
``boltz2_opt.rowpair_msa``; census ``msa_host= msa_rows=<n>/<S> msa_host_cycle_gib=``). ``ROWPAIR_MSA_M_LAYOUT=token_sharded`` — the MSA
representation is token-sharded across the ranks (seams 4-6; ``=replicated`` keeps it whole per rank; census ``msa_m= msa_cols= msa_m_gib=
msa_m_dtype= pwa_s_chunk= opm_a=local opm_b_gather_dtype= opm_b_gather_gib=``). ``ROWPAIR_CONF_PARK_ZTRUNK=1`` /
``ROWPAIR_FREE_ZTRUNK=1`` — the trunk pair shard's placement from the roll-out entry through the confidence passes is the core's
``heads.ZTrunkPlan`` (``_ztrunk_plan``; ``TrunkShard.zplan``): parked on pinned host with its device storage released before ``z_cond`` is
allocated and served per block to the conditioning and to each pass's embedding (``parked``), or — no park live, one sample — overwritten in
place by the confidence pair input (``inplace``); resident with both ``=0``. Census ``conf_ztrunk_entry= conf_ztrunk= conf_ztrunk_host_gib=
conf_pairstack=inplace``; ``dict_out["z"]`` = :class:`ConsumedRows` once the rows are gone.

Mode levers at P>1 (``report()['replaced_levers']`` / ``['bypassed_levers']``): ``expandable_segments`` composes; ``xl_trans`` /
``xl_cond`` / ``xl_free`` / ``relpos_lazy`` are REPLACED — the row statements do their work by construction (transitions per row block,
the conditioning per row block, z as parked row shards, the relative position encoding per row block); the flash triangle attention
composes (a local query-row block is a valid call, reached through ``primitives.kernel_triangular_attn``); the fused ``fpf_trimul`` kernel
and the cuEquivariance TriMul take a whole pair tensor and serve no call — every TriMul is the stock projections around the core
contraction (``fpf_trimul``'s own census reads idle); the trunk kit's class-forward levers (resid, mask2) are
bypassed on every sharded layer because the layer is restated over its sub-modules. The TP line's arithmetic is therefore the STOCK
statements' in the stock dtypes with row-partitioned GEMMs (tier 2 against the n_gpu = 1 line, whose TriMul is the fused kernel).

Refused by name (the worker exits non-zero; never a dense fallback): attached at n_gpu=1; batch size != 1; a pair tensor whose storage dtype
is not fp32 at trunk entry; affinity inputs; ``TemplateModule`` (v1) checkpoints; a rank environment that disagrees with ``BOLTZ_TP``.

Contract (worker_launch / stack.evidence): ``LEVERS``; ``apply(spec=None)`` -> levers installed (process group from the launcher's rank
environment: ``opt_core.mem.rowpair.dist.init_from_env``; installs ``Boltz2.forward`` = the sharded trunk right after
``boltz.model.models.boltz2`` has executed, and the sharded ``PairformerModule.forward`` for pair stacks entered with a whole z);
``report()`` -> ``tp_report``. Switch: ``BOLTZ_TP=<P>`` (``stack.child_env``); rank / world / rendezvous variables are the launcher's ``ROWPAIR_*``.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

LEVERS = ("rowpair_tp",)
ENV_P = "BOLTZ_TP"
TAG = "[boltz2-opt]"
MODEL_MODULE = "boltz.model.models.boltz2"
_CENSUS: Dict[str, Any] = {}                          # the core's TP memory census module once resolved (stage marks; no-op unless OPT_CORE_TP_CENSUS=1)


def _mark(stage: str) -> None:
    """A stage mark for the core's TP memory census (``opt_core.mem.rowpair.census``) — instrumentation only: a core without the module marks nothing."""
    if "mod" not in _CENSUS:
        try:
            from opt_core.mem.rowpair import census as mod
        except ImportError:                          # gates instrumentation, never real work
            mod = None
        _CENSUS["mod"] = mod
    if _CENSUS["mod"] is not None:
        _CENSUS["mod"].mark(stage)          # the trunk install runs right after this module has executed (worker_launch._AfterImport)

REPLICATED = ("s", "s_inputs", "m (MSA representation) only under ROWPAIR_MSA_M_LAYOUT=replicated (token-sharded by default; S = every MSA row unless the module subsamples)",
              "msa_mask [1, S, N]", "per-token feats", "triangle bias [H, N, N] per attention call (all-gathered)", "atom-level tensors of the sampler", "model weights")
PWA_S_CHUNK_ENV = "ROWPAIR_PWA_S_CHUNK"                                   # MSA rows per all-gathered chunk of the token-sharded m in the pair-weighted averaging (default PWA_S_CHUNK)
PWA_S_CHUNK = 128                                                         # one gathered S-chunk [128, N, 64] + every head's values [128, N, 256] + this rank's gate rows: ≈1280·N bytes per row under the trunk's bf16 autocast (0.95 GB at N = 5780; ≈2304·N = 1.7 GB in fp32), against 12 GB for ONE head over all 8192 rows
HOST_ROWS = ("feats pair planes (token_bonds, type_bonds, contact_conditioning, contact_threshold): host-resident, rows to the device per block",
             "raw MSA features (msa, has_deletion, deletion_value, msa_paired) under ROWPAIR_MSA_HOST=all|rank0: host-resident (rank0: rank 0 only), the rows a cycle uses to the device per cycle (rowpair_msa)")
GATHERED = ()                                                             # no statement gathers the pair tensor: z, z_cond, the logits live and die as rows
PENDING_SEAMS = ()                                                        # seams 1-19 bound (16-19: boltz2_opt.rowpair_heads)
REPLACED_LEVERS = {                                                       # unit levers of the n_gpu = 1 line whose work the row statements do by construction at n_gpu > 1
    "xl_trans": "pair transitions run per row block of the shard (core pairstack transition_update_)",
    "xl_cond": "the diffusion conditioning is built per row block (rowpair_heads diffusion_conditioning_rows)",
    "xl_free": "z_init / z live as row shards parked and freed by the trunk driver (core trunk.run_trunk_sharded)",
    "relpos_lazy": "the relative position encoding is built per row block, never whole (_relpos_rows)",
    "fpf_trimul": "every triangle multiplication is the core's streamed contraction over stock projections (the fused whole-tensor kernel serves no call)",
    "cuequivariance_trimul": "idem: the cuEquivariance TriMul kernel is a whole-tensor kernel",
    "fpf_opm": "the outer-product-mean is the row statement _opm_add_ (core msa.opm_rows_budgeted over boltz's chunked outer statement) on every rank's rows: OuterProductMean.forward is never called, so the fused whole-tensor kernel is not installed per rank (msa_report disposition replaced_by_rowpair)",
    "fpf_pwa": "the pair-weighted averaging is the row statement _pwa_update (core msa.pwa_rows: bias from the rank's z rows) — PairWeightedAveraging.forward is never called, the fused kernel is not installed per rank (replaced_by_rowpair)",
}

REQUIRED_CALLS = ("trunk_rows", "relpos_rows", "recycles", "template_rows", "msa_blocks_rows", "pwa_rows", "opm_rows", "presharded_calls", "layers",
                  "distogram_rows", "transposes", "parameters_guarded", "zcond_rows", "dit_blocks_sharded", "sample_coords_rank0", "conf_rows")
                                                                          # every bound seam's call counter: each must read >= 1 on every rank after item 1 or the exit
                                                                          # verdict fails by name (an ACTIVE line is a claim; the counter is the evidence)
SIZE_GATES = ()                                      # R-TP-3: no threshold switches any statement to a replicated / gathered path at any N (tests/test_rowpair_gates.py)
BYPASSED_LEVERS = ("resid", "mask2 (trunk-lever class forwards: the sharded layer is restated over its sub-modules)")

_CALLS0 = {"trunk_rows": 0, "init_blocks": 0, "recycles": 0, "template_rows": 0, "msa_blocks_rows": 0, "presharded_calls": 0, "distogram_rows": 0, "gathers_in_trunk": 0,
           "gathers_named": 0, "whole_named": 0, "conf_rows": 0, "conf_heads_rows": 0, "zcond_rows": 0, "dit_blocks_sharded": 0, "pae_rows_to_rank0": 0,
           "relpos_rows": 0, "transposes": 0, "parameters_guarded": 0, "rng_bcast": 0, "rng_bcast_corrected": 0, "rng_guarded": 0, "pair_planes_to_host": 0, "host_pageable": 0, "contract_modules_checked": 0,
           "pairformer_module": 0, "pairformer_noseq_module": 0, "layers": 0, "trimul_out": 0, "trimul_in": 0, "triatt_start": 0, "triatt_end": 0,
           "transition": 0, "apb": 0, "opm_rows": 0, "pwa_rows": 0}
from . import modes, rowpair_msa

_STATE: Dict[str, Any] = {"installed": False, "P": 1, "rank": 0, "device": None, "orig": {}, "layouts": {}, "rows_census": None, "calls": dict(_CALLS0),
                         "guards": {"inputs_checked": 0, "inputs_equal": 0, "rng_checked": 0, "replicated_checked": 0}, "errors": [],   # inputs_*: rowpair_msa.feats_census per batch (the LEVER line's entry_inputs_equal=<equal>/<checked>); rng_checked: the CUDA RNG-state guard at AtomDiffusion.sample; replicated_checked: the MSA draw / mask guards (rowpair_msa)
                         "wall_s": {"sharded_modules": 0.0, "trunk": 0.0}, "t0": None, "trunk_installed": False, "contract_checked": False, "contract_stats": {}, "pair_shaped_on_device": set(), "parked_unread": set(), "tpx": None}


class Refused(RuntimeError):
    """A shape or setting this adapter does not serve, named (the worker exits non-zero; never a silent dense run)."""


# ---------------------------------------------------------------- the fused triangle kernels of the ×P line ----------------------------------------------------------------
# Two switches name a KERNEL for the two triangle operations of every row block, served by the core's providers (detected by presence:
# opt_core.mem.rowpair.triatt.attention_core / opt_core.mem.rowpair.trimul_fused.fused_trimul_fns) around THIS module's statements:
#   ROWPAIR_TRIATT_CORE = flash_triattn | cueq | torch   the triangle-attention core of each row batch: the fused flash kernel (Tier-2), the
#                            cuEquivariance kernel (the module's own `use_kernels` route, row-chunked under 2^31 elements) or the eager statement —
#                            through the core's adapter, counted (LEVER name=F1.flash_triattn); torch / unset = the module's own `mha(..., use_kernels=)`
#                            statement per row block (no adapter, no census line).
#   ROWPAIR_TRIMUL_KERNELS = fpf_v4 | torch              the row-sharded triangle multiplication: the core's fused K1 -> bmm -> K3 provider over the
#                            module's weights (Tier-2, the core's own fpf_trimul_v4 cell table; LEVER name=F2.trimul_rows) or the torch contraction around
#                            the module's own projections (torch / unset: no provider, no census line).
# The kit modes export both words at n_gpu > 1 (modes.TP_EXPORTS: flash_triattn / fpf_v4; a caller's `=torch` opts out). A named kernel the
# installed core cannot provide is a NAMED state (`unavailable:<why>`: tp_report.tpx, stack.evidence's fallback words) and this module's own
# statements serve — never a silent substitution. The names and vocabulary are the core's (its steering variables, kernel words, census fields).
TPX_TRIATT_ENV = "ROWPAIR_TRIATT_CORE"
TPX_TRIMUL_ENV = "ROWPAIR_TRIMUL_KERNELS"
TPX_WORDS = {"triatt": ("flash_triattn", "cueq", "torch"), "trimul": ("fpf_v4", "torch")}
TPX_TIER_PREFIX = "tier:"                                                   # + triatt `tier:<word>`: the core adapter's tier door (opt_core.mem.rowpair.triatt.TIER_PREFIX; the row kernels.triattn selects for the word per row
TPX_TIER_WORDS = ("big", "fast", "exact")                                 #   window — triattn_native on 9.0 by the cell table — stepping aside BY NAME to the flash path where the provider refuses; the ×P line exports tier:big, modes.TP_EXPORTS)
TPX_ENVS = {"triatt": TPX_TRIATT_ENV, "trimul": TPX_TRIMUL_ENV}
TPX_PROVIDERS = {"triatt": ("opt_core.mem.rowpair.triatt", "attention_core"), "trimul": ("opt_core.mem.rowpair.trimul_fused", "fused_trimul_fns")}


def tpx_words(environ=None) -> Dict[str, Optional[str]]:
    """``{"triatt": word | None, "trimul": word | None}`` from the two switches (None = unset/empty = this module's own statements); an unknown word is
    ``Refused`` by name."""
    env = os.environ if environ is None else environ
    out: Dict[str, Optional[str]] = {}
    for kind, name in TPX_ENVS.items():
        w = (env.get(name) or "").strip().lower() or None
        tier_ok = kind == "triatt" and w is not None and w.startswith(TPX_TIER_PREFIX) and w[len(TPX_TIER_PREFIX):] in TPX_TIER_WORDS   # the core's tier door: kernels.triattn's row for the word per row window
        if w is not None and w not in TPX_WORDS[kind] and not tier_ok:
            extra = f" | {TPX_TIER_PREFIX}<{'|'.join(TPX_TIER_WORDS)}>" if kind == "triatt" else ""
            raise Refused(f"refused: {name}={env.get(name)!r}: one of {' | '.join(TPX_WORDS[kind])}{extra} (or unset)")
        out[kind] = w
    return out


def _core_version() -> Tuple[Tuple[int, ...], str]:
    try:
        import opt_core
        v = str(getattr(opt_core, "__version__", "0"))
    except ImportError:
        return (0,), "absent"
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4]) or (0,), v


def tpx_provider(kind: str):
    """``(module, None)`` when the installed core provides the ``kind`` kernel provider — detected by PRESENCE (the module imports and carries
    the function; the core's version is recorded, never the test), else ``(None, <why>)``: ``core_missing:<module>`` / ``core_api_missing:<module>.<name>``."""
    import importlib
    modname, attr = TPX_PROVIDERS[kind]
    try:
        mod = importlib.import_module(modname)
    except ImportError as e:
        return None, f"core_missing:{modname}({type(e).__name__})"
    if not hasattr(mod, attr):
        return None, f"core_api_missing:{modname}.{attr}"
    return mod, None


def tpx_bind(environ=None) -> Dict[str, dict]:
    """Resolve the two switches ONCE per process into ``_STATE["tpx"]``: per kind ``{word, state, reason, provider}`` with ``state`` = ``off`` (unset)
    or ``torch`` (this module's own statements, no adapter), ``on`` (the word is served through the core provider) or ``unavailable`` (``reason``
    names why; this module's own statements serve). Idempotent; ``tpx_report()`` is the evidence view."""
    if _STATE.get("tpx") is not None:
        return _STATE["tpx"]
    words = tpx_words(environ)
    bound: Dict[str, dict] = {}
    for kind, word in words.items():
        ent = {"word": word, "state": "off", "reason": None, "provider": None, "env": TPX_ENVS[kind]}
        if word == "torch":
            ent["state"] = "torch"                                         # this module's own statements, no adapter: the ×P line as it was before the fused kernels, byte for byte
        elif word is not None:
            mod, why = tpx_provider(kind)
            ent.update(state="on" if mod is not None else "unavailable", reason=why, provider=mod)
        bound[kind] = ent
    _STATE["tpx"] = bound
    return bound


def _plain(x):
    """JSON-safe copy of a census mapping (tuples -> lists, unknown objects -> str)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_plain(v) for v in x]
    return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def tpx_report() -> Dict[str, Any]:
    """``tp_report.tpx``: per kind ``{word, state, reason, env}`` + the core's version and, when a provider served, its census line
    (``lines``: the core renders them in the LEVER grammar; printed here with the kit's tag)."""
    bound = _STATE.get("tpx") or {}
    _, vs = _core_version()
    out: Dict[str, Any] = {"core": vs, "lines": []}
    for kind in TPX_ENVS:
        ent = bound.get(kind) or {"word": None, "state": "off", "reason": None, "env": TPX_ENVS[kind]}
        out[kind] = {k: ent.get(k) for k in ("word", "state", "reason", "env")}
        prov = ent.get("provider")
        if prov is not None and ent.get("state") == "on":
            emit = getattr(prov, "emit_core_line", None) or getattr(prov, "emit_line", None)
            desc = getattr(prov, "describe_core", None) or getattr(prov, "describe", None)
            try:
                line = emit(TAG.strip("[]")) if emit is not None else None       # the core prints its ONE LEVER line here (rank transcript) and returns it; it brackets the tag itself
                if isinstance(line, str) and line:
                    out["lines"].append(line)
                out[kind]["census"] = _plain(desc()) if desc is not None else None  # the core ledger's fields: served / fallback / fallback_by / cells / kernel …
            except Exception as e:  # noqa: BLE001 — the report must always be writable
                _STATE["errors"].append(f"tpx {kind} census: {type(e).__name__}: {e}")
    return out


def _say(msg: str) -> None:
    sys.stderr.write(msg + "\n"); sys.stderr.flush()


def n_gpu() -> int:
    try:
        return max(1, int(os.environ.get(ENV_P, "1") or "1"))
    except ValueError:
        raise Refused(f"refused: {ENV_P}={os.environ.get(ENV_P)!r} is not an integer") from None


# ---------------------------------------------------------------- the core ----------------------------------------------------------------
def _core():
    from opt_core.mem.rowpair import dist as D, shard as SH, transition as TRN, triatt as TA, trimul as TM
    return D, SH, TRN, TA, TM


def _core_mod(name: str):
    """``opt_core.mem.rowpair.<name>`` or Refused(core_missing:<name>) — a seam whose core module is absent refuses by name, never degrades."""
    import importlib
    try:
        return importlib.import_module(f"opt_core.mem.rowpair.{name}")
    except ImportError as e:
        raise Refused(f"refused: reason=core_missing:opt_core.mem.rowpair.{name} ({type(e).__name__}: {e}) — the pinned opt_core does not carry this seam") from None


def _layout(N: int):
    D = _core()[0]
    lay = _STATE["layouts"].get(N)
    if lay is None:
        lay = D.default_ctx(int(N))                                      # balanced row parts aligned to ROWPAIR_CHUNK_ALIGN (boltz pins no chunk grid: align 1); refuses by name when (N, P) has no row grid
        _STATE["layouts"][N] = lay
        if _STATE["rows_census"] is None:                                # once per process, all ranks present: the rows tile 0:N exactly once
            from opt_core.mem.rowpair import evidence as EV
            _STATE["rows_census"] = dict(EV.rows_census(lay))
    return lay


def _R(lay) -> int:
    return int(getattr(lay, "R", None) if getattr(lay, "R", None) is not None else lay.n_loc)


def _gather_rows(x_loc, lay, dim: int):
    """All ranks' rows of ``x_loc`` concatenated along ``dim`` (core shard.gather_rows / unshard_rows) — the tests' assembly of a reference;
    no statement of the line calls it. Counted (``gathers_named``): the evidence gate fails a run whose count is not zero."""
    SH = _core()[1]
    g = getattr(SH, "gather_rows", None) or getattr(SH, "unshard_rows")
    _STATE["calls"]["gathers_named"] += 1
    return g(x_loc.contiguous(), lay, dim=dim) if "dim" in g.__code__.co_varnames else g(x_loc.movedim(dim, 0).contiguous(), lay).movedim(0, dim)


def _transpose_shard(z4, lay):
    """Rows ``[r0, r1)`` of z^T for this rank's z rows ``z4 [1, R, N, C]`` (core ring.transpose_shard: all-to-all, budgeted / streamed)."""
    try:
        RG = _core_mod("ring")
        return RG.transpose_shard(z4.contiguous(), lay)
    except Refused:
        D = _core()[0]
        return D.transpose_shards(z4[0].contiguous(), lay).unsqueeze(0)


# ---------------------------------------------------------------- the sharded layer statement (seams 8-13) ----------------------------------------------------------------
def _trimul_fns(mod):
    """TriangleMultiplication{Outgoing,Incoming}.forward (triangular_mult.py:105-125 / :194-214) as the core's ``TriMulFns``: ``proj`` = the
    input LayerNorm, the gated input projection and the mask on a block of ORIGINAL pair values, the fp32 split into the a / b halves
    (``torch.chunk(x.float(), 2)``: a = the first ``dim`` channels); ``out`` = ``p_out(norm_out(x))`` of a contraction tile; ``gate`` =
    ``sigmoid(g_out(norm_in(z)))`` read from the original values. The contraction and its schedule are the core's."""
    import torch
    C = int(mod.p_out.weight.shape[1])                                     # dim: the hidden width of each half (p_in: dim -> 2 dim)

    def proj(z_block, mask_block, is_a: bool):
        x = mod.norm_in(z_block)
        x = mod.p_in(x) * mod.g_in(x).sigmoid()
        x = x * mask_block
        a, b = torch.chunk(x.float(), 2, dim=-1)
        return a if is_a else b

    def out(x):
        return mod.p_out(mod.norm_out(x))

    def gate(z_block):
        return mod.g_out(mod.norm_in(z_block)).sigmoid()

    TM = _core()[4]
    fns = TM.TriMulFns(proj=proj, out=out, gate=gate, C_h=C)              # this kit's statement: the torch contraction around the module's projections
    tpx = tpx_bind()["trimul"]
    if tpx["word"] == "fpf_v4" and tpx["provider"] is not None:           # ROWPAIR_TRIMUL_KERNELS=fpf_v4 with a core that provides it: the fused K1 -> bmm -> K3 rows over the
        from .trimul import weights_of                                     # module's weights (the P=1 exact TriMul adapter's canonical tensors, opt_core.trimul.WEIGHT_KEYS); `fns` is its named fallback
        return tpx["provider"].fused_trimul_fns(weights_of(mod), fns, eps=float(mod.norm_in.eps))
    return fns


def _bias5(tb_full):
    """[N, N, H] -> [1, 1, H, N, N]: attention.py:150-153 (permute_final_dims(linear(x), (2, 0, 1)).unsqueeze(-4)) on the gathered bias."""
    return tb_full.permute(2, 0, 1).unsqueeze(0).unsqueeze(-4)


def _q_rows(N: int, use_kernels: bool) -> int:
    """The query-row block of the stock statement: the module's chunk rule (pairformer.py:182-186 / trunkv2.py:595-604: 128 rows above
    const.chunk_size_threshold tokens, 512 at or below; under the kernel the stock statement is one call — the row batch is then the
    same 128 / 512 grid, a valid independent call per batch); ``ROWPAIR_TRIATT_QBLOCK`` overrides (printed in the schedule census by the core)."""
    pinned = os.environ.get("ROWPAIR_TRIATT_QBLOCK", "").strip()
    if pinned:
        return max(1, int(pinned))
    from boltz.data import const
    return 128 if N > const.chunk_size_threshold else 512


def _triatt_fns(mod, use_kernels: bool):
    """TriangleAttention{Starting,Ending}Node.forward (attention.py:137-178) as the core's ``TriAttFns``: ``ln`` = the input LayerNorm of pair
    rows, ``bias`` = the triangle-bias projection of LayerNorm'd rows (``linear``: [rows, N, H]), ``attend`` = the module's own ``mha`` on a
    LayerNorm'd row batch given its mask rows and the WHOLE bias (the mode's kernel route inside ``mha``). The ending node is the same
    statement on rows of z^T and mask^T (the core's driver transposes)."""

    def attend(x_rows, mask_rows, tb_full, _blk):
        m5 = mask_rows.unsqueeze(0)[..., :, None, None, :]                  # [1, rows, 1, 1, N]  (attention.py:146-147)
        return mod.mha(x_rows.unsqueeze(0), x_rows.unsqueeze(0), _bias5(tb_full), mod.inf * (m5 - 1), m5, use_kernels=use_kernels)[0]

    tpx = tpx_bind()["triatt"]
    if tpx["state"] == "on":                                               # ROWPAIR_TRIATT_CORE=<word> with a core that provides it: the SAME statement (boltz Attention.forward,
        attend = _triatt_attend_tpx(mod, tpx["provider"], tpx["word"])    # primitives.py:338-363) with its attention core served by the core's adapter, counted
    TA = _core()[3]
    return TA.TriAttFns(ln=mod.layer_norm, bias=lambda x_rows: mod.linear(x_rows), attend=attend)


def _triatt_attend_tpx(mod, RA, kernel: str):
    """``TriAttFns.attend`` of ``mod`` (TriangleAttention{Starting,Ending}Node) restated at the q/k/v level — boltz Attention.forward (primitives.py:
    338-363): ``_prep_qkv`` without the scale (its ``use_kernels`` branch: the kernel applies ``scale = c_hidden ** -0.5``), the attention core, the
    head transpose, ``_wrap_up`` (sigmoid gate, output projection) — with the core served by ``RA.attention_core(stock, kernel=…)``: ``flash_triattn`` /
    ``cueq`` apply the scale inside the kernel on the unscaled q exactly as the module's cuEquivariance route does; ``stock`` (the ``torch`` word and every
    named fallback) is the module's eager statement given that q: ``_attention(q / sqrt(c_hidden), k, v, [mask_bias, triangle_bias])`` (``_prep_qkv``'s
    own division, then primitives._attention). Layout ``bnhsd`` = boltz's ``[1, rows, H, N, c]``; the key mask is read off the additive mask bias."""
    from boltz.model.layers.triangular_attention import primitives as P
    mha = mod.mha
    root = math.sqrt(mha.c_hidden)

    def stock(q, k, v, biases):
        return P._attention(q / root, k, v, biases)

    core = RA.attention_core(stock, kernel=kernel, min_tokens=0, scale=1.0 / root, layout="bnhsd", mask_from="bias0", tri_bias="bias1")

    def attend(x_rows, mask_rows, tb_full, _blk):
        x = x_rows.unsqueeze(0)                                            # [1, rows, N, C]
        m5 = mask_rows.unsqueeze(0)[..., :, None, None, :]                # [1, rows, 1, 1, N]
        q, k, v = mha._prep_qkv(x, x, apply_scale=False)                  # [1, rows, H, N, c] each, q unscaled (primitives.py:342-346 under use_kernels)
        o = RA.attend_query_blocks(core, q, k, v, [mod.inf * (m5 - 1), _bias5(tb_full)], None)
        o = o.transpose(-2, -3)                                            # [1, rows, N, H, c]
        return mha._wrap_up(o, x)[0]

    return attend


def _apb_fn(layer, mask):
    """The sequence track's attention with pair bias (pairformer.py:106-110; AttentionPairBias.forward, attentionv2.py:84-110) as the pair-block
    driver's ``apb(s, z_shard, layout) -> s``: pre-norm, queries = this rank's token rows (their bias rows are the shard's), keys / values from
    the whole s, the output rows all-gathered (core ``transition.apb_local_queries``); fp32 (the stock autocast-off region)."""
    import torch
    TRN = _core()[2]
    att = layer.attention
    maskf = mask.float()

    def attn_fn(q_rows, s_full, z_rows):
        B = s_full.shape[0]
        q = att.proj_q(q_rows).view(B, -1, att.num_heads, att.head_dim)
        k = att.proj_k(s_full).view(B, -1, att.num_heads, att.head_dim)
        v = att.proj_v(s_full).view(B, -1, att.num_heads, att.head_dim)
        bias = att.proj_z(z_rows.unsqueeze(0).float())
        g = att.proj_g(q_rows).sigmoid()
        with torch.autocast("cuda", enabled=False):
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (att.head_dim ** 0.5) + bias.float()
            attn = attn + (1 - maskf[:, None, None].float()) * -att.inf
            attn = attn.softmax(dim=-1)
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        o = o.reshape(B, -1, att.c_s)
        return att.proj_o(g * o)

    def apb(s, z_shard, lay):
        with torch.autocast("cuda", enabled=False):
            s_normed = layer.pre_norm_s(s.float())
            s = s.float() + TRN.apb_local_queries(attn_fn, s_normed, z_shard, lay)
        _STATE["calls"]["apb"] += 1
        return s

    def single_transition(s):
        with torch.autocast("cuda", enabled=False):
            s = s + layer.transition_s(s)
            s = layer.s_post_norm(s)
        return s

    return apb, single_transition


def _draw_dropout(dropout: float, zs, lay, training: bool, lead: int = 1, columnwise: bool = False) -> None:
    """The stock per-sub-layer dropout-mask draw (dropout.get_dropout_mask, pairformer.py:77-101): in eval the mask is all ones and unused,
    but each draw advances the CUDA generator by ``lead * N`` values — drawn here at the WHOLE tensor's shape ([lead, N, N, 1] view, no
    memory) so the generator advances on every rank alike and as the dense statement does."""
    from boltz.model.layers.dropout import get_dropout_mask
    zfull = zs.new_empty((1, 1, 1, 1)).expand(int(lead), lay.N, lay.N, 1)
    get_dropout_mask(dropout, zfull, training, columnwise=columnwise)


def _block_fns(layer, lay, use_kernels: bool, mask=None):
    """One PairformerLayer / PairformerNoSeqLayer as the core's ``PairBlockFns`` (``pairstack.bind``): the two triangle multiplications, the
    two triangle attentions and the pair transition of THIS layer's sub-modules at the core's streamed schedules; for a Pairformer layer
    (``layer.attention`` present) also the sequence track. ``chunk`` = the stock triangle-attention row chunk; ``inplace_chunk`` = N (boltz's
    triangle multiplication is one whole-tensor statement: its column extent is N)."""
    PS = _core_mod("pairstack")
    apb, single_transition = _apb_fn(layer, mask) if (mask is not None and hasattr(layer, "attention")) else (None, None)
    return PS.bind(trimul_out=_trimul_fns(layer.tri_mul_out), trimul_in=_trimul_fns(layer.tri_mul_in),
                   triatt_start=_triatt_fns(layer.tri_att_start, use_kernels), triatt_end=_triatt_fns(layer.tri_att_end, use_kernels),
                   transition=lambda x_rows, _mask_u: layer.transition_z(x_rows), chunk=_q_rows(lay.N, use_kernels),
                   apb=apb, single_transition=single_transition, trimul_kw={"inplace_chunk": int(lay.N)}, stats=_STATE["contract_stats"])


def _pair_layer(layer, zs, pm, lay, use_kernels: bool, lead: int = 1, s=None, mask=None, maskT=None):
    """The pair track of PairformerLayer.forward / PairformerNoSeqLayer.forward (pairformer.py:76-104 / :255-281) on the shard ``zs [R, N, C]``
    (fp32), eval mode (dropout mask = fp32 ones: ``z + ones * sub(z)`` promotes the bf16 sub-layer output exactly as the in-place add does):
    z += tri_mul_out; z += tri_mul_in; z += tri_att_start; z += tri_att_end; z += transition_z — in place, through the core's ONE pair-block
    driver (``pairstack.pair_block_``); with ``s``: then the sequence track (local query rows). Returns ``(zs, s)``."""
    PS = _core_mod("pairstack")
    for cw in (False, False, False, True):                                 # the four dropout-mask draws of the layer (values unused in eval; the generator advances as stock's)
        _draw_dropout(layer.dropout, zs, lay, layer.training, lead, columnwise=cw)
    fns = _block_fns(layer, lay, use_kernels, mask if s is not None else None)
    zs2, s = PS.pair_block_(fns, zs, pm[lay.r0:lay.r1], lay, s=s, maskT_shard=maskT, transition_mask=False)
    assert zs2.data_ptr() == zs.data_ptr(), "pair_block_ must update the shard in place"
    for k in ("trimul_out", "trimul_in", "triatt_start", "triatt_end", "transition"):
        _STATE["calls"][k] += 1
    _STATE["calls"]["layers"] += 1
    return zs, s


def _presharded_stack(layers, zs, pm, lay, use_kernels: bool, s=None, mask=None, lead: int = 1):
    """The ONE pair-stack entry of this adapter: a Pairformer / PairformerNoSeq stack entered with a shard ``zs [R, N, C]`` and left as a
    shard (no gather at either end; core ``pairstack``: the rows of mask^T computed once per stack). ``s`` given: the trunk stack's sequence
    track per layer (local queries). Returns ``zs`` or ``(s, zs)``."""
    PS = _core_mod("pairstack")
    t0 = time.time()
    mrows = pm[lay.r0:lay.r1]
    maskT = PS.mask_transposed(mrows, lay)
    for layer in layers:
        zs, s = _pair_layer(layer, zs, pm, lay, use_kernels, lead, s=s, mask=mask, maskT=maskT)
    _STATE["calls"]["presharded_calls"] += 1; _STATE["wall_s"]["sharded_modules"] += time.time() - t0
    return (s, zs) if s is not None else zs


# ---------------------------------------------------------------- the born-sharded trunk (seams 1-7, 14, 15) ----------------------------------------------------------------
PAIR_PLANES = {"token_bonds": -3, "type_bonds": -2, "contact_conditioning": -3, "contact_threshold": -2}   # feats key -> row dim of its [B, N, N(, F)] plane
PARK_UNREAD = ("disto_target",)                       # the featurizer's other N x N plane at inference: the distogram LOSS target [B, N, N, E, 64] f32 (256·N² B; boltz/model/loss/distogramv2.py is its
                                                       # only reader) — read by nothing at inference: parked on the host whole, once per item (_pair_planes_to_host)


def _pair_planes_to_host(feats: dict) -> dict:
    """The featurizer's N x N planes of ``feats`` (``PAIR_PLANES``: token_bonds [B,N,N,1] f32, type_bonds [B,N,N] i64, contact_conditioning
    [B,N,N,C] f32, contact_threshold [B,N,N] f32 — about 40 B per pair) moved to pinned host memory ONCE; every statement reads ROWS of them
    through :func:`_feats_rows` (the core's ``trunk.feature_rows``: a contiguous row slab is what moves to the device). Returns the dict of
    host planes (``feats`` is updated in place to hold them); a plane already on the host is kept as is. The planes are NAMED, never
    recognised by shape: a per-token feature whose width equals N has a pair plane's shape (res_type / profile / msa / template_restype are
    one-hot over const.num_tokens = 33: ``[1, 33, 33]`` at N = 33) and the model reads it on the device — moving it is a device error in the
    first statement that touches it (the template rows' ``torch.cat([a_tij, res_type_i, res_type_j])``). ``PARK_UNREAD`` (disto_target, the
    distogram loss's [B, N, N, E, 64] target: 256·N² bytes nothing reads at inference) leaves the device whole, by name. Any OTHER device tensor
    shaped like a token-pair plane is left where it is and named once in the census (``pair_shaped_on_device``: Boltz 2.2.1's inference
    featurizer emits no N x N plane besides ``PAIR_PLANES`` and ``PARK_UNREAD`` — its 85 batch keys audited; a key listed there at large N is
    device memory this line does not shard, by name)."""
    host = {}
    N = int(feats["token_pad_mask"].shape[-1])
    for k in PARK_UNREAD:                                                  # unread N x N planes: off the device whole, once (a reader would fail by device, loudly)
        t = feats.get(k)
        if t is not None and getattr(t, "is_cuda", False):
            feats[k] = t.detach().to("cpu")
            _STATE["calls"]["pair_planes_to_host"] += 1; _STATE["parked_unread"].add(k)
    for k, t in list(feats.items()):                                       # pair-SHAPED device tensors outside PAIR_PLANES / PARK_UNREAD: named, never moved (see above)
        if k not in PAIR_PLANES and k not in PARK_UNREAD and hasattr(t, "dim") and t.dim() >= 3 and sum(1 for d in t.shape[1:] if int(d) == N) >= 2 and getattr(t, "is_cuda", False):
            _STATE["pair_shaped_on_device"].add(k)
    for k in PAIR_PLANES:
        t = feats.get(k)
        if t is None or not hasattr(t, "dim"):
            continue
        if t.is_cuda:
            h = t.detach().to("cpu", non_blocking=False)
            try:
                h = h.pin_memory()
            except RuntimeError:                        # no pinned pool (CPU-only box): pageable host copy, named in the census
                _STATE["calls"]["host_pageable"] += 1
            feats[k] = h
            _STATE["calls"]["pair_planes_to_host"] += 1
        host[k] = feats[k]
    return host


def _feats_rows(feats: dict, g0: int, g1: int, device) -> dict:
    """``feats`` with each pair plane replaced by its rows ``[g0, g1)`` on ``device`` (core ``trunk.feature_rows``; a view when the plane is
    already on the device). The per-token features are untouched (replicated, small)."""
    TRK = _core_mod("trunk")
    out = dict(feats)
    for k, rd in PAIR_PLANES.items():
        if k in out and hasattr(out[k], "dim"):
            out[k] = TRK.feature_rows(out[k], g0, g1, device, row_dim=rd)
    return out


def _relpos_rows(rel_pos, feats: dict, g0: int, g1: int):
    """RelativePositionEncoder.forward (encodersv2.py:49-121, Algorithm 3) for pair ROWS ``[g0, g1)`` only: every statement of the stock
    forward with its ``i`` operand sliced (``x[:, g0:g1, None]`` against ``x[:, None, :]``) — the same one-hots, the same ``where`` clauses
    (incl. the cyclic-period offset and the sym-id clause), the same Linear — ``[B, g1-g0, N, token_z]``. Nothing ``N x N`` is built."""
    import torch
    from torch.nn.functional import one_hot
    TRK = _core_mod("trunk")
    b_same_chain = TRK.same_rows(feats["asym_id"], g0, g1)
    b_same_residue = TRK.same_rows(feats["residue_index"], g0, g1)
    b_same_entity = TRK.same_rows(feats["entity_id"], g0, g1)
    d_residue = feats["residue_index"][:, g0:g1, None] - feats["residue_index"][:, None, :]
    if rel_pos.cyclic_pos_enc and torch.any(feats["cyclic_period"] > 0):
        period = torch.where(feats["cyclic_period"] > 0, feats["cyclic_period"], torch.zeros_like(feats["cyclic_period"]) + 10000)
        d_residue = (d_residue - period * torch.round(d_residue / period)).long()
    d_residue = torch.clip(d_residue + rel_pos.r_max, 0, 2 * rel_pos.r_max)
    d_residue = torch.where(b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * rel_pos.r_max + 1)
    a_rel_pos = one_hot(d_residue, 2 * rel_pos.r_max + 2)
    d_token = torch.clip(feats["token_index"][:, g0:g1, None] - feats["token_index"][:, None, :] + rel_pos.r_max, 0, 2 * rel_pos.r_max)
    d_token = torch.where(b_same_chain & b_same_residue, d_token, torch.zeros_like(d_token) + 2 * rel_pos.r_max + 1)
    a_rel_token = one_hot(d_token, 2 * rel_pos.r_max + 2)
    d_chain = torch.clip(feats["sym_id"][:, g0:g1, None] - feats["sym_id"][:, None, :] + rel_pos.s_max, 0, 2 * rel_pos.s_max)
    d_chain = torch.where((~b_same_entity) if rel_pos.fix_sym_check else b_same_chain, torch.zeros_like(d_chain) + 2 * rel_pos.s_max + 1, d_chain)
    a_rel_chain = one_hot(d_chain, 2 * rel_pos.s_max + 2)
    p = rel_pos.linear_layer(torch.cat([a_rel_pos.float(), a_rel_token.float(), b_same_entity.unsqueeze(-1).float(), a_rel_chain.float()], dim=-1))
    _STATE["calls"]["relpos_rows"] += 1
    return p


def _zT_rows(z_loc, lay):
    """Rows ``[r0, r1)`` of ``z^T`` for this rank (``[1, R, N, C]``): the core's shard transpose (one all-to-all of the shard). The distogram
    head and the confidence PDE head both read ``z + z^T``; a caller that needs both passes the same tensor on."""
    _STATE["calls"]["transposes"] += 1
    return _transpose_shard(z_loc, lay)


def _guard_parameters_replicated(model) -> None:
    """Every rank must hold the SAME weights (a rank that loaded another checkpoint, or a constructor that drew from an unpinned RNG, would
    fold garbage into the ring contractions): a checksum of the parameters is proven identical across the ranks once per model (core
    ``trunk.guard_replicated``: one small collective; refused by name on a mismatch)."""
    import torch
    TRK = _core_mod("trunk")
    with torch.no_grad():
        chk = torch.stack([p.detach().double().sum() for p in model.parameters()] or [torch.zeros((), dtype=torch.float64)])
    TRK.guard_replicated(chk, "model_parameters")
    _STATE["calls"]["parameters_guarded"] += 1


def _check_attention_contract(model) -> None:
    """The sub-module contract the row statements assume, checked once per model BEFORE any shard is built (refused by name otherwise):
    every triangle attention's ``mha`` carries ``linear_q / linear_k / linear_v / linear_o`` (+ ``linear_g`` or None) — the fused-projection
    lever ``qkvg`` adds a buffer beside them and keeps them, a lever that removed them would break the statement; every AttentionPairBias
    carries ``proj_q / proj_k / proj_v / proj_g / proj_z / proj_o``; every TriangleMultiplication ``norm_in / p_in / g_in / norm_out / p_out /
    g_out``."""
    from torch import nn
    need = {"TriangleAttention": ("layer_norm", "linear", "mha"), "Attention": ("linear_q", "linear_k", "linear_v", "linear_o"),
            "AttentionPairBias": ("proj_q", "proj_k", "proj_v", "proj_g", "proj_z", "proj_o"),
            "TriangleMultiplicationOutgoing": ("norm_in", "p_in", "g_in", "norm_out", "p_out", "g_out"),
            "TriangleMultiplicationIncoming": ("norm_in", "p_in", "g_in", "norm_out", "p_out", "g_out")}
    n = 0
    for mod in model.modules():
        req = need.get(type(mod).__name__)
        if req is None:
            continue
        n += 1
        for a in req:
            sub = getattr(mod, a, None)
            if not isinstance(sub, nn.Module):
                raise Refused(f"refused: attention_contract: {type(mod).__name__}.{a} is {type(sub).__name__}, not a module — a lever rebound the "
                              f"sub-modules the row-sharded statement calls (boltz2_opt.rowpair under n_gpu={_STATE['P']})")
    _STATE["calls"]["contract_modules_checked"] += n


def _z_init_rows(model, s_inputs, feats, g0: int, g1: int):
    """boltz2.py:420-429 on GLOBAL rows [g0, g1): the same sum in the same order and dtypes (bf16 outer sums, bf16 rel-pos / bond terms under
    the mode's autocast, the fp32 contact-conditioning term promotes) -> ``[1, g1-g0, N, 128]``; a result whose storage dtype is not fp32 is
    refused by name (fp32 storage is the sharded domain: the shard's in-place adds promote exactly as stock's ``z + ones_fp32 * sub(z)``)."""
    import torch
    fr = _feats_rows(feats, g0, g1, s_inputs.device)                        # the pair planes' rows on the device (host-resident planes)
    z_init = model.z_init_1(s_inputs)[:, g0:g1, None] + model.z_init_2(s_inputs)[:, None, :]
    z_init = z_init + _relpos_rows(model.rel_pos, feats, g0, g1)
    z_init = z_init + model.token_bonds(fr["token_bonds"].float())
    if model.bond_type_feature:
        z_init = z_init + model.token_bonds_type(fr["type_bonds"].long())
    z_init = z_init + model.contact_conditioning(fr)
    if z_init.dtype != torch.float32:
        raise Refused(f"refused: pair tensor storage dtype {z_init.dtype} at trunk entry under n_gpu={_STATE['P']} (fp32 storage is the sharded domain)")
    return z_init


def _template_rows(tm, z_loc, feats, pair_mask, lay, use_kernels: bool):
    """TemplateV2Module.forward (trunkv2.py:411-509) producing rows [r0, r1): every pair feature is built with its ``i`` operand sliced to the
    rows (distances ``cdist(cb_i_rows, cb_all)``, frame vectors ``R_j^T (ca_i - t_j)``, the cb / frame / visibility pair masks, the residue
    types), the 2-block template pair stack runs presharded one template at a time, the template mean and output projection are row-local."""
    import torch
    from torch.nn.functional import one_hot
    r0, r1 = lay.r0, lay.r1
    res_type = feats["template_restype"]
    frame_rot = feats["template_frame_rot"]
    frame_t = feats["template_frame_t"]
    frame_mask = feats["template_mask_frame"]
    cb_coords = feats["template_cb"]
    ca_coords = feats["template_ca"]
    cb_mask = feats["template_mask_cb"]
    visibility_ids = feats["visibility_ids"]
    template_mask = feats["template_mask"].any(dim=2).float()
    num_templates = template_mask.sum(dim=1)
    num_templates = num_templates.clamp(min=1)
    b_cb_mask = (cb_mask[:, :, r0:r1, None] * cb_mask[:, :, None, :])[..., None]
    b_frame_mask = (frame_mask[:, :, r0:r1, None] * frame_mask[:, :, None, :])[..., None]
    B, T = res_type.shape[:2]  # noqa: N806
    if int(B) != 1:
        raise Refused(f"refused: TemplateV2Module under n_gpu={_STATE['P']}: batch {B} — batch size 1 is the sharded domain")
    tmlp_pair_mask = (visibility_ids[:, :, r0:r1, None] == visibility_ids[:, :, None, :]).float()
    with torch.autocast(device_type="cuda", enabled=False):
        cb_dists = torch.cdist(cb_coords[:, :, r0:r1], cb_coords)
        boundaries = torch.linspace(tm.min_dist, tm.max_dist, tm.num_bins - 1)
        boundaries = boundaries.to(cb_dists.device)
        distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
        distogram = one_hot(distogram, num_classes=tm.num_bins)
        frame_rot_ = frame_rot.unsqueeze(2).transpose(-1, -2)
        frame_t_ = frame_t.unsqueeze(2).unsqueeze(-1)
        ca_rows = ca_coords[:, :, r0:r1].unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(frame_rot_, (ca_rows - frame_t_))
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
        unit_vector = unit_vector.squeeze(-1)
        a_tij = torch.cat([distogram, b_cb_mask, unit_vector, b_frame_mask], dim=-1)
        a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1)
        res_type_i = res_type[:, :, r0:r1, None].expand(-1, -1, -1, res_type.size(2), -1)
        res_type_j = res_type[:, :, None, :].expand(-1, -1, r1 - r0, -1, -1)
        a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
        a_tij = tm.a_proj(a_tij)
    v = tm.z_proj(tm.z_norm(z_loc[:, None])) + a_tij                     # [1, T, R, N, 64]
    del a_tij
    v = v.view(B * T, *v.shape[2:])
    pm = pair_mask[0]
    outs = []
    for t in range(int(T)):                                              # one template's pair stack at a time through the presharded driver (templates are independent
        vin = v[t].clone(memory_format=torch.contiguous_format)         # batch entries); the driver adds in place, stock keeps v for the residual -> a private copy in
        outs.append(_presharded_stack(tm.pairformer.layers, vin, pm, lay, use_kernels))
    v = v + (outs[0].unsqueeze(0) if int(T) == 1 else torch.stack(outs, 0))   # trunkv2.py:498  v = v + pairformer(v)
    del outs
    v = tm.v_norm(v)
    v = v.view(B, T, *v.shape[1:])
    template_mask = template_mask[:, :, None, None, None]
    num_templates = num_templates[:, None, None, None]
    u = (v * template_mask).sum(dim=1) / num_templates.to(v)
    u = tm.u_proj(tm.relu(u))
    _STATE["calls"]["template_rows"] += 1
    return u                                                             # [1, R, N, 128]


def _msa_chunks(N: int, training: bool) -> dict:
    """MSAModule.forward's eval chunk plan (trunkv2.py:593-611), keyed on the WHOLE N (never on a shard's row count)."""
    from boltz.data import const
    if training:
        return dict(chunk_heads_pwa=False, chunk_size_transition_z=None, chunk_size_transition_msa=None, chunk_size_outer_product=None, chunk_size_tri_attn=None)
    if N > const.chunk_size_threshold:
        return dict(chunk_heads_pwa=True, chunk_size_transition_z=64, chunk_size_transition_msa=32, chunk_size_outer_product=4, chunk_size_tri_attn=128)
    return dict(chunk_heads_pwa=False, chunk_size_transition_z=None, chunk_size_transition_msa=None, chunk_size_outer_product=None, chunk_size_tri_attn=512)


TEMPL_SKIP_ENV = "ROWPAIR_TEMPL_SKIP"      # =0: every template pass runs the row-sharded template stack; unset/1 (default at n_gpu > 1): a pass whose template mask
                                         #  has no slot set (boltz2_opt.templskip.dummy_pass — the `templ_skip` lever's own decision) gets the module's update from its output
                                         #  projection of the zero aggregation on THIS rank's rows, after the skipped Pairformer's random draws are replayed (templskip's helpers):
                                         #  no [T, R, N, ·] template slabs, no 2-block c=64 pair stack per template per cycle; census templ_rows=elided:all_dummy|stack:<why>


def _template_rows_elided_(tm, z_loc, feats, lay) -> bool:
    """The ×P row statement of the `templ_skip` lever (boltz2_opt.templskip, exact class at n_gpu = 1): when no template slot of the pass is set,
    TemplateV2Module's update ``u_proj(relu((v * template_mask).sum(1) / num_templates))`` is ``u_proj(relu(0))`` whatever the stack computed —
    served here on rows ``[r0, r1)`` as ``u_proj(relu(zeros [B, R, N, template_dim]))`` (the module's own tail on the values it would meet; a
    bias-free u_proj makes it the zero update) after :func:`templskip._replay_dropout_draws` advances the CUDA generator exactly as the skipped
    2-block Pairformer's dropout-mask draws would (the sharded stack's `_draw_dropout` issues those same whole-shape draws). Returns True when
    the pass was elided (z_loc updated in place), False when the stack must run (a live template, no mask, ``ROWPAIR_TEMPL_SKIP=0``)."""
    import torch
    from opt_core.mem.rowpair import evidence as EV
    from . import templskip as TS
    word = (os.environ.get(TEMPL_SKIP_ENV, "") or "1").strip().lower()
    if word in ("0", "off"):
        EV.record_schedule(templ_rows="stack:env_off"); return False
    d = TS.dummy_pass(feats)
    if d is None:
        EV.record_schedule(templ_rows="stack:no_mask"); return False
    if not d:
        EV.record_schedule(templ_rows="stack:live_templates"); return False
    tmask = feats["template_mask"]
    B = int(z_loc.shape[0]) if z_loc.dim() >= 4 else 1
    T = int(tmask.shape[1]) if (torch.is_tensor(tmask) and tmask.dim() >= 2) else 1
    N, R = int(lay.N), int(lay.r1) - int(lay.r0)
    draws = TS._replay_dropout_draws(tm, B * T, N, z_loc.device)             # the skipped stack's generator draws, whole-shape as stock ([B*T, N, 1, 1] x3, [B*T, 1, N, 1] x1 per layer)
    dt_v = torch.float32 if (z_loc.is_cuda and torch.is_autocast_enabled()) else tm.u_proj.weight.dtype   # v_norm's output dtype (templskip._elided_update's rule)
    dt_u = torch.promote_types(dt_v, tmask.dtype) if torch.is_tensor(tmask) else dt_v
    lead = tuple(z_loc.shape[:-3])
    u = torch.zeros(lead + (R, N, int(tm.u_proj.in_features)), dtype=dt_u, device=z_loc.device)
    z_loc.add_(tm.u_proj(tm.relu(u)))
    EV.record_schedule(templ_rows="elided:all_dummy", templ_slots=T, templ_draws_replayed=int(draws))
    _STATE["calls"]["template_rows"] += 1                                   # the template seam ran (its all-dummy statement): the exit verdict's REQUIRED_CALLS evidence, as _template_rows counts
    _STATE["calls"]["templ_elided"] = _STATE["calls"].get("templ_elided", 0) + 1
    return True


def _pwa_s_chunk() -> int:
    """MSA rows per all-gathered chunk of the token-sharded m in the pair-weighted averaging: ``ROWPAIR_PWA_S_CHUNK`` (a positive integer) or
    :data:`PWA_S_CHUNK`; refused by name when malformed. Printed by the core (``pwa_s_chunk``)."""
    v = (os.environ.get(PWA_S_CHUNK_ENV) or "").strip()
    if not v:
        return PWA_S_CHUNK
    try:
        n = int(v)
    except ValueError:
        n = 0
    if n < 1:
        raise Refused(f"refused: {PWA_S_CHUNK_ENV}={v!r}: a positive integer (MSA rows per gathered chunk)")
    return n


def _pwa_all_heads_fns(mod, H: int, c_h: int, rows=None, norm: bool = False):
    """The all-heads-at-once PairWeightedAveraging statements (pair_averaging.py:114-135) as the core's ``pwa_rows`` callables:
    ``values_fn(m_chunk)`` = every head's values ``[H, chunk, N, c_h]`` (``proj_m``) + the gates (``proj_g(...).sigmoid()``);
    ``attend_fn(w, state, g0, g1)`` = ``'hij,hsjd->hsid'`` for query rows ``[g0, g1)`` times their gate rows. ``rows=(r0, r1)`` (the
    token-sharded m): the gates are computed for this rank's tokens only — the rows ``attend_fn`` reads (``g[:, g0 - r0:g1 - r0]``) — and
    ``norm=True`` applies ``norm_m`` to the gathered chunk (:73, per (row, token)) instead of to a whole m."""
    import torch
    r0 = rows[0] if rows else 0

    def values_fn(m_chunk):
        mc = mod.norm_m(m_chunk) if norm else m_chunk
        n = int(mc.shape[0])
        v = mod.proj_m(mc).reshape(n, int(mc.shape[1]), H, c_h).permute(2, 0, 1, 3)   # [H, chunk, N, c_h]
        return v, mod.proj_g(mc[:, rows[0]:rows[1]] if rows else mc).sigmoid()        # gates [chunk, R|N, H*c_h]

    def attend_fn(w, state, g0, g1):
        v, g = state
        o = torch.einsum("hij,hsjd->hsid", w, v)                           # [H, chunk, q, c_h]
        o = o.permute(1, 2, 0, 3).reshape(int(v.shape[1]), g1 - g0, H * c_h)
        return g[:, g0 - r0:g1 - r0] * o

    return values_fn, attend_fn


def _pwa_update(mod, m4, z_loc, pair_mask, chunk_heads: bool, lay, tok: bool = False):
    """PairWeightedAveraging.forward (pair_averaging.py:73-135) with the pair logits from this rank's z rows, as the core's ``pwa_bias_rows`` +
    ``pwa_rows`` engine callables (batch dim squeezed: boltz predicts at B = 1). The logits ``proj_z(norm_z(z_rows))`` of this rank's rows,
    heads first, plus the rows' mask bias (one pass over the shard in row blocks: ``[H, R, N]``); then, as stock: with ``chunk_heads`` (N
    above const.chunk_size_threshold, eval) the per-head statements — the head's value / gate weight slices (``m @ W[h].T``), its softmax rows
    times values per local query-row block, the head's slice of ``proj_o`` accumulated in head order; without it all heads at once (``proj_m``
    / ``proj_g`` / ``proj_o`` whole). Replicated m (``tok=False``): one head at a time over all S rows (the transient is one head's ``[S, N,
    c_h]``, as stock's), the token rows of the m-update all-gathered — returns ``[1, S, N, c_m]``. Token-sharded m (``tok=True``, ``m4 [1, S,
    R, c_m]``): the core gathers one S-chunk of m at a time (``pwa_s_chunk``: :func:`_pwa_s_chunk`; ``norm_m`` per chunk — no whole normed
    copy), every head's values are that chunk's (the same per-head statements), the gates and output rows are this rank's tokens — returns
    ``[1, S, R, c_m]``, nothing of the update gathered."""
    import torch
    MS = _core_mod("msa")
    mask = pair_mask[0]                                                    # [N, N]
    H, c_h = int(mod.num_heads), int(mod.c_h)
    softmax = lambda x: torch.softmax(x, dim=-1)  # noqa: E731

    def prep_fn(z_rows, g0, g1):                                           # :74 + :95-97 / :115-117 on rows: LN, proj_z (all heads), heads first, the rows' mask bias
        b = mod.proj_z(mod.norm_z(z_rows))                                 # [rows, N, H]
        return b.permute(2, 0, 1) + (1 - mask[g0:g1][None]) * -mod.inf     # [H, rows, N]

    bias = MS.pwa_bias_rows(prep_fn, z_loc[0], lay)                        # [H, R, N]: the logits of this rank's query rows
    if tok:                                                                # the token-sharded m: ONE pass over S-chunks serves every head
        r0 = lay.r0
        kw = dict(s_chunk=_pwa_s_chunk(), m_layout="token_sharded", softmax_fn=softmax)
        if chunk_heads:                                                    # :75-113: the per-head statements, on the gathered chunk [chunk, N, c_m] (normed per chunk: :73 is per (row, token))
            W = [(mod.proj_m.weight[h * c_h:(h + 1) * c_h, :], mod.proj_g.weight[h * c_h:(h + 1) * c_h, :], mod.proj_o.weight[:, h * c_h:(h + 1) * c_h]) for h in range(H)]

            def values_fn(m_chunk):                                        # :90-91, :101-102 per head: the chunk's values (every token) and gates (this rank's tokens: the rows read)
                mc = mod.norm_m(m_chunk); mq = mc[:, r0:lay.r1]
                return [(mc @ w_m.T, (mq @ w_g.T).sigmoid()) for w_m, w_g, _ in W]   # H x ([chunk, N, c_h], [chunk, R, c_h])

            def attend_fn(w, state, g0, g1):                               # :105-108 per head for query rows [g0, g1) (global, inside this rank's tokens): 'ij,sjd->sid' times the gate rows
                return torch.cat([g[:, g0 - r0:g1 - r0] * torch.einsum("ij,sjd->sid", w[h], v) for h, (v, g) in enumerate(state)], dim=-1)   # [chunk, q, H*c_h]

            def out_fn(o):                                                 # :109-112: o_out += o_h @ W_o[h].T, in head order
                upd = None
                for h, (_, _, w_o) in enumerate(W):
                    part = o[..., h * c_h:(h + 1) * c_h] @ w_o.T
                    upd = part if upd is None else upd.add_(part)
                return upd

            upd = MS.pwa_rows(m4[0], bias, lay, values_fn=values_fn, attend_fn=attend_fn, out_fn=out_fn, **kw)   # [S, R, c_m]
        else:                                                              # :114-135: all heads at once
            values_fn, attend_fn = _pwa_all_heads_fns(mod, H, c_h, rows=(r0, lay.r1), norm=True)
            upd = MS.pwa_rows(m4[0], bias, lay, values_fn=values_fn, attend_fn=attend_fn, out_fn=mod.proj_o, **kw)
        _STATE["calls"]["pwa_rows"] += 1
        return upd.unsqueeze(0)
    m = mod.norm_m(m4[0])                                                  # [S, N, c_m]  (pair_averaging.py:73: per (row, token); the replicated m normed whole, as stock)
    if chunk_heads:                                                        # replicated m, :75-113: heads sequentially
        upd = None
        for h in range(H):
            w_m = mod.proj_m.weight[h * c_h:(h + 1) * c_h, :]; w_g = mod.proj_g.weight[h * c_h:(h + 1) * c_h, :]
            w_o = mod.proj_o.weight[:, h * c_h:(h + 1) * c_h]

            def values_fn(m_chunk, w_m=w_m, w_g=w_g):                     # :90-91, :101-102: this head's values and gates of the (replicated) m chunk
                return m_chunk @ w_m.T, (m_chunk @ w_g.T).sigmoid()       # [chunk, N, c_h] each

            def attend_fn(w, state, g0, g1):                               # :105-108 for query rows [g0, g1): 'bhij,bhsjd->bhsid' with h = 1, times the gate rows
                v, g = state
                o = torch.einsum("ij,sjd->sid", w[0], v)                   # [chunk, q, c_h]
                return g[:, g0:g1] * o

            part = MS.pwa_rows(m, bias[h:h + 1], lay, values_fn=values_fn, attend_fn=attend_fn, out_fn=lambda o_full, w_o=w_o: o_full @ w_o.T, softmax_fn=softmax)
            upd = part if upd is None else upd.add_(part)                  # :109-112: o_out += o_chunks @ W_o[h].T, in head order
            del part
    else:                                                                  # :114-135: all heads at once
        values_fn, attend_fn = _pwa_all_heads_fns(mod, H, c_h)
        upd = MS.pwa_rows(m, bias, lay, values_fn=values_fn, attend_fn=attend_fn, out_fn=mod.proj_o, softmax_fn=softmax)
    _STATE["calls"]["pwa_rows"] += 1
    return upd.unsqueeze(0)


def opm_operands(mod, m4, msa_mask, chunk_size, count_mask=None):
    """The boltz OuterProductMean statement split for row blocking (outer_product_mean.py:49-98) — the ONE statement the row-sharded trunk
    (``_opm_add_``, n_gpu > 1) runs: returns ``(a, b, outer_fn)`` where ``a`` /
    ``b`` are the masked projections on the whole replicated ``m`` (``[B, S, N, c_hidden]``, :53-57) and ``outer_fn(a_rows, b_all, g0, g1)`` maps
    ``a`` token rows ``[S, rows, c]`` + all of ``b`` ``[S, N, c]`` + the block's GLOBAL rows ``[g0, g1)`` to the OPM output rows ``[rows, N, C_z]``:
    the pair count ``num_mask`` for those rows in stock's 64-sequence slabs (:59-68 / :90-91), boltz's chunked outer statement
    (``'bsic,bsjd->bijcd'`` per c_hidden chunk, the division, the sliced output projection, the bias: :71-88; eval with a chunk size) or the
    unchunked one (:92-98). Per-element arithmetic = the stock statement's; the launch's M is the row block. ``count_mask`` (the token-sharded
    caller only, ``_opm_add_``): ``m4`` and ``msa_mask`` are then this rank's token COLUMNS (``a`` / ``b`` = the local rows of the stock
    operands) while the pair count reads the whole mask ``count_mask [B, S, N]``; ``None`` = ``msa_mask`` is whole (the replicated m, the
    n_gpu = 1 lever: the statement as it always read)."""
    import torch
    m = m4
    mask = msa_mask.unsqueeze(-1).to(m)
    mn = mod.norm(m)
    a = mod.proj_a(mn) * mask
    b = mod.proj_b(mn) * mask
    del mn
    chunked = chunk_size is not None and not mod.training
    cmask = mask if count_mask is None else count_mask.unsqueeze(-1).to(m)  # [B, S, N, 1]: the pair count's operand (every token column)

    def num_mask_rows(g0, g1):                                              # :59-68 / :90-91 for rows [g0, g1): sum_s mask[s, j] mask[s, i]
        if chunked:
            nm = None
            for i in range(0, cmask.shape[1], 64):
                blk = (cmask[:, i:i + 64, None, :] * cmask[:, i:i + 64, g0:g1, None]).sum(1)
                nm = blk if nm is None else nm + blk
            return nm.clamp(min=1)
        return (cmask[:, :, None, :] * cmask[:, :, g0:g1, None]).sum(1).clamp(min=1)

    def outer_fn(a_rows, b_all, g0, g1):                                   # a_rows [S, rows, c], b_all [S, N, c] -> [rows, N, C_z]
        a4, b4 = a_rows.unsqueeze(0), b_all.unsqueeze(0)
        num_mask = num_mask_rows(g0, g1)
        if chunked:                                                         # :71-88
            z_out = None
            for i in range(0, mod.c_hidden, chunk_size):
                a_chunk = a4[:, :, :, i:i + chunk_size]
                sliced_weight_proj_o = mod.proj_o.weight[:, i * mod.c_hidden:(i + chunk_size) * mod.c_hidden]
                zz = torch.einsum("bsic,bsjd->bijcd", a_chunk, b4)
                zz = zz.reshape(*zz.shape[:3], -1)
                zz = zz / num_mask
                z_out = zz.to(m) @ sliced_weight_proj_o.T if z_out is None else z_out + zz.to(m) @ sliced_weight_proj_o.T
            z_out = z_out + mod.proj_o.bias
            return z_out[0]
        zz = torch.einsum("bsic,bsjd->bijcd", a4.float(), b4.float())      # :92-98
        zz = zz.reshape(*zz.shape[:3], -1)
        zz = zz / num_mask
        return mod.proj_o(zz.to(m))[0]

    return a, b, outer_fn


def _opm_b_for_gather(b):
    """The OPM ``b`` operand as it crosses the ranks (:func:`_opm_add_`, token-sharded m): under the trunk's bf16 autocast ``b`` is the bf16
    ``proj_b`` output times the {0, 1} mask promoted to fp32 (outer_product_mean.py:55-57) — every value bf16-exact — so it is gathered as bf16
    (2 bytes a value; ``outer_fn``'s einsum casts its operands to bf16 under the same autocast: the values it reads are unchanged). A ``b`` that
    is not bf16-exact, or no autocast (the CPU logic tests), crosses as it stands. Returns ``(tensor, dtype word)``; census ``opm_b_gather_dtype=``."""
    import torch
    try:
        autocast = bool(torch.is_autocast_enabled(b.device.type))
    except TypeError:                                                       # torch < 2.4 spells the CUDA query without the device argument
        autocast = bool(torch.is_autocast_enabled()) and b.is_cuda
    if autocast and b.dtype == torch.float32:
        b16 = b.to(torch.bfloat16)
        if torch.equal(b16.to(torch.float32), b):
            return b16, "bf16"
        del b16
    return b, str(b.dtype).replace("torch.", "")


def _opm_add_(mod, z, m4, msa_mask, chunk_size, lay, tok: bool = False):
    """``z += OuterProductMean.forward(m)`` rows IN PLACE on the shard ``z [1, R, N, C_z]`` (outer_product_mean.py:49-98; trunkv2.py:751): the
    statement of :func:`opm_operands` as the core's ``outer_fn`` over budgeted row blocks (core msa.opm_rows_budgeted, ``add=True``: the block
    is agreed across ranks, printed in the schedule census, and lands in the shard block by block — the OPM transient never exceeds one row block).
    Token-sharded m (``tok=True``, ``m4 [1, S, R, c_m]``): ``a`` is this rank's token rows as they stand (``a_local``), ``b`` is all-gathered
    once (``[S, N, c_hidden]``; bf16 under the trunk's autocast: ``64·S·N`` bytes — :func:`_opm_b_for_gather`; census ``opm_b_gather_dtype=
    opm_b_gather_gib=``), the pair count reads the whole mask."""
    MS = _core_mod("msa")
    if not tok:
        a, b, outer_fn = opm_operands(mod, m4, msa_mask, chunk_size)
        MS.opm_rows_budgeted(a[0], b[0], lay, outer_fn, C_z=int(mod.proj_o.bias.shape[0]), out=z[0], add=True, global_rows=True)
    else:
        SH = _core()[1]
        from opt_core.mem.rowpair import evidence as EV
        a, b_loc, outer_fn = opm_operands(mod, m4, msa_mask[:, :, lay.r0:lay.r1], chunk_size, count_mask=msa_mask)   # a / b on this rank's token columns
        src, gdt = _opm_b_for_gather(b_loc[0])
        del b_loc
        b = SH.unshard_rows(src, lay, dim=1)                                # [S, N, c_hidden]: every token's b (the one operand the local output rows pair with) — a view over the gathered buffer, no second copy
        del src
        EV.record_schedule(opm_a="local", opm_b_gather_dtype=gdt, opm_b_gather_gib=round(int(b.numel()) * int(b.element_size()) / 2 ** 30, 3))
        MS.opm_rows_budgeted(a[0], b, lay, outer_fn, C_z=int(mod.proj_o.bias.shape[0]), out=z[0], add=True, global_rows=True, a_local=True)
        del b
    _STATE["calls"]["opm_rows"] += 1
    return z

def _msa_rows(msa_module, z_loc, emb, feats, pair_mask, lay, use_kernels: bool, in_place: bool = False):
    """MSAModule.forward (trunkv2.py:567-670) with z as rows, returning the module's processed pair rows (stock returns its processed z, which
    the trunk ADDS to the z it passed in — so the module works on a private copy of the shard). The MSA representation's layout is the word
    ``ROWPAIR_MSA_M_LAYOUT`` (``rowpair_msa.m_layout``): ``token_sharded`` (default) — this rank embeds its token columns ``r0:r1`` only
    (``msa_input(cols=)``, ``msa_proj``, ``s_proj`` of those tokens: ``m [1, S, R, 64]``) — or ``replicated`` (``m [1, S, N, 64]``); its input
    rows stand under the placement word ``ROWPAIR_MSA_HOST`` (``rowpair_msa.msa_input``), its subsample draw is proven identical on every rank.
    Then per block (MSALayer.forward :714-760): m += dropout * PWA(m, z rows); m += msa_transition(m) (per (row, token): on the shard, core
    ``msa.msa_transition_rows`` names the layout); z += OPM rows in place; z = the block's pair layer presharded, in place. The dropout mask
    draw reads ``m[:, :, 0:1, 0:1]`` (``[1, S, 1, 1]``: S numbers from the generator under either layout, as stock)."""
    import torch
    from boltz.model.layers.dropout import get_dropout_mask
    from opt_core.mem.rowpair import evidence as EV
    MS = _core_mod("msa")
    N = int(lay.N)
    ch = _msa_chunks(N, msa_module.training)
    word = rowpair_msa.m_layout()
    tok = word == "token_sharded"
    r0, r1 = int(lay.r0), int(lay.r1)
    m, msa_mask = rowpair_msa.msa_input(msa_module, feats, z_loc.device, _STATE["guards"], cols=((r0, r1) if tok else None))   # trunkv2.py:614-634 under the placement word (device | host): the input rows after the draw (this rank's columns when token-sharded), the mask rows (whole)
    m = msa_module.msa_proj(m)
    m = m + msa_module.s_proj(emb[:, r0:r1] if tok else emb).unsqueeze(1)   # :636-637 (s_proj is per token: this rank's tokens when token-sharded)
    m_census = lambda: EV.record_schedule(msa_m=word, msa_m_gib=round(int(m.numel()) * int(m.element_size()) / 2 ** 30, 3), msa_m_dtype=str(m.dtype).replace("torch.", ""))  # noqa: E731
    m_census()
    pm = pair_mask[0]
    z = z_loc if in_place else z_loc.clone(memory_format=torch.contiguous_format)   # the module's own pair tensor (stock: z inside the module; the trunk keeps its z for the residual) — or, `in_place`, THIS shard (the caller holds the trunk's z in a host park: msa_fn / _msa_entry_park)
    for i in range(msa_module.msa_blocks):
        layer = msa_module.layers[i]
        msa_dropout = get_dropout_mask(layer.msa_dropout, m, layer.training)
        m = m + msa_dropout * _pwa_update(layer.pair_weighted_averaging, m, z, pair_mask, ch["chunk_heads_pwa"], lay, tok)
        if tok:                                                             # :749 per (row, token) on the token shard; the core names the layout (census msa_transition / msa_m)
            m = m + MS.msa_transition_rows(lambda mm, _r0, _r1, layer=layer: layer.msa_transition(mm, ch["chunk_size_transition_msa"]), m, lay,
                                               shard_tokens=True, token_dim=-2, m_layout="token_sharded")
        else:
            m = m + layer.msa_transition(m, ch["chunk_size_transition_msa"])
        _opm_add_(layer.outer_product_mean, z, m, msa_mask, ch["chunk_size_outer_product"], lay, tok)
        if i == 0:
            m_census()                                                      # the resident m as the blocks carry it (its dtype after the first residual)
        zs = _presharded_stack([layer.pairformer_layer], z[0], pm, lay, use_kernels)     # in place on the module's rows
        assert zs.data_ptr() == z.data_ptr()
        _STATE["calls"]["msa_blocks_rows"] += 1
    return z


MSA_PARK_ENV = "ROWPAIR_MSA_PARK_Z"      # =0: the MSA module works on a device CLONE of the shard; unset/1 (default at n_gpu > 1): the trunk's
                                         #  z rows are parked on pinned host at the module's entry and the module updates the shard IN PLACE — one [R, N, 128] fp32 shard resident
                                         #  through the MSA module instead of two (census msa_z_entry=parked:<where>|device_clone msa_z_entry_gib= msa_z_park_s= msa_z_readd_s=)


def _msa_entry_park(z_loc):
    """boltz2.py:473 ``z = z + msa_module(z, ...)``: the module's pair tensor starts as a copy of the trunk's z and the trunk adds the module's
    result to ITS z — two operands of one residual. At n_gpu > 1 the trunk's z rows go to a pinned host park (core ``trunk.ShardPark``, the
    z_init park's mechanism, device copy KEPT) so the module can update the shard in place; :func:`_add_parked_rows_` adds the parked rows back.
    Returns the park, or None when the clone statement stands (``ROWPAIR_MSA_PARK_Z=0``, a CPU shard — the logic tests — or a non-contiguous
    shard), named in the census either way. Arithmetic per element is the clone statement's: the same module operations on the same values,
    then the same two fp32 addends (addition commutes exactly)."""
    from opt_core.mem.rowpair import evidence as EV
    word = (os.environ.get(MSA_PARK_ENV, "") or "1").strip().lower()
    if word in ("0", "off", "clone") or not z_loc.is_cuda or not z_loc.is_contiguous():
        EV.record_schedule(msa_z_entry="device_clone" + ("" if word in ("0", "off", "clone") else (":cpu" if not z_loc.is_cuda else ":noncontiguous")))
        return None
    TRK = _core_mod("trunk")
    park = TRK.ShardPark(z_loc, park=True, name="z_msa_entry", release_device=False)
    if not park.park:                                                       # the park declined (nothing was copied): the clone statement stands
        EV.record_schedule(msa_z_entry="device_clone:park_declined")
        return None
    EV.record_schedule(msa_z_entry="parked:" + str(park.where), msa_z_entry_gib=round(park.nbytes / 2 ** 30, 3), msa_z_park_s=park.park_s)
    _STATE["calls"]["msa_z_parked"] = _STATE["calls"].get("msa_z_parked", 0) + 1
    return park


def _add_parked_rows_(z_loc, park) -> None:
    """``z_loc += parked rows``, row block by row block (core ``shard.choose_block_rows``: the ROWPAIR_ROWBLK_MB staging block), then the park is
    released (its pinned bytes return to the pool)."""
    import time as _time
    import torch
    from opt_core.mem.rowpair import evidence as EV
    SHD = _core_mod("shard")
    R, N, C = int(z_loc.shape[-3]), int(z_loc.shape[-2]), int(z_loc.shape[-1])
    rows, _src = SHD.choose_block_rows(N=N, C=C, elem_bytes=int(z_loc.element_size()), n_max=R)
    t0 = _time.time()
    for i0 in range(0, R, rows):
        i1 = min(R, i0 + rows)
        z_loc[..., i0:i1, :, :].add_(park.block(i0, i1))
    if z_loc.is_cuda:
        torch.cuda.synchronize(z_loc.device)
    park.release()
    EV.record_schedule(msa_z_readd_s=round(_time.time() - t0, 3), msa_z_readd_rows=int(rows))


def _distogram_rows(dm, z_loc, lay):
    """DistogramModule.forward (trunkv2.py:811-828) on rows: ``(z + z^T)[:, r0:r1] = z_loc + (rows r0:r1 of z^T)`` — the transposed rows come
    from the other ranks by the core's shard transpose (all-to-all) — then the stock Linear and reshape. Returns ``[1, R, N, nd, bins]``."""
    if _sym_blocks_streamed():                                                  # (z + z^T) row block by row block off the core's block-streamed transpose —
        import torch                                                            # no whole transposed shard, no whole sum (2 x 512·N²/P B less at this stage)
        from opt_core.mem.rowpair import evidence as EV
        D = _core_mod("dist")
        R, N = int(z_loc.shape[1]), int(z_loc.shape[2])
        rows = _sym_zT_step(lay, N, int(z_loc.shape[-1]), int(z_loc.element_size()))  # ONE value on every rank (0.3.35: clamped to the WIDEST shard, not this rank's rows)
        out, n = None, 0
        for i0, i1, zT_blk in D.transpose_blocks(z_loc[0], lay, step=int(rows)):   # COLLECTIVE: every rank walks the same max_q ceil(R_q/step) rounds (exhausted here)
            n += 1
            if i1 <= i0:
                continue
            x = torch.empty_like(z_loc[:, i0:i1])                               # [1, w, N, C] in the shard's own contiguous row layout (the Linear below then runs the GEMM the whole
            torch.add(z_loc[:, i0:i1], zT_blk.unsqueeze(0), out=x)              #  statement ran on these rows: bitwise; a transposed-stride operand would take another GEMM path)
            y = dm.distogram(x).reshape(1, i1 - i0, N, dm.num_distograms, dm.num_bins)
            if out is None:
                out = torch.empty((1, R, N, dm.num_distograms, dm.num_bins), dtype=y.dtype, device=y.device)
            out[:, i0:i1] = y
            del x, y, zT_blk
        if out is None:                                                         # a rank with no rows (never at the supported P; kept total)
            out = dm.distogram(z_loc).reshape(1, R, N, dm.num_distograms, dm.num_bins)
        EV.record_schedule(disto_zT="blocks", disto_zT_blocks=int(n), disto_zT_rows=int(rows))
        _STATE["calls"]["transposes"] += 1                                      # the shard WAS transposed (block-streamed): REQUIRED_CALLS evidence of the transpose seam
        _STATE["calls"]["distogram_rows"] += 1
        return out
    zt = _zT_rows(z_loc, lay)
    x = z_loc + zt
    del zt
    out = dm.distogram(x).reshape(x.shape[0], x.shape[1], x.shape[2], dm.num_distograms, dm.num_bins)
    _STATE["calls"]["distogram_rows"] += 1
    return out


def _sym_zT_step(lay, N: int, C: int, elem_bytes: int) -> int:
    """The row-block step of the block-streamed (z + zᵀ) reader (``_distogram_rows``): the core's ROWPAIR_ROWBLK_MB staging block
    (``shard.choose_block_rows``) clamped to the layout's WIDEST shard ``lay.n_max`` (= max_q R_q) — never to this rank's own row count.
    ``dist.transpose_blocks`` walks ``max_q ceil(R_q / step)`` all-to-all rounds, a COLLECTIVE schedule, so the step must be one value on
    every rank: a step clamped to the local ``R`` would, on a ragged layout below the budget (N ≲ 1.4 K at P = 2; N = 199 → 100 | 99),
    give the short rank one round more — one unmatched all-to-all, every later collective mis-paired, and the NCCL watchdog firing in
    the confidence stack. The PDE twin (``rowpair_heads._zzT_row_blocks``) clamps to max_q R_q likewise.
    Row-block GEMM shapes per rank do not depend on the clamp in any regime (one block of R_q rows below the budget, budget rows above):
    the bytes produced are the same; only the round count is rank-independent. ×1 is inert by
    construction: ``--n_gpu 1`` never installs the row-sharded trunk (this statement does not run), and on a single-shard layout
    R == Rmax, so the clamp changes nothing."""
    n_max = int(getattr(lay, "n_max", None) or max(int(lay.nrows(q)) for q in range(int(lay.P))))
    rows, _src = _core_mod("shard").choose_block_rows(N=int(N), C=int(C), elem_bytes=int(elem_bytes), n_max=max(1, n_max))
    return max(1, int(rows))


SYM_ZT_ENV = "ROWPAIR_SYM_ZT"                # blocks (default at n_gpu > 1) | whole: how the two (z + z^T) readers (the trunk's DistogramModule rows here, the confidence PDE head in
                                             #  rowpair_heads) come by the transposed rows — `whole` = ONE shard transpose each, a whole [N/P, N, C] transposed copy + the whole sum
                                             #  resident; `blocks` = the core's block-streamed transpose (dist.transpose_blocks: one all-to-all per row block,
                                             #  transient O(N · rows · C); pure data movement, the same addends per element). Census disto_zT= / pde_zT= blocks|whole


def _sym_blocks_streamed() -> bool:
    word = (os.environ.get(SYM_ZT_ENV, "") or "blocks").strip().lower()
    if word not in ("blocks", "whole", "1", "0"):
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"refused: {SYM_ZT_ENV}={word!r}: one of blocks | whole (or unset)", lever="n_gpu")
    if word not in ("blocks", "1"):
        return False
    cm = _core_mod("dist").comm()                                             # the core's block-streamed transpose assembles each block with ONE all_to_all per round; a comm
    if not bool(getattr(cm, "has_all_to_all", True)):                          #  without it (gloo: the CPU logic tests) would all-GATHER the whole [N, N, C] tensor instead —
        from opt_core.mem.rowpair import evidence as EV                        #  there the whole-shard transpose statement (one [R, N, C] transposed copy, never N x N) stands, named
        EV.record_schedule(sym_zT=f"whole:no_all_to_all:{getattr(cm, 'backend', None) or type(cm).__name__}")
        return False
    return True


def _trunk_sharded(model, feats, recycling_steps: int):
    """Boltz2.forward's trunk (boltz2.py:414-491) with z born and kept as rows, driven by the core's ONE trunk driver
    (``opt_core.mem.rowpair.trunk.run_trunk_sharded``: init on rows, recycling in place per row block, the RNG-stream guard before the MSA
    module, no gather) over this engine's statements as callables. Stock computes the s recycle before the z recycle; the driver runs it after
    the MSA module — the two statements are independent (the template / MSA modules read ``s_inputs``, not ``s``), so the arithmetic is stock's.
    Returns ``(s_inputs, s, z_loc, pdistogram_loc, mask, pair_mask, lay)``."""
    import torch
    TRK = _core_mod("trunk")
    t0 = time.time()
    s_inputs = model.input_embedder(feats)
    s_init = model.s_init(s_inputs)
    N = int(s_inputs.shape[1])
    if int(s_inputs.shape[0]) != 1:
        raise Refused(f"refused: Boltz2.forward under n_gpu={_STATE['P']}: batch {int(s_inputs.shape[0])} — batch size 1 is the sharded domain")
    lay = _layout(N)
    if not _STATE["contract_checked"]:
        _check_attention_contract(model)
        _guard_parameters_replicated(model)
        _STATE["contract_checked"] = True
    _mark("trunk_entry")
    _pair_planes_to_host(feats)                                            # the featurizer's N x N planes leave the device; rows come back per block
    mask = feats["token_pad_mask"].float()
    pair_mask = mask[:, :, None] * mask[:, None, :]
    pm = pair_mask[0]
    C_z = int(model.z_init_2.weight.shape[0])
    template_module = msa_module = pairformer_module = None
    if model.run_trunk_and_structure:
        if model.use_templates:
            template_module = model.template_module._orig_mod if (model.is_template_compiled and not model.training) else model.template_module  # noqa: SLF001
            if type(template_module).__name__ != "TemplateV2Module":
                raise Refused(f"refused: {type(template_module).__name__} under n_gpu={_STATE['P']}: the row-sharded template statement serves TemplateV2Module")
        msa_module = model.msa_module._orig_mod if (model.is_msa_compiled and not model.training) else model.msa_module  # noqa: SLF001
        pairformer_module = model.pairformer_module._orig_mod if (model.is_pairformer_compiled and not model.training) else model.pairformer_module  # noqa: SLF001

    def init_rows_fn(g0, g1):
        _STATE["calls"]["init_blocks"] += 1
        return _z_init_rows(model, s_inputs, feats, g0, g1)

    def recycle_update_fn(z_rows):                                       # boltz2.py:455  z_recycle(z_norm(z)) on a row block
        return model.z_recycle(model.z_norm(z_rows))

    def single_recycle_fn(s, cycle):                                       # boltz2.py:454 (replicated)
        _STATE["calls"]["recycles"] += 1
        return s_init + model.s_recycle(model.s_norm(s))

    def template_fn(z_loc, cycle):                                         # boltz2.py:464  z = z + template_module(z, feats, pair_mask)  (added in place on the shard)
        if not _template_rows_elided_(template_module, z_loc, feats, lay):          # an all-dummy template pass: the module's update is its output projection of the zero aggregation, per row (templskip's statement)
            z_loc.add_(_template_rows(template_module, z_loc, feats, pair_mask, lay, model.use_kernels))
        _mark("after_template")
        return z_loc

    def msa_fn(z_loc, cycle):                                              # boltz2.py:473  z = z + msa_module(z, s_inputs, feats)  (the module returns its processed pair tensor; added in place)
        park = _msa_entry_park(z_loc)                                      # the trunk's z (the residual's other operand) on pinned host while the module updates THIS shard in place; None = the clone statement
        if park is None:
            z_loc.add_(_msa_rows(msa_module, z_loc, s_inputs, feats, pair_mask, lay, model.use_kernels))
        else:
            _msa_rows(msa_module, z_loc, s_inputs, feats, pair_mask, lay, model.use_kernels, in_place=True)   # z_loc <- the module's processed pair rows
            _add_parked_rows_(z_loc, park)                                 # z_loc += the trunk's z, row block by row block from the host park (the same two addends per element as the clone statement)
        _mark("after_msa")
        return z_loc

    def pairstack_fn(s, z_loc, cycle):                                     # boltz2.py:483  s, z = pairformer_module(s, z, mask, pair_mask): presharded, in place on the shard
        s, zs = _presharded_stack(pairformer_module.layers, z_loc[0], pm, lay, model.use_kernels, s=s, mask=mask)
        assert zs.data_ptr() == z_loc.data_ptr()
        _mark("after_pairstack")
        return s, z_loc

    _STATE["calls"]["trunk_rows"] += 1                                     # one row-sharded trunk per item
    if model.run_trunk_and_structure:
        out = TRK.run_trunk_sharded(lay, n_cycles=int(recycling_steps) + 1, init_rows_fn=init_rows_fn,
                                    init_like=s_inputs.new_empty((1, N, C_z), dtype=torch.float32), recycle_update_fn=recycle_update_fn,
                                    s_init=s_init, single_recycle_fn=single_recycle_fn, template_fn=template_fn if template_module is not None else None,
                                    msa_fn=msa_fn, pairstack_fn=pairstack_fn, s_input=s_inputs,
                                    init_rows=None,                        # the core's row-block size (agreed budget): relpos / bonds / contact rows are built per block
                                    gather="none", census_fn=None, log=_say)
        s, z_loc = out.s, out.z
    else:                                                                  # stock: s, z stay zeros when the trunk does not run
        s = torch.zeros_like(s_init)
        z_loc = s_inputs.new_zeros((1, _R(lay), N, C_z), dtype=torch.float32)
    pd_loc = _distogram_rows(model.distogram_module, z_loc, lay)
    _mark("distogram")
    _STATE["wall_s"]["trunk"] += time.time() - t0
    return s_inputs, s, z_loc, pd_loc, mask, pair_mask, lay


class TrunkShard:
    """What the trunk hands to the heads (seams 16-19): this rank's rows of the trunk pair tensor and everything row statements need.

    ``z_loc``      ``[1, R, N, 128]`` fp32, rows ``[lay.r0, lay.r1)`` of the trunk's z (after the last Pairformer block) — read-only for heads
    ``pd_loc``     ``[1, R, N, num_distograms, bins]`` the distogram logits of the same rows
    ``s`` ``s_inputs``  replicated single representations; ``mask`` ``[1, N]``, ``pair_mask`` ``[1, N, N]`` (token pad mask; small, replicated)
    ``lay``        the core Layout (``r0 r1 R N P rank`` …); ``feats`` the batch with its N x N planes HOST-resident (rows via ``_feats_rows``)
    ``zplan``      the placement of ``z_loc`` across the roll-out and the confidence passes (core ``heads.ZTrunkPlan``, ``_ztrunk_plan``): heads read
                   trunk rows through ``zplan.source()`` / ``zplan.begin(i)`` — the device tensor, or its host park while the device storage is
                   released — and ``z_loc`` itself only for metadata (shape / dtype / device)
    Row statements for heads: ``_relpos_rows(model.rel_pos | confidence_module.rel_pos, feats, g0, g1)``, ``_feats_rows(feats, g0, g1, device)``,
    ``_zT_rows(z_loc, lay)`` (rows of z^T: distogram / PDE), ``_presharded_stack(layers, z_rows[R,N,C], pair_mask[N,N], lay, use_kernels, s=, mask=)``."""
    __slots__ = ("s_inputs", "s", "z_loc", "pd_loc", "mask", "pair_mask", "lay", "feats", "zplan")

    def __init__(self, s_inputs, s, z_loc, pd_loc, mask, pair_mask, lay, feats, zplan=None):
        self.s_inputs, self.s, self.z_loc, self.pd_loc, self.mask, self.pair_mask, self.lay, self.feats = s_inputs, s, z_loc, pd_loc, mask, pair_mask, lay, feats
        self.zplan = zplan


ZTRUNK_LEVERS = ("ROWPAIR_FREE_ZTRUNK", "ROWPAIR_CONF_PARK_ZTRUNK")     # the core's words (heads.ENV_FREE_ZTRUNK / ENV_CONF_PARK_ZTRUNK); modes.TP_EXPORTS sets both on the line


def _ztrunk_plan(T: "TrunkShard", passes: int):
    """The placement of the trunk pair shard ``T.z_loc`` from the roll-out entry through the confidence passes: the core's ONE statement
    (``opt_core.mem.rowpair.heads.ZTrunkPlan``) decides per pass — ``inplace`` (the confidence pair input overwrites the shard's own storage:
    one sample per pass at the shard's last use), ``parked:<where>`` (the shard copied to pinned host and its device storage RELEASED before the
    pass's pair input — and, parked at the roll-out entry, before the conditioned pair rows — are allocated; rows served per block), or
    ``resident`` (both levers off) — from ``ROWPAIR_FREE_ZTRUNK`` / ``ROWPAIR_CONF_PARK_ZTRUNK``. ``passes`` = the confidence passes of this
    forward (one per diffusion sample: ``rowpair_heads.confidence_rows`` embeds one sample per pass)."""
    return _core_mod("heads").ZTrunkPlan(T.z_loc, passes=max(1, int(passes)), name="z_trunk", log=_say)


def _ztrunk_close(T: "TrunkShard"):
    """End of the forward: close the plan (a live host park is dropped; the census words are the core's record), keep its record for
    ``report()``, and return what ``dict_out["z"]`` carries — ``T.z_loc`` while it holds the trunk rows, else :class:`ConsumedRows`."""
    plan = T.zplan
    plan.close()
    rec = dict(plan.record())
    _STATE["ztrunk"] = rec
    if plan.consumed:
        words = [str(w) for w in (rec.get("words") or []) if w]
        return ConsumedRows("z", ",".join(words) or "consumed", T.z_loc.shape)
    return T.z_loc


class ConsumedRows:
    """What ``dict_out["z"]`` carries once the confidence stage consumed the trunk pair rows (embedded in place, or parked and dropped:
    ``zplan.consumed``): the census word, and a refusal BY NAME for any tensor use (the writer reads ``z`` only with ``write_embeddings``, which
    this line does not set; a caller that wants the trunk rows back runs with ``ROWPAIR_FREE_ZTRUNK=0 ROWPAIR_CONF_PARK_ZTRUNK=0``)."""
    __slots__ = ("name", "word", "shape")

    def __init__(self, name: str, word: str, shape):
        self.name, self.word, self.shape = str(name), str(word), tuple(int(x) for x in shape)

    def __repr__(self):
        return f"ConsumedRows({self.name} {self.shape}: {self.word})"

    def __getattr__(self, attr):
        if attr.startswith("__"):                                              # protocol probes (copy / pickle / numpy) keep their AttributeError semantics
            raise AttributeError(attr)
        raise Refused(f"refused: dict_out[{self.name!r}].{attr}: the trunk pair rows {self.shape} were consumed by the confidence stage ({self.word}); "
                      f"run with {' '.join(k + '=0' for k in ZTRUNK_LEVERS)} to keep them (e.g. for write_embeddings)")


HEADS: Dict[str, Optional[Callable]] = {"diffusion_conditioning_rows": None, "sample_sharded": None, "confidence_rows": None}
"""Seams 16-19 (``boltz2_opt.rowpair_heads``): ``diffusion_conditioning_rows(model, T: TrunkShard) -> diffusion_conditioning dict`` (seam 18:
z_cond rows, token pair-bias rows, atom-encoder pair windows from bands), ``sample_sharded(model, T, diffusion_conditioning, **sample_kw) ->
struct_out`` (seam 19: the DiffusionTransformer on local query rows), ``confidence_rows(model, T, x_pred, multiplicity, run_sequentially) ->
confidence dict`` (seams 16-17: z_conf born from the trunk rows, heads row-reduced, [N,N] outputs to rank 0's host). A name left None is
refused by name at its call site."""


def _bind_heads() -> None:
    """Bind ``boltz2_opt.rowpair_heads.HEADS`` (the three head seams). A package without the module leaves every seam unbound (each refused by
    name at its call site); a module that fails to import for any other reason raises here (never a quiet unbound)."""
    import importlib
    try:
        RH = importlib.import_module("boltz2_opt.rowpair_heads")
    except ModuleNotFoundError as e:
        if e.name != "boltz2_opt.rowpair_heads":
            raise
        _STATE["heads_module"] = "absent"
        return
    bound = []
    for k, fn in getattr(RH, "HEADS", {}).items():
        if k in HEADS and callable(fn):
            HEADS[k] = fn; bound.append(k)
    _STATE["heads_module"] = "bound:" + ",".join(sorted(bound))


def _head(name: str):
    """The bound callable of head seam ``name``; an unbound seam is refused by name (no head runs on a gathered pair tensor)."""
    fn = HEADS.get(name)
    if fn is None:
        raise Refused(f"refused: heads seam {name} unbound under n_gpu={_STATE['P']} (boltz2_opt.rowpair_heads: {_STATE.get('heads_module')}) — "
                      "every head runs on pair rows; nothing gathers z")
    return fn


def sync_policy() -> str:
    """The replicated-tensor sync policy word the kit set for this worker (registry rowpair_tp sync_switch = the core's
    ``ROWPAIR_DIFF_NOISE_SYNC``): ``bcast`` (rank 0 authoritative at every sync point; the det-0 route's policy) or ``guard`` (strict: a
    mismatch is refused by name)."""
    from .registry import LEVERS
    w = os.environ.get(LEVERS["rowpair_tp"]["sync_switch"], LEVERS["rowpair_tp"]["sync_default"]).strip().lower()
    if w not in ("bcast", "guard"):
        raise Refused(f"refused: {LEVERS['rowpair_tp']['sync_switch']}={w!r}: bcast | guard")
    return w


def _generators_from_rank0() -> None:
    """The replicated-draw policy of this line (census ``diff_noise=seed_bcast``): at the entry of every sharded forward the CPU default
    generator's and this rank's CUDA generator's STATES of rank 0 replace every rank's (one object broadcast through the core's comm), so the
    draws that follow — identical statements with identical shapes on every rank — are identical by construction whatever a rank consumed
    before (data loading, featurization); the sampler's final coordinates are still proven equal (``diffusion.sync_replicated``, guard). A
    rank whose state differed from rank 0's is counted (``rng_bcast_corrected``), never silent."""
    import torch
    comm = _core_mod("dist").comm()
    mine = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
    ref = comm.broadcast_obj({k: (v.cpu() if v is not None else None) for k, v in mine.items()}, src=0)
    differed = (not torch.equal(mine["cpu"], ref["cpu"])) or (mine["cuda"] is not None and ref["cuda"] is not None and not torch.equal(mine["cuda"].cpu(), ref["cuda"]))
    if sync_policy() == "guard":                                           # strict: every rank's generators must already agree
        bad = [i for i, d in enumerate(comm.allgather_obj(bool(differed))) if d]
        _STATE["calls"]["rng_guarded"] += 1
        if bad:
            raise Refused(f"refused: generator states differ from rank 0's on ranks {bad} at the sharded forward's entry (sync policy guard)")
        return
    torch.set_rng_state(ref["cpu"])
    if mine["cuda"] is not None and ref["cuda"] is not None:
        torch.cuda.set_rng_state(ref["cuda"])
    _STATE["calls"]["rng_bcast"] += 1
    if differed:
        _STATE["calls"]["rng_bcast_corrected"] += 1


def _forward_sharded(self, feats, recycling_steps: int = 0, num_sampling_steps=None, multiplicity_diffusion_train: int = 1, diffusion_samples: int = 1,
                     max_parallel_samples=None, run_confidence_sequentially: bool = False):
    """Boltz2.forward (boltz2.py:401-606, the predict path) at n_gpu > 1: the trunk row-sharded (``_trunk_sharded``), then the heads on the rows
    through ``HEADS`` (seams 16-19, ``boltz2_opt.rowpair_heads``); an unbound head seam is refused by name."""
    import torch
    if self.training:
        raise Refused(f"refused: Boltz2.forward in training mode under n_gpu={_STATE['P']}: inference is the sharded domain")
    TG = sys.modules.get("boltz2_opt.templates")                           # the template guard's model-entry census (templates.check_batch: the TEMPLATES line, the
    if TG is not None and TG.report().get("installed"):                    # ALL DUMMY event) runs here as it runs at the one-GPU Boltz2.forward the guard wraps — this
        TG.check_batch(feats)                                              # forward replaces that one; the census reads the records and masks only (no draw, no collective)
    _generators_from_rank0()                                               # every later draw (dropout masks, MSA subsample, diffusion noise, augmentation) starts from rank 0's state
    with torch.set_grad_enabled(False):
        T = TrunkShard(*_trunk_sharded(self, feats, recycling_steps), feats)
        T.zplan = _ztrunk_plan(T, passes=diffusion_samples if self.confidence_prediction else 1)   # the trunk shard's placement through the roll-out and the confidence passes (core heads.ZTrunkPlan)
        s, s_inputs = T.s, T.s_inputs
        dict_out = {"s": s}
        if self.run_trunk_and_structure and not self.skip_run_structure:
            diffusion_conditioning = _head("diffusion_conditioning_rows")(self, T)          # seam 18 on rows
            _STATE["calls"]["zcond_rows"] += 1
            _mark("diffusion_start")
            sample_kw = dict(s_trunk=s.float(), s_inputs=s_inputs.float(), feats=feats, num_sampling_steps=num_sampling_steps, atom_mask=feats["atom_pad_mask"].float(),
                             multiplicity=diffusion_samples, max_parallel_samples=max_parallel_samples, steering_args=self.steering_args,
                             diffusion_conditioning=diffusion_conditioning)
            with torch.autocast("cuda", enabled=False):
                struct_out = _head("sample_sharded")(self, T, **sample_kw)              # seam 19: DiffusionTransformer queries on local rows
                _STATE["calls"]["dit_blocks_sharded"] += 1
                dict_out.update(struct_out)
            if self.predict_bfactor:
                dict_out["pbfactor"] = self.bfactor_module(s)
            _mark("diffusion_peak")
        if self.confidence_prediction:
            _mark("confidence")
            x_pred = dict_out["sample_atom_coords"].detach() if not self.skip_run_structure else feats["coords"].repeat_interleave(diffusion_samples, 0)
            dict_out.update(_head("confidence_rows")(self, T, x_pred, diffusion_samples, run_confidence_sequentially))   # seams 16-17 on rows
            _STATE["calls"]["conf_rows"] += 1
        dict_out["pdistogram"] = T.pd_loc                                              # this rank's ROWS (the writer does not read it; named in the census)
        dict_out["z"] = _ztrunk_close(T)                                               # idem: the rows (resident / restored), or ConsumedRows once the confidence stage consumed them
    _mark("done")
    return dict_out


# ---------------------------------------------------------------- contract ----------------------------------------------------------------
def _install_trunk(module=None) -> None:
    """``Boltz2.forward`` = the sharded trunk (idempotent). Runs right after ``boltz.model.models.boltz2`` has executed, or at once if it already has."""
    if _STATE["trunk_installed"]:
        return
    B2 = module if module is not None else sys.modules.get(MODEL_MODULE)
    if B2 is None:
        import importlib
        B2 = importlib.import_module(MODEL_MODULE)
    _STATE["orig"]["boltz2_forward"] = B2.Boltz2.forward
    B2.Boltz2.forward = _forward_sharded
    _bind_heads()
    _install_rng_guard(_core()[0])                                         # the sampler's module (diffusionv2) is imported by the model module: guard it now
    _STATE["trunk_installed"] = True
    _say(f"{TAG} ROWPAIR trunk installed rank={_STATE['rank']}/{_STATE['P']}: Boltz2.forward = row-sharded trunk (z born as rows through the distogram; heads {_STATE.get('heads_module')})")


def apply(spec=None) -> List[str]:
    """Initialise this rank's process group from the launcher's environment and install the sharded forwards (idempotent). Refuses by
    name at n_gpu = 1 (nothing is installed there) and when the rank environment disagrees with BOLTZ_TP."""
    if _STATE["installed"]:
        return list(LEVERS)
    P = n_gpu()
    if P <= 1:
        raise Refused(f"refused: boltz2_opt.rowpair attached at n_gpu=1 ({ENV_P} unset or 1): the adapter installs nothing at n_gpu=1")
    tpx_bind()                                                             # the fused-kernel switches, resolved once (an unknown word is refused by name here)
    D = _core()[0]
    world, rank, device = D.init_from_env()
    if int(world) != P:
        raise Refused(f"refused: {ENV_P}={P} but the rank environment says world={world} (ROWPAIR_WORLD)")
    from boltz.model.layers.attentionv2 import AttentionPairBias as APBV2
    _STATE.update(installed=True, P=P, rank=int(rank), device=str(device), apb_class=APBV2.__name__, t0=time.time())
    hook = rowpair_msa.install_batch_transfer()                            # the batch's H2D keeps the raw MSA features on the host under ROWPAIR_MSA_HOST (bound now, or on the data module's import)
    mod = sys.modules.get(MODEL_MODULE)
    if mod is not None and hasattr(mod, "Boltz2"):                        # the model module has executed (worker_launch attaches ``tp`` after it): install now
        _install_trunk(mod)
    elif mod is not None:                                                  # mid-import (attached after a module the model module imports): the class does not exist yet
        raise Refused(f"refused: boltz2_opt.rowpair attached while {MODEL_MODULE} is still executing — the attach trigger must be {MODEL_MODULE} (worker_launch.ATTACH['tp'])")
    else:                                                                  # not imported yet (tests / a late attach): the import itself runs the install
        from boltz2_opt.worker_launch import _AfterImport
        sys.meta_path.insert(0, _AfterImport(MODEL_MODULE, _install_trunk))
    _say(f"{TAG} ROWPAIR installed rank={rank}/{P} device={device} sharded=trunk(pair init, recycling, template, MSA, Pairformer, distogram) "
         f"replicated={';'.join(REPLICATED)} gathered={';'.join(GATHERED)} msa_m={rowpair_msa.m_layout()} msa_host={rowpair_msa.host_mode() or 'off'} batch_hook={'bound' if hook else 'armed'} {_core_mod('rankdata').data_form_word(rowpair_msa.FEATS_FORM)}")
    return list(LEVERS)


def _install_rng_guard(D) -> None:
    """R1 guard, always on: right before the diffusion sampler draws its noise (``AtomDiffusion.sample``, diffusionv2.py) every rank's CUDA
    generator state must be the same (seed, offset) — the ranks then draw the same noise and denoise the same sample; a mismatch is refused by
    name by the core (``allreduce_checksum``), never sampled."""
    import torch
    from boltz.model.modules import diffusionv2 as DV
    orig = DV.AtomDiffusion.sample
    if getattr(orig, "_boltz2_opt_rng_guard", False):
        return

    def sample(self, *a, **kw):
        if torch.cuda.is_available() and str(_STATE.get("device") or "").startswith("cuda"):
            st = torch.cuda.get_rng_state().to(device="cuda", dtype=torch.float32)
            D.allreduce_checksum(st, "cuda_rng_state@AtomDiffusion.sample")
            _STATE["guards"]["rng_checked"] += 1
        return orig(self, *a, **kw)

    sample._boltz2_opt_rng_guard = True
    _STATE["orig"]["diffusion_sample"] = orig
    DV.AtomDiffusion.sample = sample


def peak_gib() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return round(float(torch.cuda.max_memory_allocated()) / 2 ** 30, 3)
    except Exception:
        pass
    return None



def tp_exports() -> Dict[str, Optional[str]]:
    """The ×P line's placement words (``modes.TP_EXPORTS`` names) as this rank's process carries them — the values the core's statements read;
    what each DID is the core's schedule census (``report()["schedule"]``: ``park_z_init=…``)."""
    return {k: os.environ.get(k) for k in modes.TP_EXPORTS}


def report() -> Dict[str, Any]:
    """``tp_report``: what this rank installed and did — read by stack.evidence (installed, the seam call census, the rows census, errors) and
    by report.lever_lines (the LEVER line's evidence fields)."""
    lay_facts = {str(n): (l.facts() if hasattr(l, "facts") else repr(l)) for n, l in _STATE["layouts"].items()}
    sched = {}
    try:
        from opt_core.mem.rowpair import evidence as EV
        sched = dict(EV.schedule())
    except Exception as e:  # noqa: BLE001 — the report must always be writable
        _STATE["errors"].append(f"schedule: {type(e).__name__}: {e}")
    return {"installed": bool(_STATE["installed"]), "applied": list(LEVERS) if _STATE["installed"] else [], "n_gpu": int(_STATE["P"]),
            "tp_exports": tp_exports(), "msa_host": rowpair_msa.report(), "ztrunk": _STATE.get("ztrunk"),
            "rank": int(_STATE["rank"]), "device": _STATE["device"], "trunk_installed": bool(_STATE["trunk_installed"]),
            "sharded": ["pair_init", "recycling", "TemplateV2Module", "MSAModule", "PairformerModule (trunk)", "DistogramModule", "PairformerModule (confidence: sharded at entry, gathered at exit)"],
            "replicated": list(REPLICATED), "host_rows": list(HOST_ROWS), "heads_bound": {k: (v is not None) for k, v in HEADS.items()},
            "gathered": ([] if all(v is not None for v in HEADS.values()) else list(GATHERED)),
            "pending_seams": ([] if all(v is not None for v in HEADS.values()) else list(PENDING_SEAMS)),
            "required_calls": list(REQUIRED_CALLS), "replaced_levers": sorted(REPLACED_LEVERS), "replaced_levers_why": dict(REPLACED_LEVERS), "bypassed_levers": list(BYPASSED_LEVERS), "heads_module": _STATE.get("heads_module"), "pair_shaped_on_device": sorted(_STATE.get("pair_shaped_on_device") or ()),   # device tensors shaped like a token-pair plane outside PAIR_PLANES / PARK_UNREAD: named, never moved (_pair_planes_to_host)
            "parked_unread": sorted(_STATE.get("parked_unread") or ()),                   # PARK_UNREAD planes that left the device whole (disto_target)
             "sync_policy": sync_policy() if _STATE["installed"] else None, "diff_noise": ("bcast_rank0_state+rng_state_bcast" if (_STATE["installed"] and sync_policy() == "bcast") else "guard"),
            "layouts": lay_facts, "rows_census": _STATE["rows_census"], "calls": dict(_STATE["calls"]), "schedule": sched,
            "peak_alloc_gib": peak_gib(), "wall_s": dict(_STATE["wall_s"]), "guards": dict(_STATE["guards"]),
            "tpx": tpx_report(), "errors": list(_STATE["errors"])}


def reset_for_tests() -> None:
    """Restore every patched attribute and clear the process state (the tests' fixture)."""
    orig = _STATE["orig"]
    try:
        B2 = sys.modules.get(MODEL_MODULE)
        if B2 is not None and "boltz2_forward" in orig:
            B2.Boltz2.forward = orig["boltz2_forward"]
        if "diffusion_sample" in orig:
            from boltz.model.modules import diffusionv2 as DV
            DV.AtomDiffusion.sample = orig["diffusion_sample"]
    except ImportError:
        pass
    rowpair_msa.reset_for_tests()
    _STATE["ztrunk"] = None
    _STATE.update({"installed": False, "P": 1, "rank": 0, "device": None, "orig": {}, "layouts": {}, "rows_census": None, "calls": dict(_CALLS0),
                   "guards": {"inputs_checked": 0, "inputs_equal": 0, "rng_checked": 0, "replicated_checked": 0}, "errors": [], "wall_s": {"sharded_modules": 0.0, "trunk": 0.0},
                   "t0": None, "trunk_installed": False, "contract_checked": False, "contract_stats": {}, "pair_shaped_on_device": set(), "parked_unread": set(), "tpx": None})

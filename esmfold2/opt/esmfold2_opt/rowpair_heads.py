"""The row statements of the esmfold2 kit's `n_gpu > 1` line (the launch / install / pair-stack side is `rowpair.py`).

Under P > 1 ranks the pair representation z ``[B, N, N, c_z]`` (B = 1, c_z = 256) is ROW-SHARDED from the statement that creates it to the last
statement that reads it: rank r holds global rows ``r0:r1`` (``opt_core.mem.rowpair.dist.Layout``) of z, z_init, lm_z, the relative-position and
token-bond encodings, the pair mask, the parcae-coda output, the distogram logits, the diffusion pair conditioning and the confidence pair
state. Every function below is the STOCK statement (Biohub transformers fork, ``stock/PINS.json``; files ``modeling_esmfold2.py`` = M and
``modeling_esmfold2_common.py`` = C) with the LEFT pair operand restricted to this rank's rows; per-element arithmetic equals the dense
statement, GEMM shapes differ only in the row count. Data movement is opt_core's (``rowpair.{dist, shard, ring, diffusion, trunk}``); nothing
here calls ``torch.distributed``.

REPLICATED BY DESIGN (every rank holds the whole tensor; named in the schedule census): the token single representations ``x_inputs`` /
``s`` ``[B, N, ·]``, the ESMC language-model activations (sequence-only), the MSA representation ``m`` (full_msa; ``rowpair_msa.py``), the atom
tensors and the atom sliding-window attention masks (``atom_swa.py`` names its own treatment), the token-level diffusion state ``a``
``[B·S, N, 768]``, the sampled coordinates, and the ``[B, N, N]`` fp32 pair mask of the trunk (c-less; sharded before every pair stack).

NUMERICS: the ``n_gpu > 1`` line is tier 2 by name. The pair-state RNG draw (``M._init_pair_state``) and the
per-loop LM-pair dropout draw are ROWS OF THE STOCK CUDA PHILOX STREAM (``opt_core.mem.rowpair.rng``; the default generator advances by exactly
the stock full-tensor increment on every rank) so a fold's random stream is the single-GPU stream and the output does not depend on P; on a
device without that replica (CPU test ranks) the draw is this rank's rows from a private generator seeded off ONE default-generator draw —
NAMED ``pair_rng=rank_rows`` in the census, never silent. The confidence scalars (pTM / ipTM / per-chain-pair ipTM) are masked row sums of the TM
map computed on this rank's rows, then a max over the gathered ``[S, N]`` per-row vector (pTM, ipTM) or an all-reduce of per-rank partial
sums (per-chain-pair ipTM; summation order differs from the dense statement — inside the tier-2 word); no ``[N, N]`` tensor is gathered
or replicated on any device. PAE (an output the writer stages) goes to rank 0's host memory sample by sample; PDE and the logits stay rows.
With ``num_diffusion_samples = S > 1`` the confidence statement runs ONCE PER SAMPLE (``confidence_forward_rows`` loops
``confidence_forward_rows_batched`` at S = 1 and assembles the outputs in the stock ``b·S + s`` batch order): the stock statement's
``[B·S, R, N, 256]`` fp32 pair rows and ``[B·S, R, N, 64]`` logits rows are never materialised for S > 1 — the per-sample arithmetic is the
S = 1 statement's own (the head has no cross-sample term); the per-sample PAE / PDE LOGITS rows are dropped by name (``pae_logits`` /
``pde_logits`` = None, ``conf_logits=dropped`` in the returned census: upstream's processor and the kit's writer read ``pae`` / ``plddt`` /
the scalars, never the logits); ``conf_samples=loop:<S>`` in the schedule census. S = 1 is the batched statement itself.
"""
from __future__ import annotations

import math
import os
from typing import Callable, Optional

# opt_core census hook: instrumentation only (an older core lacks it: then a no-op).
try:
    from opt_core.mem.rowpair import census as _CENSUS
except ImportError:                                                  # gates instrumentation only, never real work
    _CENSUS = None


def mark(stage: str) -> None:
    """``opt_core.mem.rowpair.census.mark(stage)`` when the core ships it (no-op unless OPT_CORE_TP_CENSUS=1)."""
    if _CENSUS is not None:
        _CENSUS.mark(stage)


def _stock():
    from transformers.models.esmfold2 import modeling_esmfold2 as M, modeling_esmfold2_common as C
    return M, C


# =========================================================================================== the pair representation born as rows
def relpos_rows(enc, lay, residue_index, asym_id, sym_id, entity_id, token_index):
    """Rows ``r0:r1`` of ``C.ResIdxAsymIdSymIdEntityIdEncoding.forward`` (C:2062-2111): every ``x.unsqueeze(2)`` (the row operand) becomes
    ``x[:, r0:r1].unsqueeze(2)``; the column operands, the clips, the ``where`` conditions, the one-hot widths and the final Linear are the
    stock statements."""
    import torch
    import torch.nn.functional as F
    r0, r1 = lay.r0, lay.r1
    bij_same_chain = asym_id[:, r0:r1].unsqueeze(2) == asym_id.unsqueeze(1)
    bij_same_residue = residue_index[:, r0:r1].unsqueeze(2) == residue_index.unsqueeze(1)
    bij_same_entity = entity_id[:, r0:r1].unsqueeze(2) == entity_id.unsqueeze(1)

    dij_residue = residue_index[:, r0:r1].unsqueeze(2) - residue_index.unsqueeze(1)
    dij_residue = torch.clip(dij_residue + enc.n_relative_residx_bins, 0, 2 * enc.n_relative_residx_bins)
    dij_residue = torch.where(bij_same_chain, dij_residue, 2 * enc.n_relative_residx_bins + 1)
    relative_residue = F.one_hot(dij_residue.long(), 2 * enc.n_relative_residx_bins + 2)

    dij_token = token_index[:, r0:r1].unsqueeze(2) - token_index.unsqueeze(1)
    dij_token = torch.clip(dij_token + enc.n_relative_residx_bins, 0, 2 * enc.n_relative_residx_bins)
    dij_token = torch.where(bij_same_chain & bij_same_residue, dij_token, 2 * enc.n_relative_residx_bins + 1)
    relative_token = F.one_hot(dij_token.long(), 2 * enc.n_relative_residx_bins + 2)

    dij_chain = sym_id[:, r0:r1].unsqueeze(2) - sym_id.unsqueeze(1)
    dij_chain = torch.clip(dij_chain + enc.n_relative_chain_bins, 0, 2 * enc.n_relative_chain_bins)
    dij_chain = torch.where(bij_same_chain, 2 * enc.n_relative_chain_bins + 1, dij_chain)
    relative_chain = F.one_hot(dij_chain.long(), 2 * enc.n_relative_chain_bins + 2)

    feats = torch.cat([relative_residue.float(), relative_token.float(), bij_same_entity.float().unsqueeze(-1), relative_chain.float()], dim=-1)   # the stock feature order
    return enc.embed(feats)


RELPOS_SOURCE_ANCHORS = (                                              # test_rowpair_heads pins these statements to the stock text (source guard)
    "bij_same_chain = asym_id.unsqueeze(2) == asym_id.unsqueeze(1)",
    "dij_chain = torch.where(",
    "self.embed = nn.Linear(total_feats, d_pair, bias=False)",
)


BORN_ROWS = 64                                                       # the launch grid (rows per block) of every row-blocked fp32 statement of this line: the
#   born-rows statements' [rows, N, F] one-hot / outer-product intermediates are block-sized, only the [R, N, c_z] results are shard-sized; ONE
#   definition — the hosted line's row-blocked statements (EF2_XL add-on) bind the same value
BORN_ROWS_ENV = "EF2_ROWPAIR_BORN_ROWS"                              # override of BORN_ROWS (diagnostics)


class _RowSpan(object):
    """Rows ``r0:r1`` (a sub-span of this rank's rows) with the layout's ``N``: what the row statements below read off a layout."""

    def __init__(self, lay, r0: int, r1: int):
        self.r0, self.r1, self.N, self.P, self.rank = int(r0), int(r1), int(lay.N), int(getattr(lay, "P", 1)), int(getattr(lay, "rank", 0))
        self.R = self.r1 - self.r0


def in_row_blocks(fn, lay, rows=None):
    """``fn(span) -> [B, span.R, N, ...]`` evaluated over consecutive row sub-spans of this rank's rows and written into one ``[B, R, N, ...]``
    result (row-local statements: identical per element to one call over all rows)."""
    blk = int(rows or int(os.environ.get(BORN_ROWS_ENV, "") or BORN_ROWS))
    out = None
    for l0 in range(0, int(lay.R), max(1, blk)):
        l1 = min(int(lay.R), l0 + blk)
        part = fn(_RowSpan(lay, lay.r0 + l0, lay.r0 + l1))
        if out is None:
            out = part.new_empty((int(part.shape[0]), int(lay.R)) + tuple(int(s) for s in part.shape[2:]))
        out[:, l0:l1] = part
        del part
    return out


def single_to_pair_rows(s2p, lay, x):
    """Rows of ``C.SingleToPair.forward`` (C:2124-2131): ``x = downproject(x)``; ``cat([x_i * x_j, x_i - x_j])`` with ``i`` restricted to this
    rank's rows; ``output_mlp`` (Linear / GELU / Linear over the last dim)."""
    import torch
    x = s2p.downproject(x)
    xi = x[:, lay.r0:lay.r1]
    x = torch.cat([(xi.unsqueeze(2) * x.unsqueeze(1)), (xi.unsqueeze(2) - x.unsqueeze(1))], dim=3)
    return s2p.output_mlp(x)


def lm_pair_rows(shim, lay, hidden_states):
    """Rows of ``C.LanguageModelShim.forward`` (C:2166-2181; ``lm_dropout`` = 0 as the model calls it): the token-level statements verbatim
    (``base_z_linear``, the softmax-weighted layer combine — replicated ``[B, N, d_z]``), then ``base_z_mlp`` = SingleToPair on rows ->
    LayerNorm (last dim)."""
    lm_z = shim.base_z_linear(hidden_states)                          # [B, L, layers+1, d_z]
    weights = shim.base_z_combine.softmax(0)
    lm_z = (weights @ lm_z).squeeze(-2)                               # [B, L, d_z]
    layers = list(shim.base_z_mlp)

    def rows_of(span):
        z = single_to_pair_rows(layers[0], span, lm_z)
        for layer in layers[1:]:
            z = layer(z)
        return z
    return in_row_blocks(rows_of, lay)


LM_SOURCE_ANCHORS = ("lm_z = (weights @ lm_z).squeeze(-2)", "[(x.unsqueeze(2) * x.unsqueeze(1)), (x.unsqueeze(2) - x.unsqueeze(1))]")


def z_init_rows(model, lay, x_inputs):
    """Rows of ``z_init = self.z_init_1(x_inputs).unsqueeze(2) + self.z_init_2(x_inputs).unsqueeze(1)`` (M:950-952) =
    ``opt_core.mem.rowpair.trunk.outer_sum_rows``."""
    from opt_core.mem.rowpair import trunk as RTK
    return RTK.outer_sum_rows(model.z_init_1(x_inputs), model.z_init_2(x_inputs), lay.r0, lay.r1)


def token_bonds_rows(model, lay, token_bonds):
    """Rows of ``self.token_bonds(token_bonds.float())`` (M:961; ``token_bonds`` is ``nn.Linear(1, c_z)`` over ``[B, N, N, 1]``)."""
    tb = token_bonds[:, lay.r0:lay.r1]
    return model.token_bonds(tb.float())


def pair_mask_rows(lay, tok_mask):
    """Rows of ``pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()`` (M:986)."""
    from opt_core.mem.rowpair import trunk as RTK
    return RTK.pair_mask_rows(tok_mask.float(), lay.r0, lay.r1)


# ----------------------------------------------------------------------------------------------- the two RNG draws of the trunk, as rows
def _rng_core():
    """``opt_core.mem.rowpair.rng`` (rows of a stock full-tensor CUDA Philox draw); None only where torch has no CUDA build of it importable
    (then ``pair_rng=rank_rows`` is the NAMED path)."""
    try:
        from opt_core.mem.rowpair import rng as RNG
        return RNG
    except ImportError:
        return None


def pair_rng_mode(device) -> str:
    """``philox_rows``: rows of the stock stream (CUDA + the core replica); ``rank_rows``: private per-rank generator (CPU ranks; NAMED tier-2)."""
    return "philox_rows" if (getattr(device, "type", str(device)) == "cuda" and _rng_core() is not None) else "rank_rows"


def _rank_generator(torch, device):
    """A private generator seeded off ONE draw of the default generator (identical on every rank, so the default stream stays in step)."""
    seed = int(torch.randint(0, 2 ** 62, (1,)).item())
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1_000_003 * _rank_of_default())
    return gen


def _rank_of_default() -> int:
    from opt_core.mem.rowpair import dist as RD
    return int(RD.world()[1])


def init_pair_state_rows(lay, z_init_rows_t):
    """Rows of ``M.ESMFold2Model._init_pair_state`` (M:748-752): ``std = sqrt(2 / (5 c_z))``; ``trunc_normal_(empty fp32 [B, N, N, c_z], 0, std,
    -3 std, 3 std).to(z_init.dtype)`` restricted to rows ``r0:r1``. ``philox_rows``: the values ARE rows of the stock draw and the default CUDA
    generator advances by the stock full-tensor increment (``opt_core.mem.rowpair.rng.trunc_normal_rows``). ``rank_rows``: a private draw."""
    import torch
    import torch.nn as nn
    B, R, N, C = (int(v) for v in z_init_rows_t.shape)
    std = math.sqrt(2.0 / (5.0 * C))
    mode = pair_rng_mode(z_init_rows_t.device)
    if mode == "philox_rows":                                         # rows of the stock draw; the generator ends where the stock call leaves it
        RNG = _rng_core()
        if B != 1:
            from opt_core.mem.rowpair import RowpairRefused
            raise RowpairRefused(f"init_pair_state_rows: batch {B} != 1 — one item per fold call is the row-sharded line's domain (the Philox row replica is written per item): "
                                 "fold the items separately or use --n_gpu 1")
        res = RNG.trunc_normal_shard(lay, C, std=std, a=-3 * std, b=3 * std, device=z_init_rows_t.device, out_dtype=z_init_rows_t.dtype)
        rows = res[0] if isinstance(res, tuple) else res                # (rows, info): rows of the stock draw; rounds agreed across ranks; generator finished as stock
        return rows.reshape(B, R, N, C), mode
    gen = _rank_generator(torch, z_init_rows_t.device)
    state = torch.empty((B, R, N, C), dtype=torch.float32, device=z_init_rows_t.device)
    nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=gen)
    return state.to(dtype=z_init_rows_t.dtype), mode


def lm_dropout_rows(lay, lm_z_rows, p: float, N: int):
    """Rows of ``F.dropout(lm_z, p=p, training=True)`` (M:781). ``philox_rows``: the keep mask is rows of the stock fused-dropout Philox stream
    and the generator advances as stock (``rng.dropout_keep_rows``); ``rank_rows``: a private Bernoulli draw. Either way ``out = lm_z * keep /
    (1 - p)`` per element (the stock kernel's arithmetic)."""
    import torch
    B, R, _, C = (int(v) for v in lm_z_rows.shape)
    mode = pair_rng_mode(lm_z_rows.device)
    if mode == "philox_rows" and lm_z_rows.dtype == torch.float32 and B == 1 and (N * N * C) % 4 == 0:
        RNG = _rng_core()
        out, _mask, _info = RNG.dropout_rows(lm_z_rows.contiguous(), N, C, lay.r0, lay.r1, p=float(p), device=lm_z_rows.device, return_mask=False)
        return out, mode                                              # == rows of F.dropout(lm_z, p, training=True) bit for bit; generator advanced as stock
    if mode == "philox_rows":
        mode = "rank_rows:dropout_dtype"                              # NAMED: the replica is the fp32 VEC=4 kernel path; other dtypes take the private draw
    gen = _rank_generator(torch, lm_z_rows.device)
    keep = torch.rand(lm_z_rows.shape, generator=gen, device=lm_z_rows.device, dtype=torch.float32) >= p
    return lm_z_rows * keep.to(lm_z_rows.dtype) / (1.0 - p), mode


# =========================================================================================== the distogram on rows
def distogram_rows(model, lay, z_rows):
    """``self.distogram_head(z + z.transpose(-2, -3))`` (M:1030) on rows: rows ``r0:r1`` of ``zᵀ`` are this rank's COLUMNS of everyone's rows =
    one all-to-all (``opt_core.mem.rowpair.ring.transpose_shard``); the head (Linear) is per element. Returns logits rows ``[B, R, N, bins]``."""
    from opt_core.mem.rowpair import ring as RR
    zt = RR.transpose_shard(z_rows.contiguous(), lay)
    out = model.distogram_head(z_rows + zt)
    del zt
    return out


# =========================================================================================== the diffusion module on rows
def diffusion_sync_mode(torch) -> str:
    """The replicated-tensor sync policy BY DET LEVEL, set by the kit: ``bcast`` at det 0 (rank 0 authoritative at every sync point),
    ``guard`` under the deterministic recipe (``--det >= 1``: torch's deterministic algorithms on; a guard failure there is a real defect).
    Census word ``diff_noise=``: ``bcast_rank0_state`` | ``guard``."""
    return "guard" if torch.are_deterministic_algorithms_enabled() else "bcast"


def decide_schedule(dm, lay, sched_box: dict, z_trunk, num_diffusion_samples: int):
    """The roll-out's row schedule (``opt_core.mem.rowpair.diffusion.DiffusionSchedule``), decided once per roll-out on every rank."""
    import torch
    from opt_core.mem.rowpair import diffusion as RDF
    cond, tt = dm.conditioning, dm.token_transformer
    H, dh, S = int(tt.attn_blocks[0].num_heads), int(tt.attn_blocks[0].head_dim), max(1, int(num_diffusion_samples))
    core = dit_rows_admit(RDF, dit_rows_word(), heads=H, head_dim=dh, samples=S, device=z_trunk.device, torch=torch)   # the rows route's attention core, once per roll-out
    sched_box["dit_rows"] = core
    kw = dict(attn_core=core["attn_core"]) if core["attn_core"] is not None else {}
    sched = RDF.DiffusionSchedule.decide(lay, c_z=int(z_trunk.shape[-1]), c_in=int(cond.z_input_norm.normalized_shape[0]),
                                       c_cond=int(cond.z_proj.out_features), H=H, S=S, n_blocks=len(tt.attn_blocks), elt=4, **kw)
    sched_box["sched"] = sched
    sched_box["rollouts"] = sched_box.get("rollouts", 0) + 1
    return sched


def dit_rows_admit(RDF, word: str, *, heads: int, head_dim: int, samples: int, device, torch) -> dict:
    """The rows route's query-block attention core for this roll-out (``EF2_ROWPAIR_DIT_ROWS``): the shared core's static admission
    (``opt_core.mem.rowpair.diffusion.dit_rows_core``, opt_core >= 0.5.216) of the word on this device / operand class -> ``attn_core``
    ('kernel': the schedule keeps all local rows as one query block) + the selection the face reuses per call; an older core or the word
    ``engine`` = the materialised statement (nothing bound). The token transformer's operands are bf16 under this line's autocast."""
    out = {"word": word, "attn_core": None, "selection": None, "why": None, "bound": False}
    if word in ("", "engine", "off", "none", "stock"):
        out["why"] = "engine:env" if os.environ.get(ENV_DIT_ROWS, "").strip() else "engine"
        return out
    if not hasattr(RDF, "dit_attention_rows"):                              # opt_core < 0.5.216: the statement, named
        out["why"] = "engine:core_without_rows_face"
        return out
    if hasattr(RDF, "dit_rows_census"):
        RDF.dit_rows_census(reset=True)                                     # per roll-out counts on the fold line
    if hasattr(RDF, "dit_bias_census"):
        RDF.dit_bias_census(reset=True)
    core, sel, why = RDF.dit_rows_core(word, dtype=torch.bfloat16, heads=int(heads), head_dim=int(head_dim), samples=int(samples), device=device)
    out.update(attn_core=core, selection=sel, why=why, bound=True)
    return out


def cond_rows(dm, lay, z_trunk, relative_position_encoding, sched=None):
    """Step 1 of ``C.DiffusionModule.forward`` on rows: the conditioned pair ROWS ``[B, R, N, c_z]`` fp32 — ``cat(z_trunk, relpos) ->
    z_input_norm -> z_proj -> (+ transitions under bf16 autocast)`` per row block (``opt_core.mem.rowpair.diffusion.pair_cond_rows`` over the
    stock statements C:1380-1385)."""
    import torch
    from opt_core.mem.rowpair import diffusion as RDF
    cond = dm.conditioning
    rel_rows = relative_position_encoding

    def embed_fn(z_rows_blk, g0, g1):                                 # C:1380-1382 on a row block: cat -> z_input_norm -> z_proj
        i0, i1 = g0 - lay.r0, g1 - lay.r0
        z_pair = torch.cat([z_rows_blk.float(), rel_rows[:, i0:i1].float()], dim=-1)
        return cond.z_proj(cond.z_input_norm(z_pair))

    def _transition(block):
        def fn(zb, g0, g1):                                           # C:1383-1385: `with autocast(bf16): z = z + block(z)` (z stays fp32)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=zb.is_cuda):
                return block(zb)
        return fn
    return RDF.pair_cond_rows(embed_fn, z_trunk, lay, c_out=int(cond.z_proj.out_features),
                              rows=(sched.cond_rows if sched is not None else None), transitions=[_transition(b) for b in cond.z_transitions])


# ====================================================================== the sampler's token path WHOLE when it fits (the fused step)
SAMPLER_ROUTE_WORDS = ("auto", "whole", "rows")                             # EF2_ROWPAIR_SAMPLER
ENV_DIT_ROWS = "EF2_ROWPAIR_DIT_ROWS"                                       # the rows route's query-block attention core: big (default: the shared core's rows face
#   opt_core.kernels.apb.pair_bias_attention_rows — flash attention over this rank's query rows against all keys, the block's pair-bias rows read in
#   place, 16-bit operands / fp32 statistics; served row apb_attn) | a row word (apb_attn | sba[:tf32|ieee|tf32x3] | sdpa[:…] | naive) | engine
#   (the materialised statement: einsum logits + softmax per query sub-block). A refusal by name serves the
#   statement; the fold line carries dit_rows=<row>:<calls>[,stock:<why>:<calls>] and dit_rows_core=<the admitted core>.
DIT_ROWS_DEFAULT = "big"


def dit_rows_word() -> str:
    return (os.environ.get(ENV_DIT_ROWS, "") or DIT_ROWS_DEFAULT).strip().lower()


def dit_rows_facts(sched_box: dict) -> dict:
    """Fold-line facts of the rows route's attention core: ``dit_rows=<row>:<calls>[,stock:<why>:<calls>]`` (the shared core's census of the
    calls its rows face served this roll-out and the calls the statement served after a refusal by name), ``dit_rows_core=`` (the admitted
    core or why nothing was bound), ``dit_rows_word=``."""
    core = sched_box.get("dit_rows") or {}
    out = {"dit_rows_word": core.get("word", dit_rows_word())}
    try:                                                                     # the pair-bias producer: dit_bias=<word>:<blocks>[,engine:<why>:<n>], dit_bias_word=
        from opt_core.mem.rowpair import diffusion as _RDF
        if hasattr(_RDF, "dit_bias_census"):
            out["dit_bias"] = _census_token(_RDF.dit_bias_census()); out["dit_bias_word"] = _RDF.dit_bias_word()
    except Exception as e:                                                   # census only
        out["dit_bias"] = "unknown:" + type(e).__name__
    if core.get("bound"):
        from opt_core.mem.rowpair import diffusion as RDF
        out["dit_rows"] = RDF.dit_rows_census() if hasattr(RDF, "dit_rows_census") else "unknown"
        sel = core.get("selection")
        why = _census_token(core.get("why"))
        out["dit_rows_core"] = (core.get("attn_core") or "none") + (":" + str(getattr(sel, "row", "")) if sel is not None else "") + (":" + why if why else "")
    else:
        out["dit_rows"] = "engine"
        out["dit_rows_core"] = _census_token(core.get("why")) or "engine"
    return out


def _census_token(v, limit: int = 96) -> str:
    """A fold-line value: one token (whitespace -> '_'), bounded."""
    t = "_".join(str(v or "").split())
    return t if len(t) <= limit else t[:limit] + "…"

ROUTE_WHOLE_SMALL_FRACTION = 0.5                                            # `auto` also takes the whole route when live + need stays under this fraction of the card: a fold that
#   small cannot run out of memory and its reach is not in question (the trunk-peak clause alone would send the SMALLEST folds to the rows route,
#   where the trunk peaked lowest); named `whole:small` on the fold line


def sampler_route_decide(dm, lay, *, S: int, word: str, dit_ready: str, has_cache: bool, device, torch):
    """Which token path the diffusion sampler takes this fold, agreed across ranks: ``whole`` = the one-GPU line's fused step
    (``ef2_dit``: the 12 blocks' pair biases hoisted once per roll-out from the WHOLE conditioned pair, flash attention with the bias in place)
    fed the conditioned pair gathered from every rank's rows in bf16 — taken when that fits inside memory the trunk ALREADY peaked at on
    every rank (it cannot raise the item's per-rank peak) and inside the free device memory with margin; ``rows`` = the row-sharded token
    transformer (:func:`dm_forward_rows`). ``word``: ``auto`` (decide) | ``whole`` (whenever the fused step is installed) | ``rows``.
    Returns ``(route, facts)``; ``facts`` land on the fold line (``sampler_route=… sampler_need_gib=… sampler_headroom_gib=…``)."""
    N, R, P = lay.N, lay.R, lay.P
    cond, tt = dm.conditioning, dm.token_transformer
    nb, H, c_z = len(tt.attn_blocks), int(tt.attn_blocks[0].num_heads), int(cond.z_proj.out_features)
    st = getattr(dm, "_dit", None)
    cfg = dict(getattr(st, "cfg", {}) or {})
    bias_elt = 4 if (cfg.get("attn") == "sdpa" and cfg.get("attn_precision") != "bf16") else 2
    gib = float(1 << 30)
    rows32, rows16 = R * N * c_z * 4, R * N * c_z * 2                       # the conditioned rows (fp32) and their bf16 cast
    whole = P * lay.Rmax * N * c_z * 2                                       # the gathered bf16 pair (the all_gather buffer itself, padded to Rmax rows per rank)
    biases = nb * H * N * N * bias_elt + H * N * N * bias_elt                # the fused step's 12 hoisted [1, H, N, N] pair biases + one block's kernel output before the store
    need = max(rows32 + rows16, rows16 + whole, whole + biases) + (1 << 30)
    facts = {"sampler_word": word, "sampler_need_gib": round(need / gib, 2)}
    ok, why = False, None
    if word == "rows":
        why = "rows:env"
    elif dit_ready != "ready":
        why = f"rows:{dit_ready}"
    elif not has_cache:
        why = "rows:no_inference_cache"
    else:
        alloc = int(torch.cuda.memory_allocated(device)); peak = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device)); free_dev = int(torch.cuda.mem_get_info(device)[0])
        headroom = peak - alloc                                              # what the trunk already peaked at above what is live now: usable without raising the item's peak
        avail = free_dev + max(0, reserved - alloc)                          # device-free + the allocator's cached-free
        facts.update(sampler_headroom_gib=round(headroom / gib, 2), sampler_avail_gib=round(avail / gib, 2))
        total = int(torch.cuda.mem_get_info(device)[1])
        facts["sampler_live_gib"] = round(alloc / gib, 2)
        if word == "whole":
            ok, why = True, "whole:env"
        elif need + (2 << 30) > avail:
            ok, why = False, "rows:over_free"
        elif need <= headroom:
            ok, why = True, "whole:fits"                                     # inside memory the trunk already peaked at: the item's per-rank peak does not move
        elif alloc + need <= ROUTE_WHOLE_SMALL_FRACTION * total:
            ok, why = True, "whole:small"                                    # a small fold: the peak moves, far inside the card
        else:
            ok, why = False, "rows:over_peak"
    if lay.P > 1:                                                            # one word for every rank (memory differs by rank): whole only if whole everywhere
        from opt_core.mem.rowpair import dist as RD
        t = torch.tensor([1 if ok else 0], dtype=torch.int64, device=device)
        RD.allreduce_(t, "min")
        agreed = bool(int(t.item()))
        if ok and not agreed:
            why = "rows:peer_over"
        ok = agreed
    facts["sampler_route"] = "whole_replicated" if ok else "sharded_rows"
    facts["sampler_route_why"] = why
    facts["sampler_S"] = int(S)
    return ("whole" if ok else "rows"), facts


def dm_forward_whole(dm, inner, lay, box: dict, DIT, x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, ref_space_uid,
                     tok_idx, s_inputs, s_trunk, z_trunk, relative_position_encoding, asym_id, residue_index, entity_id, token_index, sym_id,
                     sigma_data=None, token_attention_mask=None, num_diffusion_samples: int = 1, return_token_repr: bool = False,
                     return_atom_repr: bool = False, inference_cache=None):
    """``C.DiffusionModule.forward`` under sharding by the WHOLE route: at the roll-out's first step the conditioned pair is computed on this
    rank's rows (:func:`cond_rows`, fp32), cast to bf16 and gathered whole (``[1, N, N, c_z]`` bf16 on every rank — the one N×N tensor this
    line ever holds whole, sized inside memory the trunk already used: :func:`sampler_route_decide`), and handed to the one-GPU line's fused
    step (``inner`` = ``ef2_dit``'s forward) through the conditioning module's pair-path hook for exactly this call; the fused step hoists the
    12 blocks' pair biases from it and every later step reads them in place. Every rank runs the same statements on the same state (the noisy
    coordinates are synchronised as on the rows route), so the sample is replicated; nothing here is captured in a graph."""
    import torch
    from opt_core.mem.rowpair import diffusion as RDF
    from opt_core.mem.rowpair import shard as RS
    x_sync = RDF.sync_replicated(x_noisy.contiguous(), "diffusion_state", mode=diffusion_sync_mode(torch))
    if x_sync is not x_noisy:
        x_noisy.copy_(x_sync)
        del x_sync
    RDF.sync_replicated(x_noisy, "diffusion.x_noisy")
    cache = inference_cache
    cd = dm.conditioning
    hook_name = getattr(DIT, "PAIR_PATH_HOOK", "_ef2_pair_path")
    fresh = cache is not None and "z" not in cache
    z_whole = None
    if fresh:                                                                 # step 0 of the roll-out: the whole conditioned pair, bf16, from the ranks' rows
        mark("diffusion_start")
        sched = decide_schedule(dm, lay, box, z_trunk, num_diffusion_samples)
        z_rows = cond_rows(dm, lay, z_trunk, relative_position_encoding, sched)          # [1, R, N, c_z] fp32 — the rows route's statements
        z_bf = z_rows.to(torch.bfloat16)
        del z_rows
        z_whole = RS.unshard_rows(z_bf[0], lay, dim=0).unsqueeze(0)                      # [1, N, N, c_z] bf16: the all_gather buffer itself
        del z_bf
        box["cond_calls"] = box.get("cond_calls", 0) + 1
        box["z_whole_gib"] = round(z_whole.numel() * z_whole.element_size() / float(1 << 30), 2)
        hooks_before = int(DIT.STATS.get("z_cond_hook", 0))
        vars(cd)[hook_name] = (lambda _z_trunk, _rel, _zw=z_whole: _zw)                   # instance attribute: shadows a class-level hook (the XL add-on's) for this one call
    try:
        out = inner(x_noisy=x_noisy, t_hat=t_hat, ref_pos=ref_pos, ref_charge=ref_charge, ref_mask=ref_mask, ref_element=ref_element,
                    ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid, tok_idx=tok_idx, s_inputs=s_inputs, s_trunk=s_trunk,
                    z_trunk=z_trunk, relative_position_encoding=relative_position_encoding, asym_id=asym_id, residue_index=residue_index,
                    entity_id=entity_id, token_index=token_index, sym_id=sym_id, sigma_data=sigma_data, token_attention_mask=token_attention_mask,
                    num_diffusion_samples=num_diffusion_samples, return_token_repr=return_token_repr, return_atom_repr=return_atom_repr,
                    inference_cache=inference_cache)
    finally:
        if fresh:
            vars(cd).pop(hook_name, None)
    if fresh:
        if int(DIT.STATS.get("z_cond_hook", 0)) <= hooks_before or cache.get("z") is not z_whole:
            from opt_core.mem.rowpair import RowpairRefused
            raise RowpairRefused("sampler whole route: the fused diffusion step did not take the gathered pair through the conditioning module's "
                                 "pair-path hook (ef2_dit fell back for this call) — set EF2_ROWPAIR_SAMPLER=rows")
        mark("diffusion_peak")
    box["whole_steps"] = box.get("whole_steps", 0) + 1
    return out


def dm_forward_rows(dm, lay, sched_box: dict, x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, ref_space_uid,
                    tok_idx, s_inputs, s_trunk, z_trunk, relative_position_encoding, asym_id, residue_index, entity_id, token_index, sym_id,
                    sigma_data=None, token_attention_mask=None, num_diffusion_samples: int = 1, return_token_repr: bool = False,
                    return_atom_repr: bool = False, inference_cache=None):
    """``C.DiffusionModule.forward`` (C:1509-1615) with ``z_trunk`` and ``relative_position_encoding`` arriving as ROWS ``[B, R, N, c_z]``:
    step 1 (conditioning) caches the pair conditioning ROWS once per rollout (``opt_core.mem.rowpair.diffusion.pair_cond_rows`` over the stock
    ``z_input_norm`` / ``z_proj`` / ``z_transitions``; the stock ``DiffusionConditioning.forward`` then computes the single path verbatim and
    reads the cached rows), steps 2-4 and 6-8 are the stock statements (atoms and the token state are replicated by design), step 5 (the
    12-block token transformer) runs every ``AttentionPairBias`` with LOCAL query rows + the pair-bias rows of this rank and all-gathers the
    updated rows once per block (``diffusion.diffusion_transformer_sharded``; the fused pair-bias kernel takes a square z and is not called on
    rows: NAMED ``pair_bias=rows_standard``)."""
    import torch
    from opt_core.mem.rowpair import diffusion as RDF
    x_sync = RDF.sync_replicated(x_noisy.contiguous(), "diffusion_state", mode=diffusion_sync_mode(torch))    # every denoiser call conditions on ONE
    #   state: det 0 -> rank 0's (bcast; atoms are replicated by RECOMPUTE under non-det kernels = tier-2-identical only), det >= 1 -> guard (strict bitwise)
    if x_sync is not x_noisy:
        x_noisy.copy_(x_sync)                                             # IN PLACE: ``x_noisy`` IS the sampler's loop tensor — its update statement
        del x_sync                                                        #   (C.sample: rigid-align + the ODE step read x_noisy) integrates rank 0's state on every rank
    mark("diffusion_start")
    bsz = x_noisy.shape[0]
    sigma = dm.sigma_data if sigma_data is None else float(sigma_data)
    t = torch.as_tensor(t_hat, dtype=torch.float32, device=x_noisy.device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(bsz)
    if int(z_trunk.shape[-3]) != lay.R or int(z_trunk.shape[-2]) != lay.N:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"esmfold2 dm_forward_rows: z_trunk {tuple(z_trunk.shape)} is not this rank's row shard (R={lay.R}, N={lay.N})")
    RDF.sync_replicated(x_noisy, "diffusion.x_noisy")                # the sampler's noise draws must agree across ranks (guard by default)
    # Step 1: conditioning — pair rows once per rollout, single path = the stock module reading the cached rows
    cond = dm.conditioning
    cache = inference_cache if inference_cache is not None else {}
    if "z" not in cache:                                              # a new roll-out: decide the row schedule once (every rank reaches this point)
        sched = decide_schedule(dm, lay, sched_box, z_trunk, num_diffusion_samples)
        sched_box["bias_cache"] = RDF.PairBiasCache(enabled=sched.bias_cache)
        cache["z"] = cond_rows(dm, lay, z_trunk, relative_position_encoding, sched)
        sched_box["cond_calls"] = sched_box.get("cond_calls", 0) + 1
    s, z = cond(t_hat=t, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, relative_position_encoding=relative_position_encoding,
                sigma_data=sigma, num_diffusion_samples=num_diffusion_samples, inference_cache=cache)
    # Step 2: normalize noisy coords
    denom = torch.sqrt(t * t + sigma * sigma)
    r_noisy = x_noisy / denom[:, None, None]
    # Step 3: atom encoder (replicated by design; atom_swa.py names the window-attention treatment)
    a, q_skip, c_skip, p_skip, enc_intermediates = dm.atom_encoder(
        ref_pos=ref_pos, atom_attention_mask=ref_mask, ref_space_uid=ref_space_uid, ref_charge=ref_charge, ref_element=ref_element,
        ref_atom_name_chars=ref_atom_name_chars, atom_to_token=tok_idx, r_l=r_noisy, s_i=s_trunk, num_diffusion_samples=num_diffusion_samples,
        return_intermediates=return_atom_repr, inference_cache=inference_cache)
    # Step 4: add conditioned s
    a = a + dm.s_to_token(dm.s_step_norm(s))
    # Step 5: token transformer — local query rows per block, one all-gather per block
    a = token_transformer_rows(dm.token_transformer, lay, sched_box, a, s, z, attention_mask=token_attention_mask,
                               num_diffusion_samples=num_diffusion_samples)
    mark("diffusion_peak")
    # Step 6: token norm
    a = dm.token_norm(a)
    # Step 7: atom decoder
    r_update, dec_intermediates = dm.atom_decoder(a_i=a, q_l=q_skip, c_l=c_skip, p_lm=p_skip, atom_to_token=tok_idx, atom_attention_mask=ref_mask,
                                                  num_diffusion_samples=num_diffusion_samples, return_intermediates=return_atom_repr)
    # Step 8: compute denoised output
    sigma2 = sigma * sigma
    t2 = t * t
    out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
    out = out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update
    atom_intermediates = None
    if return_atom_repr:
        all_ints = enc_intermediates + dec_intermediates
        if all_ints:
            atom_intermediates = torch.stack(all_ints, dim=2)
    return _dm_output(dm, out, a, atom_intermediates, return_token_repr)


def _dm_output(dm, out, a, atom_intermediates, return_token_repr):
    """The stock return statement of ``DiffusionModule.forward`` (its dict keys), read off the stock source once so a key rename upstream fails
    the source guard instead of silently diverging."""
    return {"x_denoised": out, "token_repr": a if return_token_repr else None, "atom_intermediates": atom_intermediates}


DM_RETURN_ANCHOR = '"x_denoised": out,'                                       # test pins the stock return keys (source guard)


def token_transformer_rows(tt, lay, sched_box: dict, a, s, z_rows, *, attention_mask=None, num_diffusion_samples: int = 1):
    """``C.DiffusionTransformer.forward`` (C:1310-1325) = ``for attn, transition: x = x + attn(x, s, z, ...); x = x + transition(x, s)`` with
    each ``AttentionPairBias`` (C:1077-1200, the standard branch) evaluated for this rank's QUERY rows against all keys, its pair bias read off
    the bias ROWS, and the block's row update all-gathered (``opt_core.mem.rowpair.diffusion.diffusion_transformer_sharded``)."""
    import torch
    from opt_core.mem.rowpair import diffusion as RDF
    bsz, N, d_model = (int(v) for v in a.shape)
    S = int(num_diffusion_samples)
    mask = attention_mask
    if mask is not None and int(mask.shape[0]) != bsz and S > 1:      # C:1099-1105 verbatim
        mask = mask.repeat_interleave(S, dim=0)

    sched = sched_box.get("sched")
    cache = sched_box.get("bias_cache")
    rows_core = sched_box.get("dit_rows")                             # decide_schedule's admission of the rows attention core for this roll-out (None: an older adapter state)

    def fns_of(blk, trans):
        H, dh = int(blk.num_heads), int(blk.head_dim)

        def norm(x, s_):                                              # C:1087-1090
            return blk.adaln(x, s_) if s_ is not None else blk.pre_norm(x)

        def kv(xn):                                                   # C:1093-1096 (keys/values over ALL tokens: replicated x)
            k, v = blk.kv_proj(xn).chunk(2, dim=-1)
            return k.reshape(bsz, N, H, dh), v.reshape(bsz, N, H, dh)

        def bias(z_rows_blk):                                         # C:1170: pair_bias_proj(pair_norm(z)) on rows -> [B, rows, N, H]
            return blk.pair_bias_proj(blk.pair_norm(z_rows_blk))

        def attn_stock(x_q, kv_state, bias_q, rows):                  # C:1163-1181 with i = this rank's query rows g0:g1 (the materialised statement)
            g0, g1 = rows
            k, v = kv_state
            nq = g1 - g0
            q = blk.q_proj(x_q).view(bsz, nq, H, dh)
            g = torch.sigmoid(blk.g_proj(x_q)).view(bsz, nq, H, dh)
            logits = torch.einsum("... i h d, ... j h d -> ... i j h", q, k) * blk.scale
            pb = bias_q                                               # core layout [*, H, q, N] -> stock [*, q, N, H]
            if pb.dim() == 4 and int(pb.shape[1]) == H and int(pb.shape[-1]) != H:
                pb = pb.permute(0, 2, 3, 1)
            if int(pb.shape[0]) != bsz and S > 1:                    # C:1098: z.repeat_interleave(S, 0) == expand when B == 1
                pb = pb.expand(bsz, *pb.shape[1:]) if int(pb.shape[0]) == 1 else pb.repeat_interleave(S, dim=0)
            logits = logits + pb.to(dtype=logits.dtype)
            if mask is not None:
                min_val = torch.finfo(logits.dtype).min
                mask_bias = torch.where(mask.bool()[:, None, :, None], 0.0, min_val)
                logits = logits + mask_bias.to(dtype=logits.dtype)
            w = torch.softmax(logits, dim=-2).to(dtype=v.dtype)
            ctx = torch.einsum("... i j h, ... j h d -> ... i h d", w, v)
            ctx = g * ctx
            return blk.out_proj(ctx.reshape(bsz, nq, d_model))

        if rows_core is not None and rows_core.get("bound"):          # opt_core >= 0.5.216: the query block on the shared core's rows face
            def kv(xn, _kv=kv):                                       # noqa: F811 — k / v in the face's [S, H, N, D] layout (views), cast ONCE per block to bf16
                k, v = _kv(xn)                                        # when the roll-out is fp32 (the documented tier-word policy); the statement's tuple kept as stock
                return RDF.DitKV(RDF.cast16(k.permute(0, 2, 1, 3)), RDF.cast16(v.permute(0, 2, 1, 3)), (mask if mask is None else mask.bool()), stock=(k, v))

            def q_fn(x_q):                                            # C:1163: q_proj -> [S, H, q, D] (a view; the face applies D**-0.5 = blk.scale)
                return blk.q_proj(x_q).view(bsz, -1, H, dh).permute(0, 2, 1, 3)

            def out_fn(o, x_q):                                       # C:1178-1181: sigmoid(g_proj(x_q)) * ctx -> out_proj; o arrives [S, q, H*D] token-major
                return blk.out_proj(torch.sigmoid(blk.g_proj(x_q)).to(o.dtype) * o)

            attn = RDF.dit_attention_rows(q_fn=q_fn, out_fn=out_fn, stock_fn=attn_stock, num_heads=H, core_word=rows_core["word"], scale=None,
                                          stock_q_rows=(getattr(sched, "q_rows_stock", None) if sched is not None else None),
                                          selection=rows_core.get("selection"))
        else:
            attn = attn_stock

        def update(x_full, o_rows, s_, rows):                          # x = x + attn(...) (gated by s); x = x + transition(x, s) — on rows
            g0, g1 = rows
            o = o_rows
            if s_ is not None:
                o = torch.sigmoid(blk.out_gate(s_[:, g0:g1])) * o       # C:1198-1199
            x_rows = x_full[:, g0:g1] + o
            return x_rows + trans(x_rows, s_[:, g0:g1] if s_ is not None else None)

        if hasattr(RDF, "DitBias"):                                    # opt_core >= 0.5.216 (c_z 256 served from 0.5.218.1): the pair-bias producer as WEIGHTS —
            bias_into = RDF.DitBias(engine=bias, ln_weight=blk.pair_norm.weight, ln_bias=blk.pair_norm.bias,     # ROWPAIR_DIFF_BIAS=ln_proj_fp32 (this line's default: fp32-exact,
                                    weight=blk.pair_bias_proj.weight, linear_bias=blk.pair_bias_proj.bias,     # one pass, no [rows, N, H] transient) | ln_proj (bf16 MMA) | engine;
                                    eps=float(blk.pair_norm.eps))                                              # a refusal by name serves `bias` per block (dit_bias=engine:<why>)
            return RDF.DiTBlockFns(norm=norm, kv=kv, attn=attn, update=update, bias=bias, bias_into=bias_into)
        return RDF.DiTBlockFns(norm=norm, kv=kv, attn=attn, update=update, bias=bias)

    blocks = [fns_of(blk, tr) for blk, tr in zip(tt.attn_blocks, tt.transition_blocks)]
    x = RDF.diffusion_transformer_sharded(blocks, a, s, z_rows, lay, schedule=sched, bias_cache=cache)
    sched_box["tt_calls"] = sched_box.get("tt_calls", 0) + 1
    return x


# =========================================================================================== the confidence head on rows
CONF_LOGITS_DROPPED = ("pae_logits", "pde_logits")                          # the [B·S, R, N, 64] fp32 logits rows the per-sample loop (S > 1) does not keep: None in the output, named in its census (read by nothing downstream)


def confidence_forward_rows(head, lay, run_trunk: Callable, s_inputs, z, x_pred, distogram_atom_idx, token_attention_mask, atom_to_token,
                            atom_attention_mask, asym_id, mol_type, num_diffusion_samples: int = 1, relative_position_encoding=None,
                            token_bonds_encoding=None):
    """``M.ConfidenceHead.forward`` on rows for ``num_diffusion_samples = S`` structure samples. S == 1: :func:`confidence_forward_rows_batched`
    itself (the one statement; nothing else runs). S > 1: that statement ONCE PER SAMPLE at S = 1 — sample ``s`` of ``x_pred`` (the stock
    ``b·S + s`` batch order of ``M.ConfidenceHead._repeat_batch`` / ``_flatten_sample_axis``), every other input as given (they carry no
    sample axis) — and the per-sample outputs assembled along dim 0 in that same order, so every key has the stock shape and the per-sample
    numbers of the S = 1 statement; the ``[B·S, R, N, 256]`` fp32 pair rows of the batched statement never exist. The per-sample
    ``pae_logits`` / ``pde_logits`` rows (:data:`CONF_LOGITS_DROPPED`) are not kept: those keys are None and the returned census says
    ``conf_logits=dropped`` (keeping ``[B·S, R, N, 64]`` fp32 rows across the loop would make the residency S-proportional again, on device
    or host; upstream's processor and the kit's writer read neither); ``pae`` is rank 0's host ``[B·S, N, N]`` (None elsewhere). An ``x_pred``
    whose sample axis is not S, or whose item count disagrees with ``z``, is refused by name. Census:
    ``conf_samples=loop:<S>``; the fold line's ``confidence_trunk_s`` counts S trunk calls."""
    S = int(num_diffusion_samples)
    kw = dict(distogram_atom_idx=distogram_atom_idx, token_attention_mask=token_attention_mask, atom_to_token=atom_to_token,
              atom_attention_mask=atom_attention_mask, asym_id=asym_id, mol_type=mol_type, relative_position_encoding=relative_position_encoding,
              token_bonds_encoding=token_bonds_encoding)
    if S <= 1:
        return confidence_forward_rows_batched(head, lay, run_trunk, s_inputs, z, x_pred, num_diffusion_samples=num_diffusion_samples, **kw)
    import torch
    from opt_core.mem.rowpair import evidence as EV
    EV.record_schedule(conf_samples=f"loop:{S}")
    from opt_core.mem.rowpair import RowpairRefused
    if x_pred.dim() == 4 and int(x_pred.shape[1]) != S:
        raise RowpairRefused(f"esmfold2 confidence_forward_rows: x_pred {tuple(x_pred.shape)} has a sample axis of {int(x_pred.shape[1])}, not num_diffusion_samples={S}")
    x_flat = head._flatten_sample_axis(x_pred)                              # [B·S, atoms, 3] in the stock b·S + s order
    if int(x_flat.shape[0]) % S:
        raise RowpairRefused(f"esmfold2 confidence_forward_rows: x_pred {tuple(x_pred.shape)} carries {int(x_flat.shape[0])} structures, not a multiple of "
                             f"num_diffusion_samples={S}")
    B = int(x_flat.shape[0]) // S
    if int(z.shape[0]) != B:
        raise RowpairRefused(f"esmfold2 confidence_forward_rows: x_pred carries {B} item(s) x {S} samples but z has batch {int(z.shape[0])}")
    per = []
    for s in range(S):
        out_s = confidence_forward_rows_batched(head, lay, run_trunk, s_inputs, z, x_flat[s::S].contiguous(), num_diffusion_samples=1, **kw)   # rows b·S + s, b = 0..B-1
        for k in CONF_LOGITS_DROPPED:                                        # this sample's logits rows are released before the next sample allocates
            out_s[k] = None
        per.append(out_s)
    out = {"_rowpair": dict(per[0]["_rowpair"], samples=f"loop:{S}", conf_logits="dropped", **{k: "dropped" for k in CONF_LOGITS_DROPPED})}
    for k in per[0]:
        if k == "_rowpair":
            continue
        vals = [p[k] for p in per]
        out[k] = None if vals[0] is None else torch.stack(vals, dim=1).reshape((B * S,) + tuple(vals[0].shape[1:]))   # [B, ...] per sample -> [B·S, ...] in the b·S + s order (pae: None off rank 0; the dropped logits: None)
    del per
    return out


HEAD_EMBED_ROWS = 256                                                    # rows per block of the confidence head's distance-bin embedding add (a per-element lookup)


def _acc_(base, other):
    """``base + other`` accumulated IN PLACE when that is the same arithmetic — the sum's result dtype is ``base``'s own (``base`` is a
    fresh tensor of this module's, never a caller's) — else the out-of-place statement. Same values either way; one pair-rows tensor
    alive instead of two."""
    import torch
    if torch.is_tensor(other) and torch.result_type(base, other) == base.dtype and not base.requires_grad:
        return base.add_(other)
    return base + other


def add_embed_rows_(pair, embed, bins, rows: int = HEAD_EMBED_ROWS):
    """``pair + embed(bins)`` for a per-element embedding lookup (``bins`` ``[B, R, N]`` long -> ``[B, R, N, C]``), added per block of
    ``rows`` rows in place (same values as the whole-tensor statement when the sum's dtype is ``pair``'s; else that statement): the
    ``[B, R, N, C]`` lookup transient is one row block's."""
    import torch
    probe = embed(bins[:, :1, :1])
    if torch.result_type(pair, probe) != pair.dtype or pair.requires_grad:
        return pair + embed(bins)
    R = int(pair.shape[1])
    for i0 in range(0, R, max(1, int(rows))):
        i1 = min(R, i0 + max(1, int(rows)))
        pair[:, i0:i1].add_(embed(bins[:, i0:i1]))
    return pair


def add_prod_rows_(pair, out_proj, a_rows, b, rows: int = HEAD_EMBED_ROWS):
    """``pair + out_proj(a_rows[:, :, None, :] * b[:, None, :, :])`` — the outer-product pair term — added per block of ``rows`` rows in
    place (the product and its projection are per-(i, j) statements: same values as the whole-rows statement when the sum's dtype is
    ``pair``'s; else that statement): the ``[B, rows, N, c]`` product transient is one row block's."""
    import torch
    probe = out_proj(a_rows[:, :1, None, :] * b[:, None, :1, :])
    if torch.result_type(pair, probe) != pair.dtype or pair.requires_grad:
        return pair + out_proj(a_rows[:, :, None, :] * b[:, None, :, :])
    R = int(pair.shape[1])
    for i0 in range(0, R, max(1, int(rows))):
        i1 = min(R, i0 + max(1, int(rows)))
        pair[:, i0:i1].add_(out_proj(a_rows[:, i0:i1, None, :] * b[:, None, :, :]))
    return pair


def confidence_forward_rows_batched(head, lay, run_trunk: Callable, s_inputs, z, x_pred, distogram_atom_idx, token_attention_mask, atom_to_token,
                                    atom_attention_mask, asym_id, mol_type, num_diffusion_samples: int = 1, relative_position_encoding=None,
                                    token_bonds_encoding=None):
    """``M.ConfidenceHead.forward`` (M:172-350) with ``z``, ``relative_position_encoding`` and ``token_bonds_encoding`` arriving as ROWS: the
    pair state, the distance-bin embedding, the 4-block trunk (``run_trunk`` = the kit's presharded pair-stack driver), the residual add, the
    row-attention pooling (softmax over j per row: row-local), the PAE / PDE logits and the TM map are computed on rows; ``single`` rows and the
    per-row interface flags are all-gathered (``[S, N, ·]``, small); pTM / ipTM = max over the gathered ``[S, N]`` vector of row-local masked
    means; per-chain-pair ipTM = per-rank partial sums all-reduced; ``pae`` = ``[S, N, N]`` fp32 on rank 0's HOST (None on other ranks);
    ``pde``, ``pae_logits``, ``pde_logits`` stay this rank's rows (the writer stages none of them)."""
    import torch
    import torch.nn.functional as F
    from opt_core.mem.rowpair import dist as RD, shard as RS
    M, C = _stock()
    _EPS, _NONPOLYMER_ID = M._EPS, M._NONPOLYMER_ID
    gather_rep_atom_coords, gather_token_to_atom, _compute_intra_token_idx, _categorical_mean = (
        C.gather_rep_atom_coords, C.gather_token_to_atom, C._compute_intra_token_idx, C._categorical_mean)
    mark("confidence")
    r0, r1, R, N = lay.r0, lay.r1, lay.R, lay.N
    if int(z.shape[-3]) != R or int(z.shape[-2]) != N:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"esmfold2 confidence_forward_rows: z {tuple(z.shape)} is not this rank's row shard (R={R}, N={N})")

    s_inputs_normed = head.s_inputs_norm(s_inputs)
    z_base = head.z_norm(z)                                                 # fresh rows [B, R, N, c_z]: the sums below accumulate INTO it (_acc_: in place when that is the
    if relative_position_encoding is not None:                           #   same arithmetic, i.e. the sum's dtype is z_base's; else the stock out-of-place add) — one rows
        z_base = _acc_(z_base, relative_position_encoding)               #   tensor alive instead of two per statement
    if token_bonds_encoding is not None:
        z_base = _acc_(z_base, token_bonds_encoding)
    z_base = _acc_(z_base, head.s_to_z(s_inputs_normed)[:, r0:r1].unsqueeze(2))
    z_base = _acc_(z_base, head.s_to_z_transpose(s_inputs_normed).unsqueeze(1))
    z_base = add_prod_rows_(z_base, head.s_to_z_prod_out, head.s_to_z_prod_in1(s_inputs_normed)[:, r0:r1], head.s_to_z_prod_in2(s_inputs_normed))   # + prod_out(in1_i ⊗ in2_j) per row block
    pair = head._repeat_batch(z_base, num_diffusion_samples)
    del z_base
    x_pred_flat = head._flatten_sample_axis(x_pred)
    atom_to_token_m = head._repeat_batch(atom_to_token, num_diffusion_samples)
    atom_mask_m = head._repeat_batch(atom_attention_mask, num_diffusion_samples)
    rep_idx_m = head._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
    mask = head._repeat_batch(token_attention_mask, num_diffusion_samples)
    Bm = pair.shape[0]
    rep_coords = gather_rep_atom_coords(x_pred_flat, rep_idx_m)
    rep_distances = torch.cdist(rep_coords[:, r0:r1], rep_coords, compute_mode="donot_use_mm_for_euclid_dist")          # rows [Bm, R, N]
    distogram_bins = (rep_distances.unsqueeze(-1) > head.boundaries).sum(dim=-1).long()
    pair = add_embed_rows_(pair, head.dist_bin_pairwise_embed, distogram_bins)   # pair + dist_bin_pairwise_embed(bins): the lookup is per element -> added per row block in place
    mask_rows = mask[:, r0:r1]
    pair_mask = mask_rows[:, :, None].float() * mask[:, None, :].float()                                                 # rows [Bm, R, N]
    with torch.amp.autocast("cuda", enabled=pair.is_cuda, dtype=torch.bfloat16):
        pair_delta = run_trunk(head.folding_trunk, pair, pair_mask)
    pair.add_(pair_delta.float())
    del pair_delta
    # RowAttentionPooling (C:1947-1957) on rows: scores / softmax over j / pooling are per row i -> single rows, then all-gather [Bm, N, d]
    rap = head.row_attention_pooling
    scores = rap.attn_proj(pair).squeeze(-1)
    scores = scores.masked_fill(~mask[:, None, :].bool().expand_as(scores), float("-inf")) if _rap_masks_keys(rap) else scores
    single_rows = _rap_rows(rap, pair, mask, scores)
    single = RS.unshard_rows(single_rows.contiguous(), lay, dim=1)
    del single_rows, scores
    atom_mask_f = atom_mask_m.float()
    s_at_atoms = gather_token_to_atom(single, atom_to_token_m)
    s_at_atoms_ln = head.plddt_ln(s_at_atoms)
    intra_idx = _compute_intra_token_idx(atom_to_token_m)
    intra_idx = intra_idx.clamp(max=head.plddt_weight.shape[0] - 1)
    w_plddt = head.plddt_weight[intra_idx]
    plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms_ln, w_plddt)
    plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)
    L = single.shape[1]
    plddt_sum = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
    atom_count = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
    atom_mask_t = atom_mask_f.to(plddt_per_atom.dtype)
    plddt_sum.scatter_add_(1, atom_to_token_m, plddt_per_atom * atom_mask_t)
    atom_count.scatter_add_(1, atom_to_token_m, atom_mask_t)
    plddt = plddt_sum / atom_count.clamp(min=1e-6)
    complex_plddt = (plddt_per_atom * atom_mask_f).sum(dim=-1) / (atom_mask_f.sum(dim=-1) + _EPS)
    expanded_type = head._repeat_batch(mol_type, num_diffusion_samples)
    expanded_asym = head._repeat_batch(asym_id, num_diffusion_samples)
    is_ligand = (expanded_type == _NONPOLYMER_ID).float()
    inter_chain_rows = (expanded_asym[:, r0:r1].unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float()                 # rows [Bm, R, N]
    near_contact = (rep_distances < 8).float()
    interface_rows = (near_contact * inter_chain_rows * (1.0 - is_ligand[:, r0:r1]).unsqueeze(-1)).amax(dim=-1)        # per row i: complete
    interface_per_token = RS.unshard_rows(interface_rows.contiguous(), lay, dim=1)
    del near_contact, interface_rows
    iplddt_weight = torch.where(is_ligand.bool(), torch.full_like(interface_per_token, 2.0), interface_per_token)
    iplddt_weight_atoms = gather_token_to_atom(iplddt_weight.unsqueeze(-1), atom_to_token_m).squeeze(-1)
    atom_iplddt_w = atom_mask_f * iplddt_weight_atoms
    complex_iplddt = (plddt_per_atom * atom_iplddt_w).sum(dim=-1) / (atom_iplddt_w.sum(dim=-1) + _EPS)
    plddt_ca = plddt_per_atom.gather(1, rep_idx_m)
    # PAE / PDE on rows
    pae_logits = head.pae_head(head.pae_ln(pair))
    pae_rows = _categorical_mean(pae_logits, start=0.0, end=32.0).detach()
    pde_logits = head.pde_head(head.pde_ln(pair))
    pde_rows = _categorical_mean(pde_logits, start=0.0, end=32.0).detach()
    s_at_atoms_res = head.resolved_ln(s_at_atoms)
    w_res = head.resolved_weight[intra_idx]
    resolved_logits = torch.einsum("...c,...cb->...b", s_at_atoms_res, w_res)
    # pTM / ipTM / per-chain-pair ipTM (M:288-326) from the TM map ROWS: every statistic is a masked row sum (row-local on complete rows) followed
    # by a max over i (over the gathered [Bm, N] per-row vector) or a sum over (i, j) (per-rank partial sums, all-reduced) — no [N, N] tensor
    # is gathered or replicated on any device.
    n_bins = pae_logits.shape[-1]
    bin_width = 32.0 / n_bins
    bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device)
    mask_f = mask.float()
    N_res = mask_f.sum(dim=-1, keepdim=True)
    d0 = 1.24 * (N_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
    tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
    pae_probs = F.softmax(pae_logits, dim=-1)
    tm_rows = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)                                                    # rows of tm_expected [Bm, R, N]
    del pae_probs
    pair_mask_rows = mask_f[:, r0:r1].unsqueeze(-1) * mask_f.unsqueeze(-2)                                               # rows of pair_mask_2d
    ptm_rows = (tm_rows * pair_mask_rows).sum(dim=-1) / (pair_mask_rows.sum(dim=-1) + _EPS)                             # rows of ptm_per_row [Bm, R]
    ptm = RS.unshard_rows(ptm_rows.contiguous(), lay, dim=1).max(dim=-1).values
    inter_chain_rows = (expanded_asym[:, r0:r1].unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float() * pair_mask_rows
    iptm_rows = (tm_rows * inter_chain_rows).sum(dim=-1) / (inter_chain_rows.sum(dim=-1) + _EPS)
    iptm = RS.unshard_rows(iptm_rows.contiguous(), lay, dim=1).max(dim=-1).values
    del inter_chain_rows, pair_mask_rows, ptm_rows, iptm_rows
    max_chain_id = int(expanded_asym.max().item()) if Bm > 0 else 0
    n_chains = max_chain_id + 1
    num = torch.zeros(Bm, n_chains, n_chains, device=tm_rows.device, dtype=tm_rows.dtype)                                # this rank's partial sums over its rows
    den = torch.zeros(Bm, n_chains, n_chains, device=tm_rows.device, dtype=tm_rows.dtype)
    for c1 in range(n_chains):
        chain_c1 = (expanded_asym == c1).float() * mask_f
        if chain_c1.sum() == 0:
            continue
        for c2 in range(n_chains):
            chain_c2 = (expanded_asym == c2).float() * mask_f
            pair_m = chain_c1[:, r0:r1].unsqueeze(-1) * chain_c2.unsqueeze(-2)                                          # rows of the (c1, c2) pair mask
            num[:, c1, c2] = (tm_rows * pair_m).sum(dim=(-1, -2))
            den[:, c1, c2] = pair_m.sum(dim=(-1, -2))
    RD.allreduce_(num, "sum"); RD.allreduce_(den, "sum")
    pair_chains_iptm = torch.zeros_like(num)
    for c1 in range(n_chains):
        if ((expanded_asym == c1).float() * mask_f).sum() == 0:
            continue                                                                                                     # the stock loop leaves these rows at zero
        pair_chains_iptm[:, c1, :] = num[:, c1, :] / (den[:, c1, :] + _EPS)
    del tm_rows, num, den
    # PAE is an OUTPUT the writer stages ([S, N, N] fp32): its rows go to rank 0's HOST, one sample at a time (per-sample transient on rank 0
    # only); other ranks return None (the processor tolerates it; only the output rank writes). PDE and the logits stay rows.
    mark("confidence_gather")
    pae = pae_rows_to_rank0_host(pae_rows, lay)
    pde = pde_rows
    mark("done")
    return {
        "plddt_logits": plddt_logits, "plddt": plddt.detach(), "plddt_per_atom": plddt_per_atom.detach(), "plddt_ca": plddt_ca.detach(),
        "complex_plddt": complex_plddt.detach(), "complex_iplddt": complex_iplddt.detach(),
        "pae_logits": pae_logits, "pae": pae, "pde_logits": pde_logits, "pde": pde, "resolved_logits": resolved_logits,
        "ptm": ptm.detach(), "iptm": iptm.detach(), "pair_chains_iptm": pair_chains_iptm.detach(),
        "_rowpair": {"rows": (r0, r1), "pae_logits": "rows", "pde_logits": "rows", "pde": "rows", "pae": "rank0_host_rowblocks", "single": "gathered",
                     "ptm": "row_sums+gather[N]", "pair_chains_iptm": "row_partials+allreduce"},
    }




def pae_rows_to_rank0_host(pae_rows, lay):
    """``[S, R, N]`` PAE rows of every rank -> the whole ``[S, N, N]`` fp32 matrix in rank 0's pinned HOST memory (None on other ranks), one
    ``opt_core.mem.rowpair.dist.gather_rows_to_rank0_host`` per sample (the family's output-assembly primitive: row blocks land host-side; no
    ``[N, N]`` tensor exists on any device). Every rank issues the same collectives in the same order."""
    import torch
    from opt_core.mem.rowpair import dist as RD
    parts = [RD.gather_rows_to_rank0_host(pae_rows[i].contiguous(), lay) for i in range(int(pae_rows.shape[0]))]
    return torch.stack(parts) if parts and parts[0] is not None else None


def _rap_masks_keys(rap) -> bool:
    return False                                                      # the stock pooling masks inside its own forward (see _rap_rows)


def _rap_rows(rap, pair_rows, mask, _scores_unused):
    """``C.RowAttentionPooling.forward`` (C:1947-1957) on rows: the module's own forward is row-separable (its softmax runs over j for each
    row i and its mask is the KEY mask), so it is CALLED UNCHANGED on the row block with the full key mask."""
    return rap(pair_rows, mask)


CONF_SOURCE_ANCHORS = (                                                # test_rowpair_heads pins these statements to the stock text (source guard)
    "z_base = z_base + self.s_to_z(s_inputs_normed).unsqueeze(2)",
    'rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist"',
    "single = self.row_attention_pooling(pair, mask)",
    "tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)",
    "pair_chains_iptm[:, c1, c2] = (tm_expected * pair_m).sum(",
    "d0 = 1.24 * (N_res.clamp(min=19) - 15) ** (1 / 3) - 1.8",
)

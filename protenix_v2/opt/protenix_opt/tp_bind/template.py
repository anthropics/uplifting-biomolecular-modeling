"""Protenix-v2 ``TemplateEmbedder`` (pairformer.py, AF3 Algorithm 16) bound onto the shared core's row-sharded template driver
(``opt_core.mem.rowpair.template.template_embed_rows``). The pair track stays SHARDED through the embedder: every ``[T, N, N, *]`` template
pair feature exists on a rank only as its rows ``[T, R, N, *]`` — sliced ONCE at entry from dense featuriser tensors
(``slice_template_inputs_to_rows``), rebuilt per row block from per-token template keys, or synthesised as exact zeros for the
featuriser's dummy slots (an all-dummy templated run is a NAMED event, ``templ_event=all_slots_dummy``).

What is Protenix's here (the statements; the schedule is the core's):
    unit rows    ``v_t[rows] = linear_no_bias_z(layernorm_z(z[rows])) + linear_no_bias_a(at_t[rows])`` with ``at_t`` the 108-channel template
                 feature (distogram 39 | pseudo-beta mask | aatype_j one-hot 32 | aatype_i one-hot 32 | unit vector 3 | backbone mask), every
                 pair feature multiplied by the same-chain mask and the all-ones pair mask exactly as stock
    pair stack   the embedder's ``pairformer_stack.blocks`` (``c_s = 0``) through the PAIR-BLOCK CALLABLE the caller passes (the line's TP pair
                 block), then ``layernorm_v`` (one call below 2**31 elements, 128-row blocks above: ``shard.ln_rows_guarded``)
    finish       ``u = 0; u = u + v_t`` in slot order; ``u / (1e-7 + T)``; ``linear_no_bias_u(relu(u))``; ``z[rows] += u[rows]`` (in place)
Replicated by design: the per-token template keys (``template_aatype``, the 1-D masks, per-token coordinates), ``asym_id``.
Row block: ``BLOCK_ROWS`` (an explicit argument; census ``templ_block_rows … given``)."""
from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from opt_core.mem.rowpair import RowpairRefused
from opt_core.mem.rowpair import template as core_template
from opt_core.mem.rowpair.dist import Layout, require_sharded
from opt_core.mem.rowpair.evidence import record_schedule
from opt_core.mem.rowpair.shard import ln_rows_guarded
from opt_core.mem.rowpair.trunk import same_rows

DENSE_KEYS = ("template_distogram", "template_pseudo_beta_mask", "template_unit_vector", "template_backbone_frame_mask")
PB1D, BB1D = "template_pseudo_beta_mask_1d", "template_backbone_frame_mask_1d"     # per-token masks of a compacted feature dict [T, N]
DENSE_FREE_MARKER = "template_pair_dense_free"                                        # the dict carries no dense template pair tensor
REAL_MARKER = "template_real_rows"                                                    # per-token REAL template keys (coordinates, frames)
FEATURE_DIMS = {"dgram": 39, "pb2d": 1, "uv": 3, "bb2d": 1}                          # the four pair features, F per key (44 per pair)
REAL_KEYS = ("template_pb_pos", "template_pb_mask_1d", "template_ca_pos", "template_frame_R", "template_bb_mask_1d")  # per-token REAL template keys [T, N, ...]
# the tp_route.BIND row this module serves (carried module -> names)
ROUTE_ROWS = {"ptx_tp.template": ("protenix_opt.tp_bind.template", ("tp_template_embedder", "template_embedder_is_active"))}
N_AATYPE = 32                                                                         # len(protenix STD_RESIDUES_WITH_GAP)
BLOCK_ROWS = 128


def template_embedder_is_active(te, feats: Mapping[str, Any]) -> bool:
    """True iff the stock ``TemplateEmbedder.forward`` does work (else it returns the int 0)."""
    return "template_aatype" in feats and int(getattr(te, "n_blocks", 0)) >= 1


def n_slots(feats: Mapping[str, Any]) -> int:
    return int(feats["template_aatype"].shape[-2])


# ----------------------------------------------------------------------------------------------------------------- feature rows: the source
def slice_template_inputs_to_rows(feats: Mapping[str, Any], layout: Layout) -> Dict[str, Any]:
    """ENTRY seam for a dict that carries the featuriser's DENSE template pair tensors: each becomes this rank's rows ``[T, R, N, F]`` (the two
    2-D masks gain their trailing feature dim first). A dict without dense keys is returned unchanged."""
    if not any(k in feats for k in DENSE_KEYS):
        return dict(feats)
    d = dict(feats)
    for k in ("template_pseudo_beta_mask", "template_backbone_frame_mask"):
        if k in d and d[k].dim() == 3:
            d[k] = d[k].unsqueeze(-1)
    return core_template.slice_template_inputs_to_rows(d, layout, DENSE_KEYS, missing="refuse")


def slot_census(feats: Mapping[str, Any]) -> core_template.SlotCensus:
    """REAL vs DUMMY slots from whatever masks the dict carries (per-token 1-D masks; else the row-sliced 2-D masks flattened per slot, the
    verdict agreed by the driver's slot grouping)."""
    masks = [feats[k] for k in (PB1D, BB1D, "template_pb_mask_1d", "template_bb_mask_1d") if k in feats]
    if not masks:
        T = n_slots(feats)
        masks = [feats[k].reshape(T, -1) for k in ("template_pseudo_beta_mask", "template_backbone_frame_mask") if k in feats]
    if not masks:
        raise RowpairRefused("protenix template binding: the feature dict carries template_aatype but no template mask "
                             f"(none of {PB1D}, {BB1D}, template_pseudo_beta_mask, template_backbone_frame_mask)")
    return core_template.slot_census(*masks)


def _outer_rows(m, slots, g0: int, g1: int):
    """``m[t, g0:g1, None] * m[t, None, :]`` for the listed slots -> float32 ``[S, rows, N, 1]`` (the compacted 2-D mask statement)."""
    x = m[list(slots)]
    return (x[:, g0:g1, None] * x[:, None, :]).to(torch.float32).unsqueeze(-1)


def pair_rows_source(feats: Mapping[str, Any], layout: Layout, *, real_rows_fn: Optional[Callable] = None) -> core_template.TemplatePairRows:
    """The ONE source of template pair-feature rows for this dict:
       dense keys (already row-sliced by :func:`slice_template_inputs_to_rows`)  -> ``rows=``;
       per-token REAL template keys (``template_real_rows``)                      -> ``computed=`` (``real_rows_fn(feats, t, g0, g1, device) ->
                                                                                     (dgram, pb2d, uv, bb2d)`` rows of slot t; dummy slots take
                                                                                     the compacted statement);
       a compacted dict (``template_pair_dense_free``)                            -> ``computed=`` (masks = outer products of the 1-D masks,
                                                                                     distogram / unit vector exactly zero).
    Anything else is refused by name."""
    T = n_slots(feats)
    if "template_distogram" in feats:
        t = feats["template_distogram"]
        if t.dim() < 4 or int(t.shape[-3]) != layout.R or int(t.shape[-2]) != layout.N:
            raise RowpairRefused(f"protenix template binding: dense template pair features {tuple(t.shape)} must be row-sliced at trunk entry "
                                 f"(slice_template_inputs_to_rows) — rows R={layout.R} of N={layout.N}")
        rows = {"dgram": feats["template_distogram"], "pb2d": feats["template_pseudo_beta_mask"], "uv": feats["template_unit_vector"],
                "bb2d": feats["template_backbone_frame_mask"]}
        rows = {k: (v if v.dim() == 4 else v.unsqueeze(-1)) for k, v in rows.items()}
        return core_template.TemplatePairRows(layout, T, rows=rows)
    pb_key = PB1D if PB1D in feats else ("template_pb_mask_1d" if "template_pb_mask_1d" in feats else None)
    bb_key = BB1D if BB1D in feats else ("template_bb_mask_1d" if "template_bb_mask_1d" in feats else None)
    if pb_key is None or bb_key is None:
        raise RowpairRefused("protenix template binding: template features are neither dense nor compacted nor per-token real rows "
                             f"(need {DENSE_KEYS[0]} | {PB1D}+{BB1D} | {REAL_MARKER})")
    pb1d, bb1d = feats[pb_key], feats[bb_key]
    dev = pb1d.device
    real = REAL_MARKER in feats
    if real and real_rows_fn is None:
        raise RowpairRefused(f"protenix template binding: {REAL_MARKER} present but no real_rows_fn was bound (the featuriser's per-token "
                             "template rows statement)")
    cache: Dict[Any, Any] = {}

    def slot_rows(t: int, g0: int, g1: int):
        key = (t, g0, g1)
        if key not in cache:
            cache.clear()
            is_real = real and (float(pb1d[t].sum()) != 0.0 or float(bb1d[t].sum()) != 0.0)
            if is_real:
                dg, pb2d, uv, bb2d = real_rows_fn(feats, t, g0, g1, dev)
                cache[key] = (dg, pb2d.unsqueeze(-1), uv, bb2d.unsqueeze(-1))
            else:
                n = g1 - g0
                cache[key] = (torch.zeros((n, layout.N, 39), dtype=torch.float32, device=dev), _outer_rows(pb1d, [t], g0, g1)[0],
                              torch.zeros((n, layout.N, 3), dtype=torch.float32, device=dev), _outer_rows(bb1d, [t], g0, g1)[0])
        return cache[key]

    idx = {"dgram": 0, "pb2d": 1, "uv": 2, "bb2d": 3}

    def provider(k):
        return lambda slots, g0, g1: torch.stack([slot_rows(int(t), g0, g1)[idx[k]] for t in slots], dim=0)

    return core_template.TemplatePairRows(layout, T, computed={k: provider(k) for k in FEATURE_DIMS})


# ----------------------------------------------------------------------------------------------------------------- the statements
def at_rows(feats: Mapping[str, Any], acc: core_template.TemplatePairRows, slot: int, g0: int, g1: int, z_dtype) -> torch.Tensor:
    """The 108-channel template feature ``at`` of slot ``slot`` for GLOBAL rows ``[g0, g1)`` -> float32 ``[rows, N, 108]`` (stock
    ``TemplateEmbedder.forward``'s per-template block, on rows)."""
    n, N, r0 = g1 - g0, acc.N, acc.r0
    asym_id = feats["asym_id"]
    multichain_mask = same_rows(asym_id, g0, g1).to(z_dtype)                                  # rows of (asym_i == asym_j)
    pair_mask = torch.ones((n, N), dtype=z_dtype, device=asym_id.device)                      # rows of z.new_ones([N, N])
    i0, i1 = g0 - r0, g1 - r0
    dgram = acc.rows("dgram", [slot], i0, i1)[0]
    pseudo_beta_mask_2d = acc.rows("pb2d", [slot], i0, i1)[0][..., 0]
    unit_vector = acc.rows("uv", [slot], i0, i1)[0]
    backbone_mask_2d = acc.rows("bb2d", [slot], i0, i1)[0][..., 0]
    to_concat = []
    dgram = dgram * multichain_mask[..., None] * pair_mask[..., None]
    pseudo_beta_mask_2d = pseudo_beta_mask_2d * multichain_mask * pair_mask
    to_concat.append(dgram)
    to_concat.append(pseudo_beta_mask_2d.unsqueeze(-1))
    aatype = F.one_hot(feats["template_aatype"][slot], num_classes=N_AATYPE)                  # [N, 32] int64
    to_concat.append(aatype.unsqueeze(-3).expand(n, -1, -1))                                  # [i, j] = aatype[j]
    to_concat.append(aatype[g0:g1].unsqueeze(-2).expand(-1, N, -1))                           # [i, j] = aatype[i]
    unit_vector = unit_vector * multichain_mask[..., None] * pair_mask[..., None]
    to_concat.append(unit_vector)
    backbone_mask_2d = backbone_mask_2d * multichain_mask * pair_mask
    to_concat.append(backbone_mask_2d.unsqueeze(-1))
    return torch.concat(to_concat, dim=-1)


def template_update_(te, feats: Mapping[str, Any], z_shard, layout: Layout, *, pair_block: Callable, add: bool = True,
                     rows: int = BLOCK_ROWS, real_rows_fn: Optional[Callable] = None, log: Optional[Callable[[str], None]] = None,
                     triangle_multiplicative: str = "torch", triangle_attention: str = "torch", inplace_safe: bool = True,
                     chunk_size: Optional[int] = None):
    """``z_shard [R, N, c_z]`` with the template term ADDED on its rows in place (``add=True``), or a new shard holding ``z rows + term`` with
    ``z_shard`` untouched (``add=False``); ``z_shard`` itself when the embedder is inactive (stock adds the int 0). ``pair_block(block, v [R, N, c],
    layout, **kw) -> v`` is the line's TP pair block for a ``c_s = 0`` PairformerBlock."""
    if not template_embedder_is_active(te, feats):
        return z_shard
    require_sharded(layout, "protenix template binding")
    say = log if log is not None else (lambda _m: None)
    T = n_slots(feats)
    census = slot_census(feats)
    note = core_template.note_all_dummy(census, templated=REAL_MARKER in feats, tag="protenix template embedder")
    if note and layout.rank == 0:
        say(note)
    acc = pair_rows_source(feats, layout, real_rows_fn=real_rows_fn)
    keys = [feats["template_aatype"]] + [feats[k] for k in (PB1D, BB1D) + REAL_KEYS + DENSE_KEYS if k in feats]   # every per-slot input: per-token
                                                                                                                     # keys, and the row-sliced pair features
    groups = core_template.template_slot_groups(keys, T, layout, device=z_shard.device)
    z_dtype = z_shard.dtype
    kw = dict(triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)

    def unit_rows_fn(z_rows, slot: int, g: tuple):
        g0, g1 = g
        w = te.linear_no_bias_z(te.layernorm_z(z_rows))
        v = w + te.linear_no_bias_a(at_rows(feats, acc, slot, g0, g1, z_dtype))
        return v.unsqueeze(-4)

    def pair_stack_fn(u, mask_loc):
        v = u[0]
        for blk in te.pairformer_stack.blocks:
            v = pair_block(blk, v, layout, **kw)
        return ln_rows_guarded(te.layernorm_v, v, rows_dim=-3, block=BLOCK_ROWS).unsqueeze(0)

    def finish_fn(t):
        u = 0
        for k in range(T):
            u = u + t[k]
        u = u / (1e-7 + T)
        return te.linear_no_bias_u(te.relu(u))

    c_t = int(te.linear_no_bias_z.weight.shape[0]) if hasattr(te.linear_no_bias_z, "weight") else int(te.c)
    record_schedule(templ_bind="protenix_v2", templ_replicated="template_aatype,template 1-D masks,per-token template coordinates,asym_id")
    form = "dense_sliced" if DENSE_KEYS[0] in feats else "row_born"      # the pair features' provenance: row-sliced dense tensors | rows born per block from per-token keys
    say(f"{acc.describe()} real={census.real} form={form} groups={core_template.groups_word(groups)} block_rows={rows}")
    out = core_template.template_embed_rows(z_shard, layout, n_templ=T, c_t=c_t, unit_rows_fn=unit_rows_fn, pair_stack_fn=pair_stack_fn,
                                           finish_fn=finish_fn, mask_loc=None, slot_groups=groups, rows=rows, add=add, log=say)
    return out


def tp_template_embedder(te, input_feature_dict: Mapping[str, Any], z_shard, layout: Layout, *, pair_block: Optional[Callable] = None,
                         pair_mask=None, triangle_attention: str = "torch", triangle_multiplicative: str = "torch", inplace_safe: bool = False,
                         chunk_size: Optional[int] = None, add_to_z: bool = False, real_rows_fn: Optional[Callable] = None, log=None):
    """The carried seam's entry signature, IN-PLACE form only: ``add_to_z=True`` returns ``z_shard`` with the template term added on its rows (0 when
    the embedder is inactive, as stock). Refused by name: ``add_to_z=False`` (the term alone is not a product of the row driver — it yields
    ``z rows + term``; the trunk binding adds in place), a ``pair_mask`` (stock passes none: all ones), a missing ``pair_block``."""
    if not add_to_z:
        raise RowpairRefused("protenix template binding: tp_template_embedder serves the in-place form (add_to_z=True); the term-returning form is not bound")
    if pair_mask is not None:
        raise RowpairRefused("protenix template binding: the stock trunk calls TemplateEmbedder without pair_mask (all ones); a mask is not bound")
    if pair_block is None:
        raise RowpairRefused("protenix template binding: tp_template_embedder needs pair_block= (the line's TP pair block for a c_s=0 PairformerBlock)")
    if not template_embedder_is_active(te, input_feature_dict):
        return 0
    return template_update_(te, input_feature_dict, z_shard, layout, pair_block=pair_block, add=True, real_rows_fn=real_rows_fn, log=log,
                            triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe,
                            chunk_size=chunk_size)

"""OpenFold3's template embedder on row shards: ``TemplateEmbedderAllAtom`` (``latent/template_module.py``) = ``template_pair_embedder``
(``TemplatePairEmbedderAllAtom``, ``feature_embedders/template_embedders.py``: the distogram / unit-vector / restype / mask feature linears
plus ``linear_z(layer_norm_z(z))``), ``template_pair_stack`` (``TemplatePairStack`` of ``TemplatePairBlock`` = a PairBlock per slot, order by
``tri_mul_first``, then its ``layer_norm``), and the close ``sum over slots / n_templ -> relu -> linear_t`` added to z.

``z = add(z, template_embedder(batch, z, pair_mask, ...))`` runs as ``opt_core.mem.rowpair.template.template_embed_rows`` on this rank's rows:
per UNIQUE slot (``template.template_slot_groups``: slots whose features are bitwise identical are embedded once — the four predict-time
dummies collapse to one pass; ``ROWPAIR_TEMPL_NODEDUPE=1`` embeds every slot) the ``v_t`` rows come from ``unit_rows_fn`` below
(OpenFold3's ``_embed_feats`` statement restricted to GLOBAL rows ``g0:g1`` over a ``template.TemplatePairRows`` accessor: row-sliced
dense features; zero rows synthesised per block when the batch carries the lazy placeholders ``[T, 1, 1, F]`` + ``_lazy_template_pair`` and
every slot is a dummy; or, when the lazy batch also carries REAL slots' per-token precursors (``data.REAL_KEYS`` + ``_lazy_template_real``:
a templated chain), rows COMPUTED per block from them by ``template.real_template_rows`` — on the host, the line's placement, bitwise the
dense features sliced to rows),
each slot's slab ``[1, 1, n_loc, N, 64]`` runs through the slot's pair stack with the ONE pair-block driver (``pairstack.pair_stack_rows``),
and the close is added to ``z_loc`` row block by row block. ``ROWPAIR_TEMPL_PARK_Z`` / ``ROWPAIR_TEMPL_PARK_U`` park ``z_loc`` / finished
slabs on the host meanwhile (memory levers; no value changes). REPLICATED BY DESIGN: the per-token template features (``template_restype``,
the two masks, ``asym_id``; a real slot's per-token precursors) — ``templ_replicated=token_feats`` in the census. Every query travels as the
lazy placeholders (``data``): dummy slots as zero rows, a query WITH template hits as the featurizer's per-token precursors, each rank computing
its rows.
"""
from __future__ import annotations


import torch

from . import core as C
from . import pairstack as PS
from .data import LAZY_FLAG, REAL_FLAG, PB_COORDS_KEY, FRAMES_KEY, ASYM_KEY, EDGES_KEY      # the lazy featurizer's flags and REAL-slot precursors (data.featurize_template_structures_lazy)

TEMPLATE_PAIR_KEYS = ("template_distogram", "template_unit_vector")            # [1, T, N, N, 39] / [1, T, N, N, 3] (rows after slicing)

C.register("template", ("template_embed_rows", "TemplatePairRows", "slot_census", "template_slot_groups", "same_chain_rows", "real_template_rows"))


class TemplateRefused(RuntimeError):
    pass


def is_lazy(batch) -> bool:
    """The batch carries the lazy placeholders (``[.., T, 1, 1, F]`` pair keys + ``_lazy_template_pair``) instead of dense ``[T, N, N, F]``."""
    flag = batch.get(LAZY_FLAG)
    if (bool(flag.any()) if torch.is_tensor(flag) else bool(flag)):
        return True
    dg = batch.get("template_distogram")
    return dg is not None and int(dg.shape[-2]) == 1 and int(batch["template_restype"].shape[-2]) != 1


def has_real_precursors(batch) -> bool:
    """A template slot of this lazy batch is real: ``_lazy_template_real`` present AND non-zero (every item carries the flag and the ``data.REAL_KEYS``
    precursors; its value marks a real slot)."""
    flag = batch.get(REAL_FLAG)
    return flag is not None and (bool(flag.any()) if torch.is_tensor(flag) else bool(flag))


def computed_template_rows(batch, lay, n_templ, lead, uv):
    """The core's COMPUTED accessor (``template.real_template_rows``): this rank's rows of ``template_distogram`` / ``template_unit_vector`` per
    row block from the featurizer's precursors — the float64 pseudo-beta coordinates and squared bin edges (int64 bit patterns in the batch, viewed
    back), the finite fp32 backbone frames, the featurizer's per-token chain ids; masks are the batch's own token masks. Nothing ``N x N`` per
    slot is formed. The statements run on the host, the stock featurizer's device class (the line's constant ``ROWPAIR_TEMPL_FEAT_DEVICE=cpu``,
    env.LINE_CONSTANTS): bitwise the dense ``[T, N, N, F]`` features sliced to rows."""
    T = C.seam("template")
    missing = [k for k in (PB_COORDS_KEY, FRAMES_KEY, ASYM_KEY, EDGES_KEY) if k not in batch]
    if missing:
        raise TemplateRefused(f"refused: the batch flags real template slots ({REAL_FLAG}) but lacks their precursors {missing} "
                              "(data.featurize_template_structures_lazy writes all of data.REAL_KEYS)")
    pb = batch[PB_COORDS_KEY]
    if pb.dtype != torch.int64:
        raise TemplateRefused(f"refused: {PB_COORDS_KEY} is {pb.dtype}; expected int64 (float64 coordinates carried as bit patterns)")
    n_bins = int(batch["template_distogram"].shape[-1])
    e = batch[EDGES_KEY]
    if e.dtype != torch.int64 or int(e.shape[-1]) != n_bins or int(e.shape[-2]) != 2 or e.numel() != 2 * n_bins:
        raise TemplateRefused(f"refused: {EDGES_KEY} has shape {tuple(e.shape)} / {e.dtype}; expected [.., 2, n_bins={n_bins}] int64 (one query per batch)")
    edges = e.reshape(2, n_bins).view(torch.float64)
    return T.real_template_rows(lay, n_templ, pb_coords=pb.view(torch.float64), pb_mask=batch["template_pseudo_beta_mask"], frames=batch[FRAMES_KEY],
                                frame_mask=batch["template_backbone_frame_mask"], asym_id=batch[ASYM_KEY], edges=edges,
                                feature_dtype=uv.dtype, out_device=uv.device, lead=lead)


def template_rows(batch, lay):
    """The core's row accessor over this batch's template pair features: ``rows=`` (dense keys sliced to this rank's rows — a view; keys
    already holding ``R`` rows are taken as sliced), ``lazy_dummy=`` (zero rows per block; refused by name if a slot is real), or the COMPUTED
    accessor when the lazy batch carries real slots' precursors (``computed_template_rows``)."""
    T = C.seam("template")
    n_templ = int(batch["template_restype"].shape[-3])
    lead = tuple(int(v) for v in batch["template_restype"].shape[:-3])
    uv = batch["template_unit_vector"]
    if is_lazy(batch):
        if has_real_precursors(batch):
            return computed_template_rows(batch, lay, n_templ, lead, uv)
        census = T.slot_census(batch["template_pseudo_beta_mask"], batch["template_backbone_frame_mask"], slot_dim=-2)
        dims = {"template_distogram": int(batch["template_distogram"].shape[-1]), "template_unit_vector": int(uv.shape[-1])}
        return T.TemplatePairRows(lay, n_templ, lazy_dummy=census, feature_dims=dims, lead=lead, dtype=uv.dtype, device=uv.device)
    rows = {}
    for k in TEMPLATE_PAIR_KEYS:
        t = batch[k]
        if int(t.shape[-3]) == lay.N and int(t.shape[-2]) == lay.N:
            rows[k] = C.fn("shard", "shard_rows")(t, lay, dim=-3)
        elif int(t.shape[-3]) == lay.R and int(t.shape[-2]) == lay.N:
            rows[k] = t
        else:
            raise TemplateRefused(f"refused: {k} has shape {tuple(t.shape)}; expected dense [.., T, N={lay.N}, N, F], this rank's rows "
                                  f"[.., T, R={lay.R}, N, F], or the lazy placeholders [.., T, 1, 1, F] with {LAZY_FLAG} (the tp line's lazy template featurizer, data.py)")
    return T.TemplatePairRows(lay, n_templ, rows=rows, lead=lead)


def embed_feats_rows(tpe, batch, acc, slot: int, g0: int, g1: int, lay):
    """``TemplatePairEmbedderAllAtom._embed_feats`` for ONE slot and GLOBAL rows ``g0:g1``: ``[.., 1, g1-g0, N, c_t]`` (the stock term order)."""
    same = C.fn("template", "same_chain_rows")(batch["asym_id"], g0, g1)                        # bool [.., 1, rows, N, 1]: asym_i == asym_j
    dtype = batch["template_unit_vector"].dtype
    i0, i1 = g0 - lay.r0, g1 - lay.r0
    pb = batch["template_pseudo_beta_mask"][..., [slot], :]
    bb = batch["template_backbone_frame_mask"][..., [slot], :]
    pseudo_beta_pair_mask = (pb[..., g0:g1, None] * pb[..., None, :])[..., None] * same
    backbone_frame_pair_mask = (bb[..., g0:g1, None] * bb[..., None, :])[..., None] * same
    template_distogram = acc.rows("template_distogram", [slot], i0, i1)
    x, y, z = acc.rows("template_unit_vector", [slot], i0, i1).unbind(dim=-1)
    rt = batch["template_restype"][..., [slot], :, :]
    n_token = int(rt.shape[-2])
    template_restype_ti = rt[..., g0:g1, None, :].expand(*rt.shape[:-2], g1 - g0, n_token, -1)
    template_restype_tj = rt[..., None, :, :].expand(*rt.shape[:-2], g1 - g0, n_token, -1)
    a = tpe.dgram_linear(template_distogram)
    a = a + tpe.pseudo_beta_mask_linear(pseudo_beta_pair_mask)
    a = a + tpe.aatype_linear_1(template_restype_ti.to(dtype=dtype))
    a = a + tpe.aatype_linear_2(template_restype_tj.to(dtype=dtype))
    a = a + tpe.x_linear(x[..., None])
    a = a + tpe.y_linear(y[..., None])
    a = a + tpe.z_linear(z[..., None])
    a = a + tpe.backbone_mask_linear(backbone_frame_pair_mask)
    return a


def slot_groups(batch, acc, lay, device):
    """``template.template_slot_groups`` over every template feature with the slot moved to dim 0 (per-token keys replicated, pair keys this
    rank's rows — the verdict is agreed across ranks, so it is the dense verdict)."""
    n_templ = int(batch["template_restype"].shape[-3])
    keys = [batch["template_restype"].movedim(-3, 0), batch["template_pseudo_beta_mask"].movedim(-2, 0), batch["template_backbone_frame_mask"].movedim(-2, 0)]
    if acc.mode == "rows":
        keys += [acc.rows(k, list(range(n_templ)), 0, lay.R).movedim(-4, 0) for k in acc.keys()]
    elif acc.mode == "computed":                                                                    # computed rows are a function of these per-slot precursors (+ the slot-free chain ids / edges)
        keys += [batch[PB_COORDS_KEY].movedim(-3, 0), batch[FRAMES_KEY].movedim(-4, 0)]
    return C.fn("template", "template_slot_groups")(keys, n_templ, lay, device=device)


def template_embedder_add_rows_(te, batch, z_loc, pair_mask_loc, lay, census_tag=None, chunk_size=None, _mask_trans=True, inplace_safe=True, **flags):
    """``z_loc += rows r0:r1 of TemplateEmbedderAllAtom(batch, z, pair_mask, ...)`` IN PLACE, row block by row block; returns ``z_loc``
    (``[1, n_loc, N, C_z]``). ``pair_mask_loc``: ``[1, n_loc, N]``."""
    PS.refuse_kernel_flags("TemplateEmbedder", **flags)
    T = C.seam("template")
    tpe, stack = te.template_pair_embedder, te.template_pair_stack
    if getattr(stack, "chunk_size_tuner", None) is not None:
        raise TemplateRefused("refused: TemplatePairStack.tune_chunk_size=true under the tp line (runner yaml must pin tune_chunk_size=false)")
    if getattr(stack, "training", False) or torch.is_grad_enabled():
        raise TemplateRefused("refused: the template embedder on row shards is the inference path (eval mode, no grad)")
    if not inplace_safe:
        raise TemplateRefused("refused: the template embedder on shards implements inplace_safe=True (the `run_openfold predict` path)")
    comm = C.comm()
    log = comm.log if comm.verbose else None
    n_templ = int(batch["template_restype"].shape[-3])
    c_t = int(tpe.linear_z.out_features) if hasattr(tpe.linear_z, "out_features") else int(tpe.linear_z.weight.shape[0])
    acc = template_rows(batch, lay)
    if log is not None:
        log(acc.describe() + (f" {census_tag}" if census_tag else ""))
    groups = slot_groups(batch, acc, lay, device=z_loc.device)
    mask_t = pair_mask_loc.to(dtype=z_loc.dtype)                                                    # [1, n_loc, N]: one slot's pair-mask rows (stock casts to z.dtype)

    def unit_rows_fn(z_rows, slot, rows):                                                           # v_t rows: linear_z(LN(z rows)) + embedded features
        g0, g1 = rows
        zr = tpe.linear_z(tpe.layer_norm_z(z_rows))
        return zr[..., None, :, :, :] + embed_feats_rows(tpe, batch, acc, int(slot), int(g0), int(g1), lay)

    def pair_stack_fn(u, _mask_loc):                                                                # TemplatePairStack on one slot's slab [1, 1, n_loc, N, c_t] + its LayerNorm
        u4 = PS.pair_stack_rows(stack.blocks, u[..., 0, :, :, :], mask_t, lay, chunk_size=chunk_size, inplace_safe=inplace_safe, _mask_trans=_mask_trans, stack=stack)   # template_module.py:425-487 (clear_cache_between_blocks honoured)
        return C.fn("shard", "ln_rows_guarded")(stack.layer_norm, u4, rows_dim=-3)[..., None, :, :, :]   # the whole slot slab [1, n_loc, N, c_t]: row blocks from ROWPAIR_LN_GUARD_ELEMS elements (>= 2^31 from N = 16,384 at P = 8)

    def finish_fn(t):                                                                               # template_module.py close: mean over slots, relu, linear_t
        t = torch.sum(t, dim=-4) / n_templ
        t = torch.nn.functional.relu(t)
        return te.linear_t(t)

    budget = {"feat_channels": acc.transient_channels} if acc.mode == "computed" else {}                      # computed rows carry their featurizer transients inside the row block (the accessor declares the width); rows / lazy_dummy: the core's default
    return T.template_embed_rows(z_loc, lay, n_templ=n_templ, c_t=c_t, unit_rows_fn=unit_rows_fn, pair_stack_fn=pair_stack_fn, finish_fn=finish_fn,
                                 mask_loc=pair_mask_loc, slot_groups=groups, add=True, log=log, **budget)

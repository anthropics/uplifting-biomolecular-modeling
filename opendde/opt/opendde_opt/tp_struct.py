"""The STRUCTURAL-TOKEN STAGE of the ``big`` line under ``--n_gpu P > 1`` on ROW SHARDS: the structural-token expansion and the
structural refiner of OpenDDE 1.1.1 (``OpenDDE.expand_to_structural_tokens``, opendde.py:422-538) run on this rank's ROWS of the structural
pair space, fed from this rank's rows of the residue pair track — no whole ``[N, N, c_z]`` / ``[N_st, N_st, c_z]`` on any device, none on
the host, no memfrac device buffer. Per-rank resident: the residue shard ``u = N^2 c / P`` (the caller's :class:`opendde_opt.tp.RowShard`),
the structural shard ``u_st = N_st^2 c / P``, and — while the expansion runs — the ``K <= R_st`` full residue rows it reads (``K x N x c``).

Structural tokens: ``N_st`` rows (``N_st ~ 1.3-2 N``); structural row ``i`` has the residue parent ``parent(i) = parent_residue_idx[i]``.
This rank owns structural rows ``[r0_st, r1_st)`` of :func:`opendde_opt.tp_diffusion.struct_layout` (a layout of its own, independent of
the residue trunk's ``lay_res``).

Statements (each is the ENGINE's, on rows; per-element arithmetic identical to the dense statement, fp32):

  * residue rows for the expansion (:func:`struct_rows_needed`, :func:`fetch_rows_ring`): the expansion of structural rows ``[r0_st, r1_st)``
    reads the FULL residue rows ``{parent(i)}`` (all ``N`` columns) = ``K`` sorted unique rows, fetched in fixed row slabs around the ring of
    residue shards (``opt_core.mem.rowpair.dist.ring_blocks``; slab height = ``ROWPAIR_HOSTGATHER_ROWS`` / layout ``B`` (``opendde_opt.tp._hostgather_rows``), so
    every rank runs the same number of exchanges); each rank keeps only its ``K`` rows ``[K, N, c_z]``.
  * expansion (:func:`expand_rows`, :func:`expand_single`) == ``StructuralTokenExpander.forward`` (structural_tokens.py:842-883; the row-chunked
    statement ``_make_structural_pair_activations_chunked``, :726-796, which the pin's ``pair_chunk_size=128`` selects) restricted to this
    rank's structural rows, in row blocks of the expander's own ``pair_chunk_size``: per block ``pair features of the rows``
    (``_build_structural_pair_features_for_rows``, every entry ``[rows, N_st]``), ``z_res[parent(rows)][:, parent]`` (+ the role-pair
    projection ``_pair_project_by_role(row_index=global rows)`` + ``_make_pair_init_bias``) -> ``z_struct`` rows born as ``[R_st, N_st, c_z]``;
    the structural extra attention bias (``structural_pair_attn_bias``: ``_make_attention_bias``) born as ROWS ``[R_st, N_st]``. The singles
    ``s_inputs_struct`` / ``s_struct`` are replicated ``O(N_st)`` statements. The expander consumes no relative-position features: the
    structural relp is generated LAZY after it (``generate_relp(lazy=True)``: ``O(N_st)`` index vectors) and only ever materialised in rows
    by the diffusion conditioning (:mod:`opendde_opt.tp_diffusion`).
  * refiner (:func:`refine_rows`) == the pin's ``structural_token_refiner`` (``PairformerStack``: 4 blocks, WITH the single track) on the
    structural row shard through the core's ONE pair-stack driver (``opt_core.mem.rowpair.pairstack.pair_stack_``) with the SAME per-block
    binding the trunk uses (:func:`opendde_opt.tp._pair_fns`: tri-mult out = ring of projected row blocks, tri-mult in / tri-att end =
    all-to-all transposes, tri-att start = the small bias all-gather, transitions row-local, attention-pair-bias on LOCAL query rows +
    all-gather with the extra bias ROWS, single transition replicated). ``z_struct`` rows updated IN PLACE; the refined ``s_struct`` is
    what the diffusion conditioning consumes.
  * the stage entry (:func:`expand_to_structural_tokens_rows`) == ``OpenDDE.expand_to_structural_tokens`` fed a residue
    :class:`opendde_opt.tp.RowShard`: the structural feature dict (verbatim statements), the singles, ``z_struct`` as a ``RowShard`` of the
    structural layout, the extra bias as rows (marker :data:`opendde_opt.tp_diffusion.EXTRA_BIAS_ROWS_KEY`) — the form
    :func:`opendde_opt.tp_diffusion.prime` consumes.

Replicated by design (named): ``s_inputs_struct [N_st, c_s_inputs]``, ``s_struct [N_st, c_s]``, the structural per-token features and the
pair context vectors ``[N_st]``. Refusals by name (``RowpairRefused``): a ``P == 1`` / replicated layout (at ``--n_gpu 1`` the adapter
installs nothing; the engine's own stage runs), a ``z_res`` that is not a ``RowShard`` of ``lay_res``, a ``parent_residue_idx`` outside
``[0, N)`` or of the wrong length, residue rows requested outside ``[0, N)``, structural-token expansion disabled, a Fold-CP mesh.

Census (``opendde_opt.tp.STATS``): ``struct_rows`` (``R_st``), ``zres_ring_rows`` (``K``), ``struct_presharded_calls`` (+1 per refiner call
on rows), ``struct_stage='rowshard'``; the core's schedule record carries ``struct_expand_rows`` / ``struct_expand_stmt`` / ``struct_ring_slabs`` /
``struct_ring_slab_rows`` / ``struct_refiner_blocks``;
one ``STRUCT`` line per stage entry with the device peak bytes before / after.
"""
from __future__ import annotations

import sys
import time
from typing import Iterator, Tuple

__all__ = ["struct_rows_needed", "fetch_rows_ring", "expand_single", "expand_rows", "refine_rows", "expand_to_structural_tokens_rows",
           "REQUIRED_FEATURES", "REPLICATED_BY_DESIGN"]

REQUIRED_FEATURES = ("parent_residue_idx", "subtoken_role_id", "structural_token_index", "atom_to_structural_token_idx",   # opendde.py:435-445
                     "atom_to_structural_tokatom_idx", "structural_distogram_rep_atom_mask", "structural_pae_rep_atom_mask",
                     "structural_has_frame", "structural_frame_atom_index")
RESIDUE_LEVEL_FEATURES = ("token_index", "asym_id", "residue_index", "entity_id", "sym_id", "atom_to_token_idx", "atom_to_tokatom_idx",   # opendde.py:456-471
                          "has_frame", "frame_atom_index", "pae_rep_atom_mask", "distogram_rep_atom_mask")
REPLICATED_BY_DESIGN = "s_inputs_struct[N_st,c_s_inputs],s_struct[N_st,c_s],structural_token_features[N_st],pair_context[N_st]"


# ----------------------------------------------------------------------------------------------------------------- plumbing
def _core():
    from opt_core.mem.rowpair import dist as D, RowpairRefused
    return D, RowpairRefused


def _refuse(msg: str):
    from opt_core.mem.rowpair import RowpairRefused
    from . import tp as _tp
    raise RowpairRefused(f"{_tp.LEVER}: refused: structural stage: {msg}")


def _require_sharded(lay, what: str):
    """The structural n_gpu=1 rule: a ``P == 1`` / replicated / no-group layout is refused by name (the core's sentence)."""
    D, _ = _core()
    from . import tp as _tp
    return D.require_sharded(lay, f"{_tp.LEVER}: structural stage: {what}", lever=_tp.LEVER)


def _dev_peak_bytes(device) -> int:
    """The allocator's peak on ``device`` (CUDA), 0 on CPU."""
    import torch
    if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def _record(**facts) -> None:
    """Schedule facts of this stage in the core's schedule record (``opendde_opt.tp`` prints it as ``core_schedule``)."""
    try:
        from opt_core.mem.rowpair import evidence as _ev
    except ImportError:                                                             # pragma: no cover — instrumentation only
        return
    _ev.record_schedule(**facts)


# ----------------------------------------------------------------------------------------------------------------- residue rows
def struct_rows_needed(parent_idx, lay_st):
    """``LongTensor[K]``: the sorted unique RESIDUE rows the structural rows ``[lay_st.r0, lay_st.r1)`` of this rank read
    (``unique(parent_residue_idx[r0:r1])``). ``parent_idx``: ``[N_st]`` (leading size-1 dims allowed)."""
    import torch
    parent = parent_idx
    while parent.dim() > 1 and int(parent.shape[0]) == 1:
        parent = parent[0]
    if parent.dim() != 1 or int(parent.shape[0]) != int(lay_st.N):
        _refuse(f"parent_residue_idx {tuple(parent_idx.shape)} vs the structural layout's N_st={lay_st.N} ([N_st] expected)")
    rows = parent[int(lay_st.r0):int(lay_st.r1)].long()
    if rows.numel() and int(rows.min()) < 0:
        _refuse(f"parent_residue_idx has a negative entry ({int(rows.min())}) in rows [{lay_st.r0}, {lay_st.r1})")
    return torch.unique(rows, sorted=True)


def _ring_rows(lay_res) -> int:
    """Row-slab height of the residue ring: the FIXED slab of ``opendde_opt.tp._hostgather_rows`` (``ROWPAIR_HOSTGATHER_ROWS`` or the
    layout's ``B``) — identical on every rank, so every rank runs the same number of ring exchanges."""
    from . import tp as _tp
    return int(_tp._hostgather_rows(lay_res))


def _ring_row_slabs(z_shard, lay) -> Iterator[Tuple[int, object]]:
    """``(g0, block)`` for every rank's rows in FIXED row slabs around the ring (``dist.ring_blocks``; slab height :func:`_ring_rows`): ``block``
    holds global rows ``[g0, g0 + len(block))`` of the row-sharded tensor whose local rows are ``z_shard [R, N, ...]``; this rank's own slabs
    are yielded too. Fixed-slab protocol: ``ceil(Rmax / slab)``
    passes on every rank, whatever a rank does with the blocks. COLLECTIVE; consume to exhaustion."""
    D, _ = _core()
    RB, R = _ring_rows(lay), int(lay.R)
    for j0 in range(0, int(lay.Rmax), RB):
        own = z_shard[j0:min(j0 + RB, R)] if j0 < R else z_shard[:0]
        for q, x_q in D.ring_blocks(own.contiguous(), lay, rows=(j0, RB)):          # (q, rank q's rows [q0+j0, q0+j0+n)) for every q
            yield int(lay.rows(q).start) + j0, x_q


def fetch_rows_ring(z_res_shard, rows_needed, lay_res) -> Tuple[object, object]:
    """``(rows [K, N, C], index [K])``: the residue pair rows ``index = sorted unique(rows_needed)`` of the WHOLE residue track, on this rank,
    with ``rows[k] == z_res[index[k]]`` bit for bit. Every rank's shard travels once around the ring in fixed row slabs (:func:`_ring_row_slabs`),
    every rank keeps only the rows it asked for (``K <= R_st`` rows for the structural expansion); the full ``z_res`` never exists anywhere.
    COLLECTIVE: every rank must call it — the same number of slabs runs on every rank whatever it asks for (an empty request is fine), and the
    delivery accounting is a collective decision: one object all-gather of ``(K, delivered, min, max)`` after the ring, so a request outside
    ``[0, N)`` on ANY rank is refused by name on EVERY rank (no rank is left waiting in a later collective). ``z_res_shard``: this rank's rows
    ``[R, N, C]`` of ``lay_res``."""
    import bisect
    import torch
    D, _ = _core()
    _require_sharded(lay_res, "fetch_rows_ring")
    if z_res_shard.dim() != 3 or int(z_res_shard.shape[0]) != int(lay_res.R) or int(z_res_shard.shape[1]) != int(lay_res.N):
        _refuse(f"fetch_rows_ring: z_res shard {tuple(z_res_shard.shape)} vs layout R={lay_res.R} N={lay_res.N} ([R, N, C] expected)")
    idx = torch.unique(rows_needed.reshape(-1).long().to(device=z_res_shard.device), sorted=True)
    want = [int(v) for v in idx.tolist()]                                            # K host ints (K <= R_st): slab windows by bisect, no device sync in the ring loop
    K = len(want)
    N, C = int(lay_res.N), int(z_res_shard.shape[2])
    out = z_res_shard.new_empty((K, N, C))
    got, slabs = 0, len(range(0, int(lay_res.Rmax), _ring_rows(lay_res)))          # passes around the ring (the generator's fixed count)
    for g0, x_q in _ring_row_slabs(z_res_shard, lay_res):
        n = int(x_q.shape[0])
        if not n or not K:
            continue
        lo, hi = bisect.bisect_left(want, g0), bisect.bisect_left(want, g0 + n)
        if hi > lo:
            out[lo:hi] = x_q.index_select(0, idx[lo:hi] - g0)                      # exact copies of the requested rows of this slab
            got += hi - lo
    mine = (K, got, want[0] if K else None, want[-1] if K else None)
    ranks = D.comm().allgather_obj(mine)                                             # ONE tiny collective: the accounting verdict is the same on every rank
    bad = [f"rank{q}: delivered {g} of K={k} (min={mn} max={mx})" for q, (k, g, mn, mx) in enumerate(ranks) if g != k or (k and (mn < 0 or mx >= N))]
    if bad:                                                                          # total accounting: every requested row on every rank delivered exactly once, all inside [0, N)
        _refuse(f"fetch_rows_ring: residue rows requested outside [0, N={N}) or not delivered — " + "; ".join(bad))
    from . import tp as _tp
    _tp.STATS["zres_ring_rows"] = K
    _record(struct_ring_slabs=slabs, struct_ring_slab_rows=_ring_rows(lay_res))
    return out, idx


# ----------------------------------------------------------------------------------------------------------------- expansion
def _parent_role(input_feature_dict):
    """``(parent [N_st] long, role [N_st] long)`` as ``StructuralTokenExpander.forward`` reads them (structural_tokens.py:849-850)."""
    return input_feature_dict["parent_residue_idx"].long(), input_feature_dict["subtoken_role_id"].long()


def expand_single(expander, input_feature_dict, s_inputs, s):
    """``(s_inputs_struct [.., N_st, c_s_inputs], s_struct [.., N_st, c_s])`` == the single-track statements of ``StructuralTokenExpander.forward``
    (structural_tokens.py:851-859): replicated ``O(N_st)`` work, identical on every rank."""
    parent, role = _parent_role(input_feature_dict)
    s_inputs_struct = expander._gather_parent_single(s_inputs, parent) + expander.single_input_role_embedding(role).to(dtype=s_inputs.dtype)
    s_parent = expander._gather_parent_single(s, parent)
    s_struct = s_parent + expander.single_split_mlp(s_parent) + expander.single_role_embedding(role).to(dtype=s_parent.dtype)
    return s_inputs_struct, s_struct


def _expand_block_rows(expander, lay_st) -> int:
    """Row-block height of the expansion on this rank: the expander's own ``pair_chunk_size`` (the pin: 128), else all local rows — the
    engine's knob (``min(pair_chunk_size or n, n)`` in the chunked statement), bounding the per-block transient ``rows x N_st x c_z``."""
    v = getattr(expander, "pair_chunk_size", None)
    R = max(1, int(lay_st.R))
    return max(1, min(int(v), R)) if v else R


def finite_rows(t, rows: int) -> bool:
    """``bool(torch.isfinite(t).all())`` computed per block of ``rows`` along dim 0 (a ragged last block included): the same verdict without a
    ``t``-sized bool temporary (the whole-tensor form allocates one byte per element — 10 GiB for a 7,824-token ×8 structural shard)."""
    import torch
    rows = max(1, int(rows))
    n = int(t.shape[0]) if t.dim() else 1
    if t.dim() == 0:
        return bool(torch.isfinite(t))
    for i0 in range(0, n, rows):
        if not bool(torch.isfinite(t[i0:i0 + rows]).all()):
            return False
    return True


def expand_rows(expander, input_feature_dict, s_inputs, s, zres_rows, zres_index, lay_st, *, rows: int, attn_bias_out=None, out_dtype=None):
    """``z_struct`` rows ``[R_st, N_st, c_z]`` of this rank == ``StructuralTokenExpander._make_structural_pair_activations_chunked``
    (structural_tokens.py:726-796) restricted to structural rows ``[lay_st.r0, lay_st.r1)``, processed in ``rows``-row blocks (GLOBAL
    ``row_index`` handed to the engine's own per-row statements, so the per-element arithmetic is the dense statement's): per block the pair
    features of the rows, ``z_res[parent(rows)][:, parent]`` read from the fetched residue rows (``zres_rows [K, N, c_z]`` holding residue
    rows ``zres_index [K]``, :func:`fetch_rows_ring`), the role-pair projection delta, the pair init bias. ``attn_bias_out`` (``[R_st, N_st]``,
    optional): the structural extra attention bias rows (``_make_attention_bias`` of the same pair features) written in place. ``s_inputs`` /
    ``s`` are unused (parity with ``StructuralTokenExpander.forward``; the singles are :func:`expand_single`, the pair rows do not read them).
    Row-local: no communication — so its contract refusals (shapes, a parent missing from ``zres_index``) are RANK-LOCAL: callers validate the
    replicated inputs on every rank before the first collective (the stage entry does)."""
    import torch
    _require_sharded(lay_st, "expand_rows")
    parent, role = _parent_role(input_feature_dict)
    n_struct = int(parent.shape[-1])
    if n_struct != int(lay_st.N):
        _refuse(f"expand_rows: parent_residue_idx has {n_struct} structural tokens vs the layout's N_st={lay_st.N}")
    if n_struct < 1:
        raise ValueError(f"Structural pair context requires at least one structural token; got {n_struct}.")   # the engine's sentence (:740-744)
    if zres_rows.dim() != 3 or int(zres_rows.shape[0]) != int(zres_index.numel()):
        _refuse(f"expand_rows: zres_rows {tuple(zres_rows.shape)} vs zres_index [{int(zres_index.numel())}] ([K, N, c_z] + [K] expected)")
    r0, R = int(lay_st.r0), int(lay_st.R)
    row_parent = parent[r0:r0 + R]
    n_res = int(zres_rows.shape[1])
    if R and (int(row_parent.min()) < 0 or int(parent.max()) >= n_res):
        _refuse(f"expand_rows: parent_residue_idx outside [0, N={n_res}) (min={int(parent.min())} max={int(parent.max())})")
    zres_index = zres_index.to(device=parent.device, dtype=torch.long)
    pos = torch.searchsorted(zres_index, row_parent) if R else row_parent           # position of each local row's parent among the fetched rows
    if R and (int(pos.max()) >= int(zres_index.numel()) or not bool((zres_index[pos.clamp(max=max(0, int(zres_index.numel()) - 1))] == row_parent).all())):
        _refuse("expand_rows: a structural row's parent residue row is not among the fetched residue rows (zres_index) — struct_rows_needed / fetch_rows_ring contract violated")
    if attn_bias_out is not None and (tuple(attn_bias_out.shape) != (R, n_struct)):
        _refuse(f"expand_rows: attn_bias_out {tuple(attn_bias_out.shape)} vs (R_st, N_st)=({R}, {n_struct})")
    context = expander._build_structural_pair_context(input_feature_dict=input_feature_dict, role=role, parent=parent)   # [N_st] vectors
    out = zres_rows.new_empty((R, n_struct, int(zres_rows.shape[2])), dtype=out_dtype or zres_rows.dtype)   # the shard is BORN in the storage dtype (struct_pair_bf16:
    # bf16): each fp32 row block is cast on the copy below — the same values as casting the finished fp32 shard, without the fp32 [R_st, N_st, c_z] transient
    rows = max(1, int(rows))
    for i0 in range(0, R, rows):
        i1 = min(R, i0 + rows)
        row_index = torch.arange(r0 + i0, r0 + i1, device=parent.device)
        pair_features = expander._build_structural_pair_features_for_rows(context=context, row_index=row_index)   # every entry [rows, N_st]
        z_chunk = zres_rows.index_select(-3, pos[i0:i1]).index_select(-2, parent)   # == _gather_parent_pair_rows(z_res, parent, row_index) with z_res rows held as (zres_rows, zres_index)
        delta = expander._pair_project_by_role(z=z_chunk, role=role, pair_features=pair_features, row_index=row_index)
        if delta is not None:
            z_chunk = z_chunk + delta
        z_chunk = z_chunk + expander._make_pair_init_bias(pair_features, dtype=z_chunk.dtype)
        out[i0:i1].copy_(z_chunk)
        if attn_bias_out is not None:
            attn_bias_out[i0:i1].copy_(expander._make_attention_bias(pair_features, dtype=z_chunk.dtype))
        del z_chunk, delta, pair_features, row_index
    _record(struct_expand_rows=rows, struct_expand_stmt="chunked_rows")                # the pin's row-chunked statement on this rank's rows (its returned feature set: the extra bias only)
    return out


# ----------------------------------------------------------------------------------------------------------------- refiner
def refine_rows(refiner, z_st_shard, lay_st, *, s, extra_attn_bias, pair_mask_rows=None, triangle_attention: str, chunk_size=None):
    """``(s, z_st_shard)`` after the structural refiner (``PairformerStack``) on this rank's structural rows: the core's
    ``pairstack.pair_stack_`` with every block bound by :func:`opendde_opt.tp._pair_fns` (the trunk's binding: same drivers, same
    communication pattern; the triangle attention runs the module's own ``mha`` — the trunk's row-block kernel lever ``tp_triatt`` does
    not cover the refiner), ``z_st_shard [R_st, N_st, c_z]`` updated IN PLACE (the returned shard is the same tensor object when it arrived
    contiguous), ``s [.., N_st, c_s]`` replicated through the blocks' attention-pair-bias on LOCAL query rows + all-gather. ``extra_attn_bias``:
    this rank's ROWS ``[.., R_st, N_st]`` of the structural extra attention bias (or the whole ``[.., N_st, N_st]``; ``opendde_opt.tp`` slices
    what it is handed). ``pair_mask_rows``: this rank's rows ``[R_st, N_st]`` of a pair mask, or None (the pin's refiner call carries none).
    ``chunk_size``: the loop's structural refiner chunk (rows per triangle-attention / transition batch; None -> ``ROWPAIR_ATTN_ROWS``)."""
    import torch
    from opt_core.mem.rowpair import pairstack as _ps
    from . import tp as _tp
    _require_sharded(lay_st, "refine_rows")
    if z_st_shard.dim() != 3 or int(z_st_shard.shape[0]) != int(lay_st.R) or int(z_st_shard.shape[1]) != int(lay_st.N):
        _refuse(f"refine_rows: z_struct shard {tuple(z_st_shard.shape)} vs layout R_st={lay_st.R} N_st={lay_st.N} ([R_st, N_st, c_z] expected)")
    blocks = list(refiner.blocks)
    single = bool(blocks) and getattr(blocks[0], "c_s", 0) > 0
    s2, ns = (None, 0) if s is None else _tp._lead(s, keep=2)
    if single and s2 is None:
        _refuse("refine_rows: the refiner's blocks carry the single track (c_s > 0) but s is None")
    if s2 is not None and (s2.dim() != 2 or int(s2.shape[0]) != int(lay_st.N)):
        _refuse(f"refine_rows: s {tuple(s.shape)} vs N_st={lay_st.N} ([N_st, c_s] expected)")
    if pair_mask_rows is not None and tuple(pair_mask_rows.shape) != (int(lay_st.R), int(lay_st.N)):
        _refuse(f"refine_rows: pair_mask_rows {tuple(pair_mask_rows.shape)} vs (R_st, N_st)=({lay_st.R}, {lay_st.N})")
    if extra_attn_bias is not None and (int(extra_attn_bias.shape[-1]) != int(lay_st.N) or int(extra_attn_bias.shape[-2]) not in (int(lay_st.R), int(lay_st.N))):
        _refuse(f"refine_rows: extra_attn_bias {tuple(extra_attn_bias.shape)} is neither this rank's rows [.., R_st={lay_st.R}, N_st] nor whole [.., N_st={lay_st.N}, N_st]")
    if z_st_shard.is_cuda:
        torch.cuda.synchronize(z_st_shard.device)
    t0 = time.perf_counter()
    if torch.are_deterministic_algorithms_enabled() and s2 is not None:              # det-scoped, as the trunk's stack entry: the replicated s agrees bitwise across ranks or the run dies by name
        _tp._replicated(s2, f"s@struct_refiner[{len(blocks)}x{lay_st.N}]")
    z_sh = z_st_shard if z_st_shard.is_contiguous() else z_st_shard.contiguous()
    mask_sh = None if pair_mask_rows is None else pair_mask_rows.to(dtype=z_sh.dtype).contiguous()
    if s2 is not None and not s2.is_contiguous():
        s2 = s2.contiguous()
    fns = [_tp._pair_fns(blk, lay_st, triangle_attention=triangle_attention, chunk_size=chunk_size, extra_attn_bias=extra_attn_bias, trimul_rb=_tp._struct_trimul_rb(), triatt_kernel=False)
           for blk in blocks]                                                        # the refiner streams its tri-mult b slabs at ODDE_TP_STRUCT_TRIMUL_RB rows (P-invariant; N_st-sized slabs are the OOM site otherwise)
    z_sh, s2 = _ps.pair_stack_(fns, z_sh, mask_sh, lay_st, s=s2, transition_mask=False)   # opendde's pair transition takes no mask (as the trunk's call)
    if z_sh.is_cuda:
        torch.cuda.synchronize(z_sh.device)
    _tp.STATS["struct_presharded_calls"] = int(_tp.STATS.get("struct_presharded_calls") or 0) + 1
    _record(struct_refiner_blocks=len(blocks), struct_refiner_s=round(time.perf_counter() - t0, 3))
    return (None if s is None else _tp._relead(s2, ns)), z_sh


# ----------------------------------------------------------------------------------------------------------------- the stage entry
def _structural_feature_dict_pre(input_feature_dict) -> dict:
    """verbatim: ``OpenDDE.expand_to_structural_tokens`` (opendde.py:455-471) — the part BEFORE the expander call."""
    structural_feature_dict = dict(input_feature_dict)
    for residue_feature in RESIDUE_LEVEL_FEATURES:
        structural_feature_dict[f"residue_level_{residue_feature}"] = input_feature_dict[residue_feature]
    return structural_feature_dict


def _structural_feature_dict_post(input_feature_dict, structural_feature_dict, parent, structural_pair_features) -> dict:
    """verbatim: ``OpenDDE.expand_to_structural_tokens`` (opendde.py:492-519) — the part AFTER the expander call (relp generated by the caller)."""
    structural_feature_dict["token_index"] = input_feature_dict["structural_token_index"].long()
    structural_feature_dict["atom_to_token_idx"] = input_feature_dict["atom_to_structural_token_idx"].long()
    structural_feature_dict["atom_to_tokatom_idx"] = input_feature_dict["atom_to_structural_tokatom_idx"].long()
    for token_feature in ["asym_id", "residue_index", "entity_id", "sym_id"]:
        structural_feature_dict[token_feature] = input_feature_dict[token_feature].index_select(dim=-1, index=parent)
    structural_feature_dict["has_frame"] = input_feature_dict["structural_has_frame"]
    structural_feature_dict["frame_atom_index"] = input_feature_dict["structural_frame_atom_index"]
    structural_feature_dict["pae_rep_atom_mask"] = input_feature_dict["structural_pae_rep_atom_mask"].long()
    structural_feature_dict["distogram_rep_atom_mask"] = input_feature_dict["structural_distogram_rep_atom_mask"].long()
    for feature_name, feature_value in structural_pair_features.items():
        structural_feature_dict[feature_name] = feature_value
    return structural_feature_dict


def expand_to_structural_tokens_rows(model, input_feature_dict: dict, s_inputs, s, z_res_shard, *, chunk_size=None, inplace_safe: bool = False, lazy_relp: bool = True):
    """``(structural_feature_dict, s_inputs_struct, s_struct, z_struct: RowShard)`` == ``OpenDDE.expand_to_structural_tokens`` (opendde.py:422-538,
    the loop's ``lazy_relp=True`` call) under ``n_gpu > 1`` with the residue pair track arriving as this rank's :class:`opendde_opt.tp.RowShard`:
    the expansion and the refiner run on this rank's structural rows (:func:`expand_rows`, :func:`refine_rows`); ``z_struct`` leaves as a
    ``RowShard`` of :func:`opendde_opt.tp_diffusion.struct_layout` (never whole on a device, never on the host) and the structural extra
    attention bias as this rank's ROWS ``[R_st, N_st]`` in the feature dict (marker ``EXTRA_BIAS_ROWS_KEY``), the form
    :func:`opendde_opt.tp_diffusion.prime` consumes. ``chunk_size``: the loop's structural refiner chunk (opendde.py:1887-1896); ``inplace_safe``:
    parity with the pin's signature (the row statements never alias their input); ``lazy_relp``: the loop's ``True`` (a dense ``[N_st, N_st, 139]``
    relp one-hot is refused by name, never built). COLLECTIVE: every rank calls it at the same point; every refusal of its inputs fires on the
    replicated data BEFORE the first collective, so all ranks refuse alike. One ``STRUCT`` census line per call (device peak bytes before / after)."""
    import torch
    from . import tp as _tp, tp_diffusion as _td
    if not isinstance(z_res_shard, _tp.RowShard):
        _refuse(f"expand_to_structural_tokens_rows: z is {type(z_res_shard).__name__}, not the trunk's RowShard (this stage runs on the residue ROW SHARD under n_gpu>1)")
    if not lazy_relp:
        _refuse("expand_to_structural_tokens_rows: lazy_relp=False — the structural relp on rows is generated lazy (the dense [N_st, N_st, 139] one-hot is never built on a rank)")
    if not getattr(model, "enable_structural_token_expansion", False):
        _refuse("structural token expansion disabled — the structural stage on a residue row shard is not wired for enable=False")
    if getattr(model, "_maybe_foldcp_mesh", None) is not None and model._maybe_foldcp_mesh() is not None:
        _refuse("a Fold-CP mesh under n_gpu>1 (the row-sharded structural stage and upstream's Fold-CP are exclusive)")
    missing = [key for key in REQUIRED_FEATURES if key not in input_feature_dict]
    if missing:                                                                      # the engine's sentence (opendde.py:446-453)
        raise KeyError("Structural token expansion is enabled, but input_feature_dict is missing required structural feature(s): " + ", ".join(missing))
    lay_res = _require_sharded(z_res_shard.lay, "expand_to_structural_tokens_rows")
    z_res = z_res_shard.z
    dev = z_res.device
    peak0 = _dev_peak_bytes(dev)
    t0 = time.perf_counter()
    expander = model.structural_token_expander
    parent, _role = _parent_role(input_feature_dict)
    n_struct = int(parent.shape[-1])
    if parent.dim() != 1 or n_struct < 1 or int(parent.min()) < 0 or int(parent.max()) >= int(lay_res.N):   # replicated input: every rank refuses alike, BEFORE any collective
        _refuse(f"parent_residue_idx {tuple(parent.shape)} (min={int(parent.min()) if n_struct else None} max={int(parent.max()) if n_struct else None}) "
                f"is not a [N_st] index into the N={lay_res.N} residue rows")
    lay_st = _require_sharded(_td.struct_layout(n_struct), "expand_to_structural_tokens_rows")
    structural_feature_dict = _structural_feature_dict_pre(input_feature_dict)
    # ---- singles (replicated) and this rank's structural pair rows from the K residue rows they read
    s_inputs_struct, s_struct = expand_single(expander, input_feature_dict, s_inputs, s)
    rows_needed = struct_rows_needed(parent, lay_st)
    zres_rows, zres_index = fetch_rows_ring(z_res, rows_needed, lay_res)            # [K, N, c_z]: collective, fixed slab count on every rank
    K = int(zres_index.numel())
    attn_bias_rows = z_res.new_empty((int(lay_st.R), n_struct))
    pair_dtype = _tp._struct_pair_dtype()                                            # lever struct_pair_bf16: the structural shard is STORED bf16 from its first byte
    _tp.STATS["struct_pair_dtype"] = pair_dtype
    z_res_shard.park()                                                               # ROWPAIR_PARK_ZRES: the residue shard leaves the card NOW — its rows this stage reads are
    del z_res                                                                        # fetched (zres_rows); the distogram / confidence rows unpark it after the roll-out
    z_rows = expand_rows(expander, input_feature_dict, s_inputs, s, zres_rows, zres_index, lay_st, rows=_expand_block_rows(expander, lay_st),
                         attn_bias_out=attn_bias_rows, out_dtype=(torch.bfloat16 if pair_dtype == "bf16" else None))
    del zres_rows, zres_index, rows_needed
    structural_feature_dict = _structural_feature_dict_post(input_feature_dict, structural_feature_dict, parent, {_td.EXTRA_BIAS_KEY: attn_bias_rows})
    structural_feature_dict[_td.EXTRA_BIAS_ROWS_KEY] = True                          # the extra bias is BORN as this rank's rows [R_st, N_st]
    structural_feature_dict = model.relative_position_encoding.generate_relp(structural_feature_dict, lazy=lazy_relp)   # O(N_st) index vectors; rows materialised only by the diffusion conditioning
    # ---- the refiner on the structural row shard (single track carried); lever struct_pair_bf16: the shard is STORED bf16 from here and the
    # refiner's pair stack runs under bf16 autocast on it (A block, slabs, transposes all bf16); consumers upcast their rows at the read
    if pair_dtype == "bf16" and z_rows.dtype != torch.bfloat16:
        _refuse(f"structural shard dtype {z_rows.dtype} under struct_pair_bf16 (expand_rows out_dtype contract)")
    if getattr(model, "enable_structural_token_refiner", False):
        if pair_dtype == "bf16":
            with torch.autocast(device_type=z_rows.device.type, dtype=torch.bfloat16):
                s_struct, z_rows = refine_rows(model.structural_token_refiner, z_rows, lay_st, s=s_struct, extra_attn_bias=attn_bias_rows,
                                               triangle_attention=model.configs.triangle_attention, chunk_size=chunk_size)
            s_struct = s_struct.to(dtype=torch.float32)                              # the single track leaves fp32 (its consumers' dtype on every line)
            _tp.STATS["struct_bf16_calls"] += 1
        else:
            s_struct, z_rows = refine_rows(model.structural_token_refiner, z_rows, lay_st, s=s_struct, extra_attn_bias=attn_bias_rows,
                                           triangle_attention=model.configs.triangle_attention, chunk_size=chunk_size)
        if not finite_rows(z_rows, _expand_block_rows(expander, lay_st)) or not finite_rows(s_struct, _expand_block_rows(expander, lay_st)):   # finite gate on the refined shard, per row
            _refuse(f"non-finite values in the refined structural shard (struct_pair_dtype={pair_dtype}, rows {lay_st.r0}:{lay_st.r1})")   # block: never a silent NaN structure, never a shard-sized bool
    model.drop_residue_only_features_for_structural_branch(structural_feature_dict)
    if z_rows.is_cuda:
        torch.cuda.synchronize(dev)
    _tp.STATS["struct_rows"] = int(lay_st.R)
    _tp.STATS["struct_stage"] = "rowshard"
    _tp.STATS["struct_stage_s"] = round(time.perf_counter() - t0, 3)
    peak1 = _dev_peak_bytes(dev)
    P, r = int(lay_st.P), int(lay_st.rank)
    print(f"[{_tp.TAG}] STRUCT rank={r}/{P} struct_stage=rowshard N={lay_res.N} N_st={n_struct} res_rows={lay_res.r0}:{lay_res.r1} struct_rows={lay_st.r0}:{lay_st.r1} "
          f"R_st={lay_st.R} zres_ring_rows={K} expand_rows={_expand_block_rows(expander, lay_st)} refiner={'rows' if getattr(model, 'enable_structural_token_refiner', False) else 'off'} struct_pair_dtype={pair_dtype} "
          f"zstruct_shard_bytes={z_rows.numel() * z_rows.element_size()} dev_peak_before={peak0} dev_peak_after={peak1} s={_tp.STATS['struct_stage_s']}",
          file=sys.stderr, flush=True)
    return structural_feature_dict, s_inputs_struct, s_struct, _tp.RowShard(z_rows, lay_st)


def fields() -> list:
    """Census pairs of this stage (the adapter appends them to its evidence line). ``struct_R`` = this rank's structural row COUNT
    (``STATS["struct_rows"]``); the ``struct_rows`` RANGE word ``r0:r1`` is :func:`opendde_opt.tp_diffusion.fields`'."""
    from . import tp as _tp
    st = _tp.STATS
    return [("struct_stage", st.get("struct_stage")), ("struct_pair_dtype", st.get("struct_pair_dtype")), ("struct_bf16_calls", st.get("struct_bf16_calls")),
            ("struct_R", st.get("struct_rows")), ("zres_ring_rows", st.get("zres_ring_rows")),
            ("struct_presharded_calls", st.get("struct_presharded_calls", 0)), ("struct_replicated_by_design", REPLICATED_BY_DESIGN)]

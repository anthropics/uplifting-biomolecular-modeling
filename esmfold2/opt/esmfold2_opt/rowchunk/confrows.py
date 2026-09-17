"""confrows -- the sharded confidence statement issued per local row block.

The object.  ``rowpair_heads.confidence_forward_rows_batched`` runs the confidence head on this rank's
pair rows ``[Bm, R, N, c_z]``.  As it stands, ``pae_ln(pair)`` and ``pde_ln(pair)`` each materialise a
WHOLE fp32 copy of the pair rows that exists only to be consumed by the very next Linear, and the head
has no budget or block-size constant of its own (``HEAD_EMBED_ROWS = 256`` serves two other statements).
At N=6656, P=2 (R=3328, c_z=256) one such copy is 21.1 GiB per rank, and four of them are live at the
confidence phase's high-water mark.

What this does.  Source-text patch of ``rowpair_heads.confidence_forward_rows_batched`` (anchors checked,
refused by name when one is missing): the two ``ln -> head -> _categorical_mean`` pairs and the
``softmax -> tm_expected`` reduction are issued per LOCAL ROW BLOCK, straight into one preallocated
``[Bm, R, N, 64]`` logits tensor.  LayerNorm normalises over the channel axis and Linear is a per-row GEMM,
so every value is the full-shard statement's; the softmax is over the bin axis, also row-local.  The
returned tensors are identical in shape, dtype and content to the unpatched statement's.

Three flags ride on the same patch (one installer; installing them separately would let a later one
clobber a flag an earlier one set):

  ``confbf16`` -- the prologue (``z_norm(z)`` + relpos + bonds + s_to_z + s_to_z_T + prod_out) is issued
                  PER LOCAL ROW BLOCK in fp32 and stored into a BF16 ``[Bm, R, N, c_z]`` buffer, so the
                  fp32 plane never exists.  NUMERICS CHANGE: ``pair`` is stored in bf16 from the prologue
                  onward, so the confidence trunk's residual accumulates in bf16 and the row-attention
                  pooling / PAE / PDE heads read a bf16 pair (each upcasts its own row block to fp32 before
                  its LayerNorm, so the logits stay fp32).  Not equivalent to the fp32 statement; the
                  prologue's prod_out GEMM is also split on M with this lever's block size rather than
                  ``HEAD_EMBED_ROWS``, so it is not bitwise against the unpatched row blocks either.
  ``pdeskip``  -- the PDE head is not run: ``pde_logits`` (``[Bm, R, N, 64]`` fp32) and ``pde`` are
                  returned as None.  Nothing on the ``--n_gpu > 1`` route reads them: the writer stages
                  pLDDT / PAE / pTM / ipTM, ``confidence_forward_rows`` already drops ``pde_logits`` per
                  sample at S > 1 (``CONF_LOGITS_DROPPED``), and ``outputs.py`` derives nothing from the
                  confidences.  No numerics change to any value that is kept.
  ``confmem``  -- three memory edits with no numerics change beyond a block split:
                  (a) the pLDDT / resolved weight gathers ``W[intra_idx]`` (``[Bm, A, 384, 50]`` fp32) run
                      in atom blocks under a byte budget (``conf_wein``; bitwise: the einsum contracts
                      over the channel axis only);
                  (b) ``distogram_bins`` (``[Bm, R, N]`` int64) is deleted at its last use instead of
                      living across the whole confidence trunk;
                  (c) ``scores = rap.attn_proj(pair)`` is dead on this frame (``_rap_masks_keys`` returns
                      False, so the masked_fill branch is never taken and ``_rap_rows`` recomputes inside
                      the module) -- it is not computed, and the pooling is called per row block on an
                      fp32 upcast of the block (the module's own forward, unchanged: its softmax is over j
                      within a row).

Knobs: ``apply(mb=, min_N=, bf16_pair=, pde_skip=, confmem=)`` — set by ``rowchunk.install`` from EF2_ROWPAIR_CONF_MB and
EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS; nothing here reads the environment.
"""
import inspect
import sys

STATS = {"calls": 0, "head_calls": 0, "head_chunked": 0, "tm_calls": 0, "tm_chunked": 0, "add_calls": 0, "add_chunked": 0, "stock": 0,
         "row_blocks": 0, "rows": None, "R": None, "N": None,
         # the confidence-head lever group
         "prologue_calls": 0, "prologue_blocks": 0, "prologue_dtype": None,
         "embed_calls": 0, "embed_blocks": 0,
         "pde_skipped": 0, "wein_calls": 0, "wein_blocks": 0, "wein_peak_expand_GiB": 0.0,
         "rap_calls": 0, "rap_blocks": 0, "pair_dtype": None,
         # the token floor: below EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS (or off the GPU) the whole unpatched statement runs (floor_skips);
         # at or above it the row-blocked one (dispatched_blocked)
         "floor_skips": 0, "dispatched_blocked": 0}
_CFG = {"mb": 512.0,                     # target MiB per row block (rowchunk.install: EF2_ROWPAIR_CONF_MB)
        "rows": 0,                       # exact rows per block (0: derive from mb)
        "min_N": 1024,                   # below this many tokens the whole-shard statement runs (EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS)
        "bf16_pair": False,              # lever confbf16
        "pde_skip": False,               # lever pdeskip
        "confmem": False,                # lever confmem (wein + dead-tensor hygiene)
        "wein_budget": 256 * 2**20}       # confmem's atom-block byte budget for the pLDDT / resolved weight gathers
_ORIG = {}
_FEATURES = []

ANCHOR_HEADS = """    pae_logits = head.pae_head(head.pae_ln(pair))
    pae_rows = _categorical_mean(pae_logits, start=0.0, end=32.0).detach()
    pde_logits = head.pde_head(head.pde_ln(pair))
    pde_rows = _categorical_mean(pde_logits, start=0.0, end=32.0).detach()
"""
REPL_HEADS = """    pae_logits, pae_rows = _RC_CONF_HEAD_ROWS(head.pae_ln, head.pae_head, pair, _categorical_mean, torch, which="pae")
    pde_logits, pde_rows = _RC_CONF_HEAD_ROWS(head.pde_ln, head.pde_head, pair, _categorical_mean, torch, which="pde")
"""

# --- lever confbf16: the prologue, per row block, into a bf16 buffer.  Span-replaced by MARKER (the kit's
# text here carries long trailing comments; matching by first/last statement is typo-proof).
PROLOGUE_FIRST = "    z_base = head.z_norm(z)"
PROLOGUE_LAST = "    del z_base\n"
REPL_PROLOGUE = """    pair = _RC_CONF_PROLOGUE(head, z, relative_position_encoding, token_bonds_encoding, s_inputs_normed, r0, r1, num_diffusion_samples, torch)
"""

# --- lever confbf16 / confmem: the distance-bin embedding add (a bf16 `pair` makes the kit's
# add_embed_rows_ fall back to the WHOLE-tensor out-of-place statement, because result_type(bf16, fp32)
# is fp32 -- so it must be replaced when the pair is bf16), and `distogram_bins` dies at its last use.
ANCHOR_EMBED_FIRST = "    pair = add_embed_rows_(pair, head.dist_bin_pairwise_embed, distogram_bins)"
REPL_EMBED = """    pair = _RC_CONF_EMBED_ROWS(pair, head.dist_bin_pairwise_embed, distogram_bins)
    del distogram_bins
"""

# --- lever confbf16: the confidence trunk must be handed its OWN buffer.
# `FoldingTrunk.forward` (modeling_esmfold2_common.py:2704-2719) does, with the fused trimul backend on
# (this run: `backend=fused`):
#       orig_dtype = pair.dtype
#       if pair.is_cuda and fused_on and orig_dtype != torch.bfloat16: pair = pair.to(torch.bfloat16)
#       ... blocks (the rowpair trimul dispatch adds its residual IN PLACE into the tensor it is given) ...
#       if pair.dtype != orig_dtype: pair = pair.to(orig_dtype)
# With an fp32 `pair` those two casts are what make `pair_delta` a DIFFERENT object from `pair`, so the
# caller's `pair.add_(pair_delta)` is `pair + f(pair)`.  With a bf16 `pair` BOTH casts are skipped, the
# blocks would mutate the caller's own tensor, and `pair.add_(pair_delta)` would be `2 * f(pair)` -- a
# silent factor-of-two that pLDDT would not show.  The clone below is exactly the buffer the fp32 path's
# `.to(torch.bfloat16)` allocated (same size, same lifetime), and the alias is checked, not assumed.
ANCHOR_TRUNK = """        pair_delta = run_trunk(head.folding_trunk, pair, pair_mask)
"""
REPL_TRUNK = """        pair_delta = _RC_CONF_TRUNK(run_trunk, head.folding_trunk, pair, pair_mask)
"""

# --- lever confmem: the dead `scores` and the row-blocked pooling
ANCHOR_RAP = """    scores = rap.attn_proj(pair).squeeze(-1)
    scores = scores.masked_fill(~mask[:, None, :].bool().expand_as(scores), float("-inf")) if _rap_masks_keys(rap) else scores
    single_rows = _rap_rows(rap, pair, mask, scores)
"""
REPL_RAP = """    scores = None
    single_rows = _RC_CONF_RAP_ROWS(rap, pair, mask, torch)
"""

# --- lever confmem: the plddt / resolved weight gathers
ANCHOR_WPLDDT = """    w_plddt = head.plddt_weight[intra_idx]
    plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms_ln, w_plddt)
"""
REPL_WPLDDT = """    plddt_logits = _RC_CONF_WEIN(s_at_atoms_ln, head.plddt_weight, intra_idx, torch)
"""
ANCHOR_WRES = """    w_res = head.resolved_weight[intra_idx]
    resolved_logits = torch.einsum("...c,...cb->...b", s_at_atoms_res, w_res)
"""
REPL_WRES = """    resolved_logits = _RC_CONF_WEIN(s_at_atoms_res, head.resolved_weight, intra_idx, torch)
"""
ANCHOR_ADD = """    pair.add_(pair_delta.float())
"""
REPL_ADD = """    _RC_CONF_ADD_ROWS(pair, pair_delta)
"""
ANCHOR_TM = """    pae_probs = F.softmax(pae_logits, dim=-1)
    tm_rows = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)                                                    # rows of tm_expected [Bm, R, N]
    del pae_probs
"""
REPL_TM = """    tm_rows = _RC_CONF_TM_ROWS(pae_logits, tm_per_bin, F, torch)
"""


class ConfRowsRefused(RuntimeError):
    pass


def _on_device(t) -> bool:                                          # the row-blocked statements serve CUDA tensors; tests rebind this to exercise the dispatch on CPU
    return bool(getattr(t, "is_cuda", False))


def confidence_forward_rows_batched(head, lay, run_trunk, s_inputs, z, x_pred, *args, **kwargs):
    """The installed entry point (``rowpair_heads.confidence_forward_rows_batched`` after ``apply()``): ONE dispatch on the fold's token
    count. Below the floor (``N < min_N``) — or off the GPU — the kept ORIGINAL function object runs whole, so every statement of the
    confidence head (prologue dtype and GEMM split, the distance-bin embed add, the PDE head, the weight gathers, the TM reduction) is the
    unchunked route's, byte for byte: a below-floor fold is identical to the route without these levers in every output. At or above the
    floor the row-blocked function built by ``apply()`` runs. ``N`` is the layout's (``lay.N``), else ``z``'s second-to-last extent."""
    n = getattr(lay, "N", None)
    N = int(n) if n is not None else int(z.shape[-2])
    if N < int(_CFG["min_N"]) or not _on_device(z):
        STATS["floor_skips"] += 1
        return _ORIG["fn"](head, lay, run_trunk, s_inputs, z, x_pred, *args, **kwargs)
    STATS["dispatched_blocked"] += 1
    return _ORIG["patched"](head, lay, run_trunk, s_inputs, z, x_pred, *args, **kwargs)


confidence_forward_rows_batched.__confrows__ = True



def _rows_for(R, N, C, elt):
    if _CFG["rows"] > 0:
        return max(1, min(int(_CFG["rows"]), R))
    return max(1, min(R, int(_CFG["mb"] * (1 << 20)) // max(1, N * C * elt)))


def conf_head_rows(ln, lin, pair, categorical_mean, torch, start=0.0, end=32.0, which=None):
    """`lin(ln(pair))` and its categorical mean, per local row block.  Row axis is -3.

    `which="pde"` with the `pdeskip` lever on returns `(None, None)`: the PDE head is not run at all and
    the two output keys are None (nothing on this tree reads them -- see the module docstring).
    Each row block is upcast to fp32 before the LayerNorm, so the logits are fp32 whatever `pair`'s
    storage dtype is (`.float()` on an fp32 tensor returns the tensor itself: no copy, no change)."""
    STATS["head_calls"] += 1
    STATS["pair_dtype"] = str(pair.dtype).replace("torch.", "")
    if which == "pde" and _CFG["pde_skip"]:
        STATS["pde_skipped"] += 1
        return None, None
    Bm, R, N, C = int(pair.shape[0]), int(pair.shape[-3]), int(pair.shape[-2]), int(pair.shape[-1])
    if N < _CFG["min_N"] or not pair.is_cuda:
        STATS["stock"] += 1
        lg = lin(ln(pair))
        return lg, categorical_mean(lg, start=start, end=end).detach()
    rows = _rows_for(R, N, C, 4)
    STATS.update(rows=rows, R=R, N=N)
    logits = mean = None
    for s in range(0, R, rows):
        e = min(s + rows, R)
        lg = lin(ln(pair[..., s:e, :, :].float()))
        cm = categorical_mean(lg, start=start, end=end).detach()
        if logits is None:
            logits = torch.empty((Bm, R, N, int(lg.shape[-1])), dtype=lg.dtype, device=lg.device)
            mean = torch.empty((Bm, R, N), dtype=cm.dtype, device=cm.device)
        logits[..., s:e, :, :] = lg
        mean[..., s:e, :] = cm
        del lg, cm
        STATS["row_blocks"] += 1
    STATS["head_chunked"] += 1
    return logits, mean


def conf_tm_rows(pae_logits, tm_per_bin, F, torch):
    """`(softmax(pae_logits) * tm_per_bin).sum(-1)` per local row block: the whole [Bm, R, N, 64] fp32
    probability copy never exists."""
    STATS["tm_calls"] += 1
    Bm, R, N = int(pae_logits.shape[0]), int(pae_logits.shape[-3]), int(pae_logits.shape[-2])
    if N < _CFG["min_N"] or not pae_logits.is_cuda:
        STATS["stock"] += 1
        p = F.softmax(pae_logits, dim=-1)
        return (p * tm_per_bin[:, None, None, :]).sum(dim=-1)
    rows = _rows_for(R, N, int(pae_logits.shape[-1]), pae_logits.element_size())
    out = torch.empty((Bm, R, N), dtype=pae_logits.dtype, device=pae_logits.device)
    for s in range(0, R, rows):
        e = min(s + rows, R)
        p = F.softmax(pae_logits[..., s:e, :, :], dim=-1)
        out[..., s:e, :] = (p * tm_per_bin[:, None, None, :]).sum(dim=-1)
        del p
        STATS["row_blocks"] += 1
    STATS["tm_chunked"] += 1
    return out


def conf_add_rows(pair, pair_delta):
    """`pair.add_(pair_delta.float())` per local row block.  `pair_delta` is the bf16 output of the
    confidence trunk; `.float()` on the whole shard materialises a SECOND fp32 [Bm, R, N, c_z] -- 21.13 GiB
    at N=6656, P=2 -- that lives only for the add.  The add is elementwise, so a row block is exact."""
    STATS["add_calls"] += 1
    R, N, C = int(pair.shape[-3]), int(pair.shape[-2]), int(pair.shape[-1])
    if N < _CFG["min_N"] or not pair.is_cuda:
        STATS["stock"] += 1
        pair.add_(pair_delta.to(pair.dtype))
        return
    rows = _rows_for(R, N, C, 4)
    for s in range(0, R, rows):
        e = min(s + rows, R)
        pair[..., s:e, :, :].add_(pair_delta[..., s:e, :, :].to(pair.dtype))
        STATS["row_blocks"] += 1
    STATS["add_chunked"] += 1


# ---------------------------------------------------------------------------- lever confbf16
def conf_prologue_rows(head, z, relpos, bonds, sn, r0, r1, S, torch):
    """The confidence head's pair prologue, per LOCAL ROW BLOCK, straight into the output buffer:

        z_base = z_norm(z) + relpos + bonds + s_to_z(sn)_i + s_to_z_T(sn)_j + prod_out(in1_i * in2_j)

    The arithmetic of every block is the kit's, in fp32, in the kit's order; only the STORAGE dtype of the
    result changes (bf16 under the `confbf16` lever).  The whole fp32 [Bm, R, N, c_z] plane -- the object
    the P=2 L=7680 OOM is holding when the confidence trunk asks for its next 256 MiB -- never exists.

    NOT bitwise against the kit's statement: the kit issues the `prod_out` GEMM in blocks of
    HEAD_EMBED_ROWS=256 rows (`add_prod_rows_`) and this issues it in blocks of `_rows_for(...)`, so the
    GEMM is split on M differently.  With `confbf16` the result is additionally rounded to bf16."""
    STATS["prologue_calls"] += 1
    B, R, N, C = int(z.shape[0]), int(z.shape[-3]), int(z.shape[-2]), int(z.shape[-1])
    dt = torch.bfloat16 if _CFG["bf16_pair"] else torch.float32
    STATS["prologue_dtype"] = str(dt).replace("torch.", "")
    a = head.s_to_z(sn)[:, r0:r1].unsqueeze(2)                     # [B, R, 1, C]
    b = head.s_to_z_transpose(sn).unsqueeze(1)                     # [B, 1, N, C]
    in1 = head.s_to_z_prod_in1(sn)[:, r0:r1]
    in2 = head.s_to_z_prod_in2(sn)
    out = torch.empty((B, R, N, C), dtype=dt, device=z.device)
    rows = _rows_for(R, N, C, 4)
    for s in range(0, R, rows):
        e = min(s + rows, R)
        zb = head.z_norm(z[..., s:e, :, :].float())
        if relpos is not None:
            zb = zb + relpos[..., s:e, :, :]
        if bonds is not None:
            zb = zb + bonds[..., s:e, :, :]
        zb = zb + a[..., s:e, :, :]
        zb = zb + b
        zb = zb + head.s_to_z_prod_out(in1[:, s:e, None, :] * in2[:, None, :, :])
        out[..., s:e, :, :].copy_(zb)
        del zb
        STATS["prologue_blocks"] += 1
    return head._repeat_batch(out, S)


def conf_trunk(run_trunk, trunk, pair, pair_mask):
    """`run_trunk(trunk, pair, pair_mask)` on a buffer the trunk may own.  See ANCHOR_TRUNK: with a bf16
    `pair` the trunk skips its own cast copy, so it is handed a clone instead; the returned delta is
    REFUSED if it still aliases `pair` (the caller's next statement is `pair.add_(pair_delta)`)."""
    STATS["trunk_calls"] = STATS.get("trunk_calls", 0) + 1
    import torch
    arg = pair
    if pair.dtype is not torch.float32:
        arg = pair.clone()
        STATS["trunk_clones"] = STATS.get("trunk_clones", 0) + 1
    out = run_trunk(trunk, arg, pair_mask)
    if out is pair or (hasattr(out, "data_ptr") and out.data_ptr() == pair.data_ptr()):
        raise ConfRowsRefused("confrows: the confidence trunk returned a tensor aliasing the caller's "
                              "`pair` -- `pair.add_(pair_delta)` would double it; refusing")
    return out


def conf_embed_rows(pair, embed, bins):
    """`pair += dist_bin_pairwise_embed(bins)` per local row block, IN PLACE, with the lookup cast to
    `pair`'s dtype.  The kit's `add_embed_rows_` does the same thing -- but only when
    `result_type(pair, probe) == pair.dtype`; with a bf16 `pair` and an fp32 embedding table that test
    fails and it falls back to the WHOLE-tensor out-of-place `pair + embed(bins)`, i.e. exactly the fp32
    plane `confbf16` exists to avoid.  Elementwise, so a row block is exact; the cast is the only
    difference from the kit's in-place branch (and it is a no-op when `pair` is fp32)."""
    STATS["embed_calls"] += 1
    R, N, C = int(pair.shape[-3]), int(pair.shape[-2]), int(pair.shape[-1])
    rows = _rows_for(R, N, C, 4)
    for s in range(0, R, rows):
        e = min(s + rows, R)
        pair[..., s:e, :, :].add_(embed(bins[..., s:e, :]).to(pair.dtype))
        STATS["embed_blocks"] += 1
    return pair


# ---------------------------------------------------------------------------- lever confmem
def conf_rap_rows(rap, pair, mask, torch):
    """`RowAttentionPooling(pair_rows, mask)` per local row block, on an fp32 upcast of the block.

    The kit calls the module's own forward on the WHOLE row shard (`_rap_rows`), having first computed a
    `scores = rap.attn_proj(pair)` that `_rap_masks_keys(rap) -> False` makes DEAD (the masked_fill branch
    is never taken and the module recomputes its own scores inside its forward).  The module's forward is
    called here UNCHANGED on each row block -- its softmax runs over j within a row, so a row block is the
    whole-shard statement -- and the dead `scores` is never computed."""
    STATS["rap_calls"] += 1
    R, N, C = int(pair.shape[-3]), int(pair.shape[-2]), int(pair.shape[-1])
    rows = _rows_for(R, N, C, 4)
    outs = []
    for s in range(0, R, rows):
        e = min(s + rows, R)
        outs.append(rap(pair[..., s:e, :, :].float(), mask))
        STATS["rap_blocks"] += 1
    return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


def conf_wein(x, W, idx, torch):
    """`einsum("...c,...cb->...b", x, W[idx])` in ATOM blocks under a byte budget -- the pLDDT-weight gather on
    this frame.  `W` is [K, C, Bn] (1.75 MB); `W[idx]` expands it to one C x Bn matrix PER ATOM, ~4.8 GiB
    of fp32 at N=7680, live for one einsum.  Atom rows are independent (the contraction is over C only),
    so the block split is BITWISE."""
    STATS["wein_calls"] += 1
    B, A, C = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    Bn = int(W.shape[-1])
    per_atom = C * Bn * W.element_size() * B
    r = max(256, min(A, _CFG["wein_budget"] // max(1, per_atom)))
    if r >= A:
        return torch.einsum("...c,...cb->...b", x, W[idx])
    out = torch.empty((B, A, Bn), dtype=x.dtype, device=x.device)
    for s in range(0, A, r):
        e = min(s + r, A)
        w = W[idx[:, s:e]]
        out[:, s:e] = torch.einsum("...c,...cb->...b", x[:, s:e], w)
        del w
        STATS["wein_blocks"] += 1
    STATS["wein_peak_expand_GiB"] = round(max(STATS["wein_peak_expand_GiB"], r * per_atom / 2**30), 4)
    return out


def _span_replace(src, first, last, repl, what):
    """Replace the span from the line beginning `first` through the end of the line `last`, requiring each
    marker to occur EXACTLY ONCE.  Used where the kit's text carries long trailing comments that a
    hand-copied anchor would get subtly wrong."""
    if src.count(first) != 1 or src.count(last) != 1:
        raise ConfRowsRefused("confrows: the %s span markers are not unique in the kit's "
                              "confidence_forward_rows_batched (first=%d, last=%d) -- refusing"
                              % (what, src.count(first), src.count(last)))
    i = src.index(first)
    j = src.index(last) + len(last)
    if j <= i:
        raise ConfRowsRefused("confrows: the %s span markers are out of order -- refusing" % what)
    return src[:i] + repl + src[j:]


def apply(mb=None, rows=None, min_N=None, bf16_pair=None, pde_skip=None, confmem=None):
    import esmfold2_opt.rowpair_heads as H
    for k, v in (("mb", mb), ("rows", rows), ("min_N", min_N),
                 ("bf16_pair", bf16_pair), ("pde_skip", pde_skip), ("confmem", confmem)):
        if v is not None:
            _CFG[k] = v
    if "fn" in _ORIG:
        return {"already": True, "features": list(_FEATURES)}
    fn = H.confidence_forward_rows_batched
    src = inspect.getsource(fn)
    for name, anc in (("heads", ANCHOR_HEADS), ("tm", ANCHOR_TM), ("add", ANCHOR_ADD)):
        if src.count(anc) != 1:
            raise ConfRowsRefused("confrows: the kit's confidence_forward_rows_batched does not carry the "
                                  "%s anchor exactly once (found %d) -- refusing" % (name, src.count(anc)))
    new_src = src.replace(ANCHOR_HEADS, REPL_HEADS).replace(ANCHOR_TM, REPL_TM).replace(ANCHOR_ADD, REPL_ADD)
    _FEATURES[:] = ["confrows"]
    if _CFG["pde_skip"]:
        _FEATURES.append("pdeskip")                       # no source edit: conf_head_rows(which="pde") returns (None, None)
    if _CFG["bf16_pair"]:
        new_src = _span_replace(new_src, PROLOGUE_FIRST, PROLOGUE_LAST, REPL_PROLOGUE, "prologue")
        if new_src.count(ANCHOR_TRUNK) != 1:
            raise ConfRowsRefused("confrows: the kit's confidence_forward_rows_batched does not carry the "
                                  "trunk anchor exactly once (found %d) -- refusing" % new_src.count(ANCHOR_TRUNK))
        new_src = new_src.replace(ANCHOR_TRUNK, REPL_TRUNK)
        _FEATURES.append("confbf16")
    if _CFG["bf16_pair"] or _CFG["confmem"]:
        # the embed add MUST be replaced when the pair is bf16 (the kit's in-place branch refuses on
        # dtype and falls back to a whole-tensor fp32 add); under confmem alone it is the `del bins`
        new_src = _span_replace(new_src, ANCHOR_EMBED_FIRST, ANCHOR_EMBED_FIRST, REPL_EMBED, "embed")
        if "embed" not in _FEATURES:
            _FEATURES.append("embed_rows+del_bins")
    if _CFG["confmem"]:
        for name, anc, rep in (("rap", ANCHOR_RAP, REPL_RAP), ("w_plddt", ANCHOR_WPLDDT, REPL_WPLDDT),
                               ("w_res", ANCHOR_WRES, REPL_WRES)):
            if new_src.count(anc) != 1:
                raise ConfRowsRefused("confrows: the kit's confidence_forward_rows_batched does not carry the "
                                      "%s anchor exactly once (found %d) -- refusing" % (name, new_src.count(anc)))
            new_src = new_src.replace(anc, rep)
        _FEATURES.append("confmem")
    g = H.__dict__
    g["_RC_CONF_HEAD_ROWS"] = conf_head_rows
    g["_RC_CONF_TM_ROWS"] = conf_tm_rows
    g["_RC_CONF_ADD_ROWS"] = conf_add_rows
    g["_RC_CONF_PROLOGUE"] = conf_prologue_rows
    g["_RC_CONF_EMBED_ROWS"] = conf_embed_rows
    g["_RC_CONF_RAP_ROWS"] = conf_rap_rows
    g["_RC_CONF_WEIN"] = conf_wein
    g["_RC_CONF_TRUNK"] = conf_trunk
    ns = {}
    exec(compile(new_src, "<confrows:%s>" % H.__file__, "exec"), g, ns)
    new_fn = ns["confidence_forward_rows_batched"]
    new_fn.__confrows__ = True
    _ORIG["fn"] = fn                                                   # the whole statement: what runs below the token floor
    _ORIG["patched"] = new_fn                                          # the row-blocked statement: what runs at or above it
    # confidence_forward_rows resolves it as a module global at call time (rowpair_heads.py:672, :688)
    H.confidence_forward_rows_batched = confidence_forward_rows_batched      # the floor dispatch (this module's entry point), not new_fn directly
    return {"version": "2.0", "rebound": "esmfold2_opt.rowpair_heads.confidence_forward_rows_batched",
            "features": list(_FEATURES), "src_lines": src.count("\n"), "new_src_lines": new_src.count("\n"),
            "mb": _CFG["mb"], "rows": _CFG["rows"], "min_N": _CFG["min_N"],
            "bf16_pair": _CFG["bf16_pair"], "pde_skip": _CFG["pde_skip"], "confmem": _CFG["confmem"],
            "wein_budget_MiB": _CFG["wein_budget"] // 2**20}


def unapply():
    import esmfold2_opt.rowpair_heads as H
    _ORIG.pop("patched", None)
    if "fn" in _ORIG:
        H.confidence_forward_rows_batched = _ORIG.pop("fn")


def stats():
    return dict(STATS)

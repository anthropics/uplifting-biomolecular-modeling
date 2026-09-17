"""Row-local statements with NO communication under row sharding — each returns on rows ``[r0, r1)`` exactly what the dense statement
returns on those rows, so the shard is simply the input. What this module adds is the row-block schedule (bounded transients, launches
below 2**31 elements) and the two statements whose 'row-local' form needs one replicated operand:

    transition_rows(fn, z_shard, rows, add)        pair transition / LayerNorm / any pointwise sub-module in local row blocks (in place ``+=`` with
                                                   ``add=True``) — :func:`opt_core.mem.rowpair.shard.local_rows_`
    opm_rows(a, b, layout, outer_fn, rows, out, add, global_rows, row_dim)
                                                   outer-product-mean OUTPUT rows: ``out[i, j] = outer_fn(a[:, i-block], b[:, all j])`` for this
                                                   rank's rows i — ``a``/``b`` are the (replicated) MSA-derived operands ``[S, N, c]``; the
                                                   contraction over sequences S is whole per (i, j) (no cross-rank sums); ``add=True`` accumulates
                                                   the rows IN PLACE into the z shard block by block (the engine's ``z += OPM(m)``)
    apb_local_queries(attn_fn, s_full, z_shard, layout, gather)
                                                   attention-pair-bias on the SINGLE representation with the pair bias from local rows: queries =
                                                   this rank's token rows (bias rows local), keys / values from the replicated ``s``; the output
                                                   rows are all-gathered back to a replicated ``s`` update (``gather=True``). P-invariance holds
                                                   iff ``attn_fn``'s per-query arithmetic is independent of the query count (stated per kernel)
    max_rows(N, C)                                 the int32 launch budget for ``[rows, N, C]`` blocks (:func:`opt_core.mem.rowpair.trimul.max_rows_per_launch`)
    transition_face_fn(weights | w_*, word, n_tokens, engine_fn, residual, ...)
                                                   the pair transition of a row block through the shared core's ONE transition
                                                   provider (:func:`opt_core.kernels.transition.transition`) by a TIER WORD, resolved at
                                                   ``n_tokens = N`` (the cell the one-GPU line consults at that N): weights packed ONCE, the face
                                                   called per row block on ``x2d = [rows*N, c]`` (sub-blocked below the int32 hidden budget
                                                   :func:`max_rows_hidden`), mask folded, optional residual epilogue; a Refusal by name is answered
                                                   with ``engine_fn`` (today's engine statement), said in the census
    max_rows_hidden(N, c, factor)                  rows per face call so the hidden operand ``[rows*N, factor*c]`` stays below 2**31 elements
"""
from __future__ import annotations

from typing import Callable, Optional

from . import RowpairRefused
from ._torch import torch
from .dist import is_dist, require_sharded, Layout, all_gather_rows
from .evidence import record_schedule
from .shard import local_rows_, produce_rows_
from .trimul import max_rows_per_launch

__all__ = ["transition_rows", "opm_rows", "apb_local_queries", "max_rows", "transition_face_fn", "transition_face_word", "max_rows_hidden", "INT32_MAX",
           "ENV_TRANSITION_FACE", "TRANSITION_FACE_OFF_WORDS"]

INT32_MAX = 2 ** 31 - 1
ENV_TRANSITION_FACE = "ROWPAIR_TRANSITION_FACE"      # the tier / row word transition_face_fn serves when its `word` is None; unset / off / engine / 0 = OFF (today's engine statement)
TRANSITION_FACE_OFF_WORDS = ("", "0", "off", "engine", "none", "no", "false")


def transition_face_word(word: Optional[str] = None) -> str:
    """The transition-face word in force: ``word`` given wins, else ``ROWPAIR_TRANSITION_FACE``; an OFF word (unset, ``0``, ``off``, ``engine``, ...)
    answers ``""`` = the face is not bound (today's engine statement). Any other word is handed to the provider, which refuses unknown words by name."""
    import os
    w = (os.environ.get(ENV_TRANSITION_FACE, "") if word is None else str(word)).strip()
    return "" if w.lower() in TRANSITION_FACE_OFF_WORDS else w


def max_rows(N: int, C: int) -> int:
    return max_rows_per_launch(N, C)


def max_rows_hidden(N: int, c: int, factor: int = 4, cap_elems: int = INT32_MAX) -> int:
    """Rows of an ``[rows, N, c]`` pair block one transition-face call may take so that its HIDDEN operand ``[rows*N, factor*c]`` stays below
    ``cap_elems`` elements (the carried kernels index rows / elements in int32): ``cap_elems // (N * factor * c)``, at least 1. P-invariant."""
    return max(1, int(cap_elems) // max(1, int(N) * int(factor) * int(c)))


def transition_face_fn(weights=None, *, word: Optional[str] = None, n_tokens: int, engine_fn: Optional[Callable[[object, object], object]] = None,
                       residual: bool = False, w_o=None, w_a=None, w_b=None, w_ab=None, ln_w=None, ln_b=None, eps: float = 1e-5, device=None,
                       family: str = "pair", form: str = "swiglu", name: str = "pair_transition", log: Optional[Callable[[str], None]] = None):
    """Build the ``transition(x_rows, mask_u_rows)`` callable of :func:`.pairstack.bind` over the shared core's transition PROVIDER
    (:mod:`opt_core.kernels.transition`, the face the one-GPU fast/big lines call) by the tier ``word``. Nothing here is a
    kernel or a cell: the provider's table decides, per card / dtype / (c, hidden) / ``n_tokens``, which carried row serves (``v2`` in place with the
    row mask and the residual folded, ``pf`` / ``lnl`` / ``v1`` ..., or the stock statement), exactly as it decides for the one-GPU line at that N.

    * ``word``: the provider's tier / row word (``big`` ``fast`` ``v2`` ...); None reads ``ROWPAIR_TRANSITION_FACE`` (:func:`transition_face_word`).
      OFF (unset / ``0`` / ``off`` / ``engine``) returns ``engine_fn`` ITSELF — today's binding byte for byte, nothing packed, nothing recorded — so a
      kit binds this call unconditionally and the line's lever word turns the face on (``engine_fn=None`` with the face off is refused by name).
    * ``weights``: a packed :class:`opt_core.kernels.transition.Weights` (:func:`~opt_core.kernels.transition.pack`), or the canonical tensors
      ``w_a`` / ``w_b`` (or ``w_ab``) ``[hidden, c]``, ``w_o [c, hidden]``, ``ln_w`` / ``ln_b`` / ``eps`` packed here ONCE (``device``: where the
      per-row operand packs live; None = the weights' device).
    * ``n_tokens`` = the fold's N (``layout.N``), NOT the block's row count: the cell consulted — hence the launch configuration — is the one the
      one-GPU line resolves at that N (a block ``[rows, N, c]`` is ``rows*N`` rows OF that fold). A size above the table's largest measured size
      resolves ``beyond_measured`` (census ``transition_face_beyond=1``): recorded, never a new key.
    * per call: ``x_rows [rows, N, c]`` -> ``x2d = [rows*N, c]`` (a view; sub-blocked so ``[rows*N, hidden]`` stays below 2**31 elements,
      :func:`max_rows_hidden`), ``mask_u_rows [rows, N, 1]`` -> the face's row mask ``[rows*N]``; ``residual=False`` returns the DELTA (today's
      contract of :func:`.pairstack.transition_update_`); ``residual=True`` asks the face for ``x + delta`` written IN PLACE into the block
      (``out=x2d``: the kernel's residual epilogue, one pass) and the callable carries ``residual_in_fn = True`` so :func:`.pairstack.bind` drives
      :func:`.pairstack.transition_update_` in its residual form.
    * a :class:`~opt_core.kernels.transition.Refusal` (by name: ``x_not_cuda`` on a CPU tensor, an unmeasured card, a dtype / mask / stride the row
      does not take, ...) is answered with ``engine_fn(x_rows, mask_u_rows) -> delta`` — TODAY's engine statement the kit binds — for that call and
      every later one (sticky; census ``transition=engine:fallback(<kind>)``, one log line); ``engine_fn=None`` answers with the provider's own
      stock statement row (``torch_swiglu``: the op in torch on any device).
    Census: ``transition=face:<word>`` at bind, then ``transition_face=<the Selection's line>`` / ``transition_face_cell`` / ``transition_face_beyond``
    after the first served call, ``transition_face_calls`` / ``transition_face_cap_rows``. Numerics: the served row's class at that cell (tolerance-
    class vs the engine statement unless the tier's winner IS the statement); P-invariant (row-local, the same row on every rank)."""
    word = transition_face_word(word)
    if word == "":                                                        # OFF: today's engine statement, unwrapped (bind sees no residual_in_fn: today's lambda)
        if engine_fn is None:
            raise RowpairRefused(f"transition_face_fn({name}): the face is off ({ENV_TRANSITION_FACE} unset / off) and no engine_fn was bound — "
                                 "bind the engine's transition statement as engine_fn (it serves whenever the face is off or refuses)")
        return engine_fn
    from ...kernels import transition as KT                              # noqa: WPS433 — stdlib-only at import; torch enters at the first call
    W = weights if weights is not None else KT.pack(w_o=w_o, w_a=w_a, w_b=w_b, w_ab=w_ab, ln_w=ln_w, ln_b=ln_b, eps=eps, device=device)
    c, hidden = int(W.c), int(W.hidden)
    N = int(n_tokens)
    if N < 1:
        raise RowpairRefused(f"transition_face_fn({name}): n_tokens={n_tokens}: the fold's token count (layout.N) is required")
    factor = max(1, hidden // max(1, c))
    cap = max_rows_hidden(N, c, factor)
    state = {"served": 0, "engine": 0, "refusal": None, "sel": None, "line": None}
    record_schedule(transition=f"face:{word}", transition_face_ntokens=N, transition_face_cap_rows=cap)
    say = log if log is not None else (lambda _m: None)

    def _engine(x_rows, mask_u_rows):
        """today's statement for these rows -> delta"""
        state["engine"] += 1
        if engine_fn is not None:
            return engine_fn(x_rows, mask_u_rows)
        x2d = x_rows.reshape(-1, c)
        m1 = mask_u_rows.reshape(-1) if mask_u_rows is not None else None
        y2d, _sel = KT.transition(x2d, W, word="torch_swiglu", mask=m1, n_tokens=N, family=family, form=form)
        return y2d.reshape(x_rows.shape)

    def _refuse(r, x_rows, s0):
        kind = str(getattr(r, "kind", None) or r).split(" ")[0]
        state["refusal"] = kind
        record_schedule(transition=f"engine:fallback({kind})", transition_face_refusal=str(r))
        say(f"[transition] {name}: the provider refused word={word!r} by name ({r}); rows from {s0} of this block and every later block run "
            f"the engine statement ({'engine_fn' if engine_fn is not None else 'torch_swiglu row'})")

    def fn(x_rows, mask_u_rows):
        rows = int(x_rows.shape[0])
        if int(x_rows.shape[-1]) != c:
            raise RowpairRefused(f"transition_face_fn({name}): x rows {tuple(x_rows.shape)} vs packed c={c}")
        if state["refusal"] is not None:                                 # sticky named fallback: today's statement
            delta = _engine(x_rows, mask_u_rows)
            if residual:
                x_rows.add_(delta)
                return x_rows
            return delta
        out = x_rows if residual else torch.empty_like(x_rows)
        s0 = 0
        try:
            while s0 < rows:
                s1 = min(rows, s0 + cap)
                xb = x_rows[s0:s1]
                x2d = xb.view(-1, c)                                     # a view: the block is contiguous (refused by .view otherwise, never a silent copy)
                m1 = mask_u_rows[s0:s1].reshape(-1) if mask_u_rows is not None else None
                if residual:
                    y2d, sel = KT.transition(x2d, W, word=word, mask=m1, residual=True, out=x2d, n_tokens=N, family=family, form=form)
                    if y2d.data_ptr() != x2d.data_ptr():                 # a row that does not write `out` answered with a new tensor: land it
                        x2d.copy_(y2d)
                else:
                    o2d = out[s0:s1].view(-1, c)
                    y2d, sel = KT.transition(x2d, W, word=word, mask=m1, out=o2d, n_tokens=N, family=family, form=form)
                    if y2d.data_ptr() != o2d.data_ptr():
                        o2d.copy_(y2d)
                del y2d
                if state["sel"] is None:
                    state["sel"] = sel
                    state["line"] = sel.line() if hasattr(sel, "line") else str(sel)
                    record_schedule(transition_face=state["line"], transition_face_cell=str(getattr(sel, "cell_key", None)),
                                    transition_face_beyond=int(bool(getattr(sel, "beyond_measured", False))))
                    say(f"[transition] {name}: {state['line']} n_tokens={N} rows/call<={cap} residual={'epilogue' if residual else 'delta'}"
                        + (" BEYOND the table's largest measured size (recorded; no key added)" if getattr(sel, "beyond_measured", False) else ""))
                state["served"] += 1
                s0 = s1
        except KT.Refusal as r:                                          # by name -> the engine statement for the rest of this block, then sticky
            _refuse(r, x_rows, s0)
            rest = x_rows[s0:]
            delta = _engine(rest, mask_u_rows[s0:] if mask_u_rows is not None else None)
            if residual:
                rest.add_(delta)
            else:
                out[s0:].copy_(delta)
            del delta
        record_schedule(transition_face_calls=state["served"], transition_engine_calls=state["engine"])
        return out

    fn.residual_in_fn = bool(residual)
    fn.face_state = state
    fn.weights = W
    fn.word = word
    fn.n_tokens = N
    fn.cap_rows = cap
    return fn


def transition_rows(fn: Callable, z_shard, layout: Layout, rows: Optional[int] = None, *, add: bool = False, ledger=None,
                    key: Optional[str] = None):
    """``z_shard[block] = fn(z_shard[block])`` (``+=`` with ``add``) per local row block of ``rows`` rows (None: one block) of a SHARDED layout
    (P == 1 refused by name: the engine's own transition runs at n_gpu=1). Returns ``z_shard``."""
    require_sharded(layout, "transition_rows")
    return local_rows_(fn, z_shard, rows, add=add, ledger=ledger, key=key)


def opm_rows(a, b, layout: Layout, outer_fn: Callable, rows: Optional[int] = None, out=None, *, add: bool = False,
             global_rows: bool = False, row_dim: int = 1, align: Optional[int] = None, a_local: bool = False, allow_unsharded: bool = False):
    """Outer-product-mean OUTPUT rows of this rank: for local row blocks ``[i0, i1)`` (global ``[g0, g1) = [r0+i0, r0+i1)``),
    ``y = outer_fn(a_blk, b)`` — ``outer_fn(a_blk, b, g0, g1)`` with ``global_rows=True``, for statements that slice a replicated per-row
    operand themselves (the OPM mask normalisation ``norm[i, j] = sum_s mask[s, i] mask[s, j]`` of rows ``[g0, g1)``) — where ``a_blk =
    a.narrow(row_dim, g0, g1-g0)`` is the token-row block of the replicated operand ``a`` (``[S, N, c]``: ``row_dim=1``; an engine layout
    ``[*, N, S, c]``: ``row_dim=-3``), ``b`` the whole replicated operand, and ``y`` the ``[rows, N, C_z]`` output rows (mean over S, output
    projection and normalisation included; token rows on dim 0). The contraction over S is whole per ``(i, j)``: no cross-rank sums.
    ``a_local=True``: ``a`` holds only THIS rank's token rows (``R`` on ``row_dim`` — the operand of a token-sharded MSA representation;
    ``b`` is then the all-gathered operand) and blocks are taken at local offsets ``[g0-r0, g1-r0)``.

    ``add=False``: ``out[i0:i1] = y`` (``out`` allocated ``[R, N, C_z]`` on the first block when None) -> the OPM rows of this rank.
    ``add=True`` : ``out[i0:i1] += y`` with ``out`` = the z SHARD ``[R, N, C_z]`` (required; :func:`opt_core.mem.rowpair.dist.zadd`) — the
                   engine's ``z += OPM(m)`` residual lands IN PLACE row block by row block, so the ``[R, N, C_z]`` OPM transient never exceeds
                   one row block. Returns ``out``.
    ``rows``: the row block (None: one block = all local rows); ``align``: the engine's chunk size (None: the layout's ``align`` /
    ``ROWPAIR_CHUNK_ALIGN``) — blocks are :func:`opt_core.mem.rowpair.shard.produce_rows_` blocks: they start at multiples of ``align`` from a
    chunk-aligned ``r0``, so an engine statement that chunks ``a_blk``'s rows itself walks its GLOBAL chunk grid exactly
    (:func:`opt_core.mem.rowpair.msa.opm_rows_budgeted` chooses, aligns and prints the block).
    ``allow_unsharded=True``: an UNSHARDED layout (``Layout(N, 1, 0)`` / a replicated layout / no process group) is accepted and the same
    statement walks the whole row range ``[0, N)`` block by block on this one device — the single-GPU row-blocked OPM a kit binds as its own
    named memory lever (no ``[N, N, c·c]`` intermediate; no collective is issued; ``a_local`` is refused there: nothing is sharded to gather).
    Without the flag an unsharded layout is refused by name (the adapter installs nothing at ``n_gpu=1``)."""
    unsharded = bool(allow_unsharded) and (layout.P == 1 or layout.replicated or not is_dist())
    if unsharded:
        if a_local:
            raise RowpairRefused("opm_rows: allow_unsharded=True with a_local=True: an unsharded layout has no token-sharded operand to take rows from")
        if not (layout.r0 == 0 and layout.R == layout.N):
            raise RowpairRefused(f"opm_rows: allow_unsharded=True needs the whole-row layout (r0=0, R=N); got r0={layout.r0} R={layout.R} N={layout.N}")
    else:
        require_sharded(layout, "opm_rows")
    rd = row_dim % a.dim()
    want = layout.R if a_local else layout.N
    if int(a.shape[rd]) != want or b.dim() <= rd or int(b.shape[rd]) != layout.N:
        raise RowpairRefused(f"opm_rows: a {tuple(a.shape)} b {tuple(b.shape)} row_dim={row_dim} a_local={a_local} vs layout N={layout.N} R={layout.R} "
                             f"(a: {'this rank`s R' if a_local else 'all N'} token rows; b: all N token rows)")
    a_off = layout.r0 if a_local else 0
    R = layout.R
    if add and out is None:
        raise RowpairRefused("opm_rows: add=True accumulates into the z shard: out=<z_shard [R, N, C_z]> is required")
    if out is not None and int(out.shape[0]) != R:
        raise RowpairRefused(f"opm_rows: out {tuple(out.shape)} must hold this rank's R={R} rows on dim 0")

    first = {}

    def rows_fn(g0: int, g1: int):
        if g0 in first:                                                    # the block already evaluated to size ``out`` (out-of-place form)
            return first.pop(g0)
        a_blk = a.narrow(rd, g0 - a_off, g1 - g0)
        y = outer_fn(a_blk, b, g0, g1) if global_rows else outer_fn(a_blk, b)
        if y.dim() < 1 or int(y.shape[0]) != g1 - g0:
            raise RowpairRefused(f"opm_rows: outer_fn returned {tuple(y.shape)} for a block of {g1 - g0} rows (output token rows belong on dim 0)")
        return y                                                           # add: == the engine's z += opm restricted to these rows

    block_rows = R if rows is None else max(1, int(rows))
    if out is None:                                                        # allocate [R, N, C_z] from the first block's piece
        from .shard import iter_row_blocks
        _b0, _b1, g0, g1 = iter_row_blocks(layout, block_rows, align)[0]
        y0 = rows_fn(g0, g1)
        out = y0.new_empty((R,) + tuple(int(v) for v in y0.shape[1:]))
        first[g0] = y0
    return produce_rows_(out, layout, rows_fn, op="add" if add else "set", block_rows=block_rows, row_dim=0, unit=align)


def apb_local_queries(attn_fn: Callable[[object, object, object], object], s_full, z_shard, layout: Layout, *, gather: bool = True):
    """Attention with pair bias over the single representation: ``out_rows = attn_fn(s_full[r0:r1] (queries), s_full (keys/values source),
    z_shard (bias rows))`` -> ``[R, c_s]``; ``gather=True`` returns the replicated ``[N, c_s]`` (all_gather_rows), else the row shard."""
    require_sharded(layout, "apb_local_queries")
    if int(s_full.shape[-2]) != layout.N:
        raise RowpairRefused(f"apb_local_queries: s {tuple(s_full.shape)} vs layout.N={layout.N}")
    q = s_full[..., layout.r0:layout.r1, :]
    out = attn_fn(q, s_full, z_shard)
    if not gather:
        return out
    if out.dim() == 2:
        return all_gather_rows(out.contiguous(), layout)
    return all_gather_rows(out.movedim(-2, 0).contiguous(), layout).movedim(0, -2)

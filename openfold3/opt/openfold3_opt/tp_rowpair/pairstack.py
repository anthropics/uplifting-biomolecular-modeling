"""OpenFold3's pair-stack classes bound onto the core's ONE pair-block driver (``opt_core.mem.rowpair.pairstack``: ``PairBlockFns`` /
``bind`` / ``pair_block_`` / ``pair_stack_`` over the streamed schedules of ``opt_core.mem.rowpair.trimul.trimul_update_`` and
``opt_core.mem.rowpair.triatt.triatt_update_``): ``PairBlock`` (``latent/base_blocks.py:221-490``; the Pairformer's, the MSA module's, the
template pair stack's and the confidence head's pair block), ``PairFormerBlock`` / ``PairFormerStack`` (``latent/pairformer.py:39-480``),
``AttentionPairBias`` (``layers/attention_pair_bias.py:34-238``).

OpenFold3's statements enter the driver as callables (per-element identical to stock; the driver owns row blocks, the ring, the transposes):
  tri-mul   ``TriMulFns(proj, out, gate, C_h)`` from ``TriangleMultiplicativeUpdate._inference_forward`` (``triangular_multiplicative_update.py
            :724-1000``): ``proj`` = ``compute_projection_helper`` (``x = layer_norm_in(z); p = linear_*_g(x); p.sigmoid_(); p *= linear_*_p(x);
            p *= mask``), ``out`` = ``linear_z(layer_norm_out(x))``, ``gate`` = ``linear_g(layer_norm_in(z)).sigmoid_()`` on ORIGINAL rows;
            ``inplace_chunk`` = the module's ``_inplace_chunk_size`` (256): the stock column grid.
  tri-att   ``TriAttFns(ln, bias, attend)`` from ``TriangleAttention.forward`` (``triangular_attention.py:105-180``; ``tri_att_end`` is a
            ``starting=True`` module in 0.4.1 — the ENDING orientation is the driver's distributed transpose, replacing the block-level
            ``z.transpose(-2, -3)`` of ``PairBlock.tri_att_start_end`` ``base_blocks.py:387/410``): ``ln`` = ``layer_norm``, ``bias`` =
            ``linear_z`` (channel-last), ``attend`` = ``mha`` with ``[inf * (mask - 1), triangle_bias]`` — query sub-blocks of
            ``ROWPAIR_TRIATT_QBLOCK`` rows through ``mha._prep_qkv`` / ``_attention`` / ``_wrap_up`` (the core's ``attend_query_blocks``).
  transition ``pair_transition(x_rows, mask=mask_rows)`` per local row block (the block is the chunk).
  apb       (Pairformer) ``AttentionPairBias`` with queries = local rows: ``layer_norm_a``, q/k/v/gate on the FULL replicated single
            representation (M = N, as stock), q sliced to this rank's rows after the projection, pair-bias rows ``linear_z(layer_norm_z(z_shard))``
            in sub-blocks of ``ROWPAIR_APB_ROWBLOCK`` rows, the attention core per ``ROWPAIR_APB_QBLOCK`` query rows, the per-row outputs
            all-gathered BEFORE ``_wrap_up`` (``opt_core.mem.rowpair.transition.apb_local_queries``), the AdaLN output gate; ``single_transition``
            replicated.
The triangle KERNELS are levers of this binding, ONE vocabulary with the core: ``TRIATT_KERNEL`` = ``flash_triattn`` (``triatt_kernel()``: the
core's row-block dispatch ``opt_core.mem.rowpair.triatt.attention_core(kernel=<word>)`` — the carried flash triangle-attention kernel on each row batch,
the torch statement above its named per-call fallback, its census the core's ``LEVER name=F1.flash_triattn …`` line; the core's opt-out
``ROWPAIR_TRIATT_CORE=torch`` runs the torch statement for every call) and ``TRIMUL_KERNELS`` = ``fpf_v4`` (``trimul_kernels()``: the core's fused provider
``opt_core.mem.rowpair.trimul_fused.fused_trimul_fns`` wrapped around the ``TriMulFns`` above — the carried fpf_trimul_v4 kernels per row block with
the kit's cell table, every declined unit running the torch statements by reason (``below_gate`` under the provider's pair-size gate, …), the
core's opt-out ``ROWPAIR_TRIMUL_KERNELS=torch``, its census ``LEVER name=F2.trimul_rows …``). Each rank prints one ``[tp_rowpair] TRIATT …`` / ``[tp_rowpair] TRIMUL …`` line at exit (tp.py parses
them into the manifest's tp block and its exit rule) plus the core's LEVER lines. Not bitwise vs the torch statements (Tier 2 kernels).
Refused by name: ``FusedTriangleMultiplicativeUpdate`` (``fuse_projection_weights``), the DS4Sci / triton / LMA attention routes on shards
(and cuEq through the model config: the kernels are this binding's, not the runner yaml's), the non-inplace tri-mul path, a live
``ChunkSizeTuner``, grad-enabled activation checkpointing, a kernel the process cannot run (no CUDA device: the core's dispatch refuses by name
with its opt-out). Shards cross this boundary as the core's ``[R, N, C]`` (OpenFold3's ``[1, R, N, C]`` shard passes ``z_loc[0]``; masks ``[R, N]``).
"""
from __future__ import annotations

import atexit
import inspect
import os
import sys

import torch
from openfold3.core.utils.tensor_utils import add

from . import core as C
from . import env as _env


class PairstackRefused(RuntimeError):
    pass


KERNEL_FLAGS = ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma")


PAIR_STREAM = True                                              # the streamed tri-att / transition form of the driver (``bind(stream=True)``)
REUSE_Z_STORAGE = True                                          # the driver writes the ending-attention result back into the shard's own storage (``pair_block_(reuse_storage=)``)
TRIATT_KERNEL = _env.KERNEL_WORDS["triatt"]                    # flash_triattn: the tp_triatt lever's word by name; on the tier door's capabilities (env.TRIATT_DOOR_CCS: 9.0 / 8.0) triatt_kernel() resolves it to env.TRIATT_DOOR_WORD (tier:big) at first bind
_TRIATT_WORD_RESOLVED = [None]                                  # the word triatt_kernel() resolved on this rank's device (env.triatt_word(cc)), once
TRIMUL_KERNELS = _env.KERNEL_WORDS["trimul"]                   # fpf_v4: the tp_trimul lever's word for the core's provider (opt_core.mem.rowpair.trimul_fused KERNEL_WORDS)


def pair_stream() -> bool:
    return PAIR_STREAM


def reuse_z_storage() -> bool:
    return REUSE_Z_STORAGE


def triatt_kernel() -> str:
    """The triangle-attention kernel of the row-sharded pair stack (``TRIATT_KERNEL``): the core's ONE row-block dispatch
    ``opt_core.mem.rowpair.triatt.attention_core(kernel=<env.triatt_word(cc)>)`` (``tier:big`` -- the core's tier door, the pre-compiled CUDA triangle-attention row per window on 9.0 / 8.0 below the int32 bias bound -- or ``flash_triattn``, the carried flash triangle-attention kernel, by name elsewhere and above the bound), OpenFold3's torch
    attention statement over query sub-blocks (``ROWPAIR_TRIATT_QBLOCK``) its named per-call fallback (``LEVER name=F1.flash_triattn … served=
    fallback= fallback_by=``); ``ROWPAIR_TRIATT_CORE=torch`` runs the torch statement for every call, counted (the core's opt-out)."""
    global TRIATT_KERNEL
    if _TRIATT_WORD_RESOLVED[0] is None:                    # the word is a function of the device class (env.triatt_word): tier:big on 9.0 / 8.0 (the core's tier door:
        cc = None                                           #  the pre-compiled CUDA triangle-attention row per window below the int32 bias bound, the flash_triattn q-block path above it or on a refusal), flash_triattn by name elsewhere
        try:
            if torch.cuda.is_available():
                cc = tuple(torch.cuda.get_device_capability())
        except Exception:                                   # noqa: BLE001 - no device context: the word by name
            cc = None
        _TRIATT_WORD_RESOLVED[0] = _env.triatt_word(cc)
        TRIATT_KERNEL = _TRIATT_WORD_RESOLVED[0]
    return TRIATT_KERNEL


def trimul_kernels() -> str:
    """The triangle-multiplication statements of the row-sharded pair stack (``TRIMUL_KERNELS``): the core's fused provider
    ``opt_core.mem.rowpair.trimul_fused.fused_trimul_fns`` around OpenFold3's torch statements (``trimul_fns``; the carried fpf_trimul_v4 kernels per
    row block; pairs below the provider's size gate, an unsupported width / dtype or a missing cell run the torch statements, counted by reason:
    ``LEVER name=F2.trimul_rows … served= fallback= fallback_by=``); ``ROWPAIR_TRIMUL_KERNELS=torch`` runs the torch statements for every unit, counted."""
    return TRIMUL_KERNELS



TRIATT = {"kernel": None, "bound": 0, "calls": 0}
"""This process's triangle-attention record: ``kernel`` (the switch word bound), ``bound`` (TriangleAttention modules bound), ``calls`` (row batches handed to the
core dispatch); served / fallback counts are the core dispatch's (``triatt_census``)."""
TRIMUL = {"kernels": None, "bound": 0}
"""This process's triangle-multiplication record: ``kernels`` (the switch word bound), ``bound`` (modules bound); launches / declines are the core provider's
(``trimul_census``)."""


def triatt_census() -> dict:
    """``TRIATT`` plus the core dispatch's counts when a kernel word other than torch is bound: ``served`` / ``fallback`` (calls), ``fallback_by``
    (reason -> n), ``impl``."""
    c = dict(TRIATT, served=0, fallback=0, fallback_by={}, impl=None)
    if TRIATT["kernel"] not in (None, "torch"):
        d = C.fn("triatt", "describe_core")()
        c.update(served=int(d.get("core_served") or 0), fallback=int(d.get("core_fallback") or 0), fallback_by=dict(d.get("fallback_by") or {}), impl=d.get("impl"))
    return c


def trimul_census() -> dict:
    """``TRIMUL`` plus the core provider's counts when ``fpf_v4`` is bound: ``served`` (kernel launches, k1 + k3), ``fallback`` (declined units),
    ``fallback_by``, the facts ``cells`` / ``k1_impl``, ``impl``."""
    c = dict(TRIMUL, served=0, fallback=0, fallback_by={}, cells=None, k1_impl=None, impl=None)
    if TRIMUL["kernels"] == "fpf_v4":
        d = C.fn("trimul_fused", "describe")()
        facts = d.get("facts") or {}
        c.update(served=int(d.get("served") or 0), fallback=int(d.get("fallback") or 0), fallback_by=dict(d.get("fallback_by") or {}),
                 cells=facts.get("cells"), k1_impl=facts.get("k1_impl"), impl=d.get("impl"), k1_launches=d.get("k1_launches"), k3_launches=d.get("k3_launches"))
    c["ablock_words"] = C.fn("trimul", "describe_ablock")()["words"]          # the A-block schedule of trimul_update_ (ra / passes / host row mirror)
    return c


def _fb(d: dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted((d or {}).items())) or "none"


def triatt_census_line() -> str:
    c = triatt_census()
    return f"TRIATT kernel={c['kernel']} bound={c['bound']} calls={c['calls']} served={c['served']} fallback={c['fallback']} fallback_by={_fb(c['fallback_by'])}"


def trimul_census_line() -> str:
    c = trimul_census()
    return (f"TRIMUL kernels={c['kernels']} bound={c['bound']} served={c['served']} fallback={c['fallback']} fallback_by={_fb(c['fallback_by'])} cells={c['cells']} k1_impl={c['k1_impl']}"
            f" {c['ablock_words']}")


_EXIT_LINE = {"registered": False}


def _emit_exit_lines() -> None:
    """At interpreter exit, once per rank: the kit's TRIATT / TRIMUL census lines (tp.py parses them into the manifest's tp block and its exit rule) and the
    core's own LEVER lines of the kernels bound (``F1.flash_triattn``, ``F2.trimul_rows``)."""
    try:
        if TRIATT["kernel"] not in (None, "torch"):
            sys.stderr.write(f"[tp_rowpair] {triatt_census_line()}\n")
            C.fn("triatt", "emit_core_line")("tp_rowpair")
        if TRIMUL["kernels"] == "fpf_v4":
            sys.stderr.write(f"[tp_rowpair] {trimul_census_line()}\n")
            C.fn("trimul_fused", "emit_line")("tp_rowpair")
        sys.stderr.flush()
    except Exception:                                       # noqa: BLE001 — interpreter teardown: the census fields carry the same numbers
        pass


def _register_exit_lines() -> None:
    if not _EXIT_LINE["registered"]:
        _EXIT_LINE["registered"] = True
        atexit.register(_emit_exit_lines)


def refuse_kernel_flags(where: str, **flags) -> None:
    on = [k for k in KERNEL_FLAGS if flags.get(k)]
    if on:
        raise PairstackRefused(f"refused: {where}: {', '.join(on)} set — row shards run OpenFold3's torch attention / tri-mul statements "
                               f"(the tp launcher's yaml pins these kernels off in every arm)")


def _attention_core():
    from openfold3.core.model.primitives import attention as _attn_mod
    return _attn_mod._attention


def _stock_inplace_chunk_size(module) -> int:
    """The ``_inplace_chunk_size`` PairBlock's call implies (``TriangleMultiplicativeUpdate.forward`` default, 256 in 0.4.1)."""
    try:
        return int(inspect.signature(type(module).forward).parameters["_inplace_chunk_size"].default)
    except Exception:                                       # noqa: BLE001
        return 256


def _squeeze_shard(z_loc):
    """OpenFold3's ``[1, R, N, C]`` shard (or a one-element box of it) -> the core's ``[R, N, C]`` view and a re-wrapper."""
    z = z_loc[0] if isinstance(z_loc, list) else z_loc
    if isinstance(z_loc, list):
        z_loc.clear()                                       # box convention: the caller handed over ownership
    lead = z.dim() - 3
    if lead < 0 or any(int(d) != 1 for d in z.shape[:lead]):
        raise PairstackRefused(f"refused: pair shard {tuple(z.shape)}: the tp line runs batch size 1 ([1, R, N, C] or [R, N, C])")
    view = z.reshape(z.shape[lead:]) if lead else z
    return view, (lambda out: out.reshape((1,) * lead + tuple(out.shape)) if lead else out)


def _squeeze_mask(mask_loc):
    if mask_loc is None:
        return None
    m = mask_loc
    while m.dim() > 2:
        if int(m.shape[0]) != 1:
            raise PairstackRefused(f"refused: pair mask shard {tuple(mask_loc.shape)}: batch size 1 expected")
        m = m[0]
    return m


# ----------------------------------------------------------------------------------------------------------------- module -> callables
def trimul_fns(tmu):
    """``TriMulFns`` of one ``TriangleMultiplication{Outgoing,Incoming}`` (unchanged weights): the in-place inference path's statements."""
    from openfold3.core.model.layers.triangular_multiplicative_update import FusedTriangleMultiplicativeUpdate
    if isinstance(tmu, FusedTriangleMultiplicativeUpdate):
        raise PairstackRefused("refused: FusedTriangleMultiplicativeUpdate (fuse_projection_weights=true) has no row-sharded schedule; the predict config uses the unfused module")
    TriMulFns = C.fn("trimul", "TriMulFns")

    def proj(z_block, mask_block, is_a):                    # compute_projection_helper (triangular_multiplicative_update.py:796-815); z's dtype (the ring schedule's byte counts are z's)
        linear_g, linear_p = (tmu.linear_a_g, tmu.linear_a_p) if is_a else (tmu.linear_b_g, tmu.linear_b_p)
        x = tmu.layer_norm_in(z_block)
        p = linear_g(x)
        p.sigmoid_()
        p *= linear_p(x)
        p *= mask_block
        return p.to(z_block.dtype)

    def out(x):                                             # x = linear_z(layer_norm_out(x)), in x's (= z's) dtype
        return tmu.linear_z(tmu.layer_norm_out(x)).to(x.dtype)

    def gate(z_block):                                      # g = linear_g(layer_norm_in(z)); g.sigmoid_(); z's dtype
        g = tmu.linear_g(tmu.layer_norm_in(z_block))
        g.sigmoid_()
        return g.to(z_block.dtype)

    stock = TriMulFns(proj, out, gate, int(tmu.linear_a_p.out_features))
    TRIMUL["kernels"] = trimul_kernels()
    TRIMUL["bound"] += 1
    from ..cells.pairfused import trimul_weights                                 # the kit's ONE OpenFold3 -> WEIGHT_KEYS name map
    eps = float(getattr(tmu.layer_norm_in, "eps", 1e-5))
    if float(getattr(tmu.layer_norm_out, "eps", eps)) != eps:
        raise PairstackRefused(f"refused: tp_trimul ({TRIMUL_KERNELS}): layer_norm_in.eps {eps} != layer_norm_out.eps {tmu.layer_norm_out.eps} (the fused provider takes one eps)")
    _register_exit_lines()
    return C.fn("trimul_fused", "fused_trimul_fns")(trimul_weights(tmu), stock, eps=eps)   # cells=None: the kernels' own loader (+ the pointer-row rule); stock_round: the gated output is bf16 before the fp32 residual, as the statements above


def triatt_fns(ta, use_high_precision: bool = False):
    """``TriAttFns`` of one ``TriangleAttention`` (``starting`` is True for both of PairBlock's modules in 0.4.1)."""
    if not bool(getattr(ta, "starting", True)):
        raise PairstackRefused("refused: TriangleAttention(starting=False): the driver orients the ending node itself (0.4.1's PairBlock uses two starting modules)")
    TriAttFns = C.fn("triatt", "TriAttFns")
    lever_rows = C.fn("triatt", "lever_rows")
    kernel = triatt_kernel()
    TRIATT["kernel"] = kernel
    TRIATT["bound"] += 1
    core = _attention_core()                                # the core's ONE row-block dispatch; q reaches it PRE-SCALED (``_prep_qkv``'s default, the torch statement's
    from opt_core.mem.rowpair import RowpairRefused        #  operand) -> scale=1.0 for the fused kernel, and the dispatch's torch fallback is OpenFold3's statement byte for byte
    try:
        dispatch = C.fn("triatt", "attention_core")(lambda q_, k_, v_, b: core(q_, k_, v_, b, use_high_precision=use_high_precision), kernel=kernel, scale=1.0,
                                                   layout="bnhsd", mask_from="bias0", tri_bias="bias1", stock_qblock=lever_rows("TRIATT_QBLOCK"))
    except RowpairRefused as e:                             # a kernel this process cannot run (no CUDA device, the carried kernels not importable): refused by name at bind with its opt-out, never a silent torch
        raise PairstackRefused(f"refused: tp_triatt ({kernel}): {e}") from None
    _register_exit_lines()

    def ln(z_rows):
        return ta.layer_norm(z_rows)

    def bias(x_rows):                                       # triangle_bias = permute_final_dims(linear_z(x), (2, 0, 1)) — channel-last here; the core permutes
        return ta.linear_z(x_rows)

    def attend(x_rows, mask_rows, tb_full, blk):
        mask_bias = (ta.inf * (mask_rows - 1))[..., :, None, None, :]                 # [rows, 1, 1, N]   (triangular_attention.py:150-153)
        tb = tb_full.movedim(-1, 0).unsqueeze(0)                                       # [1, H, N, N]      (== triangle_bias.unsqueeze(-4) on the row batch)
        mha = ta.mha                                                                   # the core's kernel dispatch: whole rows per call (it splits launches under 2^31 elements
        q, k, v = mha._prep_qkv(x_rows, x_rows)                                        #  and runs its torch fallback over ROWPAIR_TRIATT_QBLOCK query sub-blocks); [rows, H, N, d]
        TRIATT["calls"] += 1
        o = dispatch(q, k, v, [mask_bias, tb])
        del q, k, v
        return mha._wrap_up(o.transpose(-2, -3), x_rows)                               # gating + linear_o (stock)

    # the pieces `attend` is made of, for the core's just-in-time bias schedule (triatt_update_jit_: the bias stays
    # sharded [R, N, H], query blocks of the GLOBAL grid are exchanged per row-window group; attend == wrap(core(*proj(x), [mask_bias(m), view(tb)]), x) by construction)
    def proj(x_rows):
        return ta.mha._prep_qkv(x_rows, x_rows)

    def mask_bias_fn(m_rows):
        return (ta.inf * (m_rows - 1))[..., :, None, None, :]

    def wrap(o, x_rows):
        TRIATT["calls"] += 1
        return ta.mha._wrap_up(o.transpose(-2, -3), x_rows)

    try:
        return TriAttFns(ln, bias, attend, proj=proj, mask_bias=mask_bias_fn, wrap=wrap, core=dispatch, bias_transposed=False)   # openfold3 0.4.x: the driver hands the ending node z^T rows and the module projects the bias in that frame (no transpose_bias) -> False for both nodes (RXP5 §Binding)
    except TypeError:                                       # an older core's positional-only TriAttFns (MIN_CORE guards this; belt and braces)
        return TriAttFns(ln, bias, attend)


def transition_fn(pt):
    """``fn(x_rows, mask_u_rows) -> delta`` for the driver's ``transition_update_``: OpenFold3's pair transition on one row block (the block is the chunk)."""
    def fn(x_rows, mask_u_rows):
        return pt(x_rows, mask=mask_u_rows[..., 0], chunk_size=None)
    return fn


TRIATT_BIAS_AUTO_N = 23170                                   # policy: jit (bias sharded 8N²/P + q-block tiles) above the int32 bias bound — there both words run flash
ENV_TRIATT_BIAS_KIT = "OF3TP_TRIATT_BIAS"                    #  q-blocks, so jit is pure gain (−(8−8/P)·N² resident bytes per rank); at or below it gather (the tier door's the pre-compiled CUDA row takes
TRIATT_BIAS_LAST = [None]                                    #  whole rows: jit would forfeit its speed advantage). auto (default) | gather | jit; census triatt_bias_policy


def triatt_bias_policy(N: int) -> str:
    """``gather`` | ``jit`` for a pair stack over ``N`` tokens: ``OF3TP_TRIATT_BIAS`` = auto (default: jit iff N > TRIATT_BIAS_AUTO_N) | gather | jit."""
    w = os.environ.get(ENV_TRIATT_BIAS_KIT, "auto").strip().lower() or "auto"
    if w not in ("auto", "gather", "jit"):
        raise PairstackRefused(f"refused: {ENV_TRIATT_BIAS_KIT}={w!r}: one of auto | gather | jit")
    word = ("jit" if int(N) > TRIATT_BIAS_AUTO_N else "gather") if w == "auto" else w
    if TRIATT_BIAS_LAST[0] != (word, int(N)):
        TRIATT_BIAS_LAST[0] = (word, int(N))
        try:
            C.fn("evidence", "record_schedule")(triatt_bias_policy=f"{w}:{word}@N{int(N)}")
        except Exception:                                    # noqa: BLE001 - census only
            pass
    return word


def apb_rows(apb, a, z_shard, lay, s=None, mask=None, use_high_precision: bool = False):
    """``AttentionPairBias.forward(a, z, s, mask)`` (``attention_pair_bias.py:156-238``) with queries = this rank's rows and the pair bias from the
    shard; returns the REPLICATED output ``[1, N, C]`` (identical on every rank: rows all-gathered before ``_wrap_up``)."""
    apb_local_queries = C.fn("transition", "apb_local_queries")
    lever_rows = C.fn("triatt", "lever_rows")
    blocks4 = C.fn("triatt", "blocks4")
    attend_query_blocks = C.fn("triatt", "attend_query_blocks")
    core = _attention_core()
    a_ln = apb.layer_norm_a(a, s) if apb.use_ada_layer_norm else apb.layer_norm_a(a)   # AdaLN(a, s) in the DiT; plain LN in the Pairformer
    R, N = int(lay.R), int(lay.N)
    rb = lever_rows("APB_ROWBLOCK") or R
    zb = None                                                                          # bias rows [1, H, R, N] = permute(linear_z(layer_norm_z(z rows)))
    for i0, i1 in blocks4(R, rb):
        piece = apb.linear_z(apb.layer_norm_z(z_shard[i0:i1])).movedim(-1, 0)         # [H, rows, N]
        if zb is None:
            zb = piece.new_empty((1, int(piece.shape[0]), R, N))
        zb[0, :, i0:i1, :] = piece
        del piece
    mask_bias = None if mask is None else (apb.inf * (mask - 1))[..., None, None, :]   # [1, 1, 1, N]
    qblock = lever_rows("APB_QBLOCK")

    def attn_fn(q_rows_unused, s_full, z_unused):
        q, k, v = apb.mha._prep_qkv(a_ln, a_ln)                                        # FULL projections (M = N, as stock); q sliced after
        q_loc = q[..., lay.r0:lay.r1, :]
        biases = [b for b in (mask_bias, zb) if b is not None]
        o_loc = attend_query_blocks(lambda q_, k_, v_, b: core(q_, k_, v_, b, use_high_precision=use_high_precision), q_loc, k, v, biases, qblock)   # [1, H, R, d]
        del q, k, v, q_loc
        o_loc = o_loc.transpose(-2, -3)                                                # [1, R, H, d]
        return o_loc.reshape(tuple(o_loc.shape[:-2]) + (-1,)).contiguous()            # [1, R, H*d] rows (gathered along -2 by the core)

    o = apb_local_queries(attn_fn, a_ln, z_shard, lay, gather=True)                    # [1, N, H*d], replicated
    H = int(zb.shape[1])
    o = o.reshape(tuple(o.shape[:-1]) + (H, -1))                                       # [1, N, H, d]  == stock o.transpose(-2, -3)
    o = apb.mha._wrap_up(o, a_ln)                                                      # gating linear_g(a) + linear_o on the full tensor (M = N)
    if apb.use_ada_layer_norm:
        o = apb.sigmoid(apb.linear_ada_out(s)) * o
    return o


def pair_block_fns(blk, chunk_size, *, s_track=None, stats=None, triatt_bias=None):
    """``PairBlockFns`` of a stock ``PairBlock`` through the core's ``bind`` (also ``TemplatePairBlock``). ``s_track = (single_mask, inplace_safe)``
    adds the Pairformer block's single-track callables (``attn_pair_bias`` + ``single_transition`` of the enclosing ``PairFormerBlock``)."""
    bind = C.fn("pairstack", "bind")
    kw = dict(trimul_out=trimul_fns(blk.tri_mul_out), trimul_in=trimul_fns(blk.tri_mul_in), triatt_start=triatt_fns(blk.tri_att_start), triatt_end=triatt_fns(blk.tri_att_end),
              transition=transition_fn(blk.pair_transition), chunk=int(chunk_size), stream=pair_stream(),
              trimul_kw=dict(inplace_chunk=_stock_inplace_chunk_size(blk.tri_mul_out)), stats=stats)
    if triatt_bias is not None:
        kw["triatt_bias"] = triatt_bias                        # gather | jit (None = the core's ROWPAIR_TRIATT_BIAS, unset = gather)
    if not bool(getattr(blk, "tri_mul_first", True)):
        raise PairstackRefused("refused: PairBlock(tri_mul_first=False): the driver's statement order is tri-mul -> tri-att -> transition (0.4.1 predict configs use tri_mul_first)")
    return bind(**kw)


def pairformer_block_fns(pf, chunk_size, single_mask, inplace_safe=True, stats=None, triatt_bias=None):
    bind = C.fn("pairstack", "bind")
    blk = pf.pair_stack
    single_trans_mask = single_mask

    def apb(s, z_shard, lay):                               # pairformer.py:197  s = add(s, attn_pair_bias(a=s, z=z, s=None, mask=single_mask), inplace)
        return add(s, apb_rows(pf.attn_pair_bias, s, z_shard, lay, s=None, mask=single_mask), inplace=inplace_safe)

    def single_transition(s):                               # pairformer.py:212 (replicated)
        return add(s, pf.single_transition(s, mask=single_trans_mask, chunk_size=chunk_size), inplace=inplace_safe)

    return bind(trimul_out=trimul_fns(blk.tri_mul_out), trimul_in=trimul_fns(blk.tri_mul_in), triatt_start=triatt_fns(blk.tri_att_start), triatt_end=triatt_fns(blk.tri_att_end),
                transition=transition_fn(blk.pair_transition), chunk=int(chunk_size), apb=apb, single_transition=single_transition,
                trimul_kw=dict(inplace_chunk=_stock_inplace_chunk_size(blk.tri_mul_out)), stats=stats,
                **({} if triatt_bias is None else dict(triatt_bias=triatt_bias)))


# ----------------------------------------------------------------------------------------------------------------- forwards on shards
def pair_block_rows(blk, z_loc, pair_mask_loc, lay, chunk_size=None, inplace_safe=True, _mask_trans=True, _attn_chunk_size=None, **flags):
    """``PairBlock.forward`` on the shard -> rows ``r0:r1`` of the stock output (the caller's tensor, updated in place)."""
    refuse_kernel_flags("PairBlock", **flags)
    if not inplace_safe:
        raise PairstackRefused("refused: PairBlock on shards implements the inference path (inplace_safe=True) — the path `run_openfold predict` takes")
    z, rewrap = _squeeze_shard(z_loc)
    chunk = int(_attn_chunk_size or chunk_size or lay.R)
    z, _ = C.fn("pairstack", "pair_block_")(pair_block_fns(blk, chunk, triatt_bias=triatt_bias_policy(lay.N)), z, _squeeze_mask(pair_mask_loc), lay, transition_mask=bool(_mask_trans), reuse_storage=reuse_z_storage())
    return rewrap(z)


def pair_stack_rows(blocks, z_loc, pair_mask_loc, lay, chunk_size=None, inplace_safe=True, _mask_trans=True, _attn_chunk_size=None, **flags):
    """A stack of bare PairBlocks (template pair stack): ``pair_stack_`` over their ``PairBlockFns``; the shard is never gathered between blocks."""
    refuse_kernel_flags("PairStack", **flags)
    z, rewrap = _squeeze_shard(z_loc)
    chunk = int(_attn_chunk_size or chunk_size or lay.R)
    z, _ = C.fn("pairstack", "pair_stack_")([pair_block_fns(b, chunk, triatt_bias=triatt_bias_policy(lay.N)) for b in blocks], z, _squeeze_mask(pair_mask_loc), lay, transition_mask=bool(_mask_trans), reuse_storage=reuse_z_storage())
    return rewrap(z)


def pairformer_stack_rows(stack, s, z_loc, single_mask, pair_mask_loc, lay, chunk_size=None, inplace_safe=True, _mask_trans=True, _attn_chunk_size=None,
                          block_callback=None, **flags):
    """``PairFormerStack.forward`` (``pairformer.py:407-480``) -> ``(s, z_loc)``: the driver's ``pair_stack_`` over the blocks' ``PairBlockFns`` with
    the single track; s replicated and identical on all ranks. A live ``ChunkSizeTuner`` / grad-enabled checkpointing are refused by name."""
    refuse_kernel_flags("PairFormerStack", **flags)
    if getattr(stack, "chunk_size_tuner", None) is not None and chunk_size is not None:
        raise PairstackRefused("refused: PairFormerStack.tune_chunk_size=true under the tp line (runner yaml must pin tune_chunk_size=false in all five stacks; ranks would diverge)")
    if torch.is_grad_enabled() and getattr(stack, "blocks_per_ckpt", None) is not None:
        raise PairstackRefused("refused: PairFormerStack activation checkpointing (grad enabled) under the tp line — inference only")
    z, rewrap = _squeeze_shard(z_loc)
    chunk = int(_attn_chunk_size or chunk_size or lay.R)
    fns = [pairformer_block_fns(b, chunk, single_mask, inplace_safe=inplace_safe, triatt_bias=triatt_bias_policy(lay.N)) for b in stack.blocks]
    between = (lambda i: torch.cuda.empty_cache()) if getattr(stack, "clear_cache_between_blocks", False) and torch.cuda.is_available() else None
    z, s = C.fn("pairstack", "pair_stack_")(fns, z, _squeeze_mask(pair_mask_loc), lay, s=s, block_callback=block_callback, between_blocks=between,
                                            transition_mask=bool(_mask_trans), reuse_storage=reuse_z_storage())
    return s, rewrap(z)

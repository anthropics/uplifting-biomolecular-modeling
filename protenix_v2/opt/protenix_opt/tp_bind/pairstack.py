"""Protenix-v2 pair blocks on the shared core's pair-block driver (``opt_core.mem.rowpair.pairstack``): the trunk's 48 PairformerBlocks,
the MSA module's pair stack, the template embedder's PairformerStack (c_s = 0) and the confidence head's 4 blocks are ONE block shape —
triangle multiplication outgoing -> incoming -> triangle attention starting -> ending -> pair transition (+ attention-pair-bias and the
single transition when the block carries the single representation). This module packs Protenix's per-element statements into the
core's callables and calls the core's driver; the ring / banded-transpose schedules, the ending-attention orientation, the row blocking
and every collective are the core's (``trimul.trimul_update_``, ``triatt.triatt_update_``, ``pairstack.pair_block_``,
``transition.apb_local_queries``).

Per-element statements (Protenix ``protenix/model/triangular/triangular.py`` inference branch, ``modules/transformer.py`` AttentionPairBias):
    tri-mult   proj(z, m, is_a) = sigmoid(linear_{a|b}_g(LN_in(z))) * linear_{a|b}_p(LN_in(z)) * m ; out(x) = linear_z(LN_out(x)) ;
               gate(z) = sigmoid(linear_g(LN_in(z)))                                   (``TriMulFns``; C_h = c_hidden)
    tri-att    ln = layer_norm ; bias(x) = linear(x) [rows, N, H] ; attend(x, m, tb, _) = mha(q_x=x, kv_x=x, biases=[inf * (m - 1), tb^T])
               on the engine's own attention route (``triangle_attention``: torch | cuequivariance | ...)      (``TriAttFns``)
    transition fn(x_rows, _) = pair_transition(x_rows)                                 (Protenix's Transition takes no mask)
    apb        queries = this rank's rows of layernorm_a(s), keys / values = all of it, bias = linear_nobias_z(layernorm_z(z rows)) — the
               local pair rows ARE the query rows' bias rows, so no pair-shaped gather; the [R, c_s] output rows are all-gathered
Kernels on the row blocks (the single-GPU big line's kernel families on this rank's windows, bound by the core's row providers,
each with its named torch fallback COUNTED in the core's census lines this module prints at exit): the triangle multiplication's row
statements = the core's fused row TriMul (``opt_core.mem.rowpair.trimul_fused``: fpf_trimul_v4 K1 into the ring's bf16 planes + K3 tile
epilogue; gates N >= 2048, C in {128, 256}; ``ROWPAIR_TRIMUL_KERNELS=torch`` = the statements below) and the triangle attention's core per
query-row block = the core's ``triatt.attention_core`` with :data:`TP_TRIATT_CORE` (``tier:big``: the shared triangle-attention provider's
row for the tier word per window shape — triattn_native on cc 9.0, its cells elsewhere — else the flash_triattn path by name;
``ROWPAIR_TRIATT_CORE=<word>`` overrides) around the module's own ``_prep_qkv`` / ``_wrap_up``; a query block of <= 16 rows keeps the
module's own route (its small-input rule).
Replicated by design: ``s`` (the single representation), the gathered triangle bias ``[N, N, H=4]``. ``pair_mask`` is None in every
Protenix pair stack at inference (all-ones masks). Refusals by name: a pair mask, ``triangle_multiplicative`` other than ``torch`` (the
sharded triangle multiplication IS the torch statement; a kernel substitution is not a setting of the line), P = 1 / replicated layouts
(the core's structural rule — the single-GPU line never reaches this module).

Entry points keep the carried unit's call signatures so ``tp_route.ROUTES`` can name them (``ROUTE_ROWS``):
    tp_pairformer_block(block, s, z_shard, layout, **kw) -> (s, z_shard)      tp_pairformer_stack(stack, s, z_shard, layout, ...) -> (s, z_shard)
    tp_pair_stack_block(block, z_shard, layout, ...) -> z_shard               (c_s == 0 blocks: MSA / template pair stacks)
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from opt_core.mem.rowpair import RowpairRefused
from opt_core.mem.rowpair import pairstack as core_pairstack
from opt_core.mem.rowpair import transition as core_transition
from opt_core.mem.rowpair.dist import Layout
from opt_core.mem.rowpair.triatt import TriAttFns
from opt_core.mem.rowpair.trimul import TriMulFns

__all__ = ["trimul_fns", "trimul_weights", "triatt_fns", "triatt_core", "stock_attention_core", "transition_fn", "apb_fn", "block_fns", "tp_pairformer_block",
           "tp_pairformer_stack", "tp_pair_stack_block", "ROUTE_ROWS", "ENV_TRIATT_ROWS", "DEFAULT_TRIATT_ROWS", "TP_TRIATT_CORE", "SMALL_Q", "ENV_OP_TIMERS"]

ENV_TRIATT_ROWS = "ROWPAIR_TRIATT_ROWS"                  # rows per triangle-attention row batch when the caller passes no chunk_size
DEFAULT_TRIATT_ROWS = 128
TP_TRIATT_CORE = "tier:big"                           # the core attention_core kernel word of this line (ROWPAIR_TRIATT_CORE overrides inside the core)
ENV_TRIATT_STAGE = "ROWPAIR_TRIATT_STAGE"               # the core's tri-attention bias staging word (read by triatt.triatt_update_ per plane):
TRIATT_STAGE_DEFAULT = "once"                           #   this line states `once` (the kernel's bias operand staged ONCE per gathered plane per orientation, every row window
                                                        #   launching the kernel alone; bitwise the per_call outputs; census triatt_stage=once:triattn_native triatt_stage_planes=)
                                                        #   unless the caller exported the word (per_call = staged again at every row window's launch)
SMALL_Q = 16                                            # protenix triangular layers.Attention.forward: q rows <= 16 -> its own torch route
ENV_OP_TIMERS = "PTX_TP_OP_TIMERS"                      # =1: device-synchronised per-op seconds accumulate into the ``timers`` dict a caller passes (tp_bind.trunk prints them)
TAG = "protenix-opt"
_EXIT = {"armed": False}
SCHEDULE_CENSUS_PREFIXES = ("triatt_", "trimul_", "host_slab", "rank_threads", "transpose_")   # the core schedule words the exit line `TP-PAIRSTACK schedule …` reports
# the tp_route.ROUTES rows this module serves (carried module -> names): applied by tp_route
ROUTE_ROWS = {"ptx_tp.pairformer": ("protenix_opt.tp_bind.pairstack", ("tp_pairformer_block", "tp_pairformer_stack")),
              "ptx_tp.msa": ("protenix_opt.tp_bind.pairstack", ("tp_pair_stack_block",))}


# ----------------------------------------------------------------------------------------------------------- callables ----
def trimul_fns(mod) -> TriMulFns:
    """``TriMulFns`` of a Protenix ``TriangleMultiplicativeUpdate`` (outgoing or incoming: the driver picks the contraction)."""
    def proj(z_block, mask_block, is_a: bool):
        linear_g, linear_p = (mod.linear_a_g, mod.linear_a_p) if is_a else (mod.linear_b_g, mod.linear_b_p)
        pair = mod.layer_norm_in(z_block)
        p = linear_g(pair)
        p.sigmoid_()
        p *= linear_p(pair)
        p *= mask_block
        return p

    def out(x):
        return mod.linear_z(mod.layer_norm_out(x))

    def gate(z_block):
        g = mod.linear_g(mod.layer_norm_in(z_block))
        g.sigmoid_()
        return g

    fns = TriMulFns(proj, out, gate, C_h=int(mod.c_hidden))
    RF = _trimul_rows_provider()
    if RF is None:                                                                           # ROWPAIR_TRIMUL_KERNELS=torch: the statements above on the ring
        return fns
    return RF.fused_trimul_fns(trimul_weights(mod), fns, eps=float(getattr(mod.layer_norm_in, "eps", 1e-5)), cells=None, ledger=None)


def trimul_weights(mod) -> dict:
    """A Protenix ``TriangleMultiplicativeUpdate``'s tensors in the core's TriMul vocabulary (``opt_core.trimul_weights.WEIGHT_KEYS``; no biases)."""
    return dict(ln_in_w=mod.layer_norm_in.weight, ln_in_b=mod.layer_norm_in.bias, w_ag=mod.linear_a_g.weight, w_ap=mod.linear_a_p.weight,
                w_bg=mod.linear_b_g.weight, w_bp=mod.linear_b_p.weight, ln_out_w=mod.layer_norm_out.weight, ln_out_b=mod.layer_norm_out.bias,
                w_o=mod.linear_z.weight, w_og=mod.linear_g.weight)


def _trimul_rows_provider():
    """The core's fused row TriMul provider module, or None under ``ROWPAIR_TRIMUL_KERNELS=torch`` (the engineering opt-out; census then reads
    the driver's torch statements). The provider decides per launch by its documented gates (N >= 2048, C in {128, 256}, dtype) and COUNTS
    every decline; where the carried kernels cannot run at all it raises RowpairRefused by name."""
    from opt_core.mem.rowpair import trimul_fused as RF
    word = (os.environ.get(RF.ENV_KERNELS) or "fpf_v4").strip().lower()
    if word not in RF.KERNEL_WORDS:
        raise RowpairRefused(f"tp_bind.pairstack: {RF.ENV_KERNELS}={word!r}: one of {' | '.join(RF.KERNEL_WORDS)}")
    _arm_exit_lines()
    return None if word == "torch" else RF


def stock_attention_core(mha, kernel: str):
    """The module's own attention core after ``_prep_qkv`` as the core's ``stock(q, k, v, biases) -> o`` fallback for a declined row block
    (``layers.Attention.forward``'s dispatch for the run's ``triangle_attention``): ``cuequivariance`` = the cuEquivariance kernel with the
    module's scale, the fp32 triangle bias and the boolean key mask; ``torch`` (and every other engine word on rows) = q / sqrt(c_hidden) then
    ``layers._attention``. Unscaled q in (the KERNEL applies ``D ** -0.5``; the stock callable scales itself); the result in q's shape."""
    import math
    from protenix.model.triangular import layers as LY
    root = math.sqrt(mha.c_hidden)
    if kernel == "cuequivariance":
        def stock(q, k, v, biases):
            return LY.cuequivariance_triangular_attn(q[0], k[0], v[0], biases[1][0].float(), (biases[0][0] == 0).bool(), 1.0 / root).reshape(q.shape)
    else:
        def stock(q, k, v, biases):
            return LY._attention(q[0] / root, k[0], v[0], [b[0] for b in biases]).reshape(q.shape)
    return stock


def triatt_core(mod, triangle_attention: str):
    """The core's per-query-row-block attention core for this ``TriangleAttention``: :data:`TP_TRIATT_CORE` through
    ``opt_core.mem.rowpair.triatt.attention_core`` (q/k/v ``[1, rows, H, N, D]`` = ``bnhsd``, the key mask read off ``biases[0]``, the triangle
    bias = ``biases[1]``, scale None = the module's ``1/sqrt(c_hidden)``), its named fallback = :func:`stock_attention_core`."""
    from opt_core.mem.rowpair import triatt as RA
    _arm_exit_lines()
    return RA.attention_core(stock_attention_core(mod.mha, triangle_attention), kernel=TP_TRIATT_CORE, min_tokens=0, ledger=None, scale=None,
                             layout="bnhsd", mask_from="bias0", tri_bias="bias1")


def triatt_fns(mod, triangle_attention: str = "torch") -> TriAttFns:
    """``TriAttFns`` of a Protenix ``TriangleAttention`` (the starting-node statement; the driver hands the ending node rows of z^T): the
    module's LayerNorm and bias projection, and ``attend`` = its ``mha``'s own ``_prep_qkv`` -> the core's row-block attention core
    (:func:`triatt_core`) -> its own ``_wrap_up`` (gating by the query rows, head merge, output projection); a block of <= :data:`SMALL_Q`
    query rows takes the module's whole ``mha`` call (its small-input rule)."""
    core = triatt_core(mod, triangle_attention)
    mha = mod.mha

    def attend(x_rows, mask_rows, tb_full, _rows):
        m = mask_rows if mask_rows is not None else x_rows.new_ones(x_rows.shape[:-1])
        mask_bias = (mod.inf * (m - 1))[..., :, None, None, :]                              # [rows, 1, 1, N]
        triangle_bias = tb_full.permute(2, 0, 1).unsqueeze(-4)                              # [1, H, N, N] (permute_final_dims(tb, (2, 0, 1)))
        if int(x_rows.shape[-2]) <= SMALL_Q:
            return mha(q_x=x_rows, kv_x=x_rows, biases=[mask_bias, triangle_bias], triangle_attention=triangle_attention)
        q, k, v = mha._prep_qkv(x_rows, x_rows, apply_scale=False)                          # [rows, H, N, D] each, q unscaled (the core scales)
        o = core(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), [mask_bias.unsqueeze(0), triangle_bias.unsqueeze(0)])
        return mha._wrap_up(o.squeeze(0).transpose(-2, -3), x_rows)                        # [rows, N, H, D] -> gate, merge heads, linear_o

    return TriAttFns(ln=mod.layer_norm, bias=mod.linear, attend=attend)


def _arm_exit_lines() -> None:
    """Print the core's row-kernel census lines once per rank at exit (``F2.trimul_rows`` — served launches k1/k3, declines by reason — and the
    triangle-attention core line ``F1.…`` — served query blocks by kernel word, fallbacks by name)."""
    if _EXIT["armed"]:
        return
    _EXIT["armed"] = True
    import atexit

    def _emit():
        try:
            from opt_core.mem.rowpair import triatt as RA, trimul_fused as RF, evidence as EV
            sched = EV.schedule()
            words = " ".join(f"{k}={sched[k]}" for k in sorted(sched) if k.startswith(SCHEDULE_CENSUS_PREFIXES))
            os.write(2, (RF.emit_line(TAG) + "\n" + RA.emit_core_line(TAG) + "\n" + f"[{TAG}] TP-PAIRSTACK schedule {words or 'none'}\n").encode())
        except Exception as e:                                                              # census only; never masks the run's own exit
            os.write(2, f"[{TAG}] tp_bind.pairstack: census lines unavailable ({type(e).__name__}: {e})\n".encode())
    atexit.register(_emit)


def _timed(fns: "core_pairstack.PairBlockFns", timers: dict, mem: Optional[dict]) -> "core_pairstack.PairBlockFns":
    """``fns`` with every op wrapped in a device-synchronised stopwatch accumulating into ``timers[op]`` (and ``mem[op]`` = the allocator
    high-water mark after the op) — the carried block's ``timers`` / ``mem`` contract, used under :data:`ENV_OP_TIMERS` only."""
    import time
    import torch

    def wrap(name, fn):
        if fn is None:
            return None

        def timed(*a, **k):
            torch.cuda.synchronize(); t0 = time.time()
            out = fn(*a, **k)
            torch.cuda.synchronize(); timers[name] = timers.get(name, 0.0) + (time.time() - t0)
            if mem is not None:
                mem[name] = max(mem.get(name, 0), torch.cuda.max_memory_allocated())
            return out
        return timed

    return core_pairstack.PairBlockFns(trimul_out=wrap("tri_mul_out", fns.trimul_out), trimul_in=wrap("tri_mul_in", fns.trimul_in),
                                       triatt_start=wrap("tri_att_start", fns.triatt_start), triatt_end=wrap("tri_att_end", fns.triatt_end),
                                       transition=wrap("pair_transition", fns.transition), apb=wrap("attention_pair_bias", fns.apb),
                                       single_transition=wrap("single_transition", fns.single_transition))


def transition_fn(mod) -> Callable:
    """The pair transition as ``fn(x_rows, mask_u_rows) -> delta`` (Protenix's Transition is unmasked)."""
    return lambda x_rows, _mask_u: mod(x_rows)


def apb_fn(apb) -> Callable:
    """``fn(s, z_shard, layout) -> s + AttentionPairBias(a=s, z=z)`` with the queries = this rank's token rows and the bias = its pair rows."""
    if getattr(apb, "has_s", False) or getattr(apb, "cross_attention_mode", False):
        raise RowpairRefused("tp_bind.apb: the Pairformer's single attention has has_s=False and no cross attention")

    def attn(s_rows, s_full, z_rows):
        import math
        from protenix.model.modules.primitives import _attention                            # the engine's own attention statement
        att = apb.attention
        q_rows = apb.layernorm_a(s_rows)                                                     # [R, c_s]: this rank's token rows (LayerNorm is per row)
        a = apb.layernorm_a(s_full)                                                          # [N, c_s] (replicated; keys / values read all of it)
        bias = apb.linear_nobias_z(apb.layernorm_z(z_rows)).permute(2, 0, 1)                # [h, R, N]: the query rows' own bias rows
        q = att.linear_q(q_rows); k = att.linear_k(a); v = att.linear_v(a)
        q = q.view(q.shape[:-1] + (att.num_heads, -1)).transpose(-2, -3) / math.sqrt(att.c_hidden)
        k = k.view(k.shape[:-1] + (att.num_heads, -1)).transpose(-2, -3)
        v = v.view(v.shape[:-1] + (att.num_heads, -1)).transpose(-2, -3)
        o = _attention(q=q, k=k, v=v, attn_bias=bias, use_efficient_implementation=att.use_efficient_implementation, inplace_safe=False)
        return att._wrap_up(o.transpose(-2, -3), q_rows)                                    # [R, c_s]

    def fn(s, z_shard, layout: Layout):
        return s + core_transition.apb_local_queries(attn, s, z_shard, layout, gather=True)

    return fn


def block_fns(block, *, triangle_attention: str = "torch", chunk: Optional[int] = None, stats: Optional[dict] = None,
              timers: Optional[dict] = None, mem: Optional[dict] = None) -> core_pairstack.PairBlockFns:
    """``PairBlockFns`` of one Protenix pair block (``PairformerBlock``; c_s > 0 adds the single track) bound to the core's streamed schedules;
    ``timers`` / ``mem`` (dicts, under ``PTX_TP_OP_TIMERS=1``): per-op device-synchronised seconds / allocator high-water marks accumulate there."""
    from protenix_opt.tp import INPLACE_CHUNK                                            # the engine's in-place column chunk (256): the ring's global row grid
    if not (os.environ.get(ENV_TRIATT_STAGE) or "").strip():
        os.environ[ENV_TRIATT_STAGE] = TRIATT_STAGE_DEFAULT                              # the line's staging word (a caller's export wins)
    rows = int(chunk or os.environ.get(ENV_TRIATT_ROWS, "") or DEFAULT_TRIATT_ROWS)
    single = int(getattr(block, "c_s", 0) or 0) > 0
    fns = core_pairstack.bind(
        trimul_out=trimul_fns(block.tri_mul_out), trimul_in=trimul_fns(block.tri_mul_in),
        triatt_start=triatt_fns(block.tri_att_start, triangle_attention), triatt_end=triatt_fns(block.tri_att_end, triangle_attention),
        transition=transition_fn(block.pair_transition), chunk=rows, trimul_kw={"inplace_chunk": int(INPLACE_CHUNK)},
        apb=apb_fn(block.attention_pair_bias) if single else None,
        single_transition=(lambda s: s + block.single_transition(s)) if single else None,
        stats=stats)
    if timers is not None and os.environ.get(ENV_OP_TIMERS, "") == "1":
        fns = _timed(fns, timers, mem)
    return fns


def _refusals(pair_mask, triangle_multiplicative: str, what: str) -> None:
    if pair_mask is not None:
        raise RowpairRefused(f"{what}: pair_mask must be None (Protenix pair stacks carry no pair mask at inference)")
    if triangle_multiplicative != "torch":
        raise RowpairRefused(f"{what}: triangle_multiplicative={triangle_multiplicative}: the row-sharded triangle multiplication binds the module's torch "
               f"statements (served by the core's fused row kernels where they apply); run with --trimul_kernel torch — an engine kernel word is not a setting of the line")


# --------------------------------------------------------------------------------------------------------- entry points ----
def tp_pairformer_block(block, s, z_shard, layout: Layout, *, pair_mask=None, triangle_multiplicative: str = "torch", triangle_attention: str = "torch",
                        inplace_safe: bool = True, chunk_size: Optional[int] = None, shard_single_queries: bool = True,
                        trimul_kwargs: Optional[dict] = None, triatt_kwargs: Optional[dict] = None, timers: Optional[dict] = None,
                        mem: Optional[dict] = None, stats: Optional[dict] = None):
    """One PairformerBlock on this rank's pair rows -> ``(s, z_shard)``; ``inplace_safe=False`` updates a clone (same arithmetic). The single
    attention always runs with local query rows (``shard_single_queries`` is accepted for the carried signature; the replicated-queries form
    with an all-gathered ``[N, N, h]`` bias is not a setting of this binding)."""
    _refusals(pair_mask, triangle_multiplicative, "tp_pairformer_block")
    z = z_shard if inplace_safe else z_shard.clone()
    fns = block_fns(block, triangle_attention=triangle_attention, chunk=chunk_size, stats=stats, timers=timers, mem=mem)
    z, s = core_pairstack.pair_block_(fns, z, None, layout, s=s, transition_mask=False)
    return s, z


def tp_pairformer_stack(stack, s, z_shard, layout: Layout, *, block_callback: Optional[Callable] = None, pair_mask=None,
                        triangle_multiplicative: str = "torch", triangle_attention: str = "torch", inplace_safe: bool = True,
                        chunk_size: Optional[int] = None, stats: Optional[dict] = None, timers: Optional[dict] = None, mem: Optional[dict] = None, **_kw):
    """``stack.blocks`` in sequence on this rank's rows (the shard is never gathered between blocks) -> ``(s, z_shard)``."""
    _refusals(pair_mask, triangle_multiplicative, "tp_pairformer_stack")
    z = z_shard if inplace_safe else z_shard.clone()
    blocks = [block_fns(b, triangle_attention=triangle_attention, chunk=chunk_size, stats=stats, timers=timers, mem=mem) for b in stack.blocks]
    z, s = core_pairstack.pair_stack_(blocks, z, None, layout, s=s, block_callback=block_callback, transition_mask=False)
    return s, z


def tp_pair_stack_block(block, z_shard, layout: Layout, *, triangle_multiplicative: str = "torch", triangle_attention: str = "torch",
                        inplace_safe: bool = False, chunk_size: Optional[int] = None, stats: Optional[dict] = None):
    """One c_s == 0 pair block (the MSA module's and the template embedder's pair stacks) on this rank's rows -> ``z_shard``."""
    _refusals(None, triangle_multiplicative, "tp_pair_stack_block")
    if int(getattr(block, "c_s", 0) or 0) != 0:
        raise RowpairRefused("tp_pair_stack_block: the block carries a single representation (c_s > 0): use tp_pairformer_block")
    z = z_shard if inplace_safe else z_shard.clone()
    fns = block_fns(block, triangle_attention=triangle_attention, chunk=chunk_size, stats=stats)
    z, _ = core_pairstack.pair_block_(fns, z, None, layout, s=None, transition_mask=False)
    return z

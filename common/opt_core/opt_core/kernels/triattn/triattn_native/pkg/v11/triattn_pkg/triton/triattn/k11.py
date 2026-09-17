"""k11 -- triangle-attention forward for H100 written in Gluon (Triton's explicit-layout dialect): TMA + mbarrier ring,
async wgmma with cross-row software pipelining, warp specialization (2 consumer warpgroups x 64 query rows sharing the
K/V/bias tiles of one CTA + 1 TMA producer warp), bias tile shared by ROWS pair rows per CTA.

    out[b,i,h,q,:] = softmax_k( scale * q[b,i,h,q,:] . k[b,i,h,k,:] + bias[b,0,h,q,k] (+ -1e9 where mask[b,i,0,0,k]==0) ) @ v[b,i,h,k,:]

q, k, v [B, N, H, S, D] bf16/fp16 (made contiguous if they are not), bias [B, 1, H, S_q, S_k] fp32 / bf16 / fp16 (staged once per
call as 16-bit [.., S_q, S_k pad 16] tiles; an fp32 bias that is not exactly representable in q.dtype is rounded -- `info["bias_rounded"]`),
mask [B, N, 1, 1, S_k] bool or None: the batch element's OR pattern of kept keys is folded into the staged bias tile (masking.py),
rows equal to that pattern (all rows, for padding masks) run mask-free and skip the key tiles past the last kept key, any other
row (shorter prefix, interior zeros, fully masked) applies the per-key select from its first differing key tile on.  Numerics: bf16 tensor-core products with fp32 accumulation, base-2 online softmax in fp32
(scale*log2e folded into the QK^T scale, log2e into the bias tile), P rounded to v.dtype for the PV product, -1e9 select for masked
keys (a fully-masked row yields the uniform average of v, as cuEquivariance does).  No atomics, fixed reduction order.
The kernel source is generated per ROWS (Gluon, like Triton, has no arrays of register tensors): see _render().
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from .errors import Unsupported  # noqa: F401  (re-exported)
from .launch import launch
from .masking import census, mask_tables


_LOG2E = 1.4426950408889634
_GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen")

_SRC = '''
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, fence_async_shared, warpgroup_mma, warpgroup_mma_wait, warpgroup_mma_init,
)

LOG2E = gl.constexpr(1.4426950408889634)
MSEL = gl.constexpr(MSEL_LIT)          # per-key select level for masked keys (rendered per dtype: -1e9 bf16, -32768 fp16)
BM = gl.constexpr(128)           # query rows per CTA (two consumer warpgroups x 64)
ROWS = gl.constexpr(ROWS_LIT)


@gluon.jit
def _keep(x):
    """Zero-instruction use of x (tied in/out registers): extends x's register lifetime to this point, so ptxas sees no
    write-after-read hazard between the asynchronous wgmma still reading x and later values that would reuse its registers."""
    return gl.inline_asm_elementwise("", "=r,0", [x], dtype=x.dtype, is_pure=False, pack=2)


@gluon.jit
def _phase_a(acc, m_i, l_i, s, b32, mrow, m_lo, offs_n, kvm, qk_scale, USE_MASK: gl.constexpr, O_L: gl.constexpr):
    """Row phase A (FMA pipe): logits t = scale*s + bias (natural units; keys masked for every row already carry -1e9 in the
    bias tile, and with USE_MASK this row's own mask is applied per key), running max, rescale of the running output/sum by
    alpha = 2^((m_old - m_new) log2 e).  Returns t and nml = -m_new*log2e for phase B."""
    t = s * qk_scale + b32
    if USE_MASK:
        mk = gl.load(mrow + m_lo + offs_n, mask=kvm, other=1)
        t = gl.where(gl.expand_dims(mk, 0) != 0, t, MSEL)
    m_new = gl.maximum(m_i, gl.max(t, axis=1))
    alpha = gl.exp2((m_i - m_new) * LOG2E)
    nml = m_new * (-LOG2E)
    alpha_o = gl.convert_layout(alpha, gl.SliceLayout(1, O_L), assert_trivial=True)
    acc = acc * gl.expand_dims(alpha_o, 1)
    return acc, m_new, l_i * alpha, t, nml


@gluon.jit
def _phase_b(acc, l_i, t, nml, v_view, P_L: gl.constexpr):
    """Row phase B (MUFU pipe): p = 2^(t log2e - m log2e), row-sum, P -> bf16, asynchronous PV wgmma.  Returns the fp32 P
    (dead: register donor for the next QK^T accumulator) and the bf16 P (must stay live until the PV wgmma retires)."""
    p = gl.exp2(t * LOG2E + gl.expand_dims(nml, 1))
    l_new = l_i + gl.sum(p, axis=1)
    p16 = gl.convert_layout(p.to(v_view.dtype), P_L, assert_trivial=True)
    acc = warpgroup_mma(p16, v_view, acc, is_async=True)
    return acc, l_new, p16, p


@gluon.jit
def _producer(k_desc, v_desc, b_desc, k_smem, v_smem, b_smem, ready, empty,
              XARGS, bias_row, n_tiles,
              BLOCK_N: gl.constexpr, STAGES: gl.constexpr):
    """TMA producer (default partition): per key tile, ROWS K tiles + ROWS V tiles + one [128, BLOCK_N] bias tile into stage n % STAGES."""
    KV_BYTES: gl.constexpr = k_desc.block_type.nbytes
    B_BYTES: gl.constexpr = b_desc.block_type.nbytes
    st = 0
    eph = 1          # parity to wait for on `empty[st]` before refilling stage st (first lap: passes without waiting via pred)
    for n in range(n_tiles):
        mbarrier.wait(empty.index(st), eph, pred=n >= STAGES)
        rb = ready.index(st)
        mbarrier.expect(rb, 2 * ROWS * KV_BYTES + B_BYTES)
        n0 = n * BLOCK_N
PRODUCER_LOADS
        tma.async_copy_global_to_shared(b_desc, [bias_row, n0], rb, b_smem.index(st))
        st += 1
        if st == STAGES:
            st = 0
            eph ^= 1


@gluon.jit
def _consumer(q_smem, k_smem, v_smem, b_smem, ready, empty, tok_mine, tok_other, Out, Mask,
              soi, soq, smi,
              out_bh, mask_b, IARGS, r0, q0, N_ROWS, SEQ_Q, SEQ_K, n_tiles, qk_scale, n_sel,
              BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, STAGES: gl.constexpr,
              HAS_MASK: gl.constexpr, PREFETCH: gl.constexpr, WG: gl.constexpr, SPLIT: gl.constexpr, SWAP: gl.constexpr,
              PINGPONG: gl.constexpr):
    """The compute partition (SPLIT=0: one 8-warp partition = two warpgroups owning 64 query rows each, one code stream;
    SPLIT=1: one of two 4-warp partitions, rows WG*64..+64).  Software pipeline over the ROWS pair rows of a key tile, one
    "step" per row j:  wait QK^T_j | issue QK^T_{j+1} (tensor core runs a row ahead) | phase A of row j (FMA pipe: logits,
    max, rescale) | phase B of row j-1 (MUFU pipe: exponentials, row-sum, PV wgmma) -- so every warp always has independent
    FMA-pipe and MUFU-pipe work in flight.  Register hygiene for ptxas: the accumulator registers of each use_acc=False
    QK^T wgmma are donated by a dead fp32 P tile (no zero-fill), and each bf16 P operand is kept live (_keep) until its PV
    wgmma has retired, so consecutive PV wgmmas never share A registers (no false write-after-read waits)."""
    NW: gl.constexpr = gl.num_warps()
    PM: gl.constexpr = 64 if SPLIT else 128
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, BLOCK_N, 16])
    O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, HEAD_DIM, 16])
    P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
    ROW_L: gl.constexpr = gl.SliceLayout(1, S_L)
    COL_L: gl.constexpr = gl.SliceLayout(0, S_L)
    OROW_L: gl.constexpr = gl.SliceLayout(1, O_L)
    OCOL_L: gl.constexpr = gl.SliceLayout(0, O_L)
    dt: gl.constexpr = v_smem.dtype

    offs_n = gl.arange(0, BLOCK_N, layout=COL_L)
    pd = gl.zeros([PM, BLOCK_N], gl.float32, S_L)          # dead fp32 P tile: register donor for the next QK^T accumulator
    p16a = gl.zeros([PM, BLOCK_N], dt, P_L)                # P operands of the two most recent PV wgmmas (kept live until retired)
    p16b = gl.zeros([PM, BLOCK_N], dt, P_L)
    t_pv = gl.zeros([PM, BLOCK_N], gl.float32, S_L)        # phase-A output of the previous row, consumed by its phase B
    nml_pv = gl.zeros([PM], gl.float32, ROW_L)
CONSUMER_INIT
    # prologue: QK^T of row 0, tile 0
    mbarrier.wait(ready.index(0), 0)
    s_nx = warpgroup_mma(qs0, k_smem.index(0).permute((1, 0)), pd, use_acc=False, is_async=True)
    st = 0
    ph = 0
    tph = 0          # parity of this warpgroup's MUFU-phase token (PINGPONG)
    # key tiles [0, n_sel): no row of this CTA differs from the OR pattern folded into the bias -> mask-free body;
    # key tiles [n_sel, n_tiles): per-key select with each row's own mask (n_sel == n_tiles without a mask)
    for n in range(0, n_sel):
        # stage / phase of tile n+1, stage of tile n-1
        st1 = st + 1
        ph1 = ph
        if st1 == STAGES:
            st1 = 0
            ph1 = ph ^ 1
        if STAGES == 2:
            stp = st1
        else:
            stp = (st + STAGES - 1) % STAGES
        m_lo = n * BLOCK_N
        if SPLIT:
            b32 = b_smem.index(st).slice(WG * 64, 64).load(S_L).to(gl.float32)
        else:
            b32 = b_smem.index(st).load(S_L).to(gl.float32)
BODY_PLAIN
        st = st1
        ph = ph1
    if HAS_MASK:
        for n in range(n_sel, n_tiles):
            st1 = st + 1
            ph1 = ph
            if st1 == STAGES:
                st1 = 0
                ph1 = ph ^ 1
            if STAGES == 2:
                stp = st1
            else:
                stp = (st + STAGES - 1) % STAGES
            m_lo = n * BLOCK_N
            if SPLIT:
                b32 = b_smem.index(st).slice(WG * 64, 64).load(S_L).to(gl.float32)
            else:
                b32 = b_smem.index(st).load(S_L).to(gl.float32)
BODY_MASKED
            st = st1
            ph = ph1
    # last row's phase B, drain, normalise, store
CONSUMER_EPILOGUE


@gluon.jit
def _fwd(q_desc, k_desc, v_desc, b_desc, Out, Mask, RowInfo,
         sob, soi, soh, soq, smb, smi, srb, sri,
         N_ROWS, SEQ_Q, SEQ_K, H, qk_scale2,
         BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, STAGES: gl.constexpr, HAS_MASK: gl.constexpr,
         REGS_CONS: gl.constexpr, PREFETCH: gl.constexpr, SPLIT: gl.constexpr, SWAP: gl.constexpr, PINGPONG: gl.constexpr,
         VAR_TILES: gl.constexpr):
    pid_q = gl.program_id(0)
    pid_r = gl.program_id(1)
    pid_bh = gl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H
    q0 = pid_q * BM
    r0 = pid_r * ROWS
    n_tiles = gl.cdiv(SEQ_K, BLOCK_N)

    q_smem = gl.allocate_shared_memory(q_desc.dtype, [ROWS, BM, HEAD_DIM], q_desc.layout)
    k_smem = gl.allocate_shared_memory(k_desc.dtype, [STAGES * ROWS, BLOCK_N, HEAD_DIM], k_desc.layout)
    v_smem = gl.allocate_shared_memory(v_desc.dtype, [STAGES * ROWS, BLOCK_N, HEAD_DIM], v_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, [STAGES, BM, BLOCK_N], b_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    q_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    tok = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())     # PINGPONG turn tokens
    for s in gl.static_range(STAGES):
        mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=2 if SPLIT else 1)
    mbarrier.init(q_bar, count=1)
    mbarrier.init(tok.index(0), count=1)
    mbarrier.init(tok.index(1), count=1)
    fence_async_shared()

    # pair rows of this CTA (clamped duplicates past N_ROWS are computed and not stored) -> flattened (b, i, h) row index X
FWD_ROWS
    bias_row = pid_bh * SEQ_Q + q0
    out_bh = Out + b.to(gl.int64) * sob + h.to(gl.int64) * soh
    mask_b = Mask + b.to(gl.int64) * smb
    n_sel = n_tiles
    if VAR_TILES:
        # per-row (or per-batch, sri == 0) key-tile table: [tiles to visit, first tile needing the per-key select]; max / min over rows
FWD_ROWINFO
    # Q tiles (once)
    mbarrier.expect(q_bar, ROWS * q_desc.block_type.nbytes)
FWD_QLOADS
    mbarrier.wait(q_bar, 0)
    mbarrier.arrive(tok.index(0), count=1)          # warpgroup 0 holds the first MUFU turn
    mbarrier.invalidate(q_bar)

    if SPLIT:
        gl.warp_specialize([
            (_producer, (k_desc, v_desc, b_desc, k_smem, v_smem, b_smem, ready, empty, XARGS, bias_row, n_tiles, BLOCK_N, STAGES)),
            (_consumer, (q_smem, k_smem, v_smem, b_smem, ready, empty, tok.index(0), tok.index(1), Out, Mask, soi, soq, smi,
                         out_bh, mask_b, IARGS, r0, q0, N_ROWS, SEQ_Q, SEQ_K, n_tiles, qk_scale2, n_sel,
                         BLOCK_N, HEAD_DIM, STAGES, HAS_MASK, PREFETCH, 0, 1, 0, PINGPONG)),
            (_consumer, (q_smem, k_smem, v_smem, b_smem, ready, empty, tok.index(1), tok.index(0), Out, Mask, soi, soq, smi,
                         out_bh, mask_b, IARGS, r0, q0, N_ROWS, SEQ_Q, SEQ_K, n_tiles, qk_scale2, n_sel,
                         BLOCK_N, HEAD_DIM, STAGES, HAS_MASK, PREFETCH, 1, 1, SWAP, PINGPONG)),
        ], [4, 4], [REGS_CONS, REGS_CONS])
    else:
        gl.warp_specialize([
            (_producer, (k_desc, v_desc, b_desc, k_smem, v_smem, b_smem, ready, empty, XARGS, bias_row, n_tiles, BLOCK_N, STAGES)),
            (_consumer, (q_smem, k_smem, v_smem, b_smem, ready, empty, tok.index(0), tok.index(1), Out, Mask, soi, soq, smi,
                         out_bh, mask_b, IARGS, r0, q0, N_ROWS, SEQ_Q, SEQ_K, n_tiles, qk_scale2, n_sel,
                         BLOCK_N, HEAD_DIM, STAGES, HAS_MASK, PREFETCH, 0, 0, 0, 0)),
        ], [8], [REGS_CONS])
    mbarrier.invalidate(tok.index(0))
    mbarrier.invalidate(tok.index(1))
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(ready.index(s))
        mbarrier.invalidate(empty.index(s))
'''


# --------------------------------------------------------------------------------------------------------------------------
# codegen shared by k11 (one work item per CTA) and k12 (persistent work loop)
def gen_body(rows: int, indent: str, use_mask: str = "False", persistent: bool = False) -> str:
    """One key tile of the consumer: ROWS software-pipelined row steps (see _consumer.__doc__).  persistent (k12): the QK^T of
    the next item's first tile is issued in the epilogue instead of a dummy one in the last row step of the last tile."""
    if rows < 2:
        raise ValueError("ROWS >= 2 required (the row software pipeline overlaps phase A of row j with phase B of row j-1)")
    R = range(rows)
    # The K/V/bias stage of tile n-1 is free once PV_{ROWS-1}(n-1) -- issued in step j=0 of tile n -- has retired, i.e. after
    # the QK^T wait of flat step n*ROWS+2: j == 2 of tile n (ROWS >= 3), or j == 0 of tile n+1 (ROWS == 2: release tile n-2).
    if rows >= 3:
        rel_j, rel_pred, rel_stage = 2, "n > 0", "stp"
    else:
        rel_j, rel_pred, rel_stage = 0, "n > 1", "(st + STAGES - 2) % STAGES"
    o = []
    w = o.append
    for j in R:
        # --- wait QK^T_j(n): everything older retires too (incl. PV of row j-2); only the newest PV may stay in flight
        w(f"{indent}s_cu = s_nx")
        w(f"{indent}s_cu, acc{j} = warpgroup_mma_wait(num_outstanding=1, deps=[s_cu, acc{j}])")
        w(f"{indent}p16a = _keep(p16a)")
        if j == rel_j:
            w(f"{indent}# every wgmma reading the oldest live K/V/bias stage has retired -> release it (one elected arrive)")
            w(f"{indent}mbarrier.arrive(empty.index({rel_stage}), count=1, pred={rel_pred})")
        # --- issue the next QK^T (row j+1, or row 0 of tile n+1) into the registers of a dead fp32 P tile
        if j + 1 < rows:
            w(f"{indent}s_nx = warpgroup_mma(qs{j+1}, k_smem.index(st * ROWS + {j+1}).permute((1, 0)), pd, use_acc=False, is_async=True)")
        elif not persistent:
            w(f"{indent}# (for the last tile this reads a stale stage; drained after the loop, never used)")
            w(f"{indent}mbarrier.wait(ready.index(st1), ph1, pred=(n + 1) < n_tiles)")
            w(f"{indent}s_nx = warpgroup_mma(qs0, k_smem.index(st1 * ROWS).permute((1, 0)), pd, use_acc=False, is_async=True)")
        else:
            w(f"{indent}if (n + 1) < n_tiles:")
            w(f"{indent}    mbarrier.wait(ready.index(st1), ph1)")
            w(f"{indent}    s_nx = warpgroup_mma(qs0, k_smem.index(st1 * ROWS).permute((1, 0)), pd, use_acc=False, is_async=True)")
            w(f"{indent}else:")
            w(f"{indent}    s_nx = warpgroup_mma_init(pd)          # placeholder token; the next item's first QK^T is issued in the epilogue")
        # --- phase A of row j (tile n) and phase B of row j-1 (or of row ROWS-1 of tile n-1 when j == 0), in either order
        jb = j - 1 if j >= 1 else rows - 1
        vst = "st" if j >= 1 else "stp"
        mrow = f"mrow{j}" if use_mask == "True" else "mask_b"
        kvok = "kv_ok" if use_mask == "True" else "offs_n"
        line_a = (f"acc{j}, m{j}, l{j}, t_new, nml_new = _phase_a(acc{j}, m{j}, l{j}, s_cu, b32, {mrow}, m_lo, offs_n, {kvok}, qk_scale, "
                  f"USE_MASK={use_mask}, O_L=O_L)")
        lines_b = [f"acc{jb}, l{jb}, p16n, pd = _phase_b(acc{jb}, l{jb}, t_pv, nml_pv, v_smem.index({vst} * ROWS + {jb}), P_L=P_L)",
                   f"p16a = p16b", f"p16b = p16n"]

        def emit_b(ind):
            # PINGPONG: the two consumer warpgroups take turns on the MUFU-heavy phase B (token passing through two
            # mbarriers), so one warpgroup's exponentials overlap the other's FMA-pipe phase A (FA3-style ping-pong)
            w(f"{ind}if PINGPONG:")
            w(f"{ind}    mbarrier.wait(tok_mine, tph)")
            if j == 0:
                w(f"{ind}if n > 0:")
                for l_ in lines_b:
                    w(f"{ind}    {l_}")
            else:
                for l_ in lines_b:
                    w(f"{ind}{l_}")
            w(f"{ind}if PINGPONG:")
            w(f"{ind}    mbarrier.arrive(tok_other, count=1)")
            w(f"{ind}    tph = tph ^ 1")
        w(f"{indent}if SWAP:")
        emit_b(indent + "    ")
        w(f"{indent}    {line_a}")
        w(f"{indent}else:")
        w(f"{indent}    {line_a}")
        emit_b(indent + "    ")
        w(f"{indent}t_pv = t_new")
        w(f"{indent}nml_pv = nml_new")
    # the last row's PV is issued in the next iteration (or the epilogue): carry its rescaled accumulator as a token so the
    # loop-carried type is stable
    w(f"{indent}acc{rows-1} = warpgroup_mma_init(acc{rows-1})")
    return "\n".join(o)


def gen_body_masked(rows: int, indent: str) -> str:
    """One key tile with the per-key select, rows processed one after another with synchronous waits (few live registers, so
    the kernel variant that contains it does not spill in its mask-free loop); it keeps the pipelined body's invariants: on
    entry s_nx holds the QK^T of row 0 of this tile and phase B of row ROWS-1 of the previous tile is pending in t_pv/nml_pv;
    on exit the same holds for the next tile."""
    R = range(rows)
    o = []
    w = o.append
    w(f"{indent}# pending phase B of the previous tile's last row, then release that tile's stage")
    w(f"{indent}s_cu = s_nx")
    w(f"{indent}s_cu, acc0 = warpgroup_mma_wait(num_outstanding=0, deps=[s_cu, acc0])")
    w(f"{indent}p16a = _keep(p16a)")
    w(f"{indent}p16b = _keep(p16b)")
    w(f"{indent}if n > 0:")
    w(f"{indent}    acc{rows-1}, l{rows-1}, p16n, pd = _phase_b(acc{rows-1}, l{rows-1}, t_pv, nml_pv, v_smem.index(stp * ROWS + {rows-1}), P_L=P_L)")
    w(f"{indent}    p16a = p16b")
    w(f"{indent}    p16b = p16n")
    w(f"{indent}    p16b = _keep(p16b)")
    w(f"{indent}    pd = warpgroup_mma_wait(num_outstanding=0, deps=[pd])")
    w(f"{indent}mbarrier.arrive(empty.index(stp), count=1, pred=n > 0)")
    w(f"{indent}kv_ok = (m_lo + offs_n) < SEQ_K")
    for j in R:
        w(f"{indent}mrow{j} = mask_b + i{j} * smi")
        if j > 0:
            w(f"{indent}s_cu = warpgroup_mma(qs{j}, k_smem.index(st * ROWS + {j}).permute((1, 0)), pd, use_acc=False, is_async=True)")
            w(f"{indent}s_cu, acc{j} = warpgroup_mma_wait(num_outstanding=0, deps=[s_cu, acc{j}])")
        w(f"{indent}acc{j}, m{j}, l{j}, t_new, nml_new = _phase_a(acc{j}, m{j}, l{j}, s_cu, b32, mrow{j}, m_lo, offs_n, kv_ok, qk_scale, USE_MASK=True, O_L=O_L)")
        if j + 1 < rows:
            w(f"{indent}acc{j}, l{j}, p16n, pd = _phase_b(acc{j}, l{j}, t_new, nml_new, v_smem.index(st * ROWS + {j}), P_L=P_L)")
            w(f"{indent}p16a = p16b")
            w(f"{indent}p16b = p16n")
        else:
            w(f"{indent}t_pv = t_new")
            w(f"{indent}nml_pv = nml_new")
            w(f"{indent}acc{j} = warpgroup_mma_init(acc{j})")
    w(f"{indent}# prefetch row 0 of the next tile (stale stage for the last tile; drained, never used)")
    w(f"{indent}mbarrier.wait(ready.index(st1), ph1, pred=(n + 1) < n_tiles)")
    w(f"{indent}s_nx = warpgroup_mma(qs0, k_smem.index(st1 * ROWS).permute((1, 0)), pd, use_acc=False, is_async=True)")
    return "\n".join(o)


def gen_epilogue(rows: int, indent: str = "    ", persistent: bool = False) -> list:
    """After the tile loop: phase B of the last row, drain, normalise, store (rows past N_ROWS are not stored).  persistent (k12):
    after the last PV, the first QK^T of the NEXT item is issued (its Q buffer and first ring stage awaited) and left in flight
    through the drain and the stores, so the tensor core latency and the epilogue overlap across items."""
    R = range(rows)
    ep = []
    ep.append(f"{indent}# phase B of the last row of the last tile (stage of tile n_tiles-1 = the one before `st` after the final rotation)")
    ep.append(f"{indent}if STAGES == 2:")
    ep.append(f"{indent}    stl = st ^ 1")
    ep.append(f"{indent}else:")
    ep.append(f"{indent}    stl = (st + STAGES - 1) % STAGES")
    ep.append(f"{indent}if PINGPONG:")
    ep.append(f"{indent}    mbarrier.wait(tok_mine, tph)")
    ep.append(f"{indent}acc{rows-1}, l{rows-1}, p16n, pd = _phase_b(acc{rows-1}, l{rows-1}, t_pv, nml_pv, v_smem.index(stl * ROWS + {rows-1}), P_L=P_L)")
    ep.append(f"{indent}if PINGPONG:")
    ep.append(f"{indent}    mbarrier.arrive(tok_other, count=1)")
    deps = ", ".join(f"acc{r}" for r in R)
    if persistent:
        ep.append(f"{indent}# first QK^T of the next item (Q buffer qb^1, ring stage st); a harmless dummy for the last item (drained after the loop)")
        ep.append(f"{indent}qbn = qb ^ 1")
        ep.append(f"{indent}mbarrier.wait(q_ready.index(qbn), ((it + 1) // 2) & 1, pred=has_next)")
        ep.append(f"{indent}mbarrier.wait(ready.index(st), ph, pred=has_next)")
        ep.append(f"{indent}s_nx = warpgroup_mma(q_smem.index(qbn * ROWS), k_smem.index(st * ROWS).permute((1, 0)), pd, use_acc=False, is_async=True)")
        ep.append(f"{indent}{deps}{',' if rows == 1 else ''} = warpgroup_mma_wait(num_outstanding=1, deps=[{deps}])")
    else:
        ep.append(f"{indent}{deps}, s_nx = warpgroup_mma_wait(num_outstanding=0, deps=[{deps}, s_nx])")
    ep.append(f"{indent}p16a = _keep(p16a)")
    ep.append(f"{indent}p16b = _keep(p16b)")
    ep.append(f"{indent}p16n = _keep(p16n)")
    ep.append(f"{indent}offs_q = q0 + WG * 64 + gl.arange(0, PM, layout=OROW_L)")
    ep.append(f"{indent}offs_d = gl.arange(0, HEAD_DIM, layout=OCOL_L)")
    ep.append(f"{indent}o_ptr = out_bh + gl.expand_dims(offs_q.to(gl.int64) * soq, 1) + gl.expand_dims(offs_d, 0)")
    ep.append(f"{indent}q_ok = gl.expand_dims(offs_q < SEQ_Q, 1)")
    for r in R:
        ep.append(f"{indent}l_o = gl.convert_layout(l{r}, OROW_L, assert_trivial=True)")
        ep.append(f"{indent}o = (acc{r} / gl.expand_dims(l_o, 1)).to(Out.dtype.element_ty)")
        cond = "q_ok" if r == 0 else f"q_ok & ((r0 + {r}) < N_ROWS)"
        ep.append(f"{indent}gl.store(o_ptr + i{r} * soi, o, mask={cond})")
    return ep


# the part of the generated source shared with k12: imports, constants, _keep, _phase_a, _phase_b
SRC_COMMON = _SRC[: _SRC.index("@gluon.jit\ndef _producer(")]


def _render(rows: int, fp16: bool = False) -> str:
    R = range(rows)
    src = _SRC.replace("ROWS_LIT", str(rows))
    src = src.replace("XARGS", ", ".join(f"x{r}" for r in R)).replace("IARGS", ", ".join(f"i{r}" for r in R))
    # producer loads
    pl = []
    for r in R:
        pl.append(f"        tma.async_copy_global_to_shared(k_desc, [x{r} + n0, 0], rb, k_smem.index(st * ROWS + {r}))")
        pl.append(f"        tma.async_copy_global_to_shared(v_desc, [x{r} + n0, 0], rb, v_smem.index(st * ROWS + {r}))")
    src = src.replace("PRODUCER_LOADS", "\n".join(pl))
    # consumer init
    ci = []
    for r in R:
        ci.append(f"    if SPLIT:")
        ci.append(f"        qs{r} = q_smem.index({r}).slice(WG * 64, 64)")
        ci.append(f"    else:")
        ci.append(f"        qs{r} = q_smem.index({r})")
        ci.append(f"    acc{r} = warpgroup_mma_init(gl.zeros([PM, HEAD_DIM], gl.float32, O_L))")
        ci.append(f"    m{r} = gl.full([PM], float('-inf'), gl.float32, ROW_L)")
        ci.append(f"    l{r} = gl.zeros([PM], gl.float32, ROW_L)")
    src = src.replace("CONSUMER_INIT", "\n".join(ci))
    src = src.replace("BODY_PLAIN", gen_body(rows, "        ", "False"))
    src = src.replace("BODY_MASKED", gen_body_masked(rows, "            "))
    src = src.replace("CONSUMER_EPILOGUE", "\n".join(gen_epilogue(rows)))
    fr = []
    for r in R:
        fr.append(f"    i{r} = gl.minimum(r0 + {r}, N_ROWS - 1)")
        fr.append(f"    x{r} = ((b * N_ROWS + i{r}) * H + h) * SEQ_K")
    for r in R:
        fr.append(f"    i{r} = i{r}.to(gl.int64)")
    src = src.replace("FWD_ROWS", "\n".join(fr))
    # per-row key-tile info (host-computed): [n_tiles needed, first tile needing the select]; the CTA takes max / min over its rows
    fl = ["        rib = RowInfo + b.to(gl.int64) * srb", "        nt = gl.load(rib + i0 * sri)", "        ns = gl.load(rib + i0 * sri + 1)"]
    for r in R:
        if r == 0:
            continue
        fl.append(f"        nt = gl.maximum(nt, gl.load(rib + i{r} * sri))")
        fl.append(f"        ns = gl.minimum(ns, gl.load(rib + i{r} * sri + 1))")
    fl.append("        n_tiles = gl.minimum(nt, n_tiles)")
    fl.append("        n_sel = gl.minimum(ns, n_tiles)")
    src = src.replace("FWD_ROWINFO", "\n".join(fl))
    ql = []
    for r in R:
        ql.append(f"    tma.async_copy_global_to_shared(q_desc, [(x{r} // SEQ_K) * SEQ_Q + q0, 0], q_bar, q_smem.index({r}))")
    src = src.replace("FWD_QLOADS", "\n".join(ql))
    src = src.replace("MSEL_LIT", "-32768.0" if fp16 else "-1.0e9")
    return src


_MODULES: Dict[Tuple[int, bool], object] = {}


def kernel_for(rows: int, fp16: bool = False):
    mod = _MODULES.get((rows, fp16))
    if mod is None:
        src = _render(rows, fp16)
        tag = hashlib.sha1(src.encode()).hexdigest()[:10]
        # the generated module: <this dir>/_gen/<name> on a writable install (unchanged); under a
        # read-only tree _gen_file writes it under the kit's JIT root, else the temp dir; a file of
        # that name already there is compared with src, never imported: _gen_exec runs src itself
        path = _gen_file(
            f"k11_rows{rows}_{tag}.py",
            src,
        )
        name = f"triattn_k11_gen_rows{rows}_{tag}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        _gen_exec(mod, src, path)
        _MODULES[(rows, fp16)] = mod
    return mod._fwd


DEFAULT_CONFIG = dict(ROWS=3, BLOCK_N=64, STAGES=3, REGS_CONS=232, maxnreg=168, PREFETCH=1, SPLIT=1, SWAP=0, PINGPONG=1)
INFO: Dict = {}


def _ensure_dims(t, n):
    while t.dim() < n:
        t = t.unsqueeze(0)
    return t


@triton.jit
def _stage_bias(Bias, Out, OrPat, sbb, sbh, sbq, sbk, H, SQ, SK, SKt, msent,
                FOLD_MASK: tl.constexpr, PQ: tl.constexpr, PK: tl.constexpr):
    """Out[b,0,h,q,k] (16-bit, contiguous, SKt = keys padded to the key tile) = bias[b,0,h,q,k] for k < SK -- with the mask
    sentinel -1e9 where no pair row of batch element b keeps key k (OrPat[b, k] == 0) when FOLD_MASK -- and -inf for the padded
    keys k >= SK (below the sentinel: p == 0 there even for a fully-masked row)."""
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rows = pid_q * PQ + tl.arange(0, PQ)
    r_ok = rows < SQ
    src = Bias + b.to(tl.int64) * sbb + h.to(tl.int64) * sbh + rows.to(tl.int64)[:, None] * sbq
    dst = Out + (pid_bh.to(tl.int64) * SQ + rows.to(tl.int64))[:, None] * SKt
    for k0 in range(0, SKt, PK):
        cols = k0 + tl.arange(0, PK)
        inb = cols < SK
        x = tl.load(src + cols[None, :] * sbk, mask=r_ok[:, None] & inb[None, :], other=0.0).to(tl.float32)
        if FOLD_MASK:
            keep = tl.load(OrPat + b.to(tl.int64) * SK + cols, mask=inb, other=0)
            x = tl.where(keep[None, :] != 0, x, msent)
        x = tl.where(inb[None, :], x, float("-inf"))
        tl.store(dst + cols[None, :], x.to(Out.dtype.element_ty), mask=r_ok[:, None] & (cols < SKt)[None, :])


def _mask_level(dtype) -> float:
    """The logit level written for masked keys (bias-tile fold and per-key select): -1e9 for bf16 tiles (its fp32 spacing absorbs
    scale*q.k, so a batch element that keeps nothing is the uniform average); fp16 tiles cannot hold it (nor may the base-2
    softmax's FFMA rounding at |m| ~ 1e9 overflow fp16 P), so fp16 uses -32768: masked keys still weigh exactly 0 next to any kept
    key, rows that keep nothing stay finite and deterministic (uniform where the select applies, softmax(scale*q.k) where only
    the fold does -- documented)."""
    return -32768.0 if dtype == torch.float16 else -1.0e9


def _tma_view(x):
    """A 5-D [B,N,H,S,D] tensor or strided view -> (x, base2d, (RB, RI, RH, CB, CI, CH)): element (b,i,h,s,d) ==
    base2d[b*RB + i*RI + h*RH + s, b*CB + i*CI + h*CH + d] with base2d a plain 2-D [rows, W] tensor over the same storage (what a
    rank-2 TMA descriptor addresses; W = the stride of the S dim), or None when the strides cannot be expressed that way (then the
    caller copies, a named fallback).  Each outer dim is either a whole number of rows (stride % W == 0) or a 16-byte aligned column
    offset inside a row (stride < W).  Covers contiguous [B,N,H,S,D], the engines' permuted views of a [.., S, H*D] (or wider,
    fused) projection output (heads = column blocks), and column-attention views x.transpose(1, 3) of either (pair rows = column
    blocks) -- all without a copy."""
    B, N, H, S, D = x.shape
    sB, sN, sH, sS, sD = x.stride()
    if sD != 1 or sS < D or sS % 8 != 0 or x.data_ptr() % 16 != 0:
        return None
    W = sS
    rows, cols, col_extent = [], [], D
    for st, n in ((sB, B), (sN, N), (sH, H)):
        if n == 1 or st == 0:
            rows.append(0); cols.append(0)
        elif st % W == 0:
            rows.append(st // W); cols.append(0)
        elif 0 < st < W and st % 8 == 0:
            rows.append(0); cols.append(st); col_extent += (n - 1) * st
        else:
            return None
    if col_extent > W:
        return None
    n_rows = (B - 1) * rows[0] + (N - 1) * rows[1] + (H - 1) * rows[2] + S
    if n_rows >= 2 ** 31 or W >= 2 ** 31:
        return None
    avail = x.untyped_storage().nbytes() // x.element_size() - x.storage_offset()
    if n_rows * W > avail:                              # the 2-D view must lie inside x's storage
        return None
    base = torch.as_strided(x, (n_rows, W), (W, 1))
    return x, base, (rows[0], rows[1], rows[2], cols[0], cols[1], cols[2])


def prepare(q, k, v, bias, mask, scale, bn: int, need_square: bool, dims=(16, 32, 64, 128), need_dense: bool = False):
    """Argument checks + device-side staging shared by k11 / k12: contiguous q/k/v, the 16-bit bias tile buffer [B,1,H,S_q,SKt]
    (key columns padded to the key tile with -inf; the batch OR pattern of kept keys folded in as -1e9 when a mask is given), the
    per-row key-tile table (masking.mask_tables) and the strides the kernels take.  No host synchronisation."""
    q = _ensure_dims(q, 5); k = _ensure_dims(k, 5); v = _ensure_dims(v, 5); bias = _ensure_dims(bias, 5)
    if mask is not None:
        mask = _ensure_dims(mask, 5)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
    B, N, H, SQ, D = q.shape
    SK = k.shape[3]
    if k.shape != (B, N, H, SK, D) or v.shape != (B, N, H, SK, D):
        raise Unsupported(f"unsupported shape: k/v must be (B,N,H,S_kv,D); got {tuple(k.shape)} / {tuple(v.shape)}")
    if bias.shape != (B, 1, H, SQ, SK):
        raise Unsupported(f"unsupported shape: bias must be (B,1,H,S_q,S_kv); got {tuple(bias.shape)}")
    if mask is not None and mask.shape != (B, N, 1, 1, SK):
        raise Unsupported(f"unsupported shape: mask must be (B,N,1,1,S_kv); got {tuple(mask.shape)}")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise Unsupported(f"unsupported dtype {q.dtype}/{k.dtype}/{v.dtype}: q/k/v must all be bf16 or fp16")
    if D not in dims:
        raise Unsupported(f"unsupported head_dim {D} (supported: {dims})")
    if need_square and SQ != SK:
        raise Unsupported(f"unsupported S_q != S_kv ({SQ} vs {SK}): pair representations are square")
    if B * N * H * max(SQ, SK) + 256 >= 2 ** 31:
        raise Unsupported("unsupported size: B*N*H*S exceeds the int32 TMA coordinate range")
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    info = {}
    views = {}
    for nm, t in (("q", q), ("k", k), ("v", v)):
        tv = None if need_dense else _tma_view(t)
        if tv is None:
            if not t.is_contiguous():
                t = t.contiguous(); info[f"{nm}_copied"] = True      # named fallback: strides TMA cannot express (or a k11 call)
            tv = _tma_view(t)
        views[nm] = tv
    q, k, v = views["q"][0], views["k"][0], views["v"][0]
    dev = q.device
    SKt = triton.cdiv(SK, bn) * bn
    if mask is not None:
        mask_u8 = mask.view(torch.uint8)
        if mask_u8.stride(-1) != 1:
            mask_u8 = mask_u8.contiguous()
        orpat, rowinfo, n_all = mask_tables(mask_u8.view(B, N, SK) if mask_u8.is_contiguous() else mask_u8[:, :, 0, 0, :], bn)
        info["mask_census"] = lambda: census(rowinfo, n_all)   # on demand: (rows equal to the OR pattern = mask-free, rows with per-key select)
        m = dict(mask=mask_u8, smb=mask_u8.stride(0), smi=mask_u8.stride(1), rowinfo=rowinfo, srb=N * 2, sri=2, has_mask=True)
    else:
        orpat = None
        m = dict(mask=None, smb=0, smi=0, rowinfo=None, srb=0, sri=0, has_mask=False)
    if bias.dtype == q.dtype and bias.is_contiguous() and SKt == SK and mask is None:
        bias16_t = bias
    else:
        bias16_t = torch.empty((B, 1, H, SQ, SKt), dtype=q.dtype, device=dev)
        PQ, PK = 32, 128
        launch(_LAUNCHES, _stage_bias, (triton.cdiv(SQ, PQ), B * H), (
            bias, bias16_t, orpat if mask is not None else bias,
            bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4), H, SQ, SK, SKt, _mask_level(q.dtype)),
            dict(FOLD_MASK=mask is not None, PQ=PQ, PK=PK), num_warps=4)
        info["bias_staged"] = ("16-bit tiles" if bias.dtype == q.dtype else f"{bias.dtype} -> {q.dtype} tiles (lossy unless representable)") + \
                              ("; batch OR-mask folded" if mask is not None else "")
    return dict(q=q, k=k, v=v, views=views, bias16=bias16_t, B=B, N=N, H=H, SQ=SQ, SK=SK, SKt=SKt, D=D, scale=float(scale), dev=dev, info=info, **m)


_LAYOUTS: Dict = {}
_LAUNCHES: Dict = {}


def _nvmma_layout(r, c, dtype):
    """gl.NVMMASharedLayout.get_default_for([r, c], dtype), cached (its construction costs ~15 us)."""
    key = (r, c, dtype)
    lay = _LAYOUTS.get(key)
    if lay is None:
        from triton.experimental.gluon import language as gl
        gdt = gl.bfloat16 if dtype == torch.bfloat16 else gl.float16
        lay = gl.NVMMASharedLayout.get_default_for([r, c], gdt)
        _LAYOUTS[key] = lay
    return lay


def _descriptors(P, bn):
    """TMA descriptors on the tensors AS GIVEN (rank-2 views, see _tma_view) + the integer coordinate terms the kernels use:
    row(b, i, h, s) = b*RB + i*RI + h*RH + s, col(b, i, h) = b*CB + i*CI + h*CH per tensor (keys 'q', 'k', 'v')."""
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    B, N, H, SQ, SK, SKt, D = P["B"], P["N"], P["H"], P["SQ"], P["SK"], P["SKt"], P["D"]
    dt = P["q"].dtype
    BM = 128
    kv_layout = _nvmma_layout(bn, D, dt)
    qv, kv_, vv = P["views"]["q"], P["views"]["k"], P["views"]["v"]
    b2 = P["bias16"].view(B * H * SQ, SKt)
    coords = {}
    for nm, tv in (("q", qv), ("k", kv_), ("v", vv)):
        coords[nm] = tuple(int(c) for c in tv[2])
    return (TensorDescriptor.from_tensor(qv[1], [BM, D], _nvmma_layout(BM, D, dt)),
            TensorDescriptor.from_tensor(kv_[1], [bn, D], kv_layout),
            TensorDescriptor.from_tensor(vv[1], [bn, D], kv_layout),
            TensorDescriptor.from_tensor(b2, [BM, bn], _nvmma_layout(BM, bn, dt)),
            coords)


def triattn_k11(q, k, v, bias, mask=None, scale=None, *, config: Optional[Dict] = None, out_dtype=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    rows, bn, stages = int(cfg["ROWS"]), int(cfg["BLOCK_N"]), int(cfg["STAGES"])
    if rows < 2 or (rows <= 3 and stages < 3) or stages < 2:
        raise ValueError("k11 config needs ROWS >= 2, STAGES >= 2, and STAGES >= 3 when ROWS <= 3 (stage-release ordering)")
    P = prepare(q, k, v, bias, mask, scale, bn, need_square=True, dims=(16, 32), need_dense=True)
    B, N, H, SQ, SK, D = P["B"], P["N"], P["H"], P["SQ"], P["SK"], P["D"]
    out = torch.empty((B, N, H, SQ, D), dtype=out_dtype or P["q"].dtype, device=P["dev"])
    if out.numel() == 0:
        return out
    q_desc, k_desc, v_desc, b_desc, _coords = _descriptors(P, bn)
    dummy = out
    grid = (triton.cdiv(SQ, 128), triton.cdiv(N, rows), B * H)
    kern = kernel_for(rows, P["q"].dtype == torch.float16)
    ck = kern[grid](q_desc, k_desc, v_desc, b_desc, out, P["mask"] if P["has_mask"] else dummy, P["rowinfo"] if P["has_mask"] else dummy,
               out.stride(0), out.stride(1), out.stride(2), out.stride(3), P["smb"], P["smi"], P["srb"], P["sri"],
               N, SQ, SK, H, P["scale"],
               BLOCK_N=bn, HEAD_DIM=D, STAGES=stages, HAS_MASK=P["has_mask"],
               REGS_CONS=int(cfg["REGS_CONS"]), PREFETCH=int(cfg["PREFETCH"]), SPLIT=int(cfg["SPLIT"]), SWAP=int(cfg["SWAP"]),
               PINGPONG=int(cfg["PINGPONG"]) if int(cfg["SPLIT"]) else 0,
               VAR_TILES=P["has_mask"], num_warps=4, maxnreg=int(cfg["maxnreg"]))
    INFO.clear(); INFO.update(P["info"])
    if ck is not None:
        INFO["kernel"] = dict(n_regs=getattr(ck, "n_regs", None), n_spills=getattr(ck, "n_spills", None), shared=getattr(getattr(ck, "metadata", None), "shared", None))
    return out


def _gen_dirs():
    """Where a generated kernel module is written, in order: ``_GEN_DIR`` (<this directory>/_gen — a writable install: unchanged); under a
    read-only tree (a container image's file system, a shared read-only checkout) the kit's JIT root
    ``<MODEL_OPT_JIT_ROOT>[/<MODEL_OPT_STACK_KEY>]/opt_core_gen/triattn_pkg/triton/triattn/_gen``, then ``<tempdir>/opt_core_gen-uid<uid>/triattn_pkg/triton/triattn/_gen``.  The generated
    source is byte-identical wherever it lands (the Triton cache keys on the source text, not the path)."""
    rel = os.path.join("triattn_pkg", "triton", "triattn", "_gen")
    dirs = [_GEN_DIR]
    root = os.environ.get("MODEL_OPT_JIT_ROOT")
    if root:
        key = os.environ.get("MODEL_OPT_STACK_KEY")
        dirs.append(os.path.join(root, key, "opt_core_gen", rel) if key else os.path.join(root, "opt_core_gen", rel))
    import tempfile
    dirs.append(os.path.join(tempfile.gettempdir(), "opt_core_gen-uid%d" % os.getuid(), rel))
    return dirs


def _gen_file(fname: str, src: str) -> str:
    """The generated module's path: the first directory of ``_gen_dirs()`` that may hold generated source (``_gen_why_not``) and that holds
    ``src`` under ``fname`` byte for byte when this returns.  A file already there is compared with ``src``: equal bytes are kept, anything
    else is refused by name and replaced (atomically); a directory where it cannot be replaced is skipped as if absent.  The bytes on disk
    are never imported -- ``kernel_for`` runs ``src`` itself (``_gen_exec``)."""
    data = src.encode()
    err = None
    for d in _gen_dirs():
        path = os.path.join(d, fname)
        try:
            os.makedirs(d, mode=0o755, exist_ok=True)
            why = _gen_why_not(d)
            if why:
                _gen_say(f"directory {d}", f"{why}; skipped as if absent, the next location serves")
                err = err or PermissionError(f"{d}: {why}")
                continue
            found = _gen_found(path, data)
            if found is not True:
                tmp = path + f".tmp{os.getpid()}"
                if os.path.lexists(tmp):
                    os.unlink(tmp)
                with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "wb") as f:   # O_EXCL: never through a link left at that name
                    f.write(data)
                os.replace(tmp, path)
                if found is False:
                    _gen_say(f"file {path}", "its bytes were not the generated source; replaced by the generated source (nothing to fix)")
            return path
        except OSError as e:
            err = e
            found = _gen_found(path, data)
            if found is True:
                return path
            if found is False:
                _gen_say(f"file {path}", f"its bytes are not the generated source and it cannot be replaced ({e.strerror}); "
                                         "skipped as if absent, the next location serves; to use this location delete that file")
    raise err


def _gen_why_not(d: str):
    """Why directory ``d`` may not hold generated source, else None -- the cache directories' rule: group or others can write it, or its
    owner is neither this user nor root (uid 0 accepts any owner: in a container the bound directories belong to the host user)."""
    st = os.stat(d)
    if st.st_mode & 0o022:
        return f"group or others can write it (mode {st.st_mode & 0o7777:04o}; to use it: chmod go-w)"
    uid = os.getuid()
    if uid != 0 and st.st_uid not in (0, uid):
        return f"it belongs to uid {st.st_uid}, not to this user (uid {uid}) or root (to use it: make it a directory of this user)"
    return None


def _gen_found(path: str, data: bytes):
    """What is at ``path``: None -- nothing; True -- a regular file (not a link) whose bytes are exactly ``data``; False -- anything else."""
    import stat
    try:
        mode = os.lstat(path).st_mode
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return False
    try:
        if not stat.S_ISREG(mode):
            return False
        with open(path, "rb") as f:
            return f.read(len(data) + 1) == data
    except OSError:
        return False


_GEN_SAID = set()


def _gen_say(what: str, why: str) -> None:
    """One stderr line per refused directory or file, once per process."""
    if what not in _GEN_SAID:
        _GEN_SAID.add(what)
        sys.stderr.write(f"[opt_core] GENERATED-SOURCE refused {what}: {why}\n")
        sys.stderr.flush()


def _gen_exec(mod, src: str, path: str) -> None:
    """Run the generated source in ``mod``.  The code object is compiled from ``src`` -- the text this process generated -- with ``path`` as
    its file name: neither the bytes at ``path`` nor a cached .pyc beside it are read.  The line cache is primed with ``src`` first, so
    inspect.getsourcelines (how the Triton JIT reads a kernel's text) returns it without reading ``path`` either."""
    import linecache
    linecache.cache[path] = (len(src), None, src.splitlines(True), path)
    exec(compile(src, path, "exec", dont_inherit=True), mod.__dict__)

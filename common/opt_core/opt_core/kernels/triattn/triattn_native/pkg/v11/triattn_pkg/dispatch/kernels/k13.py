"""k13 -- a persistent warp-specialized Gluon kernel (k12's design: TMA + mbarrier ring, async wgmma, one 8-warp
compute partition, ROWS pair rows per work item sharing each 16-bit bias tile, persistent over work items) with the key mask
injected THROUGH THE TENSOR CORE instead of table kernels, bias folding or per-key selects:

    S = Q K^T + 1 (x) m        m[k] = 0 for kept keys, -2^30 (bf16) / -2^15 (fp16) for masked keys of THIS pair row

The rank-1 term is a second, chained asynchronous wgmma (K = 16): A = a constant [128, 16] tile whose column 0 is ones
(built once per CTA in shared memory), B = a [BLOCK_N, 16] tile per (ring stage, pair row) whose column 0 holds m, built by
the otherwise idle TMA-producer warpgroup from the raw mask bytes while it streams K/V/bias (the stage's `ready` barrier takes
two arrivals: the TMA transaction count and the producer's arrive after its shared-memory stores).  The compute partition
spends zero instructions per logit on masking, arbitrary per-row masks are exact (masked keys weigh exactly 0 once the row
keeps any key; a row that keeps nothing yields a finite deterministic average, cf. cuEquivariance's uniform average), no
staging copy of the bias is needed for the mask, and a masked call is ONE kernel launch when the bias arrives 16-bit with S a
multiple of BLOCK_N.  Key tiles past the last key kept by any of the item's ROWS rows are skipped (in-kernel scan of the rows'
mask bytes; all tiles when a row keeps nothing).

Host API as k12: triattn_k13(q, k, v, bias, mask=None, scale=None, config=...) -> out [B, N, H, S, D]; square S, D in
{16, 32}, bf16 / fp16, sm_90.  Numerics otherwise identical to k11/k12 (triton/triattn/).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
from typing import Dict, Optional

import torch
import triton

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRITON_TRACK = os.path.normpath(os.path.join(_HERE, "..", "..", "triton"))
if _TRITON_TRACK not in sys.path:
    sys.path.insert(0, _TRITON_TRACK)

from triattn.k11 import SRC_COMMON, Unsupported, gen_body, gen_epilogue, prepare, _descriptors, _nvmma_layout  # noqa: E402,F401
from triattn.k12 import _num_sms  # noqa: E402
from triattn.launch import call_key, launch  # noqa: E402

_GEN_DIR = os.path.join(_HERE, "_gen")
INFO: Dict = {}
_LAUNCHES: Dict = {}
DEFAULT_CONFIG = dict(ROWS=3, BLOCK_N=64, STAGES=3, REGS_CONS=232, maxnreg=168, CTAS_PER_SM=1)
KX = 16                    # K extent of the rank-1 mask wgmma (one bf16 k-step)

_SRC = SRC_COMMON + '''
KXD = gl.constexpr(KX_LIT)
ROWS_P2 = gl.constexpr(ROWSP2_LIT)          # rows padded to a power of two (the visible-tiles scan holds one row per warp)


@gluon.jit
def _stamp(Dbg, slot, pid, it, k: gl.constexpr, L: gl.constexpr):
    """DBG&8: write %globaltimer (ns) into Dbg[pid, it, k] for the first 16 items of each CTA."""
    z = gl.full([1], 0, gl.int32, L)
    t = gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [z], dtype=gl.int64, is_pure=False, pack=1)
    gl.store(Dbg + (pid * 16 + it) * 8 + k + gl.arange(0, 1, layout=L), t, mask=(gl.arange(0, 1, layout=L) == 0) & (it < 16))


@gluon.jit
def _fdiv(a, d, inv_d):
    """a // d for 0 <= a < 2^22 via the fp32 reciprocal (exact; avoids the ~50-instruction integer division sequence)."""
    qt = ((a.to(gl.float32) + 0.5) * inv_d).to(gl.int32)
    return qt, a - qt * d


@gluon.jit
def _item(w, n_q, n_r, H, inv_nq, inv_nr, inv_h, N_ROWS, SEQ_Q, SEQ_K):
    """Work item w -> (b, h, q0, r0); q tiles vary fastest so concurrently running CTAs share the K/V rows in L2."""
    t, pid_q = _fdiv(w, n_q, inv_nq)
    pid_bh, pid_r = _fdiv(t, n_r, inv_nr)
    b, h = _fdiv(pid_bh, H, inv_h)
    return b, h, pid_bh, pid_q * BM, pid_r * ROWS


@gluon.jit
def _visible_tiles(mask_b, smi, SEQ_K, n_tiles, ITEM_IARGS, BLOCK_N: gl.constexpr, V_L: gl.constexpr, SCW: gl.constexpr):
    """Key tiles this item visits: up to the last key kept by any of its rows; all of them if a row keeps nothing.  One
    [ROWS_P2, SCW] load per SCW keys with each pair row on its own warp (row reductions are warp shuffles)."""
    VR_L: gl.constexpr = gl.SliceLayout(1, V_L)
    VC_L: gl.constexpr = gl.SliceLayout(0, V_L)
    ridx = gl.arange(0, ROWS_P2, layout=VR_L)
ROW_PTRS
    offs = gl.arange(0, SCW, layout=VC_L)
    last_k = gl.full([ROWS_P2], -1, gl.int32, VR_L)
    for k0 in range(0, SEQ_K, SCW):
        cols = gl.expand_dims(k0 + offs, 0)
        c_ok = (cols < SEQ_K) & (gl.expand_dims(ridx, 1) >= 0)
        mk = gl.load(gl.expand_dims(rptr, 1) + cols, mask=c_ok, other=0) != 0
        last_k = gl.maximum(last_k, gl.max(gl.where(mk, cols, -1), axis=1))
    keeps = gl.min(last_k, axis=0) >= 0                       # every row keeps some key
    n_vis = gl.where(keeps, gl.max(last_k, axis=0) // BLOCK_N + 1, n_tiles)
    return n_vis


@gluon.jit
def _producer(q_desc, k_desc, v_desc, b_desc, q_smem, k_smem, v_smem, b_smem, kx_smem, nvis_smem, ready, empty, q_ready, q_empty,
              Mask, smb, smi, N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, X0, X1, X2, XR,
              QRB, QRI, QRH, QCB, QCI, QCH, KRB, KRI, KRH, KCB, KCI, KCH, VRB, VRI, VRH, VCB, VCI, VCH,
              BLOCK_N: gl.constexpr, STAGES: gl.constexpr, HAS_MASK: gl.constexpr, KXW: gl.constexpr, DBG: gl.constexpr):
    """TMA producer (default partition, 4 warps).  Per item: the ROWS Q tiles into Q buffer it % 2, then per key tile ROWS K +
    ROWS V tiles + one [128, BLOCK_N] bias tile into the ring (which never drains between items); with a mask it also writes
    the per-row [BLOCK_N, 16] additive tiles (k-slot pieces X0..XR for masked keys, 0 for kept keys, from the mask bytes) into the stage and arrives on `ready`."""
    KV_BYTES: gl.constexpr = k_desc.block_type.nbytes
    B_BYTES: gl.constexpr = b_desc.block_type.nbytes
    Q_BYTES: gl.constexpr = q_desc.block_type.nbytes
    PNW: gl.constexpr = gl.num_warps()
    KX_L: gl.constexpr = gl.BlockedLayout([1, 8], [16, 2], [PNW, 1], [1, 0])
    KROW_L: gl.constexpr = gl.SliceLayout(1, KX_L)
    KCOL_L: gl.constexpr = gl.SliceLayout(0, KX_L)
    SCW: gl.constexpr = 512
    SCL: gl.constexpr = gl.BlockedLayout([1, SCW // 32], [1, 32], [PNW, 1], [1, 0])     # pair rows on warps, keys on lanes
    ONE_L: gl.constexpr = gl.BlockedLayout([1], [32], [PNW], [0])
    pid = gl.program_id(0)
    nprog = gl.num_programs(0)
    offs_k = gl.arange(0, BLOCK_N, layout=KROW_L)
    colj = gl.expand_dims(gl.arange(0, KXD, layout=KCOL_L), 0)
    xcol = gl.where(colj == 0, X0, gl.where(colj == 1, X1, gl.where(colj == 2, X2, gl.where(colj < KXW, XR, 0.0))))   # [1, KXD] additive pieces per k-slot
PRODUCER_MKINIT
    st = 0
    eph = 1          # parity to wait for on `empty[st]` before refilling stage st (first lap passes via pred)
    g = 0            # key tiles loaded so far (ring position)
    it = 0           # items started so far
    for w in range(pid, n_items, nprog):
        b, h, pid_bh, q0, r0 = _item(w, n_q, n_r, H, inv_nq, inv_nr, inv_h, N_ROWS, SEQ_Q, SEQ_K)
ITEM_ROWS
        n_tiles = gl.cdiv(SEQ_K, BLOCK_N)
        mask_b = Mask + b.to(gl.int64) * smb
        qb = it % 2
        # Q buffer qb was last used by item it-2: wait until the consumers released it, then load this item's Q tiles
        mbarrier.wait(q_empty.index(qb), ((it // 2) + 1) & 1, pred=it >= 2)
        qrb = q_ready.index(qb)
        mbarrier.expect(qrb, ROWS * Q_BYTES)
PRODUCER_QLOADS
        if HAS_MASK:
            # while the Q tiles fly: the visible key tiles of this item -> the consumers read it from nvis_smem[qb] after
            # q_ready[qb] (released by the second arrive); the first tile's mask bytes are fetched now, later tiles one tile ahead
            if (DBG & 4) == 0:
                n_tiles = _visible_tiles(mask_b, smi, SEQ_K, n_tiles, ITEM_IARGS, BLOCK_N, SCL, SCW)
            nvis_smem.index(qb).store(gl.full([1], n_tiles, gl.int32, ONE_L))
            mbarrier.arrive(qrb, count=1)
PRODUCER_MK0
        bias_row = pid_bh * SEQ_Q + q0
        for n in range(n_tiles):
            mbarrier.wait(empty.index(st), eph, pred=g >= STAGES)
            rb = ready.index(st)
            mbarrier.expect(rb, 2 * ROWS * KV_BYTES + B_BYTES)
            n0 = n * BLOCK_N
PRODUCER_LOADS
            tma.async_copy_global_to_shared(b_desc, [bias_row, n0], rb, b_smem.index(st))
            if HAS_MASK:
                # this tile's additive mask tiles from the prefetched bytes; then prefetch the next tile's bytes
                if (DBG & 2) == 0:
PRODUCER_KX
                    fence_async_shared()
                mbarrier.arrive(rb, count=1)
                k_nx = n0 + BLOCK_N + offs_k
                k_ok = k_nx < SEQ_K
PRODUCER_MKNEXT
            g += 1
            st += 1
            if st == STAGES:
                st = 0
                eph ^= 1
        it += 1


@gluon.jit
def _consumer(q_smem, k_smem, v_smem, b_smem, kx_smem, q1_smem, nvis_smem, o_smem, o_desc, ready, empty, q_ready, q_empty, Out, Mask, Dbg,
              sob, soi, soh, soq, smb, smi,
              N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, qk_scale,
              BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, STAGES: gl.constexpr, HAS_MASK: gl.constexpr, DBG: gl.constexpr,
              EVEN_Q: gl.constexpr, KXW: gl.constexpr, AONE: gl.constexpr):
    """The compute partition (8 warps = two warpgroups x 64 query rows, one code stream), persistent over work items; per
    item the k11 schedule (software pipeline over the ROWS pair rows of each key tile, dead-P register donors, explicit P
    liveness); with a mask every QK^T wgmma is followed by the chained rank-1 mask wgmma on the same accumulator."""
    NW: gl.constexpr = gl.num_warps()
    PM: gl.constexpr = 128
    WG: gl.constexpr = 0
    SWAP: gl.constexpr = 0
    PINGPONG: gl.constexpr = 0
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, BLOCK_N, 16])
    O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, HEAD_DIM, 16])
    P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
    ROW_L: gl.constexpr = gl.SliceLayout(1, S_L)
    COL_L: gl.constexpr = gl.SliceLayout(0, S_L)
    OROW_L: gl.constexpr = gl.SliceLayout(1, O_L)
    OCOL_L: gl.constexpr = gl.SliceLayout(0, O_L)
    Q1_L: gl.constexpr = gl.BlockedLayout([1, 8], [16, 2], [NW, 1], [1, 0])
    ONE_L: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    dt: gl.constexpr = v_smem.dtype
    pid = gl.program_id(0)
    nprog = gl.num_programs(0)

    if HAS_MASK:
        # the constant A operand of the rank-1 mask wgmma: column 0 = 1, columns 1..15 = 0
        q1c = gl.expand_dims(gl.arange(0, KXD, layout=gl.SliceLayout(0, Q1_L)), 0) < KXW
        q1r = gl.expand_dims(gl.arange(0, PM, layout=gl.SliceLayout(1, Q1_L)), 1) >= 0
        q1_smem.store(gl.where(q1c & q1r, AONE, 0.0).to(dt))
        fence_async_shared()          # (the partition-wide bar.sync before the first wgmma read is inserted by the membar pass)
    offs_n = gl.arange(0, BLOCK_N, layout=COL_L)
    pd = gl.zeros([PM, BLOCK_N], gl.float32, S_L)          # dead fp32 P tile: register donor for the next QK^T accumulator
    p16a = gl.zeros([PM, BLOCK_N], dt, P_L)                # P operands of the two most recent PV wgmmas (kept live until retired)
    p16b = gl.zeros([PM, BLOCK_N], dt, P_L)
    t_pv = gl.zeros([PM, BLOCK_N], gl.float32, S_L)        # phase-A output of the previous row, consumed by its phase B
    nml_pv = gl.zeros([PM], gl.float32, ROW_L)
    st = 0           # ring stage / parity of the next key tile (continues across items)
    ph = 0
    it = 0
    for w in range(pid, n_items, nprog):
        b, h, pid_bh, q0, r0 = _item(w, n_q, n_r, H, inv_nq, inv_nr, inv_h, N_ROWS, SEQ_Q, SEQ_K)
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 0, ONE_L)
ITEM_ROWS_C
        n_tiles = gl.cdiv(SEQ_K, BLOCK_N)
        mask_b = Mask + b.to(gl.int64) * smb
        out_bh = Out + b.to(gl.int64) * sob + h.to(gl.int64) * soh
        qb = it % 2
        mbarrier.wait(q_ready.index(qb), (it // 2) & 1)          # completion it//2 of q_ready[qb]
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 1, ONE_L)
        if HAS_MASK:
            n_tiles = gl.max(nvis_smem.index(qb).load(ONE_L), axis=0)     # visible key tiles (written by the producer)
        n_sel = n_tiles
CONSUMER_INIT
        # prologue: QK^T of row 0, tile 0 of this item (ring position continues from the previous item)
        mbarrier.wait(ready.index(st), ph)
        s_nx = warpgroup_mma(qs0, k_smem.index(st * ROWS).permute((1, 0)), pd, use_acc=False, is_async=True)
        if HAS_MASK and (DBG & 1) == 0:
            s_nx = warpgroup_mma(q1_smem, kx_smem.index(st * ROWS).permute((1, 0)), s_nx, is_async=True)
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 2, ONE_L)
        for n in range(0, n_sel):
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
            b32 = b_smem.index(st).load(S_L).to(gl.float32)
BODY_PLAIN
            st = st1
            ph = ph1
            if DBG & 8:
                if n == 0:
                    _stamp(Dbg, 0, pid, it, 3, ONE_L)       # end of the first tile
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 4, ONE_L)
CONSUMER_EPILOGUE
        if DBG & 8:
            _stamp(Dbg, 0, pid, it, 5, ONE_L)
        # everything has retired: release the ring stage(s) this item still holds (its last tile; the last two when ROWS == 2)
        # and its Q buffer, so the producer keeps streaming the next items
        mbarrier.arrive(empty.index(stl), count=1)
        if ROWS == 2:
            mbarrier.arrive(empty.index((stl + STAGES - 1) % STAGES), count=1, pred=n_tiles >= 2)
        mbarrier.arrive(q_empty.index(qb), count=1)
        it += 1
    if EVEN_Q:
        tma.store_wait(0)


@gluon.jit
def _fwd(q_desc, k_desc, v_desc, b_desc, o_desc, Out, Mask, Dbg,
         sob, soi, soh, soq, smb, smi,
         N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, qk_scale, X0, X1, X2, XR,
         QRB, QRI, QRH, QCB, QCI, QCH, KRB, KRI, KRH, KCB, KCI, KCH, VRB, VRI, VRH, VCB, VCI, VCH,
         BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, STAGES: gl.constexpr, HAS_MASK: gl.constexpr,
         REGS_CONS: gl.constexpr, KXW: gl.constexpr, AONE: gl.constexpr, DBG: gl.constexpr, EVEN_Q: gl.constexpr):
    KX_SL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, KXD], k_desc.dtype)
    Q1_SL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, KXD], k_desc.dtype)
    q_smem = gl.allocate_shared_memory(q_desc.dtype, [2 * ROWS, BM, HEAD_DIM], q_desc.layout)      # double-buffered per item
    k_smem = gl.allocate_shared_memory(k_desc.dtype, [STAGES * ROWS, BLOCK_N, HEAD_DIM], k_desc.layout)
    v_smem = gl.allocate_shared_memory(v_desc.dtype, [STAGES * ROWS, BLOCK_N, HEAD_DIM], v_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, [STAGES, BM, BLOCK_N], b_desc.layout)
    kx_smem = gl.allocate_shared_memory(k_desc.dtype, [STAGES * ROWS, BLOCK_N, KXD], KX_SL)
    q1_smem = gl.allocate_shared_memory(k_desc.dtype, [BM, KXD], Q1_SL)
    nvis_smem = gl.allocate_shared_memory(gl.int32, [2, 1], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    o_smem = gl.allocate_shared_memory(o_desc.dtype, [ROWS, BM, HEAD_DIM], o_desc.layout)              # output staging for the TMA-store epilogue
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    q_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    q_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        if HAS_MASK:
            mbarrier.init(ready.index(s), count=2)          # TMA transaction bytes + the producer's arrive after its mask tiles
        else:
            mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)
    for s in gl.static_range(2):
        if HAS_MASK:
            mbarrier.init(q_ready.index(s), count=2)        # Q transaction bytes + the producer's arrive after the visible-tiles word
        else:
            mbarrier.init(q_ready.index(s), count=1)
        mbarrier.init(q_empty.index(s), count=1)
    fence_async_shared()
    gl.warp_specialize([
        (_producer, (q_desc, k_desc, v_desc, b_desc, q_smem, k_smem, v_smem, b_smem, kx_smem, nvis_smem, ready, empty, q_ready, q_empty,
                     Mask, smb, smi, N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, X0, X1, X2, XR,
                     QRB, QRI, QRH, QCB, QCI, QCH, KRB, KRI, KRH, KCB, KCI, KCH, VRB, VRI, VRH, VCB, VCI, VCH, BLOCK_N, STAGES, HAS_MASK, KXW, DBG)),
        (_consumer, (q_smem, k_smem, v_smem, b_smem, kx_smem, q1_smem, nvis_smem, o_smem, o_desc, ready, empty, q_ready, q_empty, Out, Mask, Dbg,
                     sob, soi, soh, soq, smb, smi,
                     N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, inv_nq, inv_nr, inv_h, qk_scale,
                     BLOCK_N, HEAD_DIM, STAGES, HAS_MASK, DBG, EVEN_Q, KXW, AONE)),
    ], [8], [REGS_CONS])
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(ready.index(s))
        mbarrier.invalidate(empty.index(s))
    for s in gl.static_range(2):
        mbarrier.invalidate(q_ready.index(s))
        mbarrier.invalidate(q_empty.index(s))
'''

_S_ISSUE = re.compile(r"^(?P<ind>\s*)s_nx = warpgroup_mma\(qs(?P<row>\d+), k_smem\.index\((?P<idx>[^)]*(?:\([^)]*\))?[^)]*)\)"
                      r"\.permute\(\(1, 0\)\), pd, use_acc=False, is_async=True\)$")


def _with_mask_mma(body: str) -> str:
    """After every QK^T issue of the generated body, chain the rank-1 mask wgmma on the same accumulator (under HAS_MASK)."""
    out, n_issue = [], 0
    for line in body.split("\n"):
        out.append(line)
        m = _S_ISSUE.match(line)
        if m:
            n_issue += 1
            ind, idx = m.group("ind"), m.group("idx")
            out.append(f"{ind}if HAS_MASK and (DBG & 1) == 0:")
            out.append(f"{ind}    s_nx = warpgroup_mma(q1_smem, kx_smem.index({idx}).permute((1, 0)), s_nx, is_async=True)")
    if n_issue == 0:
        raise RuntimeError("k13 codegen: no QK^T issue lines found in the generated body")
    return "\n".join(out)


def gen_epilogue_k13(rows: int, indent: str) -> list:
    """After the tile loop: phase B of the last row, drain; then either (EVEN_Q: S_q a multiple of 128, contiguous out) the
    three output tiles go registers -> o_smem -> TMA store (asynchronous; the previous item's stores are awaited first), or the
    masked per-thread global stores of k11/k12."""
    R = range(rows)
    ep = []
    ep.append(f"{indent}if STAGES == 2:")
    ep.append(f"{indent}    stl = st ^ 1")
    ep.append(f"{indent}else:")
    ep.append(f"{indent}    stl = (st + STAGES - 1) % STAGES")
    ep.append(f"{indent}acc{rows-1}, l{rows-1}, p16n, pd = _phase_b(acc{rows-1}, l{rows-1}, t_pv, nml_pv, v_smem.index(stl * ROWS + {rows-1}), P_L=P_L)")
    deps = ", ".join(f"acc{r}" for r in R)
    ep.append(f"{indent}{deps}, s_nx = warpgroup_mma_wait(num_outstanding=0, deps=[{deps}, s_nx])")
    ep.append(f"{indent}p16a = _keep(p16a)")
    ep.append(f"{indent}p16b = _keep(p16b)")
    ep.append(f"{indent}p16n = _keep(p16n)")
    ep.append(f"{indent}if EVEN_Q:")
    ep.append(f"{indent}    tma.store_wait(0)              # the previous item's output tiles have left o_smem")
    for r in R:
        ep.append(f"{indent}    l_o = gl.convert_layout(l{r}, OROW_L, assert_trivial=True)")
        ep.append(f"{indent}    o_smem.index({r}).store((acc{r} * gl.expand_dims(1.0 / l_o, 1)).to(Out.dtype.element_ty))")
    ep.append(f"{indent}    fence_async_shared()")
    for r in R:
        ep.append(f"{indent}    tma.async_copy_shared_to_global(o_desc, [orow{r} + q0, 0], o_smem.index({r}))   # rows past N_ROWS repeat row N-1 (identical bytes)")
    ep.append(f"{indent}else:")
    ep.append(f"{indent}    offs_q = q0 + WG * 64 + gl.arange(0, PM, layout=OROW_L)")
    ep.append(f"{indent}    offs_d = gl.arange(0, HEAD_DIM, layout=OCOL_L)")
    ep.append(f"{indent}    o_ptr = out_bh + gl.expand_dims(offs_q.to(gl.int64) * soq, 1) + gl.expand_dims(offs_d, 0)")
    ep.append(f"{indent}    q_ok = gl.expand_dims(offs_q < SEQ_Q, 1)")
    for r in R:
        ep.append(f"{indent}    l_o = gl.convert_layout(l{r}, OROW_L, assert_trivial=True)")
        ep.append(f"{indent}    o = (acc{r} * gl.expand_dims(1.0 / l_o, 1)).to(Out.dtype.element_ty)")
        cond = "q_ok" if r == 0 else f"q_ok & ((r0 + {r}) < N_ROWS)"
        ep.append(f"{indent}    gl.store(o_ptr + i{r} * soi, o, mask={cond})")
    return ep


def _render(rows: int) -> str:
    R = range(rows)
    src = _SRC.replace("ROWS_LIT", str(rows)).replace("KX_LIT", str(KX)).replace("ROWSP2_LIT", str(1 << (rows - 1).bit_length()))
    ir = []
    for r in R:
        ir.append(f"        i{r} = gl.minimum(r0 + {r}, N_ROWS - 1)")
        ir.append(f"        ii{r} = i{r}          # int32 copy for the TMA coordinate terms (computed inline at each load: nothing stays live)")

    for r in R:
        ir.append(f"        i{r} = i{r}.to(gl.int64)")
    irc = []
    for r in R:
        irc.append(f"        i{r} = gl.minimum(r0 + {r}, N_ROWS - 1)")
        irc.append(f"        orow{r} = ((b * N_ROWS + i{r}) * H + h) * SEQ_Q")
        irc.append(f"        i{r} = i{r}.to(gl.int64)")
    src = src.replace("ITEM_ROWS_C", "\n".join(irc)).replace("ITEM_ROWS", "\n".join(ir))
    rp = ["    rptr = mask_b + i0 * smi + ridx * 0"]
    for r in R:
        if r > 0:
            rp.append(f"    rptr = gl.where(ridx >= {r}, mask_b + i{r} * smi, rptr)")
    src = src.replace("ROW_PTRS", "\n".join(rp))
    src = src.replace("ITEM_IARGS", ", ".join(f"i{r}" for r in R))
    si, ss = [], []
    for r in R:
        si.append(f"    kept{r} = last_k * 0")
        ss.append(f"        mk = gl.load(mask_b + i{r} * smi + cols, mask=c_ok, other=0) != 0")
        ss.append(f"        kept{r} = kept{r} | gl.max(mk.to(gl.int32), axis=0)")
        ss.append(f"        last_k = gl.maximum(last_k, gl.max(gl.where(mk, cols, -1), axis=0))")
    tail = "    all_keep_something = (" + " & ".join(f"kept{r}" for r in R) + ") != 0"
    ql, pl, kx, mk0, mkn = [], [], [], [], []
    for r in R:
        ql.append(f"        tma.async_copy_global_to_shared(q_desc, [b * QRB + ii{r} * QRI + h * QRH + q0, b * QCB + ii{r} * QCI + h * QCH], qrb, q_smem.index(qb * ROWS + {r}))")
        pl.append(f"            tma.async_copy_global_to_shared(k_desc, [b * KRB + ii{r} * KRI + h * KRH + n0, b * KCB + ii{r} * KCI + h * KCH], rb, k_smem.index(st * ROWS + {r}))")
        pl.append(f"            tma.async_copy_global_to_shared(v_desc, [b * VRB + ii{r} * VRI + h * VRH + n0, b * VCB + ii{r} * VCI + h * VCH], rb, v_smem.index(st * ROWS + {r}))")
        mk0.append(f"            mk{r} = gl.load(mask_b + i{r} * smi + offs_k, mask=offs_k < SEQ_K, other=1)")
        kx.append(f"                    kx_smem.index(st * ROWS + {r}).store(gl.where(gl.expand_dims(mk{r} == 0, 1), xcol, 0.0).to(k_desc.dtype))")
        mkn.append(f"                mk{r} = gl.load(mask_b + i{r} * smi + k_nx, mask=k_ok, other=1)")
    src = src.replace("PRODUCER_QLOADS", "\n".join(ql)).replace("PRODUCER_LOADS", "\n".join(pl)).replace("PRODUCER_KX", "\n".join(kx))
    src = src.replace("PRODUCER_MK0", "\n".join(mk0)).replace("PRODUCER_MKNEXT", "\n".join(mkn))
    src = src.replace("PRODUCER_MKINIT", "\n".join(f"    mk{r} = gl.full([BLOCK_N], 1, gl.uint8, KROW_L)" for r in R))
    # the unmasked variant never touches mk*: give the loop-carried names a definition on that path too (dead code there)
    ci = []
    for r in R:
        ci.append(f"        qs{r} = q_smem.index(qb * ROWS + {r})")
        ci.append(f"        acc{r} = warpgroup_mma_init(gl.zeros([PM, HEAD_DIM], gl.float32, O_L))")
        ci.append(f"        m{r} = gl.full([PM], float('-inf'), gl.float32, ROW_L)")
        ci.append(f"        l{r} = gl.zeros([PM], gl.float32, ROW_L)")
    src = src.replace("CONSUMER_INIT", "\n".join(ci))
    src = src.replace("BODY_PLAIN", _with_mask_mma(gen_body(rows, "            ", "False")))
    src = src.replace("CONSUMER_EPILOGUE", "\n".join(gen_epilogue_k13(rows, indent="        ")))
    for ph_ in ("ITEM_", "PRODUCER_", "CONSUMER_", "SCAN_", "BODY_"):
        if ph_ in src:
            raise RuntimeError(f"k13 codegen: placeholder {ph_} left in the generated source")
    src = src.replace("MSEL_LIT", "-1.0e9")          # (k13 never selects per key: its masks go through the tensor core)
    return src


_MODULES: Dict[int, object] = {}


def kernel_for(rows: int):
    mod = _MODULES.get(rows)
    if mod is None:
        src = _render(rows)
        tag = hashlib.sha1(src.encode()).hexdigest()[:10]
        # the generated module: <this dir>/_gen/<name> on a writable install (unchanged); under a
        # read-only tree _gen_file writes it under the kit's JIT root, else the temp dir; a file of
        # that name already there is compared with src, never imported: _gen_exec runs src itself
        path = _gen_file(
            f"k13_rows{rows}_{tag}.py",
            src,
        )
        name = f"triattn_k13_gen_rows{rows}_{tag}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        _gen_exec(mod, src, path)
        _MODULES[rows] = mod
    return mod._fwd

_SENTINELS: Dict = {}


def _bf16_exact(x: float) -> bool:
    return torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).float().item() == x


def mask_sentinel(scale: float, dtype) -> tuple:
    """The additive mask term k13 injects through the MMA for masked keys: (E, aone, kxw, (x0, x1, x2, xr)); A operand
    columns < kxw hold `aone`, the masked rows of the B operand hold the pieces.

    Kept keys always weigh exactly 0 (the masked logit sits > 10^7 below any kept one).  A row that keeps NOTHING must yield
    the uniform average of v (cuEquivariance): all its logits must be identical and the base-2 softmax's FFMA(t, log2e,
    -m*log2e) must then be exactly 0, i.e. m = -2^E exactly (a power of two times log2e is exact in fp32).
    bf16: the raw accumulator receives X = x0 + x1 + x2 (three bf16-exact pieces of one fp32 number, weight 1 each), found
    by search so that fl32(X * scale + b) == -2^40 for every |b| <= 4096, while |X| ~ 2^42.5 absorbs any raw q.k up to 2^18:
    exactly uniform.  fp16 (max 65504, and P is rounded to fp16 for the PV product): one slot 2^14 x (-2^15) = -2^29, low
    enough that the FFMA's rounding cannot overflow fp16 P; such a row is finite and deterministic but NOT uniform (its
    weights follow the raw logits quantised to the accumulator spacing) -- documented in the README."""
    key = (float(scale), dtype)
    r = _SENTINELS.get(key)
    if r is not None:
        return r
    if dtype != torch.bfloat16:
        r = (29, 16384.0, 1, (-32768.0, 0.0, 0.0, 0.0))
        _SENTINELS[key] = r
        return r
    from fractions import Fraction
    sc = Fraction(float(torch.tensor(scale, dtype=torch.float32).item()))
    for E in (64,):                 # any E is equivalent for the search (self-similar); 64 absorbs raw |q.k| up to ~2^39 (> any fp16 product sum)
        x32 = torch.tensor(float(-(Fraction(2) ** E) / sc), dtype=torch.float32)
        cands = [x32.item()]
        up, dn = x32.clone(), x32.clone()
        for _ in range(64):
            up = torch.nextafter(up, torch.tensor(float("inf"))); cands.append(up.item())
            dn = torch.nextafter(dn, torch.tensor(float("-inf"))); cands.append(dn.item())
        lo, hi = -(Fraction(2) ** (E - 25)), Fraction(2) ** (E - 24)      # |t| rounds to 2^E iff |t| - 2^E in [lo, hi]
        bmax = Fraction(2) ** (E - 28)                                   # every |bias| <= 2^36 (E = 64) keeps t == -2^E
        best_c = None
        for X in cands:
            rho = abs(Fraction(X) * sc) - Fraction(2) ** E
            if lo + bmax <= rho <= hi - bmax:
                slack = min(rho - (lo + bmax), (hi - bmax) - rho)
                if best_c is None or slack > best_c[0]:
                    best_c = (slack, X)
        if best_c is not None:
            X = best_c[1]
            x0 = torch.tensor(X, dtype=torch.float32).to(torch.bfloat16).float().item()
            x1 = torch.tensor(X - x0, dtype=torch.float32).to(torch.bfloat16).float().item()
            x2 = X - x0 - x1
            if _bf16_exact(x2) and x0 + x1 + x2 == X:
                r = (E, 1.0, 3, (x0, x1, x2, 0.0))
                _SENTINELS[key] = r
                return r
    raise Unsupported(f"no exact mask sentinel found for scale={scale}")



def triattn_k13(q, k, v, bias, mask=None, scale=None, *, config: Optional[Dict] = None, out_dtype=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    rows, bn, stages = int(cfg["ROWS"]), int(cfg["BLOCK_N"]), int(cfg["STAGES"])
    if rows < 2 or (rows <= 3 and stages < 3) or stages < 2:
        raise ValueError("k13 config needs ROWS >= 2, STAGES >= 2, and STAGES >= 3 when ROWS <= 3 (stage-release ordering)")
    mask_u8 = None
    if mask is not None:
        if mask.dim() != 5:
            raise Unsupported(f"unsupported shape: mask must be [B,N,1,1,S]; got {tuple(mask.shape)}")
        mask_u8 = (mask if mask.dtype == torch.bool else mask.to(torch.bool)).view(torch.uint8)
        if mask_u8.stride(-1) != 1:
            mask_u8 = mask_u8.contiguous()
    # staging / descriptors exactly as k11/k12 but WITHOUT mask tables or folding (the mask goes through the tensor core)
    P = prepare(q, k, v, bias, None, scale, bn, need_square=True, dims=(16, 32))
    B, N, H, SQ, SK, D = P["B"], P["N"], P["H"], P["SQ"], P["SK"], P["D"]
    if mask_u8 is not None and tuple(mask_u8.shape) != (B, N, 1, 1, SK):
        raise Unsupported(f"unsupported shape: mask must be (B,N,1,1,S_kv); got {tuple(mask_u8.shape)}")
    out = torch.empty((B, N, H, SQ, D), dtype=out_dtype or P["q"].dtype, device=P["dev"])
    if out.numel() == 0:
        return out
    q_desc, k_desc, v_desc, b_desc, coords = _descriptors(P, bn)
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    o_desc = TensorDescriptor.from_tensor(out.view(B * N * H * SQ, D), [128, D], _nvmma_layout(128, D, out.dtype))
    even_q = (SQ % 128 == 0) and out.dtype in (torch.bfloat16, torch.float16) and int(cfg.get("TMA_EPI", 1)) == 1
    n_q, n_r = triton.cdiv(SQ, 128), triton.cdiv(N, rows)
    n_items = n_q * n_r * B * H
    if n_items >= (1 << 22):
        raise Unsupported(f"unsupported size: {n_items} work items (the in-kernel item decode is exact below 2^22)")
    grid = (min(n_items, _num_sms(P["dev"]) * int(cfg["CTAS_PER_SM"])),)
    has_mask = mask_u8 is not None
    E_, aone, kxw, pieces = mask_sentinel(P["scale"], P["q"].dtype) if has_mask else (0, 1.0, 1, (0.0, 0.0, 0.0, 0.0))
    dbg = torch.zeros((grid[0], 16, 8), dtype=torch.int64, device=P["dev"]) if int(cfg.get("DBG", 0)) & 8 else out
    ck = launch(_LAUNCHES, kernel_for(rows), grid, (
        q_desc, k_desc, v_desc, b_desc, o_desc, out, mask_u8 if has_mask else out, dbg,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        mask_u8.stride(0) if has_mask else 0, mask_u8.stride(1) if has_mask else 0,
        N, SQ, SK, H, n_q, n_r, n_items, 1.0 / n_q, 1.0 / n_r, 1.0 / H, P["scale"], *pieces,
        *coords["q"], *coords["k"], *coords["v"]),
        dict(BLOCK_N=bn, HEAD_DIM=D, STAGES=stages, HAS_MASK=has_mask, REGS_CONS=int(cfg["REGS_CONS"]), KXW=kxw, AONE=aone, DBG=int(cfg.get("DBG", 0)), EVEN_Q=even_q),
        key=call_key(P["q"], P["k"], P["v"], P["bias16"], mask_u8, out, extra=(rows, grid)),
        num_warps=4, maxnreg=int(cfg["maxnreg"]))
    INFO.clear(); INFO.update(P["info"])
    if int(cfg.get("DBG", 0)) & 8:
        INFO["dbg"] = dbg
    INFO["mask_mode"] = f"tensor-core injection (k13): masked logit level -2^{E_}, {kxw} k-slot(s)" if has_mask else "none"
    if ck is not None:
        INFO["kernel"] = dict(n_regs=getattr(ck, "n_regs", None), n_spills=getattr(ck, "n_spills", None),
                              shared=getattr(getattr(ck, "metadata", None), "shared", None))
    return out


def _gen_dirs():
    """Where a generated kernel module is written, in order: ``_GEN_DIR`` (<this directory>/_gen — a writable install: unchanged); under a
    read-only tree (a container image's file system, a shared read-only checkout) the kit's JIT root
    ``<MODEL_OPT_JIT_ROOT>[/<MODEL_OPT_STACK_KEY>]/opt_core_gen/triattn_pkg/dispatch/kernels/_gen``, then ``<tempdir>/opt_core_gen-uid<uid>/triattn_pkg/dispatch/kernels/_gen``.  The generated
    source is byte-identical wherever it lands (the Triton cache keys on the source text, not the path)."""
    rel = os.path.join("triattn_pkg", "dispatch", "kernels", "_gen")
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

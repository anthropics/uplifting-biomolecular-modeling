"""k12 -- k11's warp-specialized Gluon triangle-attention forward (H100) with a PERSISTENT work loop: the grid is one CTA per SM,
each CTA walks work items (q-tile, ROWS pair rows, batch*head) with a stride of the grid size, and the TMA producer runs ahead
across item boundaries (double-buffered Q tiles, one K/V/bias ring for the whole kernel), so the pipeline fill, barrier setup
and tail-wave costs of the one-item-per-CTA launch are paid once per SM instead of once per item.  Same numerics, mask modes
(batch OR pattern folded into the bias, per-row select only where a row differs), staging and API as k11 (see k11.py); the compute partition body is the shared codegen (k11.gen_body).
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

from .launch import call_key, launch
from .k11 import SRC_COMMON, gen_body, gen_body_masked, gen_epilogue, prepare, _descriptors

_GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen")
INFO: Dict = {}
_LAUNCHES: Dict = {}
DEFAULT_CONFIG = dict(ROWS=3, BLOCK_N=64, STAGES=3, REGS_CONS=232, maxnreg=168, CTAS_PER_SM=1)

_SRC = SRC_COMMON + '''

@gluon.jit
def _item(w, n_q, n_r, H, N_ROWS, SEQ_Q, SEQ_K):
    """Work item w -> (b, h, q0, r0); q tiles vary fastest so concurrently running CTAs share the K/V rows in L2."""
    pid_q = w % n_q
    t = w // n_q
    pid_r = t % n_r
    pid_bh = t // n_r
    b = pid_bh // H
    h = pid_bh % H
    return b, h, pid_bh, pid_q * BM, pid_r * ROWS


@gluon.jit
def _producer(q_desc, k_desc, v_desc, b_desc, q_smem, k_smem, v_smem, b_smem, ready, empty, q_ready, q_empty, RowInfo,
              srb, sri, N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items,
              QRB, QRI, QRH, QCB, QCI, QCH, KRB, KRI, KRH, KCB, KCI, KCH, VRB, VRI, VRH, VCB, VCI, VCH,
              BLOCK_N: gl.constexpr, STAGES: gl.constexpr, VAR_TILES: gl.constexpr, ITEMS: gl.constexpr):
    """TMA producer (default partition).  Per item: the ROWS Q tiles into Q buffer it % 2 (once the consumers released it two
    items ago), then per key tile ROWS K + ROWS V tiles + one [128, BLOCK_N] bias tile into the ring -- which never drains
    between items, so the next item's first stages load while the consumers finish the current one."""
    KV_BYTES: gl.constexpr = k_desc.block_type.nbytes
    B_BYTES: gl.constexpr = b_desc.block_type.nbytes
    Q_BYTES: gl.constexpr = q_desc.block_type.nbytes
    pid = gl.program_id(0)
    nprog = gl.num_programs(0)
    st = 0
    eph = 1          # parity to wait for on `empty[st]` before refilling stage st (first lap passes via pred)
    g = 0            # key tiles loaded so far (ring position)
    it = 0           # items started so far
    for w in range(pid, n_items, nprog):
        b, h, pid_bh, q0, r0 = _item(w, n_q, n_r, H, N_ROWS, SEQ_Q, SEQ_K)
ITEM_ROWS
        n_tiles = gl.cdiv(SEQ_K, BLOCK_N)
        n_sel = n_tiles
        if VAR_TILES:
ITEM_ROWINFO_P
        # ITEMS: 0 = every item; 1 = only items whose rows all equal the folded OR pattern (mask-free kernel variant);
        # 2 = only items with a row that needs the per-key select (HAS_MASK variant) -- the two variants split the work
        if ITEMS == 1:
            take = n_sel >= n_tiles
        elif ITEMS == 2:
            take = n_sel < n_tiles
        else:
            take = True
        if take:
          qb = it % 2
          # Q buffer qb was last used by taken item it-2: wait until the consumers released it, then load this item's Q tiles
          mbarrier.wait(q_empty.index(qb), ((it // 2) + 1) & 1, pred=it >= 2)
          qrb = q_ready.index(qb)
          mbarrier.expect(qrb, ROWS * Q_BYTES)
PRODUCER_QLOADS
          bias_row = pid_bh * SEQ_Q + q0
          for n in range(n_tiles):
              mbarrier.wait(empty.index(st), eph, pred=g >= STAGES)
              rb = ready.index(st)
              mbarrier.expect(rb, 2 * ROWS * KV_BYTES + B_BYTES)
              n0 = n * BLOCK_N
PRODUCER_LOADS
              tma.async_copy_global_to_shared(b_desc, [bias_row, n0], rb, b_smem.index(st))
              g += 1
              st += 1
              if st == STAGES:
                  st = 0
                  eph ^= 1
          it += 1


@gluon.jit
def _consumer(q_smem, k_smem, v_smem, b_smem, ready, empty, q_ready, q_empty, Out, Mask, RowInfo,
              sob, soi, soh, soq, smb, smi, srb, sri,
              N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, qk_scale,
              BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, STAGES: gl.constexpr,
              HAS_MASK: gl.constexpr, VAR_TILES: gl.constexpr, ITEMS: gl.constexpr):
    """The compute partition (8 warps = two warpgroups x 64 query rows, one code stream), persistent over work items; per
    item the k11 schedule: software pipeline over the ROWS pair rows of each key tile (wait QK^T_j | issue QK^T_{j+1} |
    phase A of row j | phase B of row j-1), dead-P register donors, explicit P liveness."""
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
    dt: gl.constexpr = v_smem.dtype
    pid = gl.program_id(0)
    nprog = gl.num_programs(0)

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
        b, h, pid_bh, q0, r0 = _item(w, n_q, n_r, H, N_ROWS, SEQ_Q, SEQ_K)
ITEM_ROWS_C
        n_tiles = gl.cdiv(SEQ_K, BLOCK_N)
        n_sel = n_tiles
        if VAR_TILES:
ITEM_ROWINFO_C
        if ITEMS == 1:
            take = n_sel >= n_tiles
        elif ITEMS == 2:
            take = n_sel < n_tiles
        else:
            take = True
        if take:
          out_bh = Out + b.to(gl.int64) * sob + h.to(gl.int64) * soh
          mask_b = Mask + b.to(gl.int64) * smb
          qb = it % 2
          mbarrier.wait(q_ready.index(qb), (it // 2) & 1)          # completion it//2 of q_ready[qb]
CONSUMER_INIT
          # prologue: QK^T of row 0, tile 0 of this item (ring position continues from the previous item)
          mbarrier.wait(ready.index(st), ph)
          s_nx = warpgroup_mma(qs0, k_smem.index(st * ROWS).permute((1, 0)), pd, use_acc=False, is_async=True)
          # key tiles [0, n_sel): mask-free body; [n_sel, n_tiles): per-key select (n_sel == n_tiles without a mask)
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
                  b32 = b_smem.index(st).load(S_L).to(gl.float32)
BODY_MASKED
                  st = st1
                  ph = ph1
CONSUMER_EPILOGUE
          # everything has retired: release the ring stage(s) this item still holds (its last tile; the last two when ROWS == 2)
          # and its Q buffer, so the producer keeps streaming the next items
          mbarrier.arrive(empty.index(stl), count=1)
          if ROWS == 2:
              mbarrier.arrive(empty.index((stl + STAGES - 1) % STAGES), count=1, pred=n_tiles >= 2)
          mbarrier.arrive(q_empty.index(qb), count=1)
          it += 1


@gluon.jit
def _fwd(q_desc, k_desc, v_desc, b_desc, Out, Mask, RowInfo,
         sob, soi, soh, soq, smb, smi, srb, sri,
         N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, qk_scale,
         QRB, QRI, QRH, QCB, QCI, QCH, KRB, KRI, KRH, KCB, KCI, KCH, VRB, VRI, VRH, VCB, VCI, VCH,
         BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr, STAGES: gl.constexpr, HAS_MASK: gl.constexpr,
         REGS_CONS: gl.constexpr, VAR_TILES: gl.constexpr, ITEMS: gl.constexpr):
    q_smem = gl.allocate_shared_memory(q_desc.dtype, [2 * ROWS, BM, HEAD_DIM], q_desc.layout)      # double-buffered per item
    k_smem = gl.allocate_shared_memory(k_desc.dtype, [STAGES * ROWS, BLOCK_N, HEAD_DIM], k_desc.layout)
    v_smem = gl.allocate_shared_memory(v_desc.dtype, [STAGES * ROWS, BLOCK_N, HEAD_DIM], v_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, [STAGES, BM, BLOCK_N], b_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    q_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    q_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)
    for s in gl.static_range(2):
        mbarrier.init(q_ready.index(s), count=1)
        mbarrier.init(q_empty.index(s), count=1)
    fence_async_shared()
    gl.warp_specialize([
        (_producer, (q_desc, k_desc, v_desc, b_desc, q_smem, k_smem, v_smem, b_smem, ready, empty, q_ready, q_empty, RowInfo,
                     srb, sri, N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items,
                     QRB, QRI, QRH, QCB, QCI, QCH, KRB, KRI, KRH, KCB, KCI, KCH, VRB, VRI, VRH, VCB, VCI, VCH, BLOCK_N, STAGES, VAR_TILES, ITEMS)),
        (_consumer, (q_smem, k_smem, v_smem, b_smem, ready, empty, q_ready, q_empty, Out, Mask, RowInfo,
                     sob, soi, soh, soq, smb, smi, srb, sri,
                     N_ROWS, SEQ_Q, SEQ_K, H, n_q, n_r, n_items, qk_scale,
                     BLOCK_N, HEAD_DIM, STAGES, HAS_MASK, VAR_TILES, ITEMS)),
    ], [8], [REGS_CONS])
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(ready.index(s))
        mbarrier.invalidate(empty.index(s))
    for s in gl.static_range(2):
        mbarrier.invalidate(q_ready.index(s))
        mbarrier.invalidate(q_empty.index(s))
'''


def _render(rows: int, fp16: bool = False) -> str:
    R = range(rows)
    src = _SRC.replace("ROWS_LIT", str(rows))
    # per-item pair rows (clamped duplicates past N_ROWS are computed and not stored) -> flattened (b, i, h) K/V row index x_r
    ir = []
    for r in R:
        ir.append(f"        i{r} = gl.minimum(r0 + {r}, N_ROWS - 1)")
        ir.append(f"        ii{r} = i{r}          # int32 copy for the TMA coordinate terms (computed inline at each load: nothing stays live)")
    for r in R:
        ir.append(f"        i{r} = i{r}.to(gl.int64)")
    irc = []
    for r in R:
        irc.append(f"        i{r} = gl.minimum(r0 + {r}, N_ROWS - 1)")
        irc.append(f"        i{r} = i{r}.to(gl.int64)")
    src = src.replace("ITEM_ROWS_C", "\n".join(irc)).replace("ITEM_ROWS", "\n".join(ir))
    for tag, want_sel in (("ITEM_ROWINFO_P", True), ("ITEM_ROWINFO_C", True)):
        fl = ["            rib = RowInfo + b.to(gl.int64) * srb", "            nt = gl.load(rib + i0 * sri)"]
        if want_sel:
            fl.append("            ns = gl.load(rib + i0 * sri + 1)")
        for r in R:
            if r == 0:
                continue
            fl.append(f"            nt = gl.maximum(nt, gl.load(rib + i{r} * sri))")
            if want_sel:
                fl.append(f"            ns = gl.minimum(ns, gl.load(rib + i{r} * sri + 1))")
        fl.append("            n_tiles = gl.minimum(nt, n_tiles)")
        if want_sel:
            fl.append("            n_sel = gl.minimum(ns, n_tiles)")
        src = src.replace(tag, "\n".join(fl))
    ql, pl = [], []
    for r in R:
        ql.append(f"          tma.async_copy_global_to_shared(q_desc, [b * QRB + ii{r} * QRI + h * QRH + q0, b * QCB + ii{r} * QCI + h * QCH], qrb, q_smem.index(qb * ROWS + {r}))")
        pl.append(f"              tma.async_copy_global_to_shared(k_desc, [b * KRB + ii{r} * KRI + h * KRH + n0, b * KCB + ii{r} * KCI + h * KCH], rb, k_smem.index(st * ROWS + {r}))")
        pl.append(f"              tma.async_copy_global_to_shared(v_desc, [b * VRB + ii{r} * VRI + h * VRH + n0, b * VCB + ii{r} * VCI + h * VCH], rb, v_smem.index(st * ROWS + {r}))")
    src = src.replace("PRODUCER_QLOADS", "\n".join(ql)).replace("PRODUCER_LOADS", "\n".join(pl))
    ci = []
    for r in R:
        ci.append(f"          qs{r} = q_smem.index(qb * ROWS + {r})")
        ci.append(f"          acc{r} = warpgroup_mma_init(gl.zeros([PM, HEAD_DIM], gl.float32, O_L))")
        ci.append(f"          m{r} = gl.full([PM], float('-inf'), gl.float32, ROW_L)")
        ci.append(f"          l{r} = gl.zeros([PM], gl.float32, ROW_L)")
    src = src.replace("CONSUMER_INIT", "\n".join(ci))
    src = src.replace("BODY_PLAIN", gen_body(rows, "              ", "False"))
    src = src.replace("BODY_MASKED", gen_body_masked(rows, "                  "))
    src = src.replace("CONSUMER_EPILOGUE", "\n".join(gen_epilogue(rows, indent="          ")))
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
            f"k12_rows{rows}_{tag}.py",
            src,
        )
        name = f"triattn_k12_gen_rows{rows}_{tag}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        _gen_exec(mod, src, path)
        _MODULES[(rows, fp16)] = mod
    return mod._fwd


_NUM_SMS: Dict = {}


def _num_sms(dev) -> int:
    n = _NUM_SMS.get(dev)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _NUM_SMS[dev] = n
    return n


def triattn_k12(q, k, v, bias, mask=None, scale=None, *, config: Optional[Dict] = None, out_dtype=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    rows, bn, stages = int(cfg["ROWS"]), int(cfg["BLOCK_N"]), int(cfg["STAGES"])
    if rows < 2 or (rows <= 3 and stages < 3) or stages < 2:
        raise ValueError("k12 config needs ROWS >= 2, STAGES >= 2, and STAGES >= 3 when ROWS <= 3 (stage-release ordering)")
    P = prepare(q, k, v, bias, mask, scale, bn, need_square=True, dims=(16, 32))
    B, N, H, SQ, SK, D = P["B"], P["N"], P["H"], P["SQ"], P["SK"], P["D"]
    out = torch.empty((B, N, H, SQ, D), dtype=out_dtype or P["q"].dtype, device=P["dev"])
    if out.numel() == 0:
        return out
    q_desc, k_desc, v_desc, b_desc, coords = _descriptors(P, bn)
    dummy = out
    n_q, n_r = triton.cdiv(SQ, 128), triton.cdiv(N, rows)
    n_items = n_q * n_r * B * H
    grid = (min(n_items, _num_sms(P["dev"]) * int(cfg["CTAS_PER_SM"])),)
    kern = kernel_for(rows, P["q"].dtype == torch.float16)

    def launch_(items: int, has_mask: bool):
        return launch(_LAUNCHES, kern, grid, (q_desc, k_desc, v_desc, b_desc, out, P["mask"] if P["has_mask"] else dummy, P["rowinfo"] if P["has_mask"] else dummy,
                      out.stride(0), out.stride(1), out.stride(2), out.stride(3), P["smb"], P["smi"], P["srb"], P["sri"],
                      N, SQ, SK, H, n_q, n_r, n_items, P["scale"],
                      *coords["q"], *coords["k"], *coords["v"]),
                      dict(BLOCK_N=bn, HEAD_DIM=D, STAGES=stages, HAS_MASK=has_mask,
                           REGS_CONS=int(cfg["REGS_CONS"]), VAR_TILES=P["has_mask"], ITEMS=items),
                      key=call_key(P["q"], P["k"], P["v"], P["bias16"], P["mask"], out, extra=(rows, grid)),
                      num_warps=4, maxnreg=int(cfg["maxnreg"]))

    if P["has_mask"]:
        # two passes over the work items, split on the device by the per-row table: the mask-free kernel variant takes the
        # items whose rows all equal the OR pattern folded into the bias (every item, for padding masks), the per-key-select
        # variant takes the rest (an empty pass costs a few microseconds of table reads)
        ck = launch_(1, False)
        launch_(2, True)
    else:
        ck = launch_(0, False)
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

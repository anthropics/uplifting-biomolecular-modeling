# Portions derived from Protenix v2.0.0 (https://github.com/bytedance/Protenix), Copyright 2024 ByteDance and/or its affiliates,
# used under the Apache License, Version 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ptx_tp.trimul -- tensor-parallel TriangleMultiplicativeUpdate (Protenix-v2,
torch statement of `TriangleMultiplicativeUpdate._inference_forward`).

tp_trimul(mod, z_shard, layout, *, outgoing, with_add=True) -> z_shard

Outgoing (x_ij = sum_k a_ik b_jk)  = CONTRACTION-ROW:  LN_in / projections /
gates are row-local on the rank's rows; b row-slabs stream around the ring;
tile x[I_sub, J_sub] = a_I . b_J^T over the FULL k in one batched matmul
(batch = hidden channels, exactly stock's `torch.matmul(a, b_chunk)` cuBLAS
call up to (m, n) extents); epilogue = stock's in-place `_inference_forward`
epilogue: permute -> layer_norm_out -> linear_z -> *= sigmoid(linear_g(
layer_norm_in(z tile))) -> z tile += (with_add) / = (else).
Incoming (x_ij = sum_k a_ki b_kj) = CONTRACTION-COL: projections of the rank's
rows k; all-to-all a[:, I_sub] -> A[c, I_sub, k] and b[:, J_slab] -> B[c, k, J]
on the row owners, then the same ring/tile loop (tiles land in row layout).

Read-before-write discipline (stock has the same hazard and solves it with the
z-cache): a for pass p is projected from rows I_sub_p BEFORE any tile of that
pass is written; b slabs are projected from the pre-update z at op start when
cached (default whenever R*N*c*2B <= PTX_TP_TRIMUL_CACHE_GB, 24 GB); g tiles
are taken from the tile's own pre-update values; whenever a provider would
have to read rows that earlier passes already updated (b not cached, or
incoming with >1 pass) a pre-update copy of the shard is used
(PTX_TP_TRIMUL_ZCOPY = auto|device|host; host = pinned, streamed back in
contiguous row blocks on a side stream).

Only the 'torch' triangle_multiplicative statement is reproduced (the fused
cuEquivariance kernel cannot be sharded bit-exactly); mask=None (trunk) or a
row shard of the [N,N] pair mask are supported.
"""
from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Tuple

import torch

from .dist import Layout, allreduce_, is_dist, zrows
from .contract import (ContractStats, colshard_inner_contract, default_rows_a, default_rows_b,
                       max_rows_per_launch, rowshard_outer_contract, INT32_MAX)

__all__ = ["tp_trimul", "agreed_rows", "reserve_pinned_pool", "stock_column_grid", "SUPPORTED_TRIANGLE_MULTIPLICATIVE"]

SUPPORTED_TRIANGLE_MULTIPLICATIVE = ("torch",)


def agreed_rows(N: int, C: int, elt_bytes: int, layout: Layout, RA: Optional[int] = None, RB: Optional[int] = None,
                device=None) -> Tuple[int, int]:
    """(RA, RB) of one tp_trimul call — the resident A block rows per pass and the streamed B slab rows — as ONE pair on every
    rank. npass = ceil(Rmax / RA) and nslab = ceil(Rmax / RB) are how many passes, b slabs, ring rounds and all-to-all windows
    the call issues, i.e. its collective SEQUENCE, so both numbers must be the same on every rank or the ranks wait in different
    collectives. A value the caller passes is taken as given (callers pass one value on every rank). A value left None is sized
    by ``contract.default_rows_a`` / ``default_rows_b`` — whose byte budget, absent an explicit one, is a share of THIS rank's
    free device memory, and free memory differs between ranks (the ragged last shard is smaller, rank 0 keeps featurization
    residue, allocator histories differ) — and is then AGREED as the minimum over ranks: one 2-element int64 all-reduce per call,
    at a point every rank reaches. The minimum is the block size that fits the most loaded rank; the sizing rule stays the
    contract module's. Without a group, at P == 1 or on a replicated layout (no shard, no collective anywhere) the local values are
    the answer. ``device``: where the 2-element tensor of the all-reduce lives (the group's device: the rank's CUDA device under
    NCCL, the CPU under gloo; default the current CUDA device when CUDA is present, else the CPU)."""
    ra = int(RA) if RA else int(default_rows_a(N, C, elt_bytes, layout.Rmax))
    rb = int(RB) if RB else int(default_rows_b(N, C, elt_bytes, layout.Rmax))
    if ra < 1 or rb < 1:
        raise ValueError(f"agreed_rows: RA={ra} RB={rb} must be >= 1")
    if (RA and RB) or not is_dist() or layout.P == 1 or layout.replicated:
        return ra, rb
    dev = device if device is not None else (torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"))
    t = torch.tensor([ra, rb], dtype=torch.int64, device=dev)
    allreduce_(t, "min")
    ra_min, rb_min = (int(v) for v in t.tolist())
    return (ra if RA else ra_min), (rb if RB else rb_min)


def _verbose() -> int:
    return int(os.environ.get("PTX_TP_VERBOSE", "1"))


def _log(layout: Layout, msg: str):
    if _verbose() and layout.rank == 0:
        print(f"[ptx_tp.trimul] {msg}", flush=True)


# --------------------------------------------------------------------------
# row-local pieces of the stock statement
# --------------------------------------------------------------------------
def _proj_rows(mod, pair_rows: torch.Tensor, mask_rows: torch.Tensor, which: str) -> torch.Tensor:
    """stock `compute_projection_helper` minus the final permute:
    pair = LN_in(pair); p = linear_g(pair); p.sigmoid_(); p *= linear_p(pair); p *= mask
    -> [rows, cols, c_hidden] (channel-last)."""
    if which == "a":
        linear_g, linear_p = mod.linear_a_g, mod.linear_a_p
    else:
        linear_g, linear_p = mod.linear_b_g, mod.linear_b_p
    pair = mod.layer_norm_in(pair_rows)
    p = linear_g(pair)
    p.sigmoid_()
    p *= linear_p(pair)
    p *= mask_rows
    return p


def _epilogue_factory(mod, z_shard, mask3, with_add: bool):
    """stock `_inference_forward` per-chunk epilogue on tile T[c, rows, cols]."""
    c_z = mod.c_z

    def epilogue(T: torch.Tensor, rows, cols):
        i0, i1 = rows
        c0, c1 = cols
        assert (i1 - i0) * (c1 - c0) * max(c_z, T.shape[0]) <= INT32_MAX
        x = T.permute(1, 2, 0)                      # permute_final_dims(x_chunk, (1, 2, 0))
        x = mod.layer_norm_out(x)
        x = mod.linear_z(x)
        z_tile = zrows(z_shard, i0, i1)[:, c0:c1]   # pre-update values of this tile
        g = mod.linear_g(mod.layer_norm_in(z_tile))
        g.sigmoid_()
        x *= g
        del g
        if with_add:
            z_tile += x                             # z[..., cols] += x_chunk
        else:
            z_tile.copy_(x)                         # z[..., cols] = x_chunk
        del x

    return epilogue


class _PinnedPool:
    """One growable pinned host buffer, handed out in regions per op call (pinning is slow:
    allocate once, reuse across calls)."""
    buf: Optional[torch.Tensor] = None

    @classmethod
    def reserve(cls, nbytes: int, pin: bool):
        if cls.buf is None or cls.buf.numel() < nbytes or (pin and not cls.buf.is_pinned()):
            env_min = int(float(os.environ.get("PTX_TP_PINNED_POOL_GB", "0")) * 1e9)
            cls.buf = None
            cls.buf = torch.empty(max(nbytes, env_min, 1), dtype=torch.uint8, device="cpu", pin_memory=pin)
        cls.off = 0

    @classmethod
    def take(cls, shape, dtype) -> torch.Tensor:
        n = 1
        for d in shape:
            n *= int(d)
        nb = n * torch.empty((), dtype=dtype).element_size()
        nb_al = (nb + 511) // 512 * 512
        assert cls.buf is not None and cls.off + nb_al <= cls.buf.numel(), "pinned pool too small (reserve first)"
        t = cls.buf[cls.off:cls.off + nb].view(dtype).view(*shape) if n > 0 else torch.empty(shape, dtype=dtype)
        cls.off += nb_al
        return t


def reserve_pinned_pool(nbytes: Optional[int] = None) -> int:
    """Pre-allocate (pin) the host pool used by bcache='host' / zcopy='host' ONCE at start-up
    (pinning costs ~1 s per GB in some sandboxes; the pool is reused by every later call).
    nbytes=None -> PTX_TP_PINNED_POOL_GB env (default 0 = nothing).  Returns pool bytes."""
    if nbytes is None:
        nbytes = int(float(os.environ.get("PTX_TP_PINNED_POOL_GB", "0")) * 1e9)
    if nbytes > 0 and (_PinnedPool.buf is None or _PinnedPool.buf.numel() < nbytes):
        _PinnedPool.reserve(nbytes, pin=torch.cuda.is_available())
    return 0 if _PinnedPool.buf is None else int(_PinnedPool.buf.numel())


class _HostSlabs:
    """Projected b slabs parked in pinned host memory once per call; get(jb) returns the slab on
    the device, prefetching the next slab on a side stream while the current one is consumed
    (H2D per pass = shard bytes; no re-projection)."""

    def __init__(self, nslab: int, make_fn, device, expected_gets: int, stats):
        self.n, self.device, self.left, self.st = nslab, device, expected_gets, stats
        self.cuda = torch.device(device).type == "cuda"
        self.host, self.pending = [], {}
        for jb in range(nslab):
            d = make_fn(jb)
            if self.cuda:
                h = _PinnedPool.take(d.shape, d.dtype)
                h.copy_(d, non_blocking=True)          # D2H on the current stream (ordered after the producer)
            else:
                h = d.clone()
            self.host.append(h)
            del d
        self.side = torch.cuda.Stream(device=device) if self.cuda else None

    def _prefetch(self, jb):
        h = self.host[jb]
        if not self.cuda:
            self.pending[jb] = (h, None)
            return
        cur = torch.cuda.current_stream(self.device)
        buf = torch.empty(h.shape, dtype=h.dtype, device=self.device)
        self.side.wait_stream(cur)                     # buffer reuse + D2H completion ordering
        with torch.cuda.stream(self.side):
            buf.copy_(h, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.side)
        self.st.h2d_bytes = getattr(self.st, "h2d_bytes", 0) + h.numel() * h.element_size()
        self.pending[jb] = (buf, ev)

    def close(self):
        """drop device-side prefetch buffers and host views (idempotent)."""
        self.pending = {}
        self.host = []
        self.left = 0

    def get(self, jb) -> torch.Tensor:
        if jb not in self.pending:
            self._prefetch(jb)
        buf, ev = self.pending.pop(jb)
        if ev is not None:
            torch.cuda.current_stream(self.device).wait_event(ev)
        self.left -= 1
        nxt = (jb + 1) % self.n
        if self.left > 0 and nxt not in self.pending:
            self._prefetch(nxt)
        return buf


class _ZSource:
    """Access to PRE-UPDATE rows of the shard.
    mode 'none'  : the live tensor (caller guarantees rows are untouched when read)
         'device': full device clone taken at op start
         'host'  : full pinned-host copy taken at op start, streamed back in contiguous row blocks
         'lazy'  : device stash of only the row blocks already handed to a pass (prefix [0, upto));
                   mark_pass(i0, i1) must be called at the start of every pass; capacity rows =
                   (passes-1)*RA (or R when the current pass's own rows are re-read while written)."""

    def __init__(self, z_shard: torch.Tensor, mode: str, capacity_rows: int = 0):
        self.mode, self.z, self.upto = mode, z_shard, 0
        self.cuda = z_shard.is_cuda
        if mode == "device":
            self.src = z_shard.clone()
        elif mode == "host":
            if self.cuda:
                self.src = _PinnedPool.take(z_shard.shape, z_shard.dtype)
                self.src.copy_(z_shard)
                self.side = torch.cuda.Stream(device=z_shard.device)
            else:
                self.src = z_shard.clone()
                self.side = None
        elif mode == "lazy":
            self.cap = max(0, min(int(capacity_rows), z_shard.shape[0]))
            self.stash = torch.empty((self.cap,) + tuple(z_shard.shape[1:]), dtype=z_shard.dtype, device=z_shard.device)
        else:
            self.src = z_shard

    def close(self):
        """drop the device stash / host copy / streaming buffers (idempotent)."""
        for name in ("stash", "src", "bufs", "pending", "full"):
            if hasattr(self, name):
                setattr(self, name, None)

    def mark_pass(self, i0: int, i1: int):
        """rows [i0, i1) are about to be (re)written by the pass starting now."""
        if self.mode != "lazy":
            return
        j1 = min(i1, self.cap)
        if j1 > i0:
            self.stash[i0:j1].copy_(self.z[i0:j1])
        self.upto = max(self.upto, min(i1, self.cap))

    def _dev_rows(self, i0, i1):
        if self.mode == "lazy":
            if i1 <= self.upto:
                return self.stash[i0:i1]
            if i0 >= self.upto:
                return self.z[i0:i1]
            return torch.cat([self.stash[i0:self.upto], self.z[self.upto:i1]], dim=0)
        return self.src[i0:i1]

    def rows(self, i0: int, i1: int) -> torch.Tensor:
        """rows i0:i1 on the compute device (contiguous row block)."""
        if self.mode != "host" or not self.cuda:
            return self._dev_rows(i0, i1) if self.mode != "host" else self.src[i0:i1].to(self.z.device)
        out = torch.empty((i1 - i0,) + tuple(self.src.shape[1:]), dtype=self.src.dtype, device=self.z.device)
        cur = torch.cuda.current_stream(self.z.device)
        self.side.wait_stream(cur)
        with torch.cuda.stream(self.side):
            out.copy_(self.src[i0:i1], non_blocking=True)
        cur.wait_stream(self.side)
        return out

    def row_blocks(self, R: int, er: int):
        """(double-buffered for host mode) iterator over contiguous row blocks (i0, i1, rows_on_device)."""
        blocks = [(i0, min(R, i0 + er)) for i0 in range(0, R, er)]
        if self.mode != "host" or not self.cuda:
            for i0, i1 in blocks:
                yield i0, i1, (self._dev_rows(i0, i1) if self.mode != "host" else self.src[i0:i1].clone())
            return
        cur = torch.cuda.current_stream(self.z.device)

        def prefetch(i0, i1):
            buf = torch.empty((i1 - i0,) + tuple(self.src.shape[1:]), dtype=self.src.dtype, device=self.z.device)
            self.side.wait_stream(cur)
            with torch.cuda.stream(self.side):
                buf.copy_(self.src[i0:i1], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.side)
            return buf, ev

        pend = prefetch(*blocks[0]) if blocks else None
        for k, (i0, i1) in enumerate(blocks):
            buf, ev = pend
            pend = prefetch(*blocks[k + 1]) if k + 1 < len(blocks) else None
            cur.wait_event(ev)
            yield i0, i1, buf




class _ColumnSweep:
    """proj(which) of this rank's rows restricted to a set of column windows, computed in ONE
    sweep over er-row blocks (module-level class: no per-call class/closure reference cycles)."""

    def __init__(self, which: str, use_copy: bool, mod, mask3, R: int, C: int, er: int, dtype, device, z_shard, zsrc_ref):
        self.which, self.use_copy = which, use_copy
        self.mod, self.mask3, self.R, self.C, self.er, self.dtype, self.device = mod, mask3, R, C, er, dtype, device
        self.z_shard, self.zsrc_ref = z_shard, zsrc_ref          # zsrc_ref: 1-element list holding the pre-update source
        self.cache, self.pending = {}, []

    def get(self, c0, c1):
        key = (c0, c1)
        if key not in self.cache:
            self.cache = self.compute(self.pending)
        return self.cache.pop(key)

    def compute(self, windows):
        outs = {w: torch.empty((self.R, w[1] - w[0], self.C), dtype=self.dtype, device=self.device) for w in windows}
        src = self.zsrc_ref[0] if self.use_copy else _ZSource(self.z_shard, "none")
        for i0, i1, blk in src.row_blocks(self.R, self.er):
            for (c0, c1), o in outs.items():
                if c1 > c0:
                    o[i0:i1].copy_(_proj_rows(self.mod, blk[:, c0:c1], self.mask3[i0:i1, c0:c1], self.which))
            del blk
        return outs

    def close(self):
        self.cache = {}
        self.pending = []
        self.mod = self.mask3 = self.z_shard = self.zsrc_ref = None


def stock_column_grid(n: int, inplace_chunk_size: int = 256):
    """Column-chunk grid of stock `_inference_forward` (global coordinates): chunks of
    inplace_chunk_size up to half_n = ceil(n/2) with the last one contracted to end at half_n,
    then chunks of inplace_chunk_size from half_n (last clipped at n)."""
    half_n = n // 2 + n % 2
    i_range = list(range(0, half_n, inplace_chunk_size))
    offs = [i2 - i1 for i1, i2 in zip(i_range, i_range[1:] + [half_n])]
    after = list(range(half_n, n, inplace_chunk_size))
    return [(i, i + o) for i, o in zip(i_range, offs)] + [(i, min(n, i + inplace_chunk_size)) for i in after]


class _GridAssembler:
    """Re-cuts streamed b blocks (arbitrary global column ranges, arriving in ring/slab order)
    into the STOCK global column chunks, so that every tile GEMM has exactly stock's (n, k) and
    operand layout (cuBLAS bf16 GEMM output bits depend on the column partition at some N, e.g.
    2104; they do not depend on m = the row count).  kind 'bT': blocks/chunks are [C, cols, N]
    (outgoing; GEMM uses chunk.transpose(-1,-2) like stock); kind 'kw': [C, N, cols] (incoming)."""

    def __init__(self, grid, C: int, N: int, kind: str, dtype, device):
        import bisect
        self.grid, self.C, self.N, self.kind = grid, C, N, kind
        self.starts = [g[0] for g in grid]
        self.dtype, self.device = dtype, device
        self.pending = {}
        self._bisect = bisect.bisect_right

    def feed(self, c0: int, blk: torch.Tensor):
        v = blk.shape[1] if self.kind == "bT" else blk.shape[2]
        c1 = c0 + v
        out = []
        idx = max(0, self._bisect(self.starts, c0) - 1)
        while idx < len(self.grid) and self.grid[idx][0] < c1:
            s, e = self.grid[idx]
            idx += 1
            if e <= c0:
                continue
            a, b = max(s, c0), min(e, c1)
            piece = blk[:, a - c0:b - c0, :] if self.kind == "bT" else blk[:, :, a - c0:b - c0]
            if a == s and b == e:
                out.append((s, e, piece.contiguous()))
                continue
            ent = self.pending.get((s, e))
            if ent is None:
                shape = (self.C, e - s, self.N) if self.kind == "bT" else (self.C, self.N, e - s)
                ent = self.pending[(s, e)] = [torch.empty(shape, dtype=self.dtype, device=self.device), 0]
            if self.kind == "bT":
                ent[0][:, a - s:b - s, :].copy_(piece)
            else:
                ent[0][:, :, a - s:b - s].copy_(piece)
            ent[1] += b - a
            if ent[1] == e - s:
                out.append((s, e, ent[0]))
                del self.pending[(s, e)]
        return out

    def finish(self):
        assert not self.pending, f"unassembled column chunks left: {sorted(self.pending)}"


def _resolve_modes(outgoing, npass, RA, R, N, C, CZ, elt, Rmax, cache_b, bcache, zcopy, is_cuda):
    """-> (bcache in {'device','host','none'}, zmode in {'none','device','host','lazy'}, lazy_capacity_rows)"""
    env_b = os.environ.get("PTX_TP_TRIMUL_BCACHE")            # device|host|none
    if bcache is None:
        if cache_b is not None:
            bcache = "device" if cache_b else (env_b if env_b in ("host", "none") else "host")
        elif env_b in ("device", "host", "none"):
            bcache = env_b
        else:
            env = os.environ.get("PTX_TP_TRIMUL_CACHE_B")
            if env is not None:
                bcache = "device" if int(env) else "host"
            else:
                fits = Rmax * N * C * elt <= float(os.environ.get("PTX_TP_TRIMUL_CACHE_GB", "24")) * 1e9
                bcache = "device" if fits else "host"
    assert bcache in ("device", "host", "none"), bcache
    reread_own = (bcache == "none")                             # b re-projected from rows while they are written
    need_copy = reread_own or ((not outgoing) and npass > 1)
    if zcopy is None:
        zcopy = os.environ.get("PTX_TP_TRIMUL_ZCOPY", "auto")
    if not need_copy:
        zmode = "none"
    elif zcopy in ("device", "host", "lazy"):
        zmode = zcopy
    else:
        zmode = "lazy"
    cap = R if reread_own else min(R, (npass - 1) * RA)
    return bcache, zmode, cap


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------
def tp_trimul(mod, z_shard: torch.Tensor, layout: Layout, *, outgoing: bool, with_add: bool = True,
              mask_shard: Optional[torch.Tensor] = None, RA: Optional[int] = None, RB: Optional[int] = None,
              cache_b: Optional[bool] = None, bcache: Optional[str] = None, zcopy: Optional[str] = None,
              inplace_chunk_size: int = 256, grid: Optional[str] = None,
              stats: Optional[ContractStats] = None) -> torch.Tensor:
    """Tensor-parallel `mod._inference_forward(z, mask, inplace_chunk_size, with_add)`
    on a row shard.  z_shard[R, N, c_z] is updated IN PLACE and returned.
    mask_shard: rows r0:r1 of the [N, N] pair mask (None = ones, as in the trunk).
    Tiling: RA (resident A rows per pass), RB (streamed B slab rows) -- one pair on every rank
    (`agreed_rows`: sized values are the minimum over ranks); b slabs cached on
    'device' (cache_b=True), parked in pinned 'host' memory with side-stream prefetch, or
    'none' (re-projected per pass/slab from a pre-update copy).  Pre-update copy (zcopy) is only
    needed for bcache='none' or incoming with >1 pass: 'lazy' (device stash of the already
    updated row blocks only, default), 'device' (full clone) or 'host' (pinned full copy).
    Tile GEMMs are issued per STOCK column chunk (`stock_column_grid(N, inplace_chunk_size)`,
    grid='stock', default; PTX_TP_TRIMUL_GRID=slab restores per-slab tiles) because the bf16
    cuBLAS GEMM is bitwise invariant to the row count but NOT to the column partition at some N.
    Env: PTX_TP_TRIMUL_ROWS_A/_ROWS_B/_ABUF_GB/_BBUF_GB/_CACHE_GB/_CACHE_B/_BCACHE/_ZCOPY/_PROJ_ROWS/_GRID."""
    assert bool(mod._outgoing) == bool(outgoing), "module direction does not match `outgoing`"
    if layout.replicated or not is_dist() or layout.P == 1:
        mask = None if mask_shard is None else mask_shard
        return mod(z_shard, mask=mask, inplace_safe=True, _add_with_inplace=with_add,
                   triangle_multiplicative="torch")
    R, N, CZ = z_shard.shape
    assert R == layout.R and N == layout.N, (tuple(z_shard.shape), repr(layout))
    C = mod.c_hidden
    dtype, device = z_shard.dtype, z_shard.device
    elt = z_shard.element_size()
    if mask_shard is None:
        mask3 = z_shard.new_ones((R, N, 1))
    else:
        mask3 = mask_shard.to(dtype=dtype).reshape(R, N, 1)
    RA, RB = agreed_rows(N, C, elt, layout, RA, RB, device)            # ONE (RA, RB) on every rank: they fix the collective sequence below
    npass = -(-layout.Rmax // RA)
    nslab = -(-layout.Rmax // RB)
    er = min(max_rows_per_launch(N, max(C, CZ)), int(os.environ.get("PTX_TP_TRIMUL_PROJ_ROWS", "256")))
    bmode, zmode, cap = _resolve_modes(outgoing, npass, RA, R, N, C, CZ, elt, layout.Rmax, cache_b, bcache, zcopy,
                                       z_shard.is_cuda)
    # pinned host pool for this call (slabs and/or z copy)
    host_bytes = (R * N * C * elt if bmode == "host" else 0) + (R * N * CZ * elt if zmode == "host" else 0)
    if host_bytes and z_shard.is_cuda:
        _PinnedPool.reserve(host_bytes + 512 * (nslab + 2), pin=True)
    zsrc = _ZSource(z_shard, zmode, cap)
    st = stats if stats is not None else ContractStats()
    st.h2d_bytes = 0
    gmode = grid or os.environ.get("PTX_TP_TRIMUL_GRID", "stock")
    assert gmode in ("stock", "slab"), gmode
    sgrid = stock_column_grid(N, int(inplace_chunk_size)) if gmode == "stock" else None
    st.mode = (f"{'outgoing' if outgoing else 'incoming'} P={layout.P} N={N} R={R} RA={RA} RB={RB} passes={npass} "
               f"slabs={nslab} bcache={bmode} zcopy={zmode}{'(cap=%d)' % cap if zmode == 'lazy' else ''} proj_rows={er} "
               f"grid={gmode}{'(%d chunks, ics=%d)' % (len(sgrid), inplace_chunk_size) if sgrid else ''}")
    _log(layout, st.mode)
    epilogue = _epilogue_factory(mod, z_shard, mask3, with_add)
    expected_gets = npass * nslab
    from .dist import alltoall_window, ring_blocks
    scratch = []          # objects holding device buffers; closed at exit even if an exception propagates
    try:
        _HS = _HostSlabs

        if outgoing:
            # ---- A provider: a rows I_sub from the LIVE shard (rows of pass p are untouched until pass p)
            def a_provider(i0, i1):
                zsrc.mark_pass(i0, i1)
                A = torch.empty((C, i1 - i0, N), dtype=dtype, device=device)
                for s0 in range(i0, i1, er):
                    s1 = min(i1, s0 + er)
                    p = _proj_rows(mod, zrows(z_shard, s0, s1), mask3[s0:s1], "a")      # [r, N(k), c]
                    A[:, s0 - i0:s1 - i0, :].copy_(p.permute(2, 0, 1))                   # permute_final_dims(p,(2,0,1))
                    del p
                return A

            def b_slab_from(src_rows_fn, j0, j1):
                # stock helper layout: [c, rows(j), N(k)] contiguous (its .transpose(-1,-2) is stock's GEMM operand)
                Bs = torch.empty((C, j1 - j0, N), dtype=dtype, device=device)
                for s0 in range(j0, j1, er):
                    s1 = min(j1, s0 + er)
                    p = _proj_rows(mod, src_rows_fn(s0, s1), mask3[s0:s1], "b")           # [r(j), N(k), c]
                    Bs[:, s0 - j0:s1 - j0, :].copy_(p.permute(2, 0, 1))                  # permute_final_dims(p,(2,0,1))
                    del p
                return Bs

            def make_slab(jb, src_rows_fn):
                j0, j1 = min(R, jb * RB), min(R, (jb + 1) * RB)
                if j1 <= j0:
                    return torch.empty((C, 0, N), dtype=dtype, device=device)
                return b_slab_from(src_rows_fn, j0, j1)

            live_rows = lambda a, b: zrows(z_shard, a, b)  # noqa: E731
            if bmode == "device":
                slabs = [make_slab(jb, live_rows) for jb in range(nslab)]
                get_slab = lambda jb: slabs[jb]  # noqa: E731
            elif bmode == "host":
                hs = _HostSlabs(nslab, lambda jb: make_slab(jb, live_rows), device, expected_gets, st)
                scratch.append(hs)
                get_slab = hs.get
            else:
                get_slab = lambda jb: make_slab(jb, zsrc.rows)  # noqa: E731  (pre-update copy)

            for p in range(npass):
                i0, i1 = min(R, p * RA), min(R, (p + 1) * RA)
                A = a_provider(i0, i1)                                            # [c, RA_v, N(k)]
                asm = _GridAssembler(sgrid, C, N, "bT", dtype, device) if sgrid else None
                st.passes += 1
                for jb in range(nslab):
                    slab = get_slab(jb)
                    st.slabs += 1
                    for q, Bq in ring_blocks(slab, layout, rows=(jb * RB, RB), dim=1):   # Bq [c, v(j), N(k)]
                        st.ring_steps += 1
                        if q != layout.rank:
                            st.bytes_streamed += C * N * RB * elt
                        v = Bq.shape[1]
                        if v == 0:
                            continue
                        c0 = layout.bounds[q][0] + jb * RB
                        pieces = asm.feed(c0, Bq) if asm is not None else [(c0, c0 + v, Bq)]
                        for (s_, e_, Bc) in pieces:
                            if i1 <= i0:
                                continue
                            assert C * (i1 - i0) * (e_ - s_) <= INT32_MAX
                            T = torch.matmul(A, Bc.transpose(-1, -2))                     # stock: matmul(a, b_chunk^T-view)
                            st.tiles += 1
                            st.matmul_flops += 2.0 * C * (i1 - i0) * (e_ - s_) * N
                            epilogue(T, (i0, i1), (s_, e_))
                            del T
                        del pieces
                    del slab
                if asm is not None:
                    asm.finish()
                del A, asm
            del get_slab
        else:
            # ---- incoming: column-window projections of my rows k (from pre-update source when needed)
            def windows_for(off, width):
                ws = []
                for d in range(layout.P):
                    d0, d1 = layout.bounds[d]
                    a, b = min(d1, d0 + off), min(d1, d0 + off + width)
                    if b > a:
                        ws.append((a, b))
                return ws

            zsrc_ref = [zsrc]
            sweep_a = _ColumnSweep("a", npass > 1, mod, mask3, R, C, er, dtype, device, z_shard, zsrc_ref)      # later passes read pre-update rows
            sweep_b = _ColumnSweep("b", bmode == "none", mod, mask3, R, C, er, dtype, device, z_shard, zsrc_ref)  # only when re-projecting b
            scratch.extend([sweep_a, sweep_b])

            def make_A(p):
                sweep_a.pending = windows_for(p * RA, RA)
                A = alltoall_window(sweep_a.get, layout, p * RA, RA, C=C, dtype=dtype, device=device, out_layout="gemm_a")
                sweep_a.cache.clear()
                return A                                                     # [c, RA_v(i), N(k)]

            def make_slab(jb):
                sweep_b.pending = windows_for(jb * RB, RB)
                S = alltoall_window(sweep_b.get, layout, jb * RB, RB, C=C, dtype=dtype, device=device, out_layout="gemm_b")
                sweep_b.cache.clear()
                return S                                                     # [c, N(k), RB_v(j)]

            if bmode == "device":
                b_slabs = [make_slab(jb) for jb in range(nslab)]
                get_slab = lambda jb: b_slabs[jb]  # noqa: E731
            elif bmode == "host":
                hs = _HostSlabs(nslab, make_slab, device, expected_gets, st)
                scratch.append(hs)
                get_slab = hs.get
            else:
                get_slab = make_slab
            for p in range(npass):
                i0, i1 = min(R, p * RA), min(R, (p + 1) * RA)
                zsrc.mark_pass(i0, i1)
                A = make_A(p)
                assert A.shape == (C, i1 - i0, N), (tuple(A.shape), (C, i1 - i0, N))
                asm = _GridAssembler(sgrid, C, N, "kw", dtype, device) if sgrid else None
                st.passes += 1
                for jb in range(nslab):
                    slab = get_slab(jb)
                    st.slabs += 1
                    for q, Bq in ring_blocks(slab, layout, rows=(jb * RB, RB), dim=2):   # Bq [c, N(k), v(j)]
                        st.ring_steps += 1
                        if q != layout.rank:
                            st.bytes_streamed += C * N * RB * elt
                        v = Bq.shape[2]
                        if v == 0:
                            continue
                        c0 = layout.bounds[q][0] + jb * RB
                        pieces = asm.feed(c0, Bq) if asm is not None else [(c0, c0 + v, Bq)]
                        for (s_, e_, Bc) in pieces:
                            if i1 <= i0:
                                continue
                            assert C * (i1 - i0) * (e_ - s_) <= INT32_MAX
                            T = torch.matmul(A, Bc)                                       # stock: matmul(a, b_chunk [c,k,w])
                            st.tiles += 1
                            st.matmul_flops += 2.0 * C * (i1 - i0) * (e_ - s_) * N
                            epilogue(T, (i0, i1), (s_, e_))
                            del T
                        del pieces
                    del slab
                if asm is not None:
                    asm.finish()
                del A, asm
            del get_slab
    finally:
        for obj in scratch:
            try:
                obj.close()
            except Exception:
                pass
        scratch.clear()
        try:
            zsrc.close()
        except Exception:
            pass
    del zsrc
    return z_shard

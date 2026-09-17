"""P5 colattn -- the column cross-attention's keys/values written ONCE, in the layout its matmuls read (mode exact).

What stock does.  GPNStarColCrossAttention.forward hands 5-D per-head views (B, L, A, ., .) of the clade keys/values to
F.scaled_dot_product_attention with the MATH backend.  Inside ATen (attention.cpp, _scaled_dot_product_attention_math) that is

    s = sqrt(scale)                      # scale = 1/sqrt(D) as the module passes it; s a double, applied as a float scalar
    q = query * s
    attn = matmul(q, key.transpose(-2, -1) * s)
    attn += attn_mask                    # in place
    attn = _safe_softmax(attn, -1)
    out = matmul(attn, value)

and at::matmul folds the (B, L, A) batch dimensions of its 5-D operands with expand().reshape(-1, ., .).  For the
transpose_for_scores views of K and V ((B, L, A, C, D) with the head axis strided INSIDE the clade axis) that reshape cannot be a
view, so every layer materialises: the scaled K^T (the `* s` pass over all of K), a contiguous copy of it in (B*L*A, D, C) order,
and a contiguous copy of V in (B*L*A, C, D) order -- three read+write passes over (B, L, C, H/2)-sized tensors that are nothing
but memory traffic -- before cuBLAS reads each operand exactly once (a batched gemv for q @ K^T with one query row, a batched
gemv/bmm for attn @ V).  At a saturating batch that traffic is the largest single item of the forward.

What this lever does.  The same torch operations with the same scalars, in the same order, on operands that are BUILT in their
final layout straight from the distinct clade rows of the unified / de-duplicated K/V GEMM (patches P3b/P4, which this lever
requires): the distinct key rows are multiplied by s BEFORE the gather (an elementwise multiply by the same float scalar gives
the same bits wherever the element sits, so scaling then gathering == gathering then scaling, bit for bit) and gathered directly
into a contiguous (B, L, A, D, C) tensor; the distinct value rows are gathered directly into a contiguous (B, L, A, C, D)
tensor.  torch.matmul's reshape of those is a view with exactly the sizes and strides of the contiguous clone stock makes, the
query operand is formed by the stock expression itself, so cuBLAS receives the same problem (transposes, m/n/k, leading
dimensions, batch strides) on the same values and runs the same kernels: scores, softmax and context are the stock's bit for
bit.  That equality is checked on the GPU by the kit's tests, never assumed; the reduced
GEMM feeding it keeps its own layer-0 self-check (patches._reduced_kv's, reproduced here on the distinct rows).

Cost.  Per layer the full-size K and V are each written once and read once (stock: written by the GEMM, read+written by the
scale pass (K), read+written by the copy, read by cuBLAS), K^T is released before V is built, and the distinct-row tables the
gathers read are a few MiB (L2-resident).  Peak transient memory of the attention is one full-size tensor instead of stock's four.
The two gathers are pure data movement: two small Triton kernels when Triton is importable and passes a one-time bit-for-bit
check against torch.gather on this device (write-bandwidth bound), else torch.gather itself (same bits, slower).

Falls back, by rule and by name, to the kit's previous column-attention forward (patches._col_forward_unified: stock SDPA call on
gathered (B, L, C, H/2) K/V, or the stock projections) whenever the reduced K/V route is not in use for a shape (route=stock,
memory step-aside, self-check reject), for output_attentions=True, and in training mode (dropout inside SDPA).
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import patches as P

# --------------------------------------------------------------------------------------
# the two gathers: distinct rows -> full-size operands in matmul layout
# --------------------------------------------------------------------------------------

_TRITON = {"state": None, "why": None}  # state: None = not probed yet | True = use Triton kernels | False = torch.gather
_kt_kernel = None
_v_kernel = None
_pv_kernel = None

try:  # Triton ships with the CUDA builds of torch; a stack without it keeps the torch.gather path (same values, slower)
    import triton
    import triton.language as tl

    @triton.jit
    def _kt_gather_kernel(out_ptr, ku_ptr, idx_ptr, SU, C: tl.constexpr, AH: tl.constexpr, BLOCK_AD: tl.constexpr, BLOCK_C: tl.constexpr):
        # out (BL, AH, C) contiguous  <-  ku (U, AH) rows of stride SU, idx (BL, C) int64:   out[bl, ad, c] = ku[idx[bl, c], ad]
        # one program = one bl x BLOCK_AD feature rows; the tile is read feature-contiguous and stored clade-contiguous (a transpose
        # through registers/shared memory that Triton lays out), so both sides move full sectors.
        bl = tl.program_id(0).to(tl.int64)
        ad = tl.program_id(1) * BLOCK_AD + tl.arange(0, BLOCK_AD)
        c = tl.arange(0, BLOCK_C)
        cm = c < C
        u = tl.load(idx_ptr + bl * C + c, mask=cm, other=0)
        val = tl.load(ku_ptr + u[None, :] * SU + ad[:, None], mask=cm[None, :], other=0.0)
        tl.store(out_ptr + bl * (AH * C) + ad[:, None] * C + c[None, :], val, mask=cm[None, :])

    @triton.jit
    def _v_gather_kernel(out_ptr, vu_ptr, idx_ptr, SU, C: tl.constexpr, A: tl.constexpr, D: tl.constexpr, BLOCK_C: tl.constexpr):
        # out (BL, A, C, D) contiguous  <-  vu (U, A*D) rows of stride SU, idx (BL, C) int64:   out[bl, a, c, :] = vu[idx[bl, c], a*D:(a+1)*D]
        # one program = one (bl, head): C rows of D contiguous values on both sides.
        bl = tl.program_id(0).to(tl.int64)
        a = tl.program_id(1)
        c = tl.arange(0, BLOCK_C)
        cm = c < C
        d = tl.arange(0, D)
        u = tl.load(idx_ptr + bl * C + c, mask=cm, other=0)
        val = tl.load(vu_ptr + u[:, None] * SU + a * D + d[None, :], mask=cm[:, None], other=0.0)
        tl.store(out_ptr + (bl * A + a) * (C * D) + c[:, None] * D + d[None, :], val, mask=cm[:, None])

    _FMA = getattr(tl, "fma", None) or tl.math.fma

    @triton.jit
    def _pv_fused_kernel(out_ptr, p_ptr, vu_ptr, idx_ptr, SU, C: tl.constexpr, A: tl.constexpr, D: tl.constexpr):
        # out (BL, A, D):  out[bl, a, :] = sum_{c=0}^{C-1} p[bl, a, c] * vu[idx[bl, c], a*D:(a+1)*D]  as ONE fp32 fma chain per element, c
        # increasing from 0 with the accumulator starting at 0 -- the summation order of cuBLAS's batched gemv for these problems (an
        # observed property of the library, so it is CHECKED per batch shape at run time against the by-construction path: see _context).
        # V is never materialised: the distinct rows are read from the (L2-resident) table, 2 KiB contiguous per clade.
        bl = tl.program_id(0).to(tl.int64)
        a = tl.arange(0, A)
        d = tl.arange(0, D)
        acc = tl.zeros((A, D), dtype=tl.float32)
        for c in tl.static_range(C):
            u = tl.load(idx_ptr + bl * C + c)
            vrow = tl.load(vu_ptr + u * SU + a[:, None] * D + d[None, :])
            pc = tl.load(p_ptr + (bl * A + a) * C + c)
            acc = _FMA(vrow, pc[:, None], acc)
        tl.store(out_ptr + (bl * A + a[:, None]) * D + d[None, :], acc)

    _kt_kernel, _v_kernel, _pv_kernel = _kt_gather_kernel, _v_gather_kernel, _pv_fused_kernel
except Exception as _e:  # noqa: BLE001 - no triton: torch.gather path
    _TRITON["state"], _TRITON["why"] = False, f"triton not importable ({type(_e).__name__})"

_BLOCK_AD = 64  # feature rows per program of the K^T gather (A*D = 512 -> 8 programs per (b, l))
COLATTN_LAYER_FACTOR = 1.25  # K-sized tensors ONE layer holds transiently on this route (K^T or V, never both, + scores / FFN activations of the
                             # target row); the memory plan of the reduced route (patches._memory_plan) budgets this instead of stock's five


def _block_c(C: int) -> int:
    return max(16, 1 << (int(C) - 1).bit_length())  # next power of two >= C


def _kt_gather_torch(ku: Tensor, idx2: Tensor, out: Tensor) -> Tensor:
    BL, C = idx2.shape
    U, AH = ku.shape
    kut = ku.t()  # (AH, U) view
    return torch.gather(kut.unsqueeze(0).expand(BL, AH, U), 2, idx2.view(BL, 1, C).expand(BL, AH, C), out=out)


def _v_gather_torch(vu: Tensor, idx2: Tensor, A: int, D: int, out: Tensor) -> Tensor:
    BL, C = idx2.shape
    U = vu.shape[0]
    vu3 = vu.unflatten(1, (A, D)).transpose(0, 1)  # (A, U, D) view
    return torch.gather(vu3.unsqueeze(0).expand(BL, A, U, D), 2, idx2.view(BL, 1, C, 1).expand(BL, A, C, D), out=out)


def _kt_gather_triton(ku: Tensor, idx2: Tensor, out: Tensor) -> Tensor:
    BL, C = idx2.shape
    U, AH = ku.shape
    if AH % _BLOCK_AD:  # feature width not a multiple of the tile: torch path (never the case for GPN-Star: A*D = 512)
        return _kt_gather_torch(ku, idx2, out)
    _kt_kernel[(BL, AH // _BLOCK_AD)](out, ku, idx2, ku.stride(0), C=C, AH=AH, BLOCK_AD=_BLOCK_AD, BLOCK_C=_block_c(C), num_warps=4)
    return out


def _v_gather_triton(vu: Tensor, idx2: Tensor, A: int, D: int, out: Tensor) -> Tensor:
    BL, C = idx2.shape
    if D & (D - 1):  # head size not a power of two: torch path (GPN-Star: D = 64)
        return _v_gather_torch(vu, idx2, A, D, out)
    _v_kernel[(BL, A)](out, vu, idx2, vu.stride(0), C=C, A=A, D=D, BLOCK_C=_block_c(C), num_warps=4)
    return out


def _triton_ok(device) -> bool:
    """One-time probe per process: the Triton gathers must compile, launch and reproduce torch.gather bit for bit on this device
    (they only move data, so this is a health check of the toolchain, not a numerical tolerance).  Any failure -> torch.gather."""
    if _TRITON["state"] is None:
        try:
            g = torch.Generator(device=device).manual_seed(1234)
            U, A, D, C, BL = 37, 8, 64, 45, 6
            ku = torch.randn((U, A * D), device=device, generator=g)
            vu2 = torch.randn((U, 2 * A * D), device=device, generator=g)[:, A * D:]  # strided rows, as a fused K|V slice would be
            idx2 = torch.randint(0, U, (BL, C), device=device, generator=g)
            a = _kt_gather_triton(ku, idx2, torch.empty((BL, A * D, C), device=device))
            b = _kt_gather_torch(ku, idx2, torch.empty((BL, A * D, C), device=device))
            c = _v_gather_triton(vu2, idx2, A, D, torch.empty((BL, A, C, D), device=device))
            d = _v_gather_torch(vu2, idx2, A, D, torch.empty((BL, A, C, D), device=device))
            torch.cuda.synchronize(device)
            ok = bool(torch.equal(a, b)) and bool(torch.equal(c, d))
            _TRITON["state"], _TRITON["why"] = ok, (None if ok else "triton gather differs from torch.gather")
        except Exception as e:  # noqa: BLE001 - no C compiler / driver mismatch / ...: keep the torch path, say why once
            _TRITON["state"], _TRITON["why"] = False, f"{type(e).__name__}: {str(e)[:160]}"
        if not _TRITON["state"]:
            warnings.warn(f"gpnstar_exact colattn: Triton gather kernels unavailable ({_TRITON['why']}); using torch.gather (same values, slower)",
                          RuntimeWarning, stacklevel=2)
    return bool(_TRITON["state"])


def gather_impl() -> str:
    return {None: "unprobed", True: "triton", False: "torch"}[_TRITON["state"]]


def gather_keys_transposed(k_rows: Tensor, idx: Tensor, A: int, D: int) -> Tensor:
    """(rows, A*D) distinct key rows (already scaled) + idx (B, L, C) -> contiguous (B, L, A, D, C): out[b,l,a,d,c] = k_rows[idx[b,l,c], a*D+d]."""
    B, L, C = idx.shape
    AH = A * D
    if k_rows.stride(-1) != 1:
        k_rows = k_rows.contiguous()
    out = torch.empty((B * L, AH, C), dtype=k_rows.dtype, device=k_rows.device)
    idx2 = idx.view(B * L, C)
    if k_rows.is_cuda and _triton_ok(k_rows.device):
        _kt_gather_triton(k_rows, idx2, out)
    else:
        _kt_gather_torch(k_rows, idx2, out)
    return out.view(B, L, A, D, C)


def gather_values_rows(v_rows: Tensor, idx2: Tensor, A: int, D: int) -> Tensor:
    """(rows, A*D) distinct value rows (row stride free) + idx2 (R, C) -> contiguous (R, A, C, D): out[r,a,c,:] = v_rows[idx2[r,c], a*D:(a+1)*D]."""
    R, C = idx2.shape
    if v_rows.stride(-1) != 1:
        v_rows = v_rows.contiguous()
    if not idx2.is_contiguous():
        idx2 = idx2.contiguous()
    out = torch.empty((R, A, C, D), dtype=v_rows.dtype, device=v_rows.device)
    if v_rows.is_cuda and _triton_ok(v_rows.device):
        _v_gather_triton(v_rows, idx2, A, D, out)
    else:
        _v_gather_torch(v_rows, idx2, A, D, out)
    return out


def gather_values(v_rows: Tensor, idx: Tensor, A: int, D: int) -> Tensor:
    """idx (B, L, C) -> contiguous (B, L, A, C, D) == ATen's contiguous copy of stock's value_layer."""
    B, L, C = idx.shape
    return gather_values_rows(v_rows, idx.view(B * L, C), A, D).view(B, L, A, C, D)


CUBLAS_BATCH_CHUNK = 65535  # cuBLAS runs a batched gemv over chunks of this many problems; the problems of the last, partial chunk get their own
                            # launch and heuristic (split-k when few), so exactly those are handed to cuBLAS itself on operands gathered for them


def context_fused(probs: Tensor, v_rows: Tensor, idx: Tensor, A: int, D: int) -> Tensor:
    """attn @ V without materialising V: probs (B, L, A, T=1, C) contiguous fp32 -> context (B, L, A, 1, D), each element ONE fma chain over
    the clades in increasing order (the order cuBLAS's gemv uses for every problem of a full 65 535-problem chunk); the P mod 65 535 trailing
    problems go to torch.matmul on V rows gathered for them alone (cuBLAS treats them as their own launch either way).  Whether the result IS
    the by-construction path's, bit for bit, is checked per batch shape before this is used (see _attend_from_rows)."""
    B, L, C = idx.shape
    BL = B * L
    P = BL * A
    if v_rows.stride(-1) != 1:
        v_rows = v_rows.contiguous()
    idx2 = idx.view(BL, C)
    p3 = probs.view(BL, A, C)
    out = torch.empty((BL, A, D), dtype=probs.dtype, device=probs.device)
    _pv_kernel[(BL,)](out, p3, v_rows, idx2, v_rows.stride(0), C=C, A=A, D=D, num_warps=4)
    r = P % CUBLAS_BATCH_CHUNK
    if r:
        bl0 = (P - r) // A
        v_tail = gather_values_rows(v_rows, idx2[bl0:], A, D).view(-1, C, D)[(P - r) - bl0 * A:]   # (r, C, D) contiguous rows
        out.view(P, 1, D)[P - r:] = torch.matmul(p3.view(P, 1, C)[P - r:], v_tail)
    return out.view(B, L, A, 1, D)


def fused_pv_applicable(probs: Tensor, idx: Tensor, A: int, D: int) -> bool:
    return (_pv_kernel is not None and bool(_TRITON["state"]) and probs.is_cuda and probs.dtype == torch.float32 and probs.dim() == 5
            and probs.shape[3] == 1 and probs.is_contiguous() and (A & (A - 1)) == 0 and (D & (D - 1)) == 0)


# --------------------------------------------------------------------------------------
# the attention, op for op as ATen's math SDPA
# --------------------------------------------------------------------------------------

COLATTN_MIN_TOKENS = 2048  # B*L below this keeps the previous column-attention forward: the forward is launch-bound there and the two extra kernel
                           # launches per layer buy nothing (H100, L=128: B=8 at parity either way, B=16 1.5x vs 1.2x, B=32 1.8x vs 1.2x)

_SAFE_SOFTMAX = getattr(torch.ops.aten, "_safe_softmax", None)  # what _scaled_dot_product_attention_math calls (torch >= 2.5); == softmax unless a row is all -inf


def _softmax_as_sdpa(x: Tensor) -> Tensor:
    return _SAFE_SOFTMAX(x, -1) if _SAFE_SOFTMAX is not None else torch.softmax(x, -1)


def _distinct_kv_checked(self, st, X: Tensor, idx: Tensor, source_embeddings: Tensor):
    """The distinct-row projections (k_u, v_u or None = compute later) for this layer, with patches._reduced_kv's layer-0 self-check
    reproduced on them (first forward of a (reduced rows, stock rows) pair: reduced GEMM == stock GEMM through the gather, bit for bit,
    K then V so one reference tensor is alive at a time).  Returns None when this forward must run the stock projections (reject)."""
    check = bool(st.validate and getattr(self, "_is_layer0", False))
    vkey = (int(X.shape[0]), int(st.kv_m_stock))
    status = st.kv_validation.get(vkey) if check else True
    if check and status is False:  # known-bad pair: stock projections, by the same policy as the unified path
        P._register_reject(st, getattr(st, "kv_mode_used", st.dedup), vkey)
        st.kv_fallback_full = True
        st.kv_input = None
        return None
    if st.debug_force_oom:  # TEST HOOK ONLY (fault injection); never set in production
        st.debug_force_oom -= 1
        raise torch.cuda.OutOfMemoryError("forced by debug_force_oom")
    Ah = self.all_head_size
    v_u = None
    if st.fuse_kv_unified:
        if not hasattr(self, "_w_kv"):
            self._w_kv = torch.cat([self.key.weight, self.value.weight], 0).contiguous()
            self._b_kv = torch.cat([self.key.bias, self.value.bias], 0).contiguous()
        kv = F.linear(X, self._w_kv, self._b_kv)
        k_u, v_u = kv[:, :Ah], kv[:, Ah:]
    else:
        k_u = self.key(X)
    if status is None:
        K_ref = self.key(source_embeddings)
        ok = P._equal_gathered(k_u, idx, K_ref)
        del K_ref
        if ok:
            if v_u is None:
                v_u = self.value(X)
            V_ref = self.value(source_embeddings)
            ok = P._equal_gathered(v_u, idx, V_ref)
            del V_ref
        if st.debug_force_reject:  # TEST HOOK ONLY
            st.debug_force_reject -= 1
            ok = False
        st.kv_validation[vkey] = ok
        mode_used = getattr(st, "kv_mode_used", st.dedup)
        st.validation_log.append({"rows": vkey[0], "stock_rows": vkey[1], "ok": ok,
                                  "mode": {False: "p3b", True: "dedup", "static": "sdedup"}.get(mode_used, str(mode_used))})
        if not ok:  # stay exact: stock projections in every layer of THIS forward; later forwards use the next option
            P._register_reject(st, mode_used, vkey)
            st.kv_fallback_full = True
            st.kv_input = None
            return None
    return k_u, v_u


def _attend_from_rows(self, X: Tensor, idx: Tensor, k_u: Tensor, v_u, hidden_states: Tensor, attention_mask: Tensor, evol_time_bias: Tensor):
    A, D = int(self.num_attention_heads), int(self.attention_head_size)
    scale = 1 / math.sqrt(self.attention_head_size)      # what the stock module passes as `scale=`
    s = math.sqrt(scale)                                   # ATen: scaling_factor = calculate_scale(query, scale).sqrt() -- a double, applied as a float scalar
    query_layer = self.transpose_for_scores(self.query(hidden_states))   # (B, L, A, T, D): the stock expression, so matmul's LHS is stock's
    mask = attention_mask + evol_time_bias.to(query_layer.dtype)         # (B, 1, 1, T, C): the stock expression
    kT = gather_keys_transposed(k_u * s, idx, A, D)        # (B, L, A, D, C) contiguous == ATen's contiguous copy of key.transpose(-2,-1)*s
    del k_u
    scores = torch.matmul(query_layer * s, kT)             # (B, L, A, T, C); matmul's reshape of kT is a view with the clone's strides
    del kT
    scores.add_(mask)                                      # ATen: attn.add_(*attn_mask)
    probs = _softmax_as_sdpa(scores)                       # ATen: at::_safe_softmax(attn, -1)
    del scores
    if v_u is None:
        v_u = self.value(X)
    ctx = _context(self, probs, v_u, idx, A, D)            # (B, L, A, T, D) == torch.matmul(probs, ATen's contiguous copy of value)
    del v_u
    ctx = ctx.transpose(-2, -3).contiguous()               # the stock epilogue
    return (ctx.view(*(ctx.size()[:-2] + (self.all_head_size,))),)


def _context(self, probs: Tensor, v_u: Tensor, idx: Tensor, A: int, D: int) -> Tensor:
    """attn @ V.  By construction: V gathered into the contiguous (B, L, A, C, D) operand stock's matmul clones, then torch.matmul (same cuBLAS
    call).  Faster, when it reproduces that bit for bit for this batch shape: context_fused (no full-size V at all).  The first forward of a
    shape decides at layer 0 by computing both and comparing with torch.equal; a shape that does not compare equal keeps the by-construction
    path (st.colattn_fused[shape] records the verdict; validation_log names it)."""
    st = self._exact_state
    key = (int(idx.shape[0]) * int(idx.shape[1]), int(idx.shape[2]))
    verdict = st.colattn_fused.get(key) if getattr(st, "colattn_fuse", True) else False
    if verdict is None and not fused_pv_applicable(probs, idx, A, D):
        verdict = st.colattn_fused[key] = False
    if verdict is True:
        return context_fused(probs, v_u, idx, A, D)
    V = gather_values(v_u, idx, A, D)                      # (B, L, A, C, D) contiguous == ATen's contiguous copy of value
    ctx = torch.matmul(probs, V)
    del V
    if verdict is None:                                    # undecided shape: decide now if this is layer 0, else stay by-construction this forward
        if getattr(self, "_is_layer0", False):
            try:
                ok = bool(torch.equal(context_fused(probs, v_u, idx, A, D), ctx))
            except Exception:  # noqa: BLE001 - compile/launch trouble: by-construction path for this shape
                ok = False
            st.colattn_fused[key] = ok
            st.validation_log.append({"rows": key[0] * key[1], "stock_rows": key[0] * key[1], "ok": ok, "mode": "colattn_pv"})
    return ctx


def _col_forward_layout(self, hidden_states, source_embeddings, attention_mask=None, evol_time_bias=None, output_attentions=False):
    """GPNStarColCrossAttention.forward for mode exact (P5).  See the module docstring."""
    if attention_mask is None or evol_time_bias is None:
        raise ValueError("attention_mask and evol_time_bias are required")
    st = self._exact_state
    X, idx = st.kv_input, st.kv_index
    if (output_attentions or self.training or not getattr(st, "colattn", False) or st.kv_fallback_full or X is None or idx is None
            or not hidden_states.is_cuda or hidden_states.dim() != 4 or idx.dim() != 3
            or hidden_states.shape[0] * hidden_states.shape[1] < int(getattr(st, "colattn_min_tokens", COLATTN_MIN_TOKENS))):
        return P._col_forward_unified(self, hidden_states, source_embeddings, attention_mask, evol_time_bias, output_attentions)
    oom = False
    try:
        kv = _distinct_kv_checked(self, st, X, idx, source_embeddings)
        if kv is not None:
            return _attend_from_rows(self, X, idx, kv[0], kv[1], hidden_states, attention_mask, evol_time_bias)
    except torch.cuda.OutOfMemoryError:  # the reduced route needs memory the stock projections do not: this shape runs stock's, by rule
        oom = True
    if oom:
        kv = None
        P._memory_fallback(st, int(st.kv_m_stock), where="reduced K/V projections")
    # self-check reject or memory fallback: kv_fallback_full is set -> the unified forward runs the stock projections + stock SDPA
    return P._col_forward_unified(self, hidden_states, source_embeddings, attention_mask, evol_time_bias, output_attentions)


def patch_col_attention_layout(model: nn.Module, enabled: bool = True) -> str:
    """Install P5 on every layer's column cross-attention.  Requires patch_unified_kv (P3b/P4) to be installed first.
    Returns the gather implementation in use ('triton' | 'torch')."""
    core = P.core_model(model)
    st = P._state(model)
    layers = list(core.encoder.layer)
    if not layers or not hasattr(layers[0].attention.col_attention.self, "_is_layer0"):
        raise RuntimeError("patch_col_attention_layout requires patch_unified_kv (the unified / de-duplicated K/V route) first")
    st.colattn = bool(enabled)
    st.layer_factor = COLATTN_LAYER_FACTOR if st.colattn else P.MEMORY_STOCK_LAYER_FACTOR
    st.colattn_min_tokens = int(COLATTN_MIN_TOKENS)
    st.colattn_fuse = True
    st.colattn_fused = {}
    dev = next(core.parameters()).device
    if dev.type == "cuda":
        _triton_ok(dev)
    st.colattn_gather = gather_impl()
    for layer in layers:
        ca = layer.attention.col_attention.self
        ca.forward = _col_forward_layout.__get__(ca, type(ca))
    return st.colattn_gather


ExactState = P.ExactState
ExactState.colattn = False          # P5 engaged
ExactState.layer_factor = P.MEMORY_STOCK_LAYER_FACTOR  # per-layer K-sized tensors the memory plan budgets for the route in use
ExactState.colattn_gather = None    # 'triton' | 'torch'
ExactState.colattn_min_tokens = COLATTN_MIN_TOKENS
ExactState.colattn_fuse = True           # try the fused attn@V per shape (checked); False = always the by-construction operand + cuBLAS
ExactState.colattn_fused = None          # {(B*L, C): verdict}

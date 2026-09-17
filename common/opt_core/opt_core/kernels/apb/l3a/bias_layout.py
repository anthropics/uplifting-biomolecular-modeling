# bias_layout.py -- lever L3a: the pair-bias layout produced ONCE per fold, read by every attention call of the rollout.
#
#   PairBiasLayout(L, H, N, dtype, device)   owns ONE buffer [L, H, N, Np]  (L = number of transformer blocks; no sample axis anywhere)
#   padded_np(N)                             the row-stride rule
#
# Row stride rule (Stage C (i) + (iii) of the L3 plan):  Np = N rounded up to a multiple of 64 elements (every bias row starts on a 128-byte line for
# 16-bit dtypes, and Np % 16 == 0 lets the compiler prove 16-byte vector loads);  when that Np is a multiple of 512 it gets 64 more (breaks the
# power-of-two row stride, e.g. 2048 -> 2112, 4096 -> 4160, 1536 -> 1600).
#
# PAD COLUMNS: ZERO-filled at construction and never written by fill_from.  They are NOT a mask: the attention kernel bounds keys by n < N with its own
# N (= q.shape[-2]) and never reads a pad column (fab_batched: full key tiles cover n < (N // BN) * BN <= N; the tail tile is masked by n < N).  This
# keeps production's semantics bitwise (production has no pad columns) and keeps a whole-buffer non-finite scan meaningful.  `pads_clean()` checks that
# nobody wrote into them.
#
# Offsets: `layer(l)` returns the [H, N, Np] VIEW of one block, so offsets inside the kernel stay below H * N * Np < 2^31 while the layer's base offset
# (l * H * N * Np elements, which passes 2^31 for N >= ~2,365 at L = 24, H = 16) lives in the 64-bit data pointer of the view.  H * N * Np >= 2^31 is
# refused by name (PairBiasLayoutTooLarge) -- the batched kernel computes every offset in int64 anyway; the refusal protects OTHER consumers of the view.
#
# torch is imported inside the functions (package import stays standard-library only).

TILE = 64          # row stride granularity (elements)
SKEW_MOD = 512     # a row stride that is a multiple of this many elements ...
SKEW = 64          # ... gets this many more


class PairBiasLayoutTooLarge(ValueError):
    """H * N * Np >= 2^31 elements in one layer."""


class PairBiasLayoutError(ValueError):
    """a source / argument that does not fit the layout."""


def padded_np(N, skew=True):
    """row stride (elements) for N keys: ceil(N / 64) * 64, plus 64 when that is a multiple of 512 (skew=False keeps the plain multiple of 64)."""
    N = int(N)
    if N < 1:
        raise PairBiasLayoutError("padded_np: N must be >= 1 (got %d)" % N)
    np_ = -(-N // TILE) * TILE
    if skew and np_ % SKEW_MOD == 0:
        np_ += SKEW
    return np_


class PairBiasLayout:
    """ONE padded pair-bias buffer [L, H, N, Np] for all L blocks of a fold; see the module header for the pad-column and offset contracts."""

    def __init__(self, L, H, N, dtype=None, device=None, skew=True):
        import torch
        self.L, self.H, self.N = int(L), int(H), int(N)
        if self.L < 1 or self.H < 1:
            raise PairBiasLayoutError("PairBiasLayout: L and H must be >= 1 (got L=%d H=%d)" % (self.L, self.H))
        self.Np = padded_np(self.N, skew=skew)
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        per_layer = self.H * self.N * self.Np
        if per_layer >= 2 ** 31:
            raise PairBiasLayoutTooLarge("PairBiasLayout: H * N * Np = %d * %d * %d = %d >= 2^31 elements in one layer (refused by name; "
                                         "split the heads or shard the rows)" % (self.H, self.N, self.Np, per_layer))
        self.buf = torch.zeros((self.L, self.H, self.N, self.Np), dtype=self.dtype, device=self.device)

    # ---- sizes
    def layer_elems(self):
        return self.H * self.N * self.Np

    def nbytes(self):
        return self.L * self.layer_elems() * self.buf.element_size()

    def layer_base_offset(self, l):
        """element offset of layer l inside the buffer (exceeds 2^31 for the late layers of a large fold; carried by the view's 64-bit pointer)."""
        return int(l) * self.layer_elems()

    # ---- views
    def _l(self, l):
        l = int(l)
        if not 0 <= l < self.L:
            raise PairBiasLayoutError("PairBiasLayout: layer %d outside [0, %d)" % (l, self.L))
        return l

    def layer(self, l):
        """[H, N, Np] view of block l (what flash_bias_attn_batched / flash_bias_attn_v2 take; row stride Np, columns >= N are pads)."""
        return self.buf[self._l(l)]

    def valid(self, l):
        """[H, N, N] view of block l with row stride Np: exactly what the PRODUCTION kernel can read through its strides (no copy)."""
        return self.buf[self._l(l)][:, :, :self.N]

    def layers(self, ls):
        """[G, H, N, Np] view for a contiguous range of layers (ls = (l0, l1)) -- e.g. as the grouped bias of the batched kernel."""
        l0, l1 = ls
        return self.buf[self._l(l0):self._l(l1 - 1) + 1]

    # ---- writers
    def fill_from(self, l, src, heads_last=True):
        """copy one block's bias into the padded layer.  src: [N, N, H] (heads-last projection output, the default) or [1, N, N, H]; with
        heads_last=False: [H, N, N].  One strided copy (a transposition for heads-last sources), done once per fold per block; pad columns untouched.
        dtype conversion follows torch's copy_ (round to nearest even), the same rounding a `.to(dtype)` of the source would apply."""
        N, H = self.N, self.H
        if heads_last:
            if src.dim() == 4 and src.shape[0] == 1:
                src = src[0]
            if tuple(src.shape) != (N, N, H):
                raise PairBiasLayoutError("PairBiasLayout.fill_from: heads-last source must be [N, N, H] = %s (got %s)" % ((N, N, H), tuple(src.shape)))
            src = src.permute(2, 0, 1)
        elif tuple(src.shape) != (H, N, N):
            raise PairBiasLayoutError("PairBiasLayout.fill_from: source must be [H, N, N] = %s (got %s)" % ((H, N, N), tuple(src.shape)))
        self.valid(l).copy_(src)
        return self

    def fill_all_from(self, srcs, heads_last=True):
        """srcs: a sequence of L per-block sources, or one stacked tensor [L, N, N, H] / [L, H, N, N]."""
        n = len(srcs)
        if n != self.L:
            raise PairBiasLayoutError("PairBiasLayout.fill_all_from: %d sources for %d layers" % (n, self.L))
        for l in range(self.L):
            self.fill_from(l, srcs[l], heads_last=heads_last)
        return self

    # ---- checks (host syncs: outside CUDA-graph capture)
    def pads_clean(self):
        """True when every pad column still holds zero."""
        if self.Np == self.N:
            return True
        return not bool(self.buf[..., self.N:].any())

    def check_finite(self, l=None):
        """writer-side non-finite check over one layer's valid region (or all layers): raises FloatingPointError naming the layer."""
        import torch
        for i in (range(self.L) if l is None else [self._l(l)]):
            n_bad = int((~torch.isfinite(self.valid(i))).sum())
            if n_bad:
                raise FloatingPointError("PairBiasLayout: layer %d holds %d non-finite bias values" % (i, n_bad))
        return self

    def __repr__(self):
        return "PairBiasLayout(L=%d, H=%d, N=%d, Np=%d, dtype=%s, device=%s, %.3f GB)" % (
            self.L, self.H, self.N, self.Np, str(self.dtype).replace("torch.", ""), self.device, self.nbytes() / 1e9)

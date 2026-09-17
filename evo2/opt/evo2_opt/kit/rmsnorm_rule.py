"""ATen's reduce geometry for the bf16 row norm the kit's RMSNorm kernel reproduces (no torch): the number of partial sums T per row."""

CHUNK = 4096           # the tail's chunk (elements); served widths are multiples of it


def _last_pow2(n: int) -> int:
    return 1 << (max(int(n), 1).bit_length() - 1)


def aten_partials(rows: int, H: int):
    """T = the number of partial sums torch's CUDA norm of a contiguous (rows, H) bf16 tensor over H forms per row (Reduce.cuh setReduceConfig,
    MAX_NUM_THREADS 512, warp 32, input_vec_size 4), or None when the scheme is not the one reproduced here."""
    if H % 4 != 0 or H // 4 <= 128:
        return None
    dim0 = H // 4
    d0 = _last_pow2(dim0) if dim0 < 512 else 512
    d1 = _last_pow2(rows) if rows < 512 else 512
    bw = min(d0, 32)
    bh = min(d1, 512 // bw)
    bw = min(d0, 512 // bh)
    T = bw
    if -(-H // bw) >= min(16 * bh, 256):                                  # the row is split across the block's bh warps as well
        T = bw * bh
        if -(-H // T) >= 256:                                            # ATen may add CTAs per row (grid-dependent): not reproduced
            return None
    if T < 32 or T > 512 or H % (4 * T) != 0:
        return None
    return T


def row_rule(rows: int, H: int):
    """T when a contiguous (rows, H) bf16 input is served by the fused kernel (width a multiple of CHUNK, T reproducible), else None."""
    if rows < 1 or H % CHUNK != 0:
        return None
    return aten_partials(rows, H)

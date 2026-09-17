"""opt_core.shape_policy — ONE place that says how many tokens a padding engine compiles for: the stock padding rules, by
name, and the token multiple the core's kernels tile at.  Standard library only.

    padded_len(n, policy, *, tile=None, max_len=None) -> int

policies
    'none'             exactly n (exact modes: the stock shape, no padding of the core's own).
    'af3_buckets'      the stock bucket list of the engine whose kit names this policy (its STOCK.md cites the stock file:line of the
                       `--buckets` default and of the bucket rule): the first bucket >= n; above the largest bucket, exactly n.
    'recompile_abs10'  the monomer recompile padding of the engine whose kit names this policy (its STOCK.md cites the stock file:line):
                       n + RECOMPILE_PADDING (10), capped at max_len when max_len is given.
    'crop_sizes'       a model crop-size list (CROP_SIZES): the first
                       size >= n; above the largest size, exactly n.
    'kernel_tile'      the next multiple of the kernel tile (`tile`, default KERNEL_TILE): the fast / big policy of a padding engine, so every
                       compiled shape is one the core's Pallas kernels serve without a ragged tail.
An unknown policy raises ValueError naming the valid set; n < 1 raises ValueError.

    KERNEL_TILE        the token multiple the core's JAX pair kernels serve on the H100 table (= kernels.fpf_pallas_serve.required_multiple
                       ('triattn', cc='9.0') = 64; kernels/pallas_attn bq = bk = 64 divides it); padded_len(..., cc=<cc>) asks the capability's own multiple.
"""
from typing import Optional

__all__ = ["KERNEL_TILE", "STOCK_BUCKETS", "CROP_SIZES", "RECOMPILE_PADDING", "POLICIES", "padded_len"]

from opt_core.kernels.fpf_pallas_serve import required_multiple as _required_multiple   # standard library at import; THE tile rule

KERNEL_TILE = _required_multiple("triattn", cc="9.0")          # 64: the H100 table (other capabilities: padded_len(..., cc=<cc>))
STOCK_BUCKETS = (32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120)
CROP_SIZES = (256, 384, 512, 768, 1024, 1536, 2048)
RECOMPILE_PADDING = 10
POLICIES = ("none", "af3_buckets", "recompile_abs10", "crop_sizes", "kernel_tile")


def _round_up(n: int, t: int) -> int:
    return ((n + t - 1) // t) * t


def padded_len(n: int, policy: str, *, tile: Optional[int] = None, max_len: Optional[int] = None, cc: Optional[str] = None) -> int:
    """The padded token count for `n` real tokens under `policy` (see the module docstring); for policy 'kernel_tile' the tile is `tile` if given,
    else the tiles-pinned multiple of compute capability `cc` (`required_multiple`), else KERNEL_TILE; `max_len` caps 'recompile_abs10'
    (the engine's own max_len)."""
    n = int(n)
    if n < 1:
        raise ValueError(f"padded_len: n must be >= 1, got {n}")
    if policy == "none":
        return n
    if policy in ("af3_buckets", "crop_sizes"):
        for b in (STOCK_BUCKETS if policy == "af3_buckets" else CROP_SIZES):
            if b >= n:
                return b
        return n
    if policy == "recompile_abs10":
        p = n + RECOMPILE_PADDING
        return min(p, int(max_len)) if max_len is not None else p
    if policy == "kernel_tile":
        t = int(tile) if tile is not None else (max(_required_multiple("triattn", cc=cc), _required_multiple("trimul", cc=cc)) if cc is not None else KERNEL_TILE)
        if t < 1:
            raise ValueError(f"padded_len: tile must be >= 1, got {t}")
        return _round_up(n, t)
    raise ValueError(f"padded_len: unknown policy {policy!r}; valid: {', '.join(POLICIES)}")

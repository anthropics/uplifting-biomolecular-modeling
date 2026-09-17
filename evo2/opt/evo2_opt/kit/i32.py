"""The 32-bit element-addressing ceiling of the Triton kernels: a shape past it is refused by name before the launch."""
import sys

__all__ = ["EXTENT", "ShapeCeiling", "strided_extent", "fits", "max_leading", "check", "line"]

EXTENT = 2 ** 31
STOCK_WORD = "canUse32BitIndexMath"        # torch's name for the same rule on the stock's F.conv1d featurizer (the word its RuntimeError carries)
PREFIX = "[evo2-kit] SHAPE CEILING"


class ShapeCeiling(RuntimeError):
    """A shape past a kit Triton kernel's 32-bit element addressing, refused before the launch (the line is the message)."""


def strided_extent(shape, strides) -> int:
    shape = [int(n) for n in shape]; strides = [int(s) for s in strides]
    if any(n <= 0 for n in shape):
        return 0
    return sum((n - 1) * s for n, s in zip(shape, strides)) + 1


def fits(*extents) -> bool:
    return max(int(e) for e in extents) <= EXTENT


def max_leading(extent_per: int, extent_rest: int) -> int:
    """Largest n >= 0 with (n - 1) * extent_per + extent_rest <= EXTENT (extent_rest = the extent of one leading-index slice)."""
    extent_per = int(extent_per); extent_rest = int(extent_rest)
    if extent_rest > EXTENT:
        return 0
    if extent_per <= 0:
        return sys.maxsize
    return (EXTENT - extent_rest) // extent_per + 1


def line(lever: str, what: str, extent: int, hint: str) -> str:
    return (f"{PREFIX} {lever}: {what} spans {int(extent):,} elements > 2^31, past the Triton kernel's 32-bit element addressing — {hint}; "
            f"refused before the launch, nothing rerouted (the stock's F.conv1d featurizer stops at this element count by the same rule: {STOCK_WORD})")


def check(lever: str, what: str, extents, hint: str) -> None:
    if fits(*extents):
        return None
    msg = line(lever, what, max(int(e) for e in extents), hint)
    print(msg, file=sys.stderr, flush=True)
    raise ShapeCeiling(msg)

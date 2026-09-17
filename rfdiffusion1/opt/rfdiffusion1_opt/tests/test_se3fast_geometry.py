"""The T2 add-on's launch-geometry table (opt/forward/se3fast_addon/rfd_se3fast/geometry.py): one row per compute capability, read by
kernels.py at launch. Loaded by path — the add-on package itself imports torch, the table module imports nothing."""
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GEOMETRY = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "se3fast_addon", "rfd_se3fast", "geometry.py"))


@pytest.fixture(scope="module")
def geo():
    spec = importlib.util.spec_from_file_location("rfd_se3fast_geometry", GEOMETRY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_capability_key_grammar(geo):
    assert geo.capability_key((9, 0)) == "9.0" and geo.capability_key((8, 0)) == "8.0" and geo.capability_key((12, 0)) == "12.0"


def test_h100_row_is_the_written_geometry(geo):
    # the 9.0 row is what the kernels launched with before the table existed: trunk 64; conv 64 | 32 | 16 by JP*KP <= 128 | <= 256 | larger
    assert geo.trunk_block_e("9.0") == 64
    assert [geo.conv_block_e("9.0", t) for t in (16, 64, 128, 129, 256, 257, 512, 4096)] == [64, 64, 64, 32, 32, 16, 16, 16]
    assert geo.NUM_WARPS == 4


def test_a100_row(geo):
    assert geo.trunk_block_e("8.0") == 128
    assert [geo.conv_block_e("8.0", t) for t in (64, 128, 129, 256, 512, 4096)] == [128, 128, 32, 32, 32, 32]


def test_unlisted_capability_takes_the_default_row(geo):
    assert geo.DEFAULT == "9.0"
    for cc in ("8.6", "8.9", "10.0", "12.0"):
        assert geo.trunk_block_e(cc) == geo.trunk_block_e("9.0")
        assert [geo.conv_block_e(cc, t) for t in (64, 256, 512)] == [geo.conv_block_e("9.0", t) for t in (64, 256, 512)]


def test_every_row_ends_open(geo):
    # the last limit of every conv row is None so any tile resolves; rows are ascending in their limits
    for cc, row in geo.CONV_BLOCK_E.items():
        limits = [l for l, _ in row]
        assert limits[-1] is None, cc
        assert limits[:-1] == sorted(limits[:-1]), cc
        assert all(isinstance(b, int) and b >= 16 and b & (b - 1) == 0 for _, b in row), cc   # BLOCK_E: a power of two >= 16 (tl.dot's M floor)
    assert set(geo.TRUNK_BLOCK_E) == set(geo.CONV_BLOCK_E) and geo.DEFAULT in geo.TRUNK_BLOCK_E

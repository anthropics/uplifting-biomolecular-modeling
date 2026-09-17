import pytest

from opt_core import shape_policy as SP


def test_none_and_tile():
    assert SP.padded_len(400, "none") == 400
    assert SP.KERNEL_TILE == 64 and SP.padded_len(400, "kernel_tile") == 448 and SP.padded_len(512, "kernel_tile") == 512 and SP.padded_len(513, "kernel_tile") == 576
    assert SP.padded_len(400, "kernel_tile", tile=128) == 512 and SP.padded_len(1, "kernel_tile") == 64


def test_af3_buckets_are_the_stock_list():
    assert SP.STOCK_BUCKETS == (32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120)
    assert SP.padded_len(1, "af3_buckets") == 32 and SP.padded_len(256, "af3_buckets") == 256 and SP.padded_len(257, "af3_buckets") == 512
    assert SP.padded_len(1401, "af3_buckets") == 1536 and SP.padded_len(5121, "af3_buckets") == 5121      # above the largest bucket: exactly n


def test_crop_sizes():
    assert SP.CROP_SIZES == (256, 384, 512, 768, 1024, 1536, 2048)
    assert SP.padded_len(200, "crop_sizes") == 256 and SP.padded_len(1025, "crop_sizes") == 1536 and SP.padded_len(2049, "crop_sizes") == 2049


def test_colabfold_recompile_padding():
    assert SP.RECOMPILE_PADDING == 10 and SP.padded_len(400, "recompile_abs10") == 410
    assert SP.padded_len(400, "recompile_abs10", max_len=405) == 405 and SP.padded_len(400, "recompile_abs10", max_len=1000) == 410


def test_unknown_policy_names_the_valid_set():
    with pytest.raises(ValueError) as e:
        SP.padded_len(400, "recompile_pct10")
    assert "valid: none, af3_buckets, recompile_abs10, crop_sizes, kernel_tile" in str(e.value)
    with pytest.raises(ValueError):
        SP.padded_len(0, "none")


def test_stdlib_only():
    src = open(SP.__file__).read()
    assert "import torch" not in src and "import jax" not in src and "numpy" not in src

"""boot loader: with out_dtype the floating tensors of a safetensors file arrive ALREADY cast (per tensor, as its bytes complete),
value for value the cast torch's .to(dtype) makes; other dtypes and already-matching tensors arrive as the file's bytes."""
import json
import os
import struct

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _write_safetensors(path, tensors):
    """a minimal safetensors writer: {name: np.ndarray} (F32 / BF16-as-uint16 / I64)"""
    tags = {np.dtype("float32"): "F32", np.dtype("int64"): "I64", np.dtype("uint16"): "BF16"}
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        b = np.ascontiguousarray(a).tobytes(); header[name] = {"dtype": tags[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}; blobs.append(b); off += len(b)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h); [f.write(b) for b in blobs]


def test_out_dtype_delivers_the_cast_torch_would_make(tmp_path):
    import importlib
    L = importlib.import_module("esmc_opt.kits.boot.loader")
    g = np.random.default_rng(0)
    big = g.standard_normal((300, 700)).astype(np.float32)             # spans several chunks at chunk_bytes 64 KiB
    small = g.standard_normal((5,)).astype(np.float32)
    ints = np.arange(12, dtype=np.int64).reshape(3, 4)
    bf = torch.randn(4, 6, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    p = str(tmp_path / "model.safetensors")
    _write_safetensors(p, {"w.big": big, "w.small": small, "idx": ints, "already_bf16": bf.view(torch.int16).numpy().view(np.uint16)})
    st = {}
    out = L.load_files([p], "cpu", out_dtype=torch.bfloat16, chunk_bytes=1 << 16, n_slots=4, n_readers=2, stats=st)
    assert out["w.big"].dtype == torch.bfloat16 and torch.equal(out["w.big"], torch.from_numpy(big).to(torch.bfloat16))      # the cast .to(dtype) makes, bit for bit
    assert torch.equal(out["w.small"], torch.from_numpy(small).to(torch.bfloat16))
    assert out["idx"].dtype == torch.int64 and torch.equal(out["idx"], torch.from_numpy(ints))                               # non-floating: the file's bytes
    assert out["already_bf16"].dtype == torch.bfloat16 and torch.equal(out["already_bf16"], bf)                              # already the target dtype: direct path
    assert st["n_cast"] == 2 and st["out_dtype"] == "torch.bfloat16" and 0 < st["staging_peak_bytes"] <= big.nbytes + small.nbytes
    plain = L.load_files([p], "cpu", chunk_bytes=1 << 16, n_slots=4, n_readers=2)                                              # no out_dtype: every tensor in the file's dtype, bytes verbatim
    assert plain["w.big"].dtype == torch.float32 and torch.equal(plain["w.big"], torch.from_numpy(big))

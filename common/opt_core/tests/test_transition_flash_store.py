"""kernels/transition: the sealed sm_90a unit's binaries for other interpreter ABIs (flash_prebuilt/: one binary per ABI key, built from the sealed
csrc, digests recorded), the face's own operand pack (no parameter version counters: serves under torch.inference_mode) and the ABI key word."""
import hashlib
import json
import os
import re

import pytest

from opt_core.kernels import transition as T

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(CORE_DIR, "opt_core", "kernels", "transition")
SEALED = os.path.join(PKG, "flash_sm90a")
STORE = os.path.join(PKG, T.FLASH_STORE)
KEY_RE = r"^torch\d+\.\d+\.\d+\+cu\d+-cpython-3\d+-x86_64-linux-gnu-sm90$"


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _sealed_manifest():
    keys = sorted(os.listdir(os.path.join(SEALED, "prebuilt")))
    return json.load(open(os.path.join(SEALED, "prebuilt", keys[0], "manifest.json"), encoding="utf-8"))


def test_store_manifest_names_the_sealed_source_and_every_binary_is_digest_consistent():
    man_p = os.path.join(STORE, "manifest.json")
    assert os.path.isfile(man_p), "flash_prebuilt/manifest.json missing"
    man = json.load(open(man_p, encoding="utf-8"))
    sealed = _sealed_manifest()
    assert man["module_name"] == sealed["module_name"]
    assert man["sources"] == sealed["source_sha256"], (man["sources"], sealed["source_sha256"])          # built from the SAME source the sealed binary was
    for fn, digest in man["sources"].items():
        assert _sha(os.path.join(SEALED, "csrc", fn)) == digest, fn
    assert man["binaries"], "no binaries recorded"
    for key, ent in man["binaries"].items():
        assert re.match(KEY_RE, key), key
        so = os.path.join(STORE, ent["so"])
        assert os.path.isfile(so), so
        assert _sha(so) == ent["so_sha256"], key
        assert ent["so"].startswith(key + "/"), ent["so"]
        assert ("cpython-%s" % ent["python_tag"][2:]) in key and ent["soabi"] in key, (key, ent["python_tag"], ent["soabi"])
        assert key.startswith("torch%s-" % ent["torch"]), (key, ent["torch"])
        for k in ("cuda", "arch", "cuda_flags", "cutlass_headers", "smem_bytes", "hidden_chunk"):
            assert ent[k] == sealed[k], (key, k, ent[k], sealed[k])                                    # same recipe, same kernel constants
        assert ent.get("loadcheck", {}).get("silu_mismatch") == 0, (key, ent.get("loadcheck"))
        for k in ("nvcc", "built_utc", "note"):
            assert k in ent, (key, k)


def test_the_sealed_directory_is_untouched_by_the_store():
    assert sorted(os.listdir(os.path.join(SEALED, "prebuilt"))) == ["torch2.13.0-cu130"]
    assert not os.path.exists(os.path.join(SEALED, T.FLASH_STORE))


def test_flash_pack_reads_no_version_counters_and_matches_the_sealed_layout():
    torch = pytest.importorskip("torch")
    FT = T.carried_module("flash_sm90a")

    class E(object):                       # stands in for the extension: the pack reads hidden_chunk() only
        @staticmethod
        def hidden_chunk():
            return int(_sealed_manifest()["hidden_chunk"])

    g = torch.Generator().manual_seed(7)
    c, h = 256, 1024
    wa = torch.randn(h, c, generator=g); wb = torch.randn(h, c, generator=g); wo = torch.randn(c, h, generator=g)
    lw = 1 + 0.1 * torch.randn(c, generator=g); lb = 0.05 * torch.randn(c, generator=g)
    dev = torch.device("cpu")
    with torch.inference_mode():           # packed AND served inside inference mode: inference tensors carry no version counter
        W = T.pack(w_a=wa, w_b=wb, w_o=wo, ln_w=lw, ln_b=lb, eps=1e-5, device=dev)
        assert W.wa16.is_inference()
        pk = T._flash_pack(W, E, dev)
        assert pk["w1p"].shape == (2 * h, c) and pk["wo"].shape == (c, h) and pk["lnw"].dtype == torch.bfloat16 and pk["nh"] == h
        assert T._flash_pack(W, E, dev) is pk                                              # cached per (Weights, device)
    W2 = T.pack(w_a=wa, w_b=wb, w_o=wo, ln_w=lw, ln_b=lb, eps=1e-5, device=dev)          # packed outside, served inside
    with torch.inference_mode():
        pk2 = T._flash_pack(W2, E, dev)
    with torch.no_grad():
        pk3 = T._flash_pack(T.pack(w_a=wa, w_b=wb, w_o=wo, ln_w=lw, ln_b=lb, eps=1e-5, device=dev), E, dev)
    for k in ("w1p", "wo", "lnw", "lnb"):
        assert torch.equal(pk[k], pk2[k]) and torch.equal(pk[k], pk3[k]), k
    # the sealed unit's own pack on an engine-module stand-in gives the same operand bytes (the layout the kernel reads)
    ns = type("M", (), {})
    m = ns(); m.c_in = c; m.training = False
    for name, w in (("linear_no_bias_a", W2.wa16), ("linear_no_bias_b", W2.wb16), ("linear_no_bias", W2.wo16)):
        lin = ns(); lin.weight = w; setattr(m, name, lin)
    m.layernorm1 = torch.nn.LayerNorm(c, eps=1e-5)
    with torch.no_grad():
        m.layernorm1.weight.copy_(W2.ln_w); m.layernorm1.bias.copy_(W2.ln_b)
    saved = FT._EXT
    try:
        FT._EXT = E
        ref = FT._pack(m, dev)
    finally:
        FT._EXT = saved
    for k in ("w1p", "wo", "lnw", "lnb"):
        assert torch.equal(pk[k], ref[k]), k
    assert pk["nh"] == ref["nh"] and pk["eps"] == ref["eps"]


def test_abi_key_word():
    pytest.importorskip("torch")
    key = T.flash_abi_key()
    assert key.startswith("torch") and key.endswith("-sm90") and "cpython-3" in key, key

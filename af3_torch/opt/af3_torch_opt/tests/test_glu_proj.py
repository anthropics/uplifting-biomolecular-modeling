"""Lever `glu_proj` (exact / fast / big): the transition's gated linear unit + output projection (+ the Pairformer block's residual add) as one
Triton kernel that restates xfold's fastnn statement byte for byte — registry / lever-set membership, the adapter's by-name step-asides spelled
as registry.EXPECTED_FALLBACKS judges them, the kernel module's invariants (int64 row offsets, one MMA class per autotune config, the widths it
serves), the in-process byte check, and the README / CHANGES rows. CPU only: the kernel's bytes are proven on the GPU (CHANGES.md)."""
import os, re
from af3_torch_opt import registry, modes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))                      # the kit dir (run.sh, README.md, opt/)
KERNELS = os.path.join(ROOT, "opt", "forward", "af3t", "kernels")


def _read(*parts):
    return open(os.path.join(ROOT, *parts)).read()


def test_registry_rows():
    assert "glu_proj" in registry.LEVERS and registry.LEVERS["glu_proj"]["kind"] == "kernel"
    assert "glu_proj" in registry.EXACT and "glu_proj" not in registry.NOT_BITWISE
    ev = registry.BITWISE_EVIDENCE["glu_proj"]
    assert ev["bitwise_vs_off"] and ev["bitwise_vs_stock_kernels"] and ev["kept"]
    assert registry.EVIDENCE["glu_proj"] == "census" and registry.IMPL["glu_proj"] == ("af3t_glu_proj.glu_proj", "kit") and registry.STRATEGY["glu_proj"] == "LOCAL.af3_torch.glu_proj"
    assert registry.EXPECTED_FALLBACKS["glu_proj"] == ("fallback:c=384", "fallback:c=128,rows<16384", "fallback:c=64,rows<16384", "fallback:c=64,card-off", "fallback:superseded:transition")


def test_mode_membership(monkeypatch):
    """exact keeps it (registry.EXACT filter of the `fastest` set), fast and big carry it (lever transition supersedes it there, by name);
    MODEL_OPT_LEVERS_OFF=glu_proj drops it from every mode."""
    monkeypatch.delenv(modes.ENV_LEVERS_OFF, raising=False)
    assert modes.resolve("exact")["levers"] == ("stepgraph", "glu_proj", "trimul_exact")
    for m in ("fast", "big"):
        assert "glu_proj" in modes.resolve(m)["levers"] and "transition" in modes.resolve(m)["levers"]
    assert "glu_proj" not in modes.resolve("off")["levers"]
    monkeypatch.setenv(modes.ENV_LEVERS_OFF, "glu_proj")
    for m in ("exact", "fast", "big"):
        r = modes.resolve(m)
        assert "glu_proj" not in r["levers"] and r["levers_off"] == ("glu_proj",) or list(r["levers_off"]) == ["glu_proj"]


def test_adapter_spells_the_expected_fallbacks_and_the_verification():
    ak = _read("opt", "forward", "af3t", "kernels", "af3_kernels.py")
    assert '"glu_proj")' in ak.split("\n_ON = set()")[0]                                              # in LEVERS
    assert "_GLU_MIN_ROWS = 16384" in ak and '_fallback(lever, "c=%d,rows<%d" % (C, _GLU_MIN_ROWS))' in ak   # -> fallback:c=128,rows<16384 / c=64,rows<16384
    assert '_fallback(lever, "c=%d" % C)' in ak.split("def _glu_proj_forward")[1].split("\ndef ")[0]        # -> fallback:c=384 (the single transition)
    body = ak.split("def _glu_proj_forward")[1].split("\n# ====")[0]
    assert "ok = bool(torch.equal(out, ref)); _GLU_CHECKED[key] = ok" in body                          # checked once per shape in process
    assert '("equal:" if ok else "differs:")' in body and '_fallback(lever, "differs:c=%d" % C)' in body   # a differing shape is served by stock BY NAME (not in EXPECTED_FALLBACKS: visible)
    assert 'if _fastnn_config.gated_linear_unit_implementation != "triton":' in body                  # the statement restated is the fastnn kernel's
    assert "if is_oom(e) or _capturing(): raise" in body
    assert '(_transition_forward_glu_aside if "glu_proj" in _ON else _transition_forward) if "transition" in _ON' in ak and 'else (_glu_proj_forward if "glu_proj" in _ON else _ORIG["transition"])' in ak
    assert 'if lever == "transition" and "glu_proj" in _ON and "transition" not in _ON:' in ak         # the block's add folds into the kernel epilogue under exact
    assert '_count("glu_proj", "fallback:superseded:transition")' in ak.split("def _transition_forward_glu_aside")[1].split("\ndef ")[0]   # fast / big: steps aside per call by name


def test_kernel_module_invariants():
    src = open(os.path.join(KERNELS, "af3t_glu_proj.py")).read()
    assert "r64 = rows.to(tl.int64)" in src and "r64[:, None] * C" in src                              # int64 row offsets before * C (XFOLD-003 class)
    assert "acc = tl.dot(h, w3t, acc)" in src and "a = tl.dot(y, w1t, a)" in src                        # chained accumulators: the tile choice cannot change the accumulation sequence
    assert "SUPPORTED_C = (64, 128)" in src and "SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)" in src
    cfgs = re.findall(r'triton\.Config\(\{"BM": (\d+), "BH": (\d+)\}, num_warps=(\d+)', src)
    assert len(cfgs) >= 4 and all(int(bm) >= 64 and int(w) in (4, 8) and int(bh) in (32, 64, 128) for bm, bh, w in cfgs), cfgs   # one MMA class per config; BH divides 128 / 256 / 512
    assert "no atomics" in src and "tl.atomic" not in src


def test_api_and_gate_rows():
    api = _read("opt", "forward", "af3t", "af3_torch", "af3_torch_api.py")
    fastest = re.search(r'"fastest": \(([^)]*)\)', api).group(1)
    assert '"glu_proj"' in fastest and '"transition"' in fastest
    assert '"glu_proj"' in re.search(r"_KERNEL_LEVERS = \(([^)]*)\)", api).group(1)
    fwd = _read("opt", "af3_torch_opt", "forward.py")
    assert '("glu_proj", "composes:row_local")' in fwd


def test_docs_rows():
    readme = _read("README.md"); changes = _read("CHANGES.md")
    assert "| `glu_proj` |" in readme
    assert "lever `glu_proj`" in changes and "MODEL_OPT_LEVERS_OFF=glu_proj" in changes

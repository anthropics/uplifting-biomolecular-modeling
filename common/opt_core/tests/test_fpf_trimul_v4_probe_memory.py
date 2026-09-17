"""kernels.fpf_trimul_v4 warm probe (reached through kernels.trimul's v4 row on the first serve of a shape class): its fp64 statement and bf16-class
yardstick are evaluated per OUTPUT BLOCK with a bounded transient (was ~2.8 GiB one-shot of whole [n, n, C|D] fp64 tensors at the 512-token probe
size), everything is released after the verdict, and a PASSED verdict is stamped through the byte gate's stamp directory chain (leaf fpf_trimul_v4)
so stamped processes / baked images skip the probe."""
import os

import pytest

torch = pytest.importorskip("torch")
try:
    from opt_core.kernels.fpf_trimul_v4 import generic as G
except Exception as e:                                                       # noqa: BLE001 -- triton absent: the package does not import here
    pytest.skip("fpf_trimul_v4 not importable here: %s" % e, allow_module_level=True)
from opt_core.kernels.trimul import native as NV


def _raw(C, D, has_bias, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float32)    # noqa: E731
    raw = dict(ln_in_w=1.0 + 0.1 * rnd(C), ln_in_b=0.1 * rnd(C), w_ag=rnd(D, C) * C ** -0.5, w_ap=rnd(D, C) * C ** -0.5, w_bg=rnd(D, C) * C ** -0.5, w_bp=rnd(D, C) * C ** -0.5,
               ln_out_w=1.0 + 0.1 * rnd(D), ln_out_b=0.1 * rnd(D), w_o=rnd(C, D) * D ** -0.5, w_og=rnd(C, C) * C ** -0.5)
    if has_bias:
        raw.update(b_ag=0.1 * rnd(D), b_ap=0.1 * rnd(D), b_bg=0.1 * rnd(D), b_bp=0.1 * rnd(D), b_o=0.1 * rnd(C), b_og=0.1 * rnd(C))
    return raw, rnd


@pytest.mark.parametrize("C,D,n,has_bias,masked", [(16, 8, 13, False, False), (8, 16, 10, True, True), (16, 16, 7, True, False)])
def test_blocked_reference_equals_the_whole_reference_elementwise(C, D, n, has_bias, masked):
    """Per-element operations are unchanged (LN over C per position, gated projections, the full-k contraction per (i, j), LN over D, gated
    out-projection): the blocked statement reproduces reference_torch on the CPU in fp64 and fp32, both directions, with / without bias, mask and
    residual, at several block sizes (incl. 1 row and blocks that do not divide n)."""
    raw, rnd = _raw(C, D, has_bias)
    z = rnd(n, n, C)
    mask = (torch.rand(n, n, generator=torch.Generator().manual_seed(3)) > 0.2).float() if masked else None
    for dtype in (torch.float64, torch.float32):
        for direction in ("outgoing", "incoming"):
            for residual in (False, True):
                whole = G.reference_torch(z, mask, direction=direction, dtype=dtype, residual=residual, **raw)
                for rows in (1, 3, n):
                    got = torch.empty_like(whole)
                    for I, J, o in G.reference_torch_blocks(z, mask, direction=direction, dtype=dtype, block_rows=rows, residual=residual, **raw):
                        got[I, J] = o
                    if dtype == torch.float64:
                        assert torch.equal(got, whole) or float((got - whole).abs().max()) <= 1e-12 * float(whole.abs().max()), (direction, rows, float((got - whole).abs().max()))
                    else:
                        assert torch.allclose(got, whole, rtol=1e-5, atol=1e-6), (direction, rows)


def test_streamed_error_statistics_match_the_whole_tensor_statistics():
    """max-abs of e_s / e_c identical to the whole-tensor statistics (the class yardstick tensor is the same whole tensor; the fp64 statement per block
    equals the whole one to fp64 rounding), sums of squares equal to fp64 re-ordering."""
    C, D, n = 16, 16, 11
    raw, rnd = _raw(C, D, False, seed=7)
    z = rnd(n, n, C)
    for direction in ("outgoing", "incoming"):
        cls = G.reference_torch(z, None, direction=direction, dtype=torch.float32, **raw)                       # the class statement (whole)
        served = (cls + 1e-3 * rnd(n, n, C)).to(torch.bfloat16)                                                  # a stand-in for the kernel's output
        ref64 = G.reference_torch(z, None, direction=direction, dtype=torch.float64, **raw)
        e_s, e_c = served.double() - ref64, cls.double() - ref64
        want = {"max_s": float(e_s.abs().max()), "max_c": float(e_c.abs().max()), "ss_s": float(e_s.pow(2).sum()), "ss_c": float(e_c.pow(2).sum()), "count": e_s.numel()}
        for rows in (2, 4, n):
            got = G.probe_error_stats(served, cls, z, None, direction=direction, raw=raw, block_rows=rows)
            assert got["count"] == want["count"]
            for k in ("max_s", "max_c"):
                assert abs(got[k] - want[k]) <= 1e-12 * max(want[k], 1e-300) + 1e-15, (k, rows, got[k], want[k])
            for k in ("ss_s", "ss_c"):
                assert abs(got[k] - want[k]) <= 1e-8 * want[k], (k, rows, got[k], want[k])        # fp64 re-ordered sums
            r_whole = (float(e_s.abs().max() / e_c.abs().max()), float(e_s.pow(2).mean().sqrt() / e_c.pow(2).mean().sqrt()))
            r_got = (got["max_s"] / got["max_c"], (got["ss_s"] / got["count"]) ** 0.5 / (got["ss_c"] / got["count"]) ** 0.5)
            assert abs(r_whole[0] - r_got[0]) <= 1e-9 * r_whole[0] and abs(r_whole[1] - r_got[1]) <= 1e-9 * r_whole[1]      # the verdict ratios


def test_block_rows_keep_one_blocks_transients_under_the_budget():
    for n, C, D in ((512, 128, 128), (512, 256, 256), (128, 128, 128), (2048, 128, 128)):
        rows = G.probe_block_rows(n, C, D, 8)
        assert 1 <= rows <= n and 6 * rows * n * max(C, D) * 8 <= max(G.PROBE_CHUNK_BYTES, 6 * n * max(C, D) * 8)
    assert G.probe_block_rows(128, 128, 128, 8) == 128                      # the 128-token probe size is ONE block (exactly the whole-tensor statement)
    assert G.probe_block_rows(512, 128, 128, 8) < 512                       # the 512-token size (descriptor cells) is blocked: whole, its fp64 statement alone was ~1 GiB


def test_a_passed_probe_verdict_is_stamped_and_a_fresh_process_takes_it_from_the_stamp(monkeypatch, tmp_path):
    """Stamp write -> fresh-interpreter hit (no probe run, nothing allocated) -> a changed package digest is another key (the probe runs again).
    Through the byte gate's directory chain with leaf fpf_trimul_v4 ($OPT_CORE_VERDICT_DIR here)."""
    monkeypatch.setenv(NV.STAMP_ENV, str(tmp_path))
    facts0 = {"schema": G.PROBE_STAMP_SCHEMA, "package": "fpf_trimul_v4", "version": "x", "digest": "aa" * 32, "C": 128, "D": 128, "bias": False, "in_f32": False,
              "sizes": [128, 512], "cell": "test-cell", "bars": [G.SAME_CLASS_RMS, G.SAME_CLASS_MAX], "cc": "9.0", "device_name": "TEST", "driver_version": 1,
              "cuda_version": "13.0", "torch": "2.13.0", "triton": "3.7.1", "python": "3.11"}
    state = {"digest": "aa" * 32, "runs": 0}
    monkeypatch.setattr(G, "probe_stamp_facts", lambda device, C, D, has_bias, in_f32, sizes: dict(facts0, digest=state["digest"], C=int(C), D=int(D), bias=bool(has_bias), in_f32=bool(in_f32), sizes=list(sizes)))
    monkeypatch.setattr(G.CELLS, "cell_for", lambda device: {"cell": "test"})
    monkeypatch.setattr(G.CELLS, "cell_word", lambda device: "test-cell")
    monkeypatch.setattr(G, "probe_sizes", lambda cfg, C, D, has_bias, in_f32: [128, 512])
    monkeypatch.setattr(G, "pack_weights", lambda **raw: {"packed": True})
    monkeypatch.setattr(G.CELLS, "run", lambda fn, cfg, device: None, raising=False)
    monkeypatch.setattr(G, "reference_torch", lambda z, mask, **kw: None)   # the class statement needs no device here

    def fake_stats(served, cls, z, mask, *, direction, raw, chunk_bytes=None, block_rows=None):
        state["runs"] += 1
        return {"max_s": 1.0, "max_c": 1.0, "ss_s": 1.0, "ss_c": 1.0, "count": 1}
    monkeypatch.setattr(G, "probe_error_stats", fake_stats)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    G._PROBES.clear()
    r1 = G.probe(torch.device("cpu"), 128, 128, False, False)                 # first process: the probe runs (4 = 2 sizes x out/in) and stamps
    assert r1["ok"] is True and state["runs"] == 4 and str(r1["stamp"]).startswith("written:")
    path = r1["stamp"].split("written:", 1)[1]
    assert path.startswith(os.path.join(str(tmp_path), G.PROBE_STAMP_LEAF) + os.sep) and os.path.isfile(path)
    G._PROBES.clear()                                                        # a fresh interpreter on the same stack: the verdict comes from the stamp, no run
    r2 = G.probe(torch.device("cpu"), 128, 128, False, False)
    assert r2["ok"] is True and r2["stamp"] == "hit" and state["runs"] == 4 and r2["ratio_max"] == r1["ratio_max"]
    G._PROBES.clear(); state["digest"] = "bb" * 32                           # any package edit (digest) / version / device / driver / torch / triton change: another key
    r3 = G.probe(torch.device("cpu"), 128, 128, False, False)
    assert state["runs"] == 8 and str(r3["stamp"]).startswith("written:") and r3["stamp"] != r1["stamp"]
    G._PROBES.clear()
    assert NV.stamp_dirs(leaf=G.PROBE_STAMP_LEAF) == [("env", os.path.join(str(tmp_path), G.PROBE_STAMP_LEAF))]     # same chain and order as the byte gate, sibling leaf


def test_probe_transient_peak_on_a_device_is_bounded_and_released():
    """Device check (skipped without CUDA): with a 4 GiB tensor resident, a forced probe of the (128, 128) bf16 class peaks well under the old
    ~2.8 GiB one-shot and leaves nothing reserved."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    resident = torch.empty(4 << 30, dtype=torch.uint8, device="cuda")
    os.environ[NV.STAMP_ENV] = "0"                                           # no stamp: the probe runs
    G._PROBES.clear()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base_a, base_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    rec = G.probe(torch.device("cuda"), 128, 128, False, False)
    peak = torch.cuda.max_memory_allocated() - base_a
    held = torch.cuda.memory_reserved() - base_r
    assert rec is not None and rec["ok"]
    assert peak < (1 << 30), peak                                            # z + served + kernel workspace + the whole bf16-class statement (~0.4 GiB at 512 tokens) + <=128 MiB of fp64 blocks (was ~2.8 GiB)
    assert held <= (64 << 20), held
    del resident


def test_the_stamp_machinery_works_and_never_raises_when_the_package_runs_under_its_BARE_name(monkeypatch, tmp_path):
    """Kits that route ``fpf_trimul_v4`` by its bare name run this file as the top-level package ``fpf_trimul_v4.generic``: the stamp chain must be
    reached by the shared core's ABSOLUTE name there (a relative import beyond the package raised ImportError inside probe() at 0.5.200.0), and
    nothing in the stamp facts / read / write may ever raise into a probe."""
    import importlib, sys
    kdir = os.path.dirname(os.path.dirname(os.path.abspath(G.__file__)))          # .../opt_core/kernels
    monkeypatch.syspath_prepend(kdir)
    for k in [k for k in sys.modules if k == "fpf_trimul_v4" or k.startswith("fpf_trimul_v4.")]:
        monkeypatch.delitem(sys.modules, k, raising=False)
    B = importlib.import_module("fpf_trimul_v4.generic")
    assert B.__name__ == "fpf_trimul_v4.generic" and B is not G
    NVb = B._stamp_api()
    assert NVb is not None and NVb.__name__ == "opt_core.kernels.trimul.native" and hasattr(NVb, "stamp_dirs")
    monkeypatch.setenv(NV.STAMP_ENV, str(tmp_path))
    facts = {"schema": B.PROBE_STAMP_SCHEMA, "package": "fpf_trimul_v4", "version": "x", "digest": "cc" * 32, "C": 128, "D": 128, "bias": False, "in_f32": False,
             "sizes": [128, 512], "cell": "c", "bars": [1.25, 2.5], "cc": "9.0", "device_name": "T", "driver_version": None, "cuda_version": "13.0",
             "torch": "2", "triton": "3", "python": "3.11"}
    assert B.probe_stamp_read(facts) is None
    p = B.probe_stamp_write(facts, {"ok": True})
    assert p and os.path.isfile(p) and B.probe_stamp_read(facts)["_path"] == p
    # a broken stamp chain (module without the api, or raising) -> None, never an exception
    monkeypatch.setattr(B, "_stamp_api", lambda: object())
    assert B.probe_stamp_read(facts) is None and B.probe_stamp_write(facts, {"ok": True}) is None
    def boom():
        raise RuntimeError("stamp chain exploded")
    monkeypatch.setattr(B, "_stamp_api", boom)
    assert B.probe_stamp_read(facts) is None and B.probe_stamp_write(facts, {"ok": True}) is None
    assert B.probe_stamp_facts("cpu", 128, 128, False, False, [128]) is None       # no device facts on a CPU host: None, no raise

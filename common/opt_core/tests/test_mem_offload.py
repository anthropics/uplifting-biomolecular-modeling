"""mem.offload: the module imports without torch; the refusal catalogue closes both ways; the lever is a registry lever (mem.apply
selects it, ctx.setting reads its settings from the kit's ctx.settings["host_park"], the census marks / skips land on the kit's
record, undo closes it); with torch (CPU line, device="cpu"): the streamed line equals the whole-tensor path on the synthetic stack
(torch.equal for GEMM-free ops, allclose + the measured equality recorded for GEMM-bearing ones), rows and cols, in place and to an
out buffer, with and without leading batch dims; a rows-only op streamed by cols is detectably wrong; the resident line round-trips;
module parking with hooks and with explicit residency; every refusal path by name; the gate; the record. CUDA tests (skipped without a
device) prove pinning, the pitched column copies against the rowloop and against the reference, and the budget refusals with real
pinned memory. Needs the core's ``opt_core.mem`` package (registry / record): on a tree without it the file skips by name."""
import json
import os
import re
import subprocess
import sys

import pytest

pytest.importorskip("opt_core.mem.registry", reason="opt_core.mem (registry.py) is not on this tree")
from opt_core import mem  # noqa: E402
from opt_core.mem import offload as O  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(HERE)


# ----------------------------------------------------------------------------------------------------------------- torch-free
def test_import_needs_no_torch():
    code = "import sys; import opt_core.mem.offload as m; assert 'torch' not in sys.modules, 'torch imported at module load'; print(m.LEVER)"
    r = subprocess.run([sys.executable, "-c", code], cwd=CORE_DIR, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": CORE_DIR})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "host_park"


def test_refusal_catalogue_closes_both_ways():
    src = open(O.__file__).read()
    raised = set(re.findall(r'(?:_refuse|OffloadRefusal)\(\s*"([a-z_]+)"', src)) | set(re.findall(r'refuse\(LEVER,\s*"([a-z_]+)"', src))
    assert raised == set(O.REFUSALS), {"raised_not_catalogued": raised - set(O.REFUSALS), "catalogued_not_raised": set(O.REFUSALS) - raised}


def test_settings_refusals_by_name():
    for kw in [dict(pin_max_gb=0, min_tokens=0), dict(pin_max_gb=1, min_tokens=-1), dict(pin_max_gb=1, min_tokens=0, block=0),
               dict(pin_max_gb=1, min_tokens=0, window=1), dict(pin_max_gb=1, min_tokens=0, cols="magic"),
               dict(pin_max_gb=1, min_tokens=0, host_reserve_gb=-1), dict(pin_max_gb=1, min_tokens=0, device="tpu"),
               dict(pin_max_gb=True, min_tokens=0)]:
        with pytest.raises(O.OffloadRefusal) as ei:
            O.Settings(**kw)
        assert ei.value.name == "bad_setting"
    s = O.Settings(pin_max_gb=2.5, min_tokens=1000)
    assert s.as_dict()["cols"] == "pitched" and s.block == 256 and s.window == 2


def _ctx(environ=None, settings=None, hooks=None, **kw):
    return mem.Ctx(prefix="ENG", tag="[eng]", environ=environ if environ is not None else {},
                   settings={"host_park": settings} if settings is not None else {},
                   hooks={"host_park": hooks if hooks is not None else {"device": "cpu"}}, **kw)


def test_registered_on_the_core_registry():
    lv = mem.LEVERS["host_park"]
    assert lv.family == "offload" and lv.exact == "measured" and lv.settings == O.SETTINGS and lv.module == "opt_core.mem.offload"
    assert set(lv.preconditions) == set(O.REFUSALS)
    assert mem.discover()["loaded"].count("offload") == 1 and "host_park" in mem.names("offload")
    assert mem.setting_ref("host_park", "pin_max_gb") == "ctx.settings['host_park']['pin_max_gb']"


def test_settings_from_ctx_core_grammar():
    env = {"ENG_BIG_HOST_PARK_PIN_MAX_GB": "99", "ENG_BIG_HOST_PARK_COLS": "magic", "ENG_BIG_HOST_PARK_BLOCK": "0"}    # never read: no effect
    s = O.Settings.from_ctx(_ctx(env, settings={"pin_max_gb": "12", "min_tokens": 1500, "cols": "rowloop", "block": "128"}))
    assert (s.pin_max_gb, s.min_tokens, s.cols, s.block, s.window, s.device) == (12.0, 1500, "rowloop", 128, 2, "cpu")   # the kit's values; strings cast
    with pytest.raises(mem.RefusalError) as ei:
        O.Settings.from_ctx(_ctx({}, settings={"min_tokens": 1, "pin_max_gb": "twelve"}))
    assert ei.value.refusal.lever == "host_park" and ei.value.refusal.precondition == "settings.host_park.pin_max_gb"
    with pytest.raises(O.OffloadRefusal) as ei:
        O.Settings.from_ctx(_ctx({}))
    assert ei.value.name == "bad_setting" and ei.value.details["missing"] == ["pin_max_gb", "min_tokens"]
    assert ei.value.details["settings"] == ["ctx.settings['host_park']['pin_max_gb']", "ctx.settings['host_park']['min_tokens']"]
    assert isinstance(ei.value, mem.RefusalError) and ei.value.refusal.precondition == "bad_setting"
    sel = mem.selection(("host_park",), switches={"host_park": True})
    assert sel.levers == ("host_park",) and sel.refusals == []                       # on for a line lever: nothing to add, no refusal


def test_mem_apply_census_and_undo():
    ctx = _ctx({}, settings={"pin_max_gb": 1.0, "min_tokens": 4, "block": "3"})
    rec = mem.apply(("host_park",), ctx, base="exact")
    assert rec.mode_line() == "big:host_park" and rec.applied[0].settings["block"] == 3 and rec.applied[0].settings["device"] == "cpu"
    assert {(x["key"], x["source"]) for x in rec.settings} >= {("block", "ctx.settings"), ("pin_max_gb", "ctx.settings"), ("window", "default")}
    assert rec.preconditions[-1]["ok"] is True and set(rec.preconditions[-1]["checked"]) == set(O.REFUSALS)
    park = O.park_of(ctx)
    assert park.ctx is ctx and park.record()["ctx"] is True
    rec.unit_begin("u1")
    assert park.gate(3) is False
    rec.unit_end()
    rec.unit_begin("u2")
    assert park.gate(10) is True
    park.park("z", pair((6, 6, 2)))
    park.stream_blocks(lambda b, a, e: b * 2, "z", "rows", exact="bitwise", reason="elementwise")
    rec.unit_end()
    c = rec.census()
    assert c["units"]["u1"]["skipped"] == {"host_park": "below_gate:3/4"} and "u1" not in c["partial_units"]
    assert c["units"]["u2"]["ran"] == ["host_park"] and c["ok"] is True
    ev = [e["detail"] for e in rec.units["u2"].events]
    assert ev == ["resident:tensor", "streamed"]
    assert mem.undo(rec) == ["host_park"] and park.closed is True
    with pytest.raises(O.OffloadRefusal) as ei:
        O.park_of(_ctx({}))
    assert ei.value.name == "not_applied"


def test_mem_apply_refusals_by_name():
    rec = mem.apply(("host_park",), _ctx({}, settings={"pin_max_gb": 1.0, "min_tokens": 4}, graphs=True), base="exact", strict=False)
    assert [r.precondition for r in rec.refused] == ["graphs"] and rec.applied == []
    rec = mem.apply(("host_park",), _ctx({}, settings={"pin_max_gb": 1.0, "min_tokens": "many"}), base="exact", strict=False)
    assert rec.refused[0].precondition == "settings.host_park.min_tokens"
    rec = mem.apply(("host_park",), _ctx({}), base="exact", strict=False)
    assert rec.refused[0].precondition == "bad_setting"
    with pytest.raises(mem.Refused):
        mem.apply(("host_park",), _ctx({}), base="exact")
    rec = mem.apply(("host_park",), _ctx({}, settings={"pin_max_gb": 1.0, "min_tokens": 4}), base="exact", switches={"host_park": False})
    assert rec.mode_line() == "big:" and rec.applied == []           # switched off by the kit: named, nothing applied
    rec = mem.apply(("host_park",), _ctx({"ENG_BIG_HOST_PARK": "0"}, settings={"pin_max_gb": 1.0, "min_tokens": 4}), base="exact")
    assert rec.mode_line() == "big:host_park"                        # the former switch VARIABLE has no effect


def test_describe_and_refusal_forms():
    d = O.describe()
    assert d["name"] == "host_park" and d["family"] == "offload" and d["lines"]["resident"]["exact"] == "bitwise" and "streamed" in d["lines"]
    assert d["refusals"] == sorted(O.REFUSALS) and d["exact_labels"] == ["bitwise", "band", "measured"]
    r = O.OffloadRefusal("not_parked", "x is not parked", name="x")
    assert isinstance(r, mem.RefusalError) and r.refusal.lever == "host_park" and r.refusal.precondition == "not_parked"
    assert r.refusal.details == {"name": "x"} and str(r) == "host_park: not_parked: x is not parked"
    g = r.gate()
    assert g.name == "host_park:not_parked" and g.ok is False and g.details == {"name": "x"}


def test_host_mem_available_reads_meminfo(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       1000 kB\nMemAvailable:    2048 kB\n")
    assert O.host_mem_available_bytes(str(p)) == 2048 * 1024
    p.write_text("MemTotal:       1000 kB\n")
    with pytest.raises(O.OffloadRefusal) as ei:
        O.host_mem_available_bytes(str(p))
    assert ei.value.name == "host_ram_unknown"
    with pytest.raises(O.OffloadRefusal) as ei:
        O.host_mem_available_bytes(str(tmp_path / "absent"))
    assert ei.value.name == "host_ram_unknown"


def test_cudart_unavailable_by_name(monkeypatch):
    monkeypatch.setattr(O, "_CUDART", {"lib": None, "path": ["x"], "tried": True})
    with pytest.raises(O.OffloadRefusal) as ei:
        O.cudart()
    assert ei.value.name == "cudart_unavailable"


def test_cudart_error_by_name(monkeypatch):
    class Lib:
        def cudaMemcpy2DAsync(self, *a):
            return 11
    monkeypatch.setattr(O, "_CUDART", {"lib": Lib(), "path": "fake", "tried": True})
    with pytest.raises(O.OffloadRefusal) as ei:
        O.memcpy2d_async(0, 0, 0, 0, 0, 0, 1, 0)
    assert ei.value.name == "cudart_error" and ei.value.details["cudaError"] == 11


# ----------------------------------------------------------------------------------------------------------------- torch (CPU line)
torch = pytest.importorskip("torch")
if torch is not None:
    sys.path.insert(0, HERE)
    import offload_harness as S  # noqa: E402  (the launcher over tests/synthetic_pair.py — the chunking stack)


@pytest.fixture(autouse=True)
def _inference_only():
    """The lever is inference-only (``grad_enabled`` refuses a streamed call under grad mode): every test runs under no_grad."""
    with torch.no_grad():
        yield

CPU = dict(pin_max_gb=1.0, min_tokens=0, block=5, window=2, device="cpu")


def cpu_park(**kw):
    p = O.HostPark(O.Settings(**{**CPU, **kw}))
    p.check()
    return p


def pair(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


@pytest.mark.parametrize("lead", [(), (1,), (2,)])
@pytest.mark.parametrize("dim", ["rows", "cols"])
@pytest.mark.parametrize("block,window", [(5, 2), (4, 3), (13, 2), (1, 2)])
def test_stream_scale_shift_bitwise(lead, dim, block, window):
    n, c = 13, 8
    m = S.ScaleShift(c)
    z = pair(lead + (n, n, c))
    ref = m(z)
    p = cpu_park(block=block, window=window)
    zt = p.park("z", z)
    out = p.park_like("out", z.shape, z.dtype)
    r = O.stream_blocks(lambda b, a, e: m(b), zt, dim, block, window, out=out, exact="bitwise", reason="GEMM-free")
    assert r is out and torch.equal(out.host, ref)
    assert torch.equal(zt.host, z)                     # the input is untouched with an out buffer
    r2 = p.stream_blocks(lambda b, a, e: m(b), "z", dim, exact="bitwise", reason="in place")
    assert r2 is zt and torch.equal(zt.host, ref)
    rec = p.record()
    assert rec["mode_ran"] == "resident:tensor+streamed" and rec["lines_ran"] == ["resident:tensor", "streamed"]
    assert [e for e in rec["events"] if e["event"] == "stream"][0]["n_blocks"] == -(-n // block)


@pytest.mark.parametrize("dim", ["rows", "cols"])
def test_stream_transition_allclose_and_equality_recorded(dim, record_property):
    n, c = 11, 8
    m = S.PairTransition(c, 2)
    z = pair((n, n, c))
    ref = m(z)
    p = cpu_park(block=4)
    zt = p.park("z", z)
    p.stream_blocks(lambda b, a, e: m(b), zt, dim, exact="measured", reason="GEMM-bearing")
    assert torch.allclose(zt.host, ref, rtol=1e-5, atol=1e-6)
    ev = [e for e in p.record()["events"] if e["event"] == "stream"][0]
    assert ev["exact"] == "measured"
    eq = bool(torch.equal(zt.host, ref))                                     # measured on this CPU/torch, recorded, never assumed
    record_property(f"transition_streamed_{dim}_bitwise_cpu", eq)
    record_property(f"transition_streamed_{dim}_max_abs_diff_cpu", float((zt.host - ref).abs().max()))


def test_rows_only_op_streamed_by_cols_is_detectably_wrong():
    n, c = 9, 8
    m = S.RowGate(c)
    z = pair((n, n, c))
    ref = m(z)
    p = cpu_park(block=3)
    zt = p.park("z", z)
    p.stream_blocks(lambda b, a, e: m(b), zt, "rows", exact="measured", reason="row-separable")
    assert torch.allclose(zt.host, ref)
    p2 = cpu_park(block=3)
    zt2 = p2.park("z", z)
    p2.stream_blocks(lambda b, a, e: m(b), zt2, "cols", exact="measured", reason="WRONG dim on purpose")
    assert not torch.allclose(zt2.host, ref)


def test_synth_stack_offloaded_equals_reference():
    torch.manual_seed(1)
    n, c = 10, 8
    stack = S.SynthStack(c=c, n=2, c_hidden=4, no_heads=2, n_blocks=2, bank_rows=32)
    z, zi = pair((n, n, c), 1), pair((n, n, c), 2)
    ref = S.run_reference(stack, z.clone(), zi, attn_block=3)
    p = cpu_park(block=4, pin_max_gb=1.0)
    p.park("z", z)
    p.park("z_init", zi)
    name = S.run_offloaded(stack, p, attn_block=3)
    got = p.get(name).host
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    rec = p.record()
    assert rec["mode_ran"] == "resident:tensor+resident:module+streamed" and rec["refusals"] == []
    assert set(rec["parked"]) == {"z", "z_init", "tmp"} and "head" in rec["parked_modules"]
    assert all(str(pp.device) == "cpu" for pp in stack.head.parameters())    # parked between calls
    p.close()
    assert rec["closed"] is False and p.record()["closed"] is True and p.record()["pinned_now_gb"] == 0.0
    json.dumps(p.record())


def test_resident_roundtrip_and_unpark():
    p = cpu_park()
    z = pair((6, 6, 4))
    p.park("z", z)
    with p.resident("z") as g:
        assert g is not p.get("z").host
        g.mul_(2.0)
    assert torch.equal(p.get("z").host, z * 2)
    with p.resident("z", writeback=False) as g:
        g.zero_()
    assert torch.equal(p.get("z").host, z * 2)
    back = p.unpark("z")
    assert torch.equal(back, z * 2) and "z" not in p.tensors and p.pool.bytes_now == 0


def test_rows_cols_accessors_and_put():
    p = cpu_park()
    z = pair((2, 7, 7, 3))
    zt = p.park("z", z)
    assert torch.equal(zt.rows(2, 5), z[:, 2:5]) and torch.equal(zt.cols(1, 4), z[:, :, 1:4])
    zt.put_rows(0, 2, torch.zeros(2, 2, 7, 3))
    zt.put_cols(6, 7, torch.ones(2, 7, 1, 3))
    exp = z.clone(); exp[:, 0:2] = 0; exp[:, :, 6:7] = 1
    assert torch.equal(zt.host, exp)
    assert torch.equal(zt.to_device(), exp)
    zt.put(exp * 3)
    assert torch.equal(zt.host, exp * 3)


def test_park_module_with_hooks_and_explicit_residency():
    torch.manual_seed(0)
    m = S.ParamBank(4, 16)
    x = pair((3, 3, 4))
    ref = m(x)
    p = cpu_park()
    mp = p.park_module(m, "head")
    assert all(id(t) == id(h) for (_, t, h) in mp.entries) or all(t.data.data_ptr() == h.data_ptr() for (_, t, h) in mp.entries)
    assert torch.equal(m(x), ref)                       # hooks bring the weights in for the call
    assert mp._dev == [] and all(t.data.data_ptr() == h.data_ptr() for (_, t, h) in mp.entries)
    with mp.resident():
        assert torch.equal(m(x), ref)
    p.unpark_module("head")
    assert torch.equal(m(x), ref) and mp._hooks and "head" not in p.modules and p.pool.bytes_now == 0
    with pytest.raises(O.OffloadRefusal) as ei:
        p.unpark_module("head")
    assert ei.value.name == "not_parked"
    with pytest.raises(O.OffloadRefusal) as ei:
        p.park_module(m, "h2", what="weights")
    assert ei.value.name == "bad_setting"


def test_refusals_by_name_on_the_cpu_line():
    p = cpu_park(pin_max_gb=1e-7)          # 107 bytes: below one 6x6x4 fp32 tensor
    z = pair((6, 6, 4))
    with pytest.raises(O.OffloadRefusal) as ei:
        p.park("z", z)
    assert ei.value.name == "pin_budget_exceeded" and p.record()["mode_ran"] == "refused:pin_budget_exceeded"
    p = cpu_park()
    with pytest.raises(O.OffloadRefusal) as ei:
        p.get("z")
    assert ei.value.name == "not_parked"
    zt = p.park("z", z)
    with pytest.raises(O.OffloadRefusal) as ei:
        p.park("z", z)
    assert ei.value.name == "already_parked"
    with pytest.raises(O.OffloadRefusal) as ei:
        p.park("zt", z.transpose(0, 1))
    assert ei.value.name == "not_contiguous"
    with pytest.raises(O.OffloadRefusal) as ei:
        p.park("meta", torch.empty(2, 2, device="meta"))
    assert ei.value.name == "device_mismatch"
    with pytest.raises(O.OffloadRefusal) as ei:
        O.stream_blocks(lambda b, a, e: b, zt, "rows", 2, 2)
    assert ei.value.name == "exact_unlabelled"
    with pytest.raises(O.OffloadRefusal) as ei:
        O.stream_blocks(lambda b, a, e: b, zt, "rows", 2, 2, exact="bitwise", reason="  ")
    assert ei.value.name == "exact_unlabelled"
    with pytest.raises(O.OffloadRefusal) as ei:
        O.stream_blocks(lambda b, a, e: b, z, "rows", 2, 2, exact="bitwise", reason="raw tensor, not parked")
    assert ei.value.name == "not_parked"
    with torch.enable_grad():
        with pytest.raises(O.OffloadRefusal) as ei:
            O.stream_blocks(lambda b, a, e: b, zt, "rows", 2, 2, exact="bitwise", reason="grad mode on")
    assert ei.value.name == "grad_enabled"
    with pytest.raises(O.OffloadRefusal) as ei:
        p.park("g", pair((6, 6, 4)).requires_grad_())
    assert ei.value.name == "grad_enabled"
    with pytest.raises(O.OffloadRefusal) as ei:
        O.stream_blocks(lambda b, a, e: b, zt, "diag", 2, 2, exact="bitwise", reason="x")
    assert ei.value.name == "layout_unsupported"
    for blk, win in [(0, 2), (2, 1)]:
        with pytest.raises(O.OffloadRefusal) as ei:
            O.stream_blocks(lambda b, a, e: b, zt, "rows", blk, win, exact="bitwise", reason="x")
        assert ei.value.name == "window_invalid"
    O.stream_blocks(lambda b, a, e: b, zt, "rows", 7, 2, exact="bitwise", reason="block > N: clamped, recorded")
    clamp = [e for e in p.record()["events"] if e["event"] == "block_clamped"]
    assert clamp and clamp[0]["block"] == 7 and clamp[0]["N"] == 6
    with pytest.raises(O.OffloadRefusal) as ei:
        O.stream_blocks(lambda b, a, e: b[..., :1], zt, "rows", 2, 2, exact="bitwise", reason="x")
    assert ei.value.name == "fn_shape_mismatch"
    with pytest.raises(O.OffloadRefusal) as ei:
        zt.rows(0, 2, out=torch.zeros(3, 6, 4))
    assert ei.value.name == "fn_shape_mismatch"
    bad = p.park("bad", pair((6, 6, 3)))
    with pytest.raises(O.OffloadRefusal) as ei:
        O.stream_blocks(lambda b, a, e: b, zt, "rows", 2, 2, out=bad, exact="bitwise", reason="x")
    assert ei.value.name == "fn_shape_mismatch"
    nonpair = p.park("np", pair((5, 6, 4)))
    with pytest.raises(O.OffloadRefusal) as ei:
        nonpair.rows(0, 1)
    assert ei.value.name == "layout_unsupported"
    with pytest.raises(O.OffloadRefusal) as ei:
        zt.put(torch.zeros(6, 6, 5))
    assert ei.value.name == "fn_shape_mismatch"
    p.close()
    with pytest.raises(O.OffloadRefusal) as ei:
        p.park("z", z)
    assert ei.value.name == "closed"
    names = [r["name"] for r in p.record()["refusals"]]
    assert names[:3] == ["not_parked", "already_parked", "not_contiguous"] and names[-1] == "closed"
    assert "grad_enabled" in names and "window_invalid" in names


def test_no_cuda_refusal_when_cuda_absent():
    if torch.cuda.is_available():
        pytest.skip("cuda present")
    p = O.HostPark(O.Settings(pin_max_gb=1, min_tokens=0))
    with pytest.raises(O.OffloadRefusal) as ei:
        p.check()
    assert ei.value.name == "no_cuda" and p.record()["mode_ran"] == "refused:no_cuda"


def test_host_ram_refusals_through_the_pool(monkeypatch):
    s = O.Settings(pin_max_gb=100, min_tokens=0, host_reserve_gb=1.0)
    pool = O.PinPool(s)
    pool.pin = True
    monkeypatch.setattr(O, "host_mem_available_bytes", lambda meminfo="/proc/meminfo": 2 * (1 << 30))
    with pytest.raises(O.OffloadRefusal) as ei:
        pool.alloc((1 << 28, 2), torch.float32, "big")      # 2 GiB + 1 GiB reserve > 2 GiB avail
    assert ei.value.name == "host_ram_insufficient"
    monkeypatch.setattr(O, "host_mem_available_bytes", lambda meminfo="/proc/meminfo": (_ for _ in ()).throw(O.OffloadRefusal("host_ram_unknown", "x")))
    with pytest.raises(O.OffloadRefusal) as ei:
        pool.alloc((4,), torch.float32, "small")
    assert ei.value.name == "host_ram_unknown"
    monkeypatch.setattr(O, "host_mem_available_bytes", lambda meminfo="/proc/meminfo": 64 * (1 << 30))

    def fail_pin(*a, **k):
        if k.get("pin_memory"):
            raise RuntimeError("CUDA error: OS call failed or operation not supported on this OS")
        return torch.zeros(1)
    monkeypatch.setattr(torch, "empty", fail_pin)
    with pytest.raises(O.OffloadRefusal) as ei:
        pool.alloc((4,), torch.float32, "pinned")
    assert ei.value.name == "pin_alloc_failed"


def test_gate_and_record():
    p = O.HostPark(O.Settings(pin_max_gb=1, min_tokens=1500, device="cpu"))
    p.check()
    assert p.gate(1000) is False and p.record()["mode_ran"] == "below_gate" and p.record()["gate_crossed"] is False
    assert p.gate(1500) is True and p.record()["mode_ran"] is None
    ev = [e for e in p.record()["events"] if e["event"] == "gate"]
    assert [e["crossed"] for e in ev] == [False, True] and ev[0]["min_tokens"] == 1500
    z = pair((4, 4, 2))
    p.park("z", z)
    assert p.record()["mode_ran"] == "resident:tensor"
    p.park_module(S.ParamBank(2, 4), "m")
    assert p.record()["mode_ran"] == "resident:tensor+resident:module"
    rec = p.record()
    for k in ("lever", "family", "settings", "mode_ran", "lines_ran", "gate_crossed", "checks", "events", "refusals", "pinned_peak_gb", "pinned_now_gb", "h2d_gb", "d2h_gb", "cudart", "parked", "parked_modules", "closed"):
        assert k in rec
    assert rec["checks"][0]["check"] == "cpu_line" and rec["pinned"] is False
    json.dumps(rec)


# ----------------------------------------------------------------------------------------------------------------- CUDA
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")


@cuda
def test_cuda_pinned_and_pitched_cols_equal_reference_and_rowloop():
    n, c = 37, 8
    z = pair((2, n, n, c)).cuda()
    for cols in ("pitched", "rowloop"):
        p = O.HostPark(O.Settings(pin_max_gb=1, min_tokens=0, block=5, window=3, cols=cols))
        p.check()
        zt = p.park("z", z)
        assert zt.host.is_pinned() and zt.pinned and p.record()["pinned"] is True
        assert (p.cudart_path is not None) == (cols == "pitched")
        g = zt.cols(3, 9)
        torch.cuda.synchronize()
        assert torch.equal(g, z[:, :, 3:9])
        zt.put_cols(0, 5, torch.full((2, n, 5, c), 7.0, device="cuda"))
        torch.cuda.synchronize()
        exp = z.clone(); exp[:, :, 0:5] = 7.0
        assert torch.equal(zt.host.cuda(), exp)
        m = S.ScaleShift(c).cuda()
        ref = m(exp)
        p.stream_blocks(lambda b, a, e: m(b), "z", "cols", exact="bitwise", reason="GEMM-free")
        assert torch.equal(zt.host.cuda(), ref)
        if cols == "pitched":
            with pytest.raises(O.OffloadRefusal) as ei:
                zt.put_cols(0, 5, torch.zeros(2, n, c, 5, device="cuda").transpose(2, 3))     # strided source: named, never a temporary
            assert ei.value.name == "layout_unsupported"
        p.release("z")                                                               # quiesces both streams before the free
        assert p.pool.bytes_now == 0
        p.close()


@cuda
def test_cuda_synth_stack_streams_equal_reference():
    torch.manual_seed(0)
    n, c = 40, 8
    stack = S.SynthStack(c=c, n=2, c_hidden=4, no_heads=2, n_blocks=2, bank_rows=64).cuda()
    z, zi = pair((n, n, c), 1).cuda(), pair((n, n, c), 2).cuda()
    ref = S.run_reference(stack, z.clone(), zi, attn_block=3)
    p = O.HostPark(O.Settings(pin_max_gb=1, min_tokens=0, block=7, window=2))
    p.check()
    p.park("z", z)
    p.park("z_init", zi)
    del z, zi
    name = S.run_offloaded(stack, p, attn_block=3)
    got = p.get(name).host.cuda()
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    rec = p.record()
    assert rec["mode_ran"] == "resident:tensor+resident:module+streamed" and rec["refusals"] == [] and rec["pinned"] and rec["cudart"]
    p.close()


@cuda
def test_cuda_module_park_hooks():
    m = S.ParamBank(8, 1024).cuda()
    x = pair((5, 5, 8)).cuda()
    ref = m(x)
    p = O.HostPark(O.Settings(pin_max_gb=1, min_tokens=0))
    p.park_module(m, "head")
    assert all(pp.device.type == "cpu" and pp.data.is_pinned() for pp in m.parameters())
    assert torch.equal(m(x), ref)
    assert all(pp.device.type == "cpu" for pp in m.parameters())
    p.unpark_module("head")
    assert all(pp.device.type == "cuda" for pp in m.parameters()) and torch.equal(m(x), ref)


@cuda
def test_cuda_pin_budget_refusal_with_real_pinning():
    p = O.HostPark(O.Settings(pin_max_gb=0.001, min_tokens=0))
    p.check()
    with pytest.raises(O.OffloadRefusal) as ei:
        p.park("z", torch.zeros(1024, 1024, device="cuda"))
    assert ei.value.name == "pin_budget_exceeded" and p.pool.bytes_now == 0


# ----------------------------------------------------------------------------------------------------------------- the ONE pinned pool
def test_the_lever_pool_is_the_package_pool_and_words_its_refusals():
    """``offload.PinPool`` is the package's one pinned host-buffer pool (``mem.torch_hostpair.PinPool``) bound to the lever's settings;
    the MemAvailable reader is the package's one reader, re-worded as this lever's ``host_ram_unknown``."""
    from opt_core.mem import torch_hostpair as H

    assert issubclass(O.PinPool, H.PinPool) and O.GiB == H.GiB == 1 << 30
    assert set(H.PIN_REFUSALS) <= set(O.REFUSALS), "every pool refusal kind is a named refusal of the lever"
    with pytest.raises(O.OffloadRefusal) as ei:
        O.host_mem_available_bytes(meminfo=os.path.join(HERE, "no-such-meminfo"))
    assert ei.value.name == "host_ram_unknown"
    with pytest.raises(H.PinRefused) as pi:
        H.host_mem_available_bytes(meminfo=os.path.join(HERE, "no-such-meminfo"))
    assert pi.value.kind == "host_ram_unknown" and isinstance(pi.value, mem.MemLeverRefused)


def test_one_pool_budget_release_pageable_policy_and_kinds(monkeypatch):
    from opt_core.mem import torch_hostpair as H

    pool = H.PinPool(1 << 20, pin=False, lever="t")                       # the CPU line: plain host tensors, budget enforced
    a = pool.alloc((256, 512), torch.float32, "a")                        # 512 KiB
    b = pool.alloc((256, 512), torch.float32, "b")                        # 1 MiB total == budget
    assert tuple(a.shape) == (256, 512) and pool.bytes_now == 1 << 20 and pool.n_alloc == 2 and not a.is_pinned()
    with pytest.raises(H.PinRefused) as ei:
        pool.alloc((1,), torch.float32, "c")
    assert ei.value.kind == "pin_budget_exceeded" and ei.value.details["tag"] == "c" and pool.bytes_now == 1 << 20 and isinstance(ei.value, mem.MemLeverRefused)
    pool.release(a)
    assert pool.bytes_now == 1 << 19 and pool.bytes_peak == 1 << 20
    pool.release(a)                                                       # releasing twice / a foreign tensor changes nothing
    pool.release(torch.zeros(3))
    assert pool.bytes_now == 1 << 19
    staging = H.PinPool(4096, pin=False, pageable=True, lever="w")        # the writer's policy: a refused request is a COUNTED pageable buffer
    p1 = staging.alloc((2048,), torch.float32, "big")                     # 8 KiB > 4 KiB budget
    p2 = staging.alloc((2048,), torch.float32, "big2")
    assert tuple(p1.shape) == (2048,) and staging.n_pageable == 2 and [e["kind"] for e in staging.events] == ["pin_budget_exceeded"] and staging.bytes_now == 0
    assert staging.alloc_counted((8,), torch.float32, "fits")[1] is None and staging.alloc_counted((2048,), torch.float32, "big3")[1] == "pin_budget_exceeded"
    # the pinned path (pin=True) with the host-RAM check and the page-lock, driven through fakes: every refusal returns its reservation
    avail = {"v": 64 << 30}

    def fake_avail():
        if avail["v"] is None:
            raise H.PinRefused("p", "host_ram_unknown", "no meminfo here")
        return avail["v"]
    pinned = H.PinPool(100 << 30, reserve_bytes=1 << 30, pin=True, lever="p", avail=fake_avail)
    avail["v"] = 2 << 30
    with pytest.raises(H.PinRefused) as ei:
        pinned.alloc((1 << 28, 2), torch.float32, "big")                  # 2 GiB + 1 GiB reserve > 2 GiB available
    assert ei.value.kind == "host_ram_insufficient" and ei.value.details["host_reserve_gb"] == 1.0 and pinned.bytes_now == 0
    avail["v"] = None
    with pytest.raises(H.PinRefused) as ei:
        pinned.alloc((4,), torch.float32, "small")
    assert ei.value.kind == "host_ram_unknown" and pinned.bytes_now == 0
    avail["v"] = 64 << 30
    real_empty = torch.empty

    def fail_pin(*a, **k):
        if k.get("pin_memory"):
            raise RuntimeError("CUDA error: OS call failed or operation not supported on this OS")
        return real_empty(*a, **k)
    monkeypatch.setattr(torch, "empty", fail_pin)
    with pytest.raises(H.PinRefused) as ei:
        pinned.alloc((4,), torch.float32, "pinned")
    assert ei.value.kind == "pin_alloc_failed" and pinned.bytes_now == 0 and pinned.n_alloc == 0
    per_call = pinned.alloc((4,), torch.float32, "plain", pin=False)      # a per-call pin=False request skips the RAM check and the page-lock
    assert tuple(per_call.shape) == (4,) and pinned.bytes_now == 16


def test_pinnedpool_alias_is_a_keyed_cache_over_the_one_pool():
    from opt_core.mem import torch_hostpair as H

    led = mem.Ledger()
    pp = H.PinnedPool(lever="hp", ledger=led)
    assert isinstance(pp.pool, H.PinPool) and pp.pool.max_bytes is None
    v = pp.get((2, 3), torch.float32, pin=True)                           # no CUDA on this box: a plain buffer, pinned=False recorded
    w = pp.get((3, 2), torch.float32, pin=True)                           # same (numel, dtype) key: the SAME buffer, another view
    if not torch.cuda.is_available():
        assert not pp.is_pinned((2, 3), torch.float32) and led.get("host_pageable_bytes") == 24
    assert tuple(v.shape) == (2, 3) and tuple(w.shape) == (3, 2) and v.data_ptr() == w.data_ptr() and pp.bytes() == 24 and pp.pool.n_alloc == 1
    pp.drop()
    assert pp.bytes() == 0 and pp.pool.bytes_now == 0

"""CPU tests for the exit rule under upstream's own kernel / dtype knobs (report.knob_states): a completed run whose user-set STOCK knob
(--triatt_kernel / --trimul_kernel other than cuequivariance, --dtype other than bf16) routed it off a lever's kernel path names that lever
`skipped reason=stock_knob:<knob>:<value>` and exits 0; the same served-0 account under DEFAULT knobs stays partial (exit 3)."""
import copy

import pytest

from protenix_v1_opt import modes, report as R, stack
from protenix_v1_opt.tests.test_exit_rule import full_account

FAST = modes.resolve("fast")
EXACT = modes.resolve("exact")
DEFAULT = dict(R.STOCK_KNOB_DEFAULTS)


def _rep(res, acc, **knobs):
    return {"active": True, "mode": res.mode, "arm": res.arm, "levers": acc, "stock_knobs": dict(DEFAULT, **knobs)}


def _lines(v, rep):
    return {l.rsplit(" lever=", 1)[1]: l for l in R.lever_lines(v["evidence"], rep)}


def test_knob_words_reduce_to_the_non_default_ones():
    assert R.stock_knobs({"stock_knobs": dict(DEFAULT)}) == {}
    assert R.stock_knobs({"stock_knobs": dict(DEFAULT, triatt_kernel="torch")}) == {"triatt_kernel": "torch"}
    assert R.stock_knobs({}) == {} and R.stock_knobs(None) == {} and R.stock_knobs({"stock_knobs": "x"}) == {}
    assert set(R.KNOB_LEVERS) == set(R.STOCK_KNOB_DEFAULTS)


@pytest.mark.parametrize("res", [FAST, EXACT], ids=["fast", "exact"])
def test_triatt_kernel_torch_idles_the_tri_attention_levers_by_name_exit_0(res):
    acc = full_account(res)
    acc["counts"]["triattn"] = {"stock:path": 1920}                     # --triatt_kernel torch: every triangle-attention call answered by the stock statement (levers_ptx1._tri_forward)
    for w in ("core:triexact", "core:tricuda"):
        acc["counts"].pop(w, None)                                       # the kernel words behind the fused block are never reached
    rep = _rep(res, acc, triatt_kernel="torch")
    v = R.verdict(rep, res.trimul, res.levers, allow_partial=False)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK, (v["partial"], v["partial_reason"])
    tri = [lv for lv in ("gblock", "gflash", "triexact", "tricuda") if lv in res.levers]
    assert tri and all(v["evidence"][lv]["knob"] == {"state": "knob-off", "flag": "triatt_kernel:torch", "knob": "triatt_kernel", "value": "torch"} for lv in tri)
    by = _lines(v, rep)
    for lv in tri:
        assert f" state=skipped reason=stock_knob:triatt_kernel:torch " in by[lv] and " served=0 " in by[lv], by[lv]
    word = "gflash" if "gflash" in res.levers else "gblock"
    assert " gated=1920 gated_by=stock:path:1920 " in by[word]
    assert " state=on " in by[res.trimul] and " state=on " in by["sg"]     # the levers the knob does not touch are judged as always
    # the same account under DEFAULT knobs: an undeclared served-0 -> partial, exit 3 (unchanged rule)
    rep0 = _rep(res, copy.deepcopy(acc))
    v0 = R.verdict(rep0, res.trimul, res.levers, allow_partial=False)
    assert word in v0["partial"] and v0["exit_code"] == R.EXIT_NOT_ACTIVE and f"{word} served 0 calls (gated {{'stock:path': 1920}})" in v0["partial_reason"]
    assert f" state=skipped reason=served0 " in _lines(v0, rep0)[word]
    # an older record without the knobs: unchanged rule
    rep1 = dict(rep0); rep1.pop("stock_knobs")
    assert R.verdict(rep1, res.trimul, res.levers, allow_partial=False)["exit_code"] == R.EXIT_NOT_ACTIVE


@pytest.mark.parametrize("res", [FAST, EXACT], ids=["fast", "exact"])
def test_trimul_kernel_torch_idles_the_trimul_word_by_name_exit_0(res):
    acc = full_account(res)
    acc["counts"]["trimul"] = {"stock:path": 96}                         # --trimul_kernel torch: the stock TriMul statement answers every call (levers_ptx1._trimul_forward)
    rep = _rep(res, acc, trimul_kernel="torch")
    v = R.verdict(rep, res.trimul, res.levers, allow_partial=False)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK, (v["partial"], v["partial_reason"])
    line = _lines(v, rep)[res.trimul]
    assert f" state=skipped reason=stock_knob:trimul_kernel:torch " in line and " served=0 " in line and " gated_by=stock:path:96 " in line, line
    v0 = R.verdict(_rep(res, copy.deepcopy(acc)), res.trimul, res.levers, allow_partial=False)
    assert v0["partial"] == [res.trimul] and v0["exit_code"] == R.EXIT_NOT_ACTIVE


@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_dtype_other_than_bf16_idles_the_autocast_gated_levers_by_name_exit_0(dtype):
    res = FAST
    acc = full_account(res)
    acc["counts"]["triattn"] = {"stock:gate": 1920}                     # no autocast: the fused block's own gate (levers_ptx1._tri_forward)
    acc["counts"]["transition"] = {"stock:gate": 400}                    # the fused transitions' dtype / autocast gate
    acc["counts"]["pf"] = {"stock:no_autocast": 96}; acc["counts"]["opm"] = {"stock:no_autocast": 8}; acc["counts"]["pwa"] = {"stock:no_autocast": 6}
    acc["counts"].pop("core:tricuda", None)
    rep = _rep(res, acc, dtype=dtype)
    v = R.verdict(rep, res.trimul, res.levers, allow_partial=False)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK, (v["partial"], v["partial_reason"])
    by = _lines(v, rep)
    idle = [lv for lv in ("gflash", "tricuda", "ttr", "pfattn", "opm_fused", "pwa_fused") if lv in res.levers]
    assert {"gflash", "ttr"} <= set(idle)
    for lv in idle:
        assert f" state=skipped reason=stock_knob:dtype:{dtype} " in by[lv], by[lv]
    assert " state=on " in by[res.trimul] and " state=on " in by["ditattn"]     # levers of the dtype set that still served are judged as always (served > 0)
    v0 = R.verdict(_rep(res, copy.deepcopy(acc)), res.trimul, res.levers, allow_partial=False)
    assert {"gflash", "ttr"} <= set(v0["partial"]) and v0["exit_code"] == R.EXIT_NOT_ACTIVE


def test_a_fallback_under_a_knob_stays_partial():
    """A knob excuses an idle lever (served 0, nothing fell back), never a failed attempt: a fallback counter is partial with or without the knob."""
    acc = full_account(FAST)
    acc["counts"]["triattn"] = {"stock:path": 100, "fallback:gflash:RuntimeError": 2}     # levers_ptx1._tri_forward: the fused block raised and the stock statement answered (a fallback word)
    v = R.verdict(_rep(FAST, acc, triatt_kernel="torch"), FAST.trimul, FAST.levers, allow_partial=False)
    assert "gflash" in v["partial"] and v["exit_code"] == R.EXIT_NOT_ACTIVE and "knob" not in v["evidence"]["gflash"]


def test_the_runner_wrap_reads_the_knobs_off_upstreams_configs():
    class Cfg(dict):
        __getattr__ = dict.__getitem__
    c = Cfg(triangle_attention="torch", triangle_multiplicative="cuequivariance", dtype="fp32", other=1)
    assert stack.stock_knobs_of(c) == {"triatt_kernel": "torch", "trimul_kernel": "cuequivariance", "dtype": "fp32"}
    class Obj:
        triangle_attention = "cuequivariance"; dtype = "bf16"
    assert stack.stock_knobs_of(Obj()) == {"triatt_kernel": "cuequivariance", "trimul_kernel": None, "dtype": "bf16"}
    assert stack.stock_knobs_of(None) == {}
    assert R.stock_knobs({"stock_knobs": stack.stock_knobs_of(Obj())}) == {}


# ------------------------------------------------------------------------------------------------ the measured --dtype fp32 census (fast, H100, 2 x 400 tokens)
def _fp32_census_0233(acc):
    """The account as kit 0.2.33 printed it under --dtype fp32: the flash block ENTERED and refused per call (fallback words), the fused
    transitions gated by name, the fused outer-product-mean kernel's assertion caught per call (error word), tricuda aside."""
    acc["counts"]["triattn"] = {"fallback:gflash:no-cell:fpf:prologue:64x4x32:9.0|3.7+off(not-measured)": 160, "fallback:gflash:dtype:float32": 2080}
    acc["counts"]["transition"] = {"stock:C=64": 40, "stock:C=128": 400, "stock:C=384": 8}
    acc["counts"]["opm"] = {"error:AssertionError": 80}
    acc["counts"]["core:tricuda"] = {"aside:k2b:cell:k2": 20}
    return acc


def _fp32_census_0236(acc):
    """The same run as the 0.2.36 gates account it: the autocast compute dtype read BEFORE any cell lookup / launch — by-name stock words only."""
    acc["counts"]["triattn"] = {"stock:dtype:float32": 2240}                          # levers_ptx1._tri_forward: trunk (c=128) and template (c=64) calls alike
    acc["counts"]["transition"] = {"stock:C=64": 40, "stock:C=128": 400, "stock:C=384": 8}
    acc["counts"]["opm"] = {"stock:autocast_dtype=float32": 80}                       # trunk2_ptx1._common_route
    acc["counts"]["pf"] = {"stock:autocast_dtype=float32": 96}; acc["counts"]["pwa"] = {"stock:autocast_dtype=float32": 6}
    acc["counts"]["core:tricuda"] = {}                                                # never reached behind the block's gate
    return acc


def test_the_measured_fp32_census_as_0_2_33_printed_it_stays_partial():
    """Fallback / error words are failed attempts, not idles: the knob does not excuse them (why the gates of 0.2.36 read the dtype first)."""
    res = FAST
    rep = _rep(res, _fp32_census_0233(full_account(res)), dtype="fp32")
    v = R.verdict(rep, res.trimul, res.levers, allow_partial=False)
    assert v["exit_code"] == R.EXIT_NOT_ACTIVE and v["partial"] == ["gflash", "opm_fused"], (v["partial"], v["partial_reason"])
    by = _lines(v, rep)
    assert " state=skipped reason=fallback " in by["gflash"] and " state=skipped reason=fallback " in by["opm_fused"]
    assert " state=skipped reason=stock_knob:dtype:fp32 " in by["ttr"] and " state=skipped reason=stock_knob:dtype:fp32 " in by["tricuda"]


def test_the_fp32_census_as_the_dtype_gates_account_it_exits_0_by_name():
    res = FAST
    rep = _rep(res, _fp32_census_0236(full_account(res)), dtype="fp32")
    v = R.verdict(rep, res.trimul, res.levers, allow_partial=False)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK, (v["partial"], v["partial_reason"])
    by = _lines(v, rep)
    for lv in ("gflash", "tricuda", "ttr", "opm_fused", "pfattn", "pwa_fused"):
        assert f" state=skipped reason=stock_knob:dtype:fp32 " in by[lv] and " served=0 " in by[lv], by[lv]
    assert " gated_by=stock:dtype:float32:2240 " in by["gflash"] and " gated_by=stock:autocast_dtype=float32:80 " in by["opm_fused"]
    assert " state=on " in by[res.trimul] and " state=on " in by["sg"] and " state=on " in by["ditattn"]
    # the same by-name account under DEFAULT knobs (no --dtype): served 0 is partial, exit 3
    v0 = R.verdict(_rep(res, _fp32_census_0236(full_account(res))), res.trimul, res.levers, allow_partial=False)
    assert {"gflash", "ttr", "opm_fused", "pfattn", "pwa_fused"} <= set(v0["partial"]) and v0["exit_code"] == R.EXIT_NOT_ACTIVE


# ------------------------------------------------------------------------------------------------ the source gates (CPU: decided before any launch)
def test_trunk2_routes_name_another_autocast_dtype_before_any_launch():
    import os, sys
    from protenix_v1_opt import kit as K
    sys.path.insert(0, os.path.join(K.kit_home(), "ptxfpf"))
    dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True                        # the kit tree carries no bytecode caches (test_kit_carry)
    try:
        import trunk2_ptx1 as T2
        for adt in ("float32", "float16"):
            assert T2.opm_route((4, 32, 64), False, dtype="float32", autocast=True, autocast_dtype=adt) == "stock:autocast_dtype=" + adt
            assert T2.pwa_route((4, 32, 64), (1, 32, 32, 128), dtype="float32", z_dtype="float32", autocast=True, autocast_dtype=adt) == "stock:autocast_dtype=" + adt
            assert T2.pf_route((32, 32, 128), True, (1, 32, 32, 128), False, dtype="float32", z_dtype="float32", autocast=True, autocast_dtype=adt) == "stock:autocast_dtype=" + adt
        assert T2.opm_route((4, 32, 64), False, dtype="float32", autocast=False, autocast_dtype="float32") == "stock:no_autocast"   # autocast off keeps its own word
        assert T2._common_route("bfloat16", True, True, False, "bfloat16") is None and T2._common_route("float32", True, True, False) is None   # bf16 autocast (fp32-stored activations): served as before
    finally:
        sys.dont_write_bytecode = dwb
        sys.path.remove(os.path.join(K.kit_home(), "ptxfpf"))


def test_the_flash_block_takes_the_stock_statement_on_an_fp32_autocast_stream_without_raising(monkeypatch):
    torch = pytest.importorskip("torch")
    import os, sys, types
    from protenix_v1_opt import kit as K
    paths = [os.path.join(K.kit_home(), d) for d in K.KIT_SYS_PATHS]
    for p in reversed(paths):
        sys.path.insert(0, p)
    dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        import levers_ptx1 as L
    except Exception as e:                                                             # the adapter's own imports (shared core kernels) unavailable in this interpreter
        sys.dont_write_bytecode = dwb
        for p in paths:
            if p in sys.path: sys.path.remove(p)
        pytest.skip("levers_ptx1 not importable here: %r" % (e,))
    try:
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True), raising=False)   # a CPU tensor posing as a CUDA one: the gate is read before any launch
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a, **k: True)
        monkeypatch.setitem(L.CFG, "gflash", True); monkeypatch.setitem(L.CFG, "gblock", False)
        monkeypatch.setitem(L._ORIG, "triattn", lambda self, x, mask, chunk_size, triangle_attention, inplace_safe: "STOCK")
        counts = L.COUNTS.setdefault("triattn", {}); saved = dict(counts); counts.clear()
        stub = types.SimpleNamespace(starting=True)
        for adt, x in ((torch.float32, torch.zeros(1, 32, 32, 128)), (torch.float16, torch.zeros(1, 32, 32, 64, dtype=torch.float16))):   # trunk c=128 and template c=64 alike
            monkeypatch.setattr(torch, "get_autocast_dtype", lambda device_type="cuda", _d=adt: _d)
            assert L._tri_forward(stub, x, mask=None, chunk_size=None, triangle_attention="cuequivariance", inplace_safe=False) == "STOCK"
        assert dict(counts) == {"stock:dtype:float32": 1, "stock:dtype:float16": 1}, dict(counts)
        assert not any(k.startswith("fallback:") for k in counts)
        counts.clear(); counts.update(saved)
    finally:
        sys.dont_write_bytecode = dwb
        for p in paths:
            if p in sys.path: sys.path.remove(p)

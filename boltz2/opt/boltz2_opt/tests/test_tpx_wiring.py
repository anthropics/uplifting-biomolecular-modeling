"""The ×P line's fused triangle kernels are wired BEHIND two switches and OFF unless named: with the switches unset the row block's callables
are today's statements object for object (no adapter, no census key change but `tpx`), an unknown word is refused by name, a named kernel the
installed core cannot provide is a NAMED state (`unavailable:<why>`) whose evidence word rides stack.evidence's fallback list, and with a core
that provides it the TriAttFns / TriMulFns are built through the core's providers with boltz's own q / weight conventions."""
import math
import types

import pytest

from .. import rowpair as RP, stack


@pytest.fixture(autouse=True)
def _reset():
    RP.reset_for_tests()
    yield
    RP.reset_for_tests()


def test_switches_unset_is_today_and_unknown_words_are_refused(monkeypatch):
    monkeypatch.delenv(RP.TPX_TRIATT_ENV, raising=False); monkeypatch.delenv(RP.TPX_TRIMUL_ENV, raising=False)
    assert RP.tpx_words() == {"triatt": None, "trimul": None}
    b = RP.tpx_bind()
    assert {k: v["state"] for k, v in b.items()} == {"triatt": "off", "trimul": "off"} and all(v["provider"] is None for v in b.values())
    rep = RP.tpx_report()
    assert rep["triatt"] == {"word": None, "state": "off", "reason": None, "env": "ROWPAIR_TRIATT_CORE"} and rep["lines"] == [] and stack.rowpair_tpx_words(rep) == []
    RP.reset_for_tests()
    assert RP.tpx_words({RP.TPX_TRIATT_ENV: " CUEQ "}) == {"triatt": "cueq", "trimul": None}, "case / blanks are not words"
    with pytest.raises(RP.Refused, match=r"ROWPAIR_TRIATT_CORE='flash': one of flash_triattn \| cueq \| torch"):
        RP.tpx_words({RP.TPX_TRIATT_ENV: "flash"})
    with pytest.raises(RP.Refused, match=r"ROWPAIR_TRIMUL_KERNELS='fpf': one of fpf_v4 \| torch"):
        RP.tpx_bind({RP.TPX_TRIMUL_ENV: "fpf"})
    assert set(RP.TPX_WORDS["triatt"]) == {"flash_triattn", "cueq", "torch"} and set(RP.TPX_WORDS["trimul"]) == {"fpf_v4", "torch"}, "the core's ONE vocabulary"


def test_a_named_kernel_the_core_cannot_provide_is_a_named_state_and_an_evidence_word(monkeypatch):
    """This tree's core predates the providers (or lacks the module / the function): state=unavailable, reason names which, the kit's own
    statement serves (provider None), and stack.evidence's per-rank fallback words carry `<switch>=<word>:unavailable:<why>`."""
    import importlib
    have_tri = hasattr(importlib.import_module("opt_core.mem.rowpair.triatt"), "attention_core")
    try:
        have_mul = hasattr(importlib.import_module("opt_core.mem.rowpair.trimul_fused"), "fused_trimul_fns")
    except ImportError:
        have_mul = False
    b = RP.tpx_bind({RP.TPX_TRIATT_ENV: "flash_triattn", RP.TPX_TRIMUL_ENV: "fpf_v4"})
    if not have_tri:                                                                # this tree's core predates the providers: detected by presence, never by version
        assert b["triatt"] == {"word": "flash_triattn", "state": "unavailable", "reason": "core_api_missing:opt_core.mem.rowpair.triatt.attention_core", "provider": None, "env": RP.TPX_TRIATT_ENV}
    if not have_mul:
        assert b["trimul"]["state"] == "unavailable" and b["trimul"]["reason"].startswith("core_missing:opt_core.mem.rowpair.trimul_fused") and b["trimul"]["provider"] is None
        assert stack.rowpair_tpx_words(RP.tpx_report())[-1].startswith("ROWPAIR_TRIMUL_KERNELS=fpf_v4:unavailable:core_missing:opt_core.mem.rowpair.trimul_fused")
    if have_tri and have_mul:
        assert b["triatt"]["state"] == b["trimul"]["state"] == "on" and stack.rowpair_tpx_words(RP.tpx_report()) == []
    fake = types.SimpleNamespace()                                                  # a module without the function: named
    monkeypatch.setitem(__import__("sys").modules, "opt_core.mem.rowpair.trimul_fused", fake)
    assert RP.tpx_provider("trimul") == (None, "core_api_missing:opt_core.mem.rowpair.trimul_fused.fused_trimul_fns")
    RP.reset_for_tests()
    b = RP.tpx_bind({RP.TPX_TRIMUL_ENV: "torch", RP.TPX_TRIATT_ENV: "torch"})      # torch = this module's own statements, no adapter, no provider (the ×P line as it was)
    assert b["trimul"] == {"word": "torch", "state": "torch", "reason": None, "provider": None, "env": "ROWPAIR_TRIMUL_KERNELS"} and stack.rowpair_tpx_words(RP.tpx_report()) == []
    assert b["triatt"] == {"word": "torch", "state": "torch", "reason": None, "provider": None, "env": "ROWPAIR_TRIATT_CORE"}


class _Lin:
    def __init__(self, n_out, n_in):
        import torch
        self.weight = torch.zeros(n_out, n_in); self.bias = torch.zeros(n_out); self.eps = 1e-5

    def __call__(self, x):
        return x @ self.weight.t()


def _fake_trimul_module(C=8):
    m = types.SimpleNamespace()
    m.norm_in = _Lin(C, C); m.p_in = _Lin(2 * C, C); m.g_in = _Lin(2 * C, C); m.norm_out = _Lin(C, C); m.p_out = _Lin(C, C); m.g_out = _Lin(C, C)
    return m


def test_trimul_fns_are_today_s_unless_fpf_v4_is_named_and_provided(monkeypatch):
    torch = pytest.importorskip("torch")
    mod = _fake_trimul_module()
    fns = RP._trimul_fns(mod)                                                       # switches unset: the core's plain TriMulFns over the module's statements
    TM = RP._core()[4]
    assert type(fns) is TM.TriMulFns and fns.C_h == 8
    RP.reset_for_tests()
    calls = {}

    class FakeRF:
        @staticmethod
        def fused_trimul_fns(weights, stock_fns, eps=1e-5, cells=None, ledger=None):
            calls.update(weights=sorted(weights), stock=stock_fns, eps=eps); return ("fused", stock_fns)

        @staticmethod
        def emit_line(tag):
            return f"[{tag}] LEVER name=F2.trimul_rows state=on impl=fpf_trimul_v4@x origin=core served=0 fallback=0"   # the core brackets the tag itself (opt_core.counters.Ledger.line)

    monkeypatch.setitem(RP._STATE, "tpx", {"triatt": {"word": None, "state": "off", "reason": None, "provider": None, "env": RP.TPX_TRIATT_ENV},
                                           "trimul": {"word": "fpf_v4", "state": "on", "reason": None, "provider": FakeRF, "env": RP.TPX_TRIMUL_ENV}})
    got = RP._trimul_fns(mod)
    assert got[0] == "fused" and type(got[1]) is TM.TriMulFns, "the provider wraps today's fns as its named fallback"
    from opt_core import trimul as CT
    assert set(CT.WEIGHT_KEYS) <= set(calls["weights"]) and calls["eps"] == 1e-5, calls
    assert RP.tpx_report()["lines"] == ["[boltz2-opt] LEVER name=F2.trimul_rows state=on impl=fpf_trimul_v4@x origin=core served=0 fallback=0"]


def test_triatt_attend_through_the_adapter_restates_boltz_attention_forward(monkeypatch):
    """With a provider, attend = _prep_qkv (unscaled q) -> attend_query_blocks(core, …) -> transpose -> _wrap_up; the adapter gets kernel=<word>,
    scale = c_hidden ** -0.5, layout bnhsd, mask from bias 0, triangle bias = bias 1; its `stock` is boltz's eager core on q / sqrt(c_hidden). With
    kernel served by `stock` the result equals the module's own mha(..., use_kernels=False) — the module's eager statement — to the bit."""
    torch = pytest.importorskip("torch")
    TA_mod = pytest.importorskip("boltz.model.layers.triangular_attention.attention")     # the installed boltz (the GPU boxes run this file too)
    TriangleAttentionStartingNode = TA_mod.TriangleAttentionStartingNode
    torch.manual_seed(0)
    mod = TriangleAttentionStartingNode(c_in=16, c_hidden=4, no_heads=2, inf=1e9).eval()
    seen = {}

    class FakeRA:
        @staticmethod
        def attention_core(stock, kernel, min_tokens=0, ledger=None, scale=None, layout="bnhsd", mask_from="bias0", tri_bias="bias1"):
            seen.update(kernel=kernel, scale=scale, layout=layout, mask_from=mask_from, tri_bias=tri_bias, min_tokens=min_tokens)
            return lambda q, k, v, biases: stock(q, k, v, biases)                  # the named fallback path: exactly the caller's q, k, v, biases

        @staticmethod
        def attend_query_blocks(core_fn, q, k, v, biases, qblock):
            seen["qblock"] = qblock; seen["q_shape"] = tuple(q.shape); return core_fn(q, k, v, list(biases))

    attend = RP._triatt_attend_tpx(mod, FakeRA, "flash_triattn")
    rows, N = 3, 5
    x = torch.randn(rows, N, 16); mask = torch.ones(rows, N); mask[:, -1] = 0; tb_full = torch.randn(N, N, 2)
    with torch.no_grad():
        got = attend(mod.layer_norm(x), mask, tb_full, (0, rows))
        m5 = mask.unsqueeze(0)[..., :, None, None, :]
        want = mod.mha(mod.layer_norm(x).unsqueeze(0), mod.layer_norm(x).unsqueeze(0), RP._bias5(tb_full), mod.inf * (m5 - 1), m5, use_kernels=False)[0]
    assert torch.equal(got, want), "the stock core given the unscaled q reproduces boltz's eager statement bit for bit"
    assert seen == {"kernel": "flash_triattn", "scale": 1.0 / math.sqrt(4), "layout": "bnhsd", "mask_from": "bias0", "tri_bias": "bias1", "min_tokens": 0, "qblock": None, "q_shape": (1, rows, 2, N, 4)}


def test_report_and_apply_surface_the_switches(monkeypatch):
    """tp_report carries `tpx` (both off by default: no other key of the report changes); apply() binds the switches before installing anything."""
    monkeypatch.delenv(RP.TPX_TRIATT_ENV, raising=False); monkeypatch.delenv(RP.TPX_TRIMUL_ENV, raising=False)
    rep = RP.report()
    assert rep["tpx"]["triatt"]["state"] == "off" and rep["tpx"]["trimul"]["state"] == "off" and rep["tpx"]["lines"] == []
    src = open(RP.__file__).read()
    assert src.index("    tpx_bind()") < src.index("    D = _core()[0]\n    world, rank, device = D.init_from_env()"), "bound before the process group"


def test_the_rowpair_lever_line_is_unchanged_by_the_switches_being_off():
    """report.lever_state renders the rowpair LEVER line's pairs from tp_report fields it names; the added `tpx` key (both off) changes nothing."""
    from .. import report
    base = {"installed": True, "n_gpu": 2, "calls": {"trunk_rows": 1}, "schedule": {"park_z_init": "host_pinned"}, "tp_exports": {}, "errors": [], "rank": 0, "peak_alloc_gib": 1.0}
    with_tpx = dict(base, tpx={"core": "0.5.17.4", "lines": [], "triatt": {"word": None, "state": "off", "reason": None, "env": RP.TPX_TRIATT_ENV},
                               "trimul": {"word": None, "state": "off", "reason": None, "env": RP.TPX_TRIMUL_ENV}})
    a = report.lever_state("rowpair_tp", {"tp_report": base}, {"n_gpu": 2}); b = report.lever_state("rowpair_tp", {"tp_report": with_tpx}, {"n_gpu": 2})
    assert a == b


def test_the_kit_modes_export_both_kernel_words_above_one_gpu_only():
    """modes.TP_EXPORTS carries the core's two steering variables (flash_triattn / fpf_v4) — set by stack.child_env at n_gpu > 1 only; they are
    the row's alone (a caller's copy is stripped with the other lever words)."""
    from .. import modes
    assert modes.TP_EXPORTS[RP.TPX_TRIATT_ENV] == "tier:big" and modes.TP_EXPORTS[RP.TPX_TRIMUL_ENV] == "fpf_v4"   # the core's tier door word
    e1, e2 = stack.child_env("big", {}, n_gpu=1), stack.child_env("big", {}, n_gpu=2)
    assert RP.TPX_TRIATT_ENV not in e1 and RP.TPX_TRIMUL_ENV not in e1
    assert e2[RP.TPX_TRIATT_ENV] == "tier:big" and e2[RP.TPX_TRIMUL_ENV] == "fpf_v4"
    assert stack.child_env("big", {RP.TPX_TRIMUL_ENV: "torch"}, n_gpu=2)[RP.TPX_TRIMUL_ENV] == "fpf_v4", "the row's word, not the caller's"
    assert RP.tpx_words(e2) == {"triatt": "tier:big", "trimul": "fpf_v4"} and RP.tpx_words(e1) == {"triatt": None, "trimul": None}
    assert RP.tpx_words({RP.TPX_TRIATT_ENV: "tier:fast"}) == {"triatt": "tier:fast", "trimul": None}          # every tier word of the core's door is a word
    with pytest.raises(RP.Refused, match=r"tier:<big\|fast\|exact>"):
        RP.tpx_words({RP.TPX_TRIATT_ENV: "tier:biggly"})


def test_a_rank_s_RowpairRefused_line_reaches_the_caller_s_stdout():
    """A x P fused lever that cannot run raises opt_core.mem.rowpair.RowpairRefused inside the rank process (the core names the lever and its one
    opt-out); worker.relay_lines carries that traceback line verbatim to the caller's transcript beside KERNELS / PHASE / PEAK, and the run exits
    failed by its units (worker exit code != 0), never degraded."""
    from .. import worker
    rank_log = ("Traceback (most recent call last):\n  File \"trunk.py\", line 1, in f\n"
                "opt_core.mem.rowpair.RowpairRefused: F2.trimul_rows (fpf_trimul_v4): cannot run in this process — triton is not importable; "
                "run the kit with --mode off, or opt this lever out with ROWPAIR_TRIMUL_KERNELS=torch\n")
    text = "[boltz2-opt big] KERNELS route=big_x2 settings=defaults verdict=PASS\n" + rank_log
    lines = worker.relay_lines(text, [])
    assert lines[0].startswith("[boltz2-opt big] KERNELS route=") and lines[1].startswith("opt_core.mem.rowpair.RowpairRefused: F2.trimul_rows (fpf_trimul_v4): cannot run")
    assert "ROWPAIR_TRIMUL_KERNELS=torch" in lines[1]

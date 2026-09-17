"""Lever L7 (the shared fused TriangleMultiplication kernel): the adapter `genie3_opt/trimul.py` over the shared core's `opt_core.trimul`
ladder and its census word in `genie3_opt/kernels.py`, wired by the batched driver's `--trimul fpf` flag — which mode fast's line carries
(modes.FAST_FLAGS): L7 is fast's TriangleMultiplication provider, never on the exact line; no package flag selects or deselects it.
L7's evidence is the KERNELS census line with an `engaged` / `partial` word and served >= 1; a request entirely under the kernel's token
floor prints the `declined:below_min_tokens` word (registry L7's declined rule) instead."""
import os
import re

from genie3_opt import kernels, modes, registry, trimul

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_l7_is_fasts_provider_and_no_package_switch():
    assert "L7" in modes.MODES["fast"].levers and modes.MODES["fast"].flags[-3:] == ("--trimul", "fpf", "--compile")
    assert "L7" not in modes.MODES["exact"].levers and "--trimul" not in modes.MODES["exact"].flags
    assert registry.LEVERS["L7"].switch == ("flag", "--trimul", "fpf") and registry.LEVERS["L7"].tier == 3
    cli_src = open(os.path.join(PKG, "cli.py"), encoding="utf-8").read()
    assert 'add_argument("--trimul"' not in cli_src and "--precision" not in cli_src                                  # the driver's flag, never the package's
    assert (trimul.SWITCH, trimul.CHOICES, trimul.DEFAULT) == (modes.TRIMUL_FLAG, ("stock", modes.TRIMUL_ON), "stock")     # one spelling: registry L7's switch, read by the driver's adapter and the mode table alike
    assert trimul.cell_row("9.0") == "9.0|*" and trimul.cell_row("8.0") == "8.0|*" and trimul.cell_row("8.9") is None and trimul.cell_row(None) is None


def test_census_word_grammar_and_the_lever_rule():
    kind, rx = registry.LEVERS["L7"].evidence
    assert kind == "log"
    ok = {"mode": "fpf", "census": {"kernel": "fpf_trimul_v4@4.2.0", "origin": "core", "served": 30, "fallback": {}}, "gate": {"ok": True, "reason": None}}
    assert trimul.word(ok) == "engaged:fpf_trimul_v4@4.2.0-core[served=30,fallback=0]"
    line = kernels.line(kernels.census(dict(ok, word=trimul.word(ok)), tf32=True))
    assert line == ("[genie3-opt] KERNELS route=g3batch trimul=engaged:fpf_trimul_v4@4.2.0-core[served=30,fallback=0] "
                    "triatt=n/a-upstream:no-triangle-attention-in-architecture cueq=n/a-upstream:not-imported deepspeed=n/a-upstream:not-imported tf32_matmul=1")
    assert re.search(rx, line)
    expected_fb = dict(ok, census=dict(ok["census"], fallback={"below_min_tokens": 6}))             # an EXPECTED fallback reason (a short binder input): a distinct census kind, counted; L7 still evidenced
    assert trimul.word(expected_fb) == "partial:fpf_trimul_v4@4.2.0-core(stock=6,stock_by=below_min_tokens:6)[served=30]" and re.search(rx, kernels.line(kernels.census(expected_fb, tf32=True)))
    assert kernels.verdict(kernels.census(expected_fb, tf32=True), "fpf") == "ok"
    refused = dict(ok, census=dict(ok["census"], served=0, fallback={"no_cell:cc(8, 0)": 30}), gate={"ok": False, "reason": "unexpected fallback reason(s) no_cell"})
    w = trimul.word(refused)
    assert w.startswith("fallback:unexpected_fallback_reason(s)_no_cell[served=0,fallback=30(") and not re.search(rx, kernels.line(kernels.census(refused, tf32=True)))
    served_none = dict(ok, census=dict(ok["census"], served=0))
    assert trimul.word(served_none).startswith("fallback:served_none[") and not re.search(rx, kernels.line(kernels.census(served_none, tf32=True)))
    floor = dict(ok, census=dict(ok["census"], served=0, fallback={"below_min_tokens": 60}), gate={"ok": False, "reason": "mode fast routed fpf_trimul_v4@4.2.0 but served 0 of 60 calls"})   # EVERY call under the kernel's floor: the core gate reads it as refused, the kit's word says declined
    assert trimul.declined(floor) == "below_min_tokens" and trimul.word(floor) == "declined:below_min_tokens[served=0,fallback=60(below_min_tokens:60)]"
    assert trimul.declined(dict(floor, census=dict(floor["census"], fallback={"below_min_tokens_v2": 60}))) is None      # exactly the expected reason, not a prefix of it
    dkind, drx, dkeys = registry.LEVERS["L7"].declined
    floor_line = kernels.line(kernels.census(dict(floor, word=trimul.word(floor)), tf32=True))
    assert dkind == "log+timings" and dkeys == ("trimul", "declined") and re.search(drx, floor_line) and not re.search(rx, floor_line) and kernels.verdict(kernels.census(dict(floor, word=trimul.word(floor)), tf32=True), "fpf") == "ok"
    mixed_zero = dict(floor, census=dict(floor["census"], fallback={"below_min_tokens": 30, "no_cell:cc(8, 9)": 30}))            # an unexpected reason among them: not a decline — the gate's refusal stands
    assert trimul.declined(mixed_zero) is None and trimul.word(mixed_zero).startswith("fallback:")
    assert trimul.declined(ok) is None and trimul.declined({"mode": "stock"}) is None
    errored = dict(ok, census=dict(ok["census"], errors={"RuntimeError": 1}), gate={"ok": False, "reason": "kernel errors RuntimeError:1"})
    assert ",errors=1(RuntimeError:1)]" in trimul.word(errored) and trimul.word(errored).startswith("fallback:")
    assert trimul.word({"mode": "stock"}) == "off-by-route:stock" == kernels.EXPECT["stock"][0] and not re.search(rx, kernels.line(kernels.census({"mode": "stock"}, tf32=False)))
    assert kernels.EXPECT["fpf"] == ("engaged:fpf_trimul_v4@", "partial:fpf_trimul_v4@", "declined:below_min_tokens[") and kernels.verdict(kernels.census(ok, tf32=True), "fpf") == "ok"
    assert kernels.verdict(kernels.census({"mode": "stock"}, tf32=True), "fpf").startswith("FAIL") and kernels.verdict(kernels.census(ok, tf32=True), "maybe").startswith("FAIL")
    assert set(kernels.CONSTANT_WORDS) == {"triatt", "cueq", "deepspeed"} and all(v.startswith("n/a-upstream:") for v in kernels.CONSTANT_WORDS.values())


def test_adapter_stands_on_the_core_ladder():
    """The kit carries no kernel and no ladder of its own: trimul.py names the core kernel, the core strategy word, the module's weights, and
    the one expected fallback reason; g3batch.py switches it before the capture and prints the census at exit (source facts — no torch here)."""
    assert trimul.KERNEL == "fpf_trimul_v4" and trimul.STRATEGY == "F2.fpf_trimul_fast" and trimul.MIN_TOKENS == 101
    assert trimul.EXPECTED_FALLBACKS == ("below_min_tokens",) and (trimul.SWITCH, trimul.CHOICES, trimul.DEFAULT) == ("--trimul", ("stock", "fpf"), "stock")
    keys = set(trimul.weights_of.__code__.co_consts) & {"ln_in_w", "w_ag", "b_ag", "w_o", "b_og"}
    assert keys == {"ln_in_w", "w_ag", "b_ag", "w_o", "b_og"}                                      # the module's biases are passed (genie3's Linear layers carry them)
    src = open(os.path.join(PKG, "trimul.py"), encoding="utf-8").read()
    assert "from opt_core import trimul as T" in src and "T.Lever(TAG, \"fast\", provider=T.fpf_v4(weights_of" in src and "import triton" not in src and "@triton" not in src
    g3b = open(os.path.join(PKG, "g3batch.py"), encoding="utf-8").read()                          # the real driver (the test tree's stand-in is _stubs.STUB_G3BATCH)
    i_enable, i_tab = g3b.index("KT.enable()"), g3b.index("tab = G.StepTables(sampler, dev)")
    assert 0 < i_enable < i_tab                                                                     # patched before the first capture
    assert 'ap.add_argument("--trimul", choices=KT.CHOICES, default=KT.DEFAULT' in g3b and "KK.line(T[\"kernels\"])" in g3b and "T[\"trimul\"] = KT.evidence()" in g3b
    assert trimul.census()["mode"] == "stock" and trimul.gate() == {"ok": True, "reason": None} and trimul.lever_line().endswith("state=off mode=stock")   # nothing enabled in this process


def test_route_names_the_core_kernel(monkeypatch):
    """The route holds the core copy by name and exports the kit's cell table before any import (no torch needed to check the gate)."""
    for k in ("FPF_TRIMUL_V4_CELLS", "FPF_TRIMUL_V4_NMIN", "FPF_TRIMUL_V4_NMAX"):
        monkeypatch.delenv(k, raising=False)
    rec = trimul.route()
    assert rec["kernel"] == "fpf_trimul_v4" and rec["routed"] and rec["core_copy"] and rec["runtime_imports"] == {}, rec   # the kernel imports no other carried kernel at run time
    assert os.environ["FPF_TRIMUL_V4_CELLS"] == trimul.CELLS_JSON and trimul.STATE["route"] is rec
    # the kit exports the cell table ONLY — no size gate of its own: the kernel's supported() decides (N >= 101, no upper limit)
    assert "FPF_TRIMUL_V4_NMIN" not in os.environ and "FPF_TRIMUL_V4_NMAX" not in os.environ

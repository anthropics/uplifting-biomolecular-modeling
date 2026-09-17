"""chunk_lift: upstream's score-budget clamp on the dynamic attention chunk lifted to upstream's own threshold-table value when the probed
device admits it — the admission arithmetic (table vs clamp vs free memory; the 40 GB and the 80 GB card), the CHUNK census line, the patch
installed at once / at import on a stand-in model module (the core's per-site patch), the gate's named event, the registry row and the
lines that carry the lever."""
import importlib
import os
import re
import sys

import pytest

from opendde_opt import chunklift, modes, registry, report as _report, stack

GIB = 1 << 30
H100 = (77 * GIB, int(79.1 * GIB))          # (free at decision time, device total): an idle 80 GB card
A100_40 = (38 * GIB, int(39.4 * GIB))       # an idle 40 GB card


# ---------------------------------------------------------------- upstream's two functions, restated verbatim (pinned against the stock's own memory-peak test values)
def test_table_and_clamp_are_upstreams():
    assert [chunklift.table_chunk(n) for n in (200, 1024, 1025, 1536, 1537, 2048, 2560, 2561)] == [None, None, 512, 512, 256, 256, 128, 32]
    # stock/src/tests/test_inference_memory_peak.py: "Threshold table says unchunked at N=1024; the budget bounds it to 256", "512 at N=1536 -> 128", "small stay unchunked"
    assert chunklift.stock_clamp(1024, chunklift.table_chunk(1024)) == 256
    assert chunklift.stock_clamp(1536, chunklift.table_chunk(1536)) == 128
    assert chunklift.stock_clamp(200, chunklift.table_chunk(200)) is None
    assert chunklift.stock_clamp(662, None) is None and chunklift.stock_clamp(663, None) == 512      # the clamp's onset on the pin's budget (2^floor(log2(450e6 / N^2)) < N)
    assert chunklift.stock_clamp(1910, 256) == 64 and chunklift.stock_clamp(2811, 32) == 32           # the structural site at N = 1,000 / 1,472


def test_need_is_the_reference_score_tensors():
    assert chunklift.need_bytes(1000, None) == 2 * 1000 * 4 * 1000 * 1000 * 4                          # [c=n, H, n, n] fp32 x (scores + softmax copy)
    assert chunklift.need_bytes(1910, 256) == 2 * 256 * 4 * 1910 * 1910 * 4
    assert chunklift.need_bytes(100, 512) == chunklift.need_bytes(100, None)                            # c never exceeds n
    assert chunklift.need_bytes(1000, 256) < chunklift.need_bytes(1000, None)


def test_admission_on_the_80_and_the_40_gb_card():
    dec = lambda n, card: chunklift.admit(n, chunklift.table_chunk(n), chunklift.stock_clamp(n, chunklift.table_chunk(n)), *card)   # noqa: E731  (the structural site's rule)
    res = lambda n, card: chunklift.admit(n, chunklift.table_chunk(n), chunklift.stock_clamp(n, chunklift.table_chunk(n)), *card, unchunked_only=True, gate=1024)   # noqa: E731  (the residue site's)
    # below the clamp's onset there is nothing to lift on either card
    assert dec(600, H100)["decision"] == "stock" and dec(600, H100)["chunk"] is None and dec(600, A100_40)["decision"] == "stock"
    # the 80 GB card admits the table (un-chunked) through N = 1024 and — at the structural site — the chunked table values beyond
    for n, want in ((663, None), (800, None), (1000, None), (1024, None), (1146, 512), (1530, 512), (1910, 256), (2560, 128)):
        d = dec(n, H100)
        assert d["decision"] == "admitted" and d["chunk"] == want, (n, d)
        r = res(n, H100)                                                      # the residue site lifts only an un-chunked table value: above the table's
        if want is None:                                                      # un-chunked gate the clamp stays, named (the sampler measured slower behind larger trunk chunks)
            assert r == d
        else:
            assert r["decision"] == f"above_gate:{n}/1024" and r["chunk"] == chunklift.stock_clamp(n, chunklift.table_chunk(n)) and r["reserve"] is None, (n, r)
    assert res(1472, H100) == {"chunk": 128, "decision": "above_gate:1472/1024", "need": chunklift.need_bytes(1472, 512), "free": H100[0], "reserve": None}
    assert res(600, H100)["decision"] == "stock" and res(2811, H100)["decision"] == "stock"          # table == clamp wins over the gate word: nothing to lift either way
    # the 40 GB card clamps earlier: un-chunked N = 1024 (32 GiB of scores) and 512 rows at 1530 do not fit 38 GiB free less the reserve
    assert dec(1000, A100_40)["decision"] == "admitted"
    assert dec(1024, A100_40) == {"chunk": 256, "decision": "clamped_by_memory", "need": chunklift.need_bytes(1024, None), "free": 38 * GIB, "reserve": chunklift.reserve_bytes(A100_40[1])}
    assert dec(1530, A100_40)["decision"] == "clamped_by_memory" and dec(1530, A100_40)["chunk"] == 128
    # the structural site sees the trunk's residents: less free memory clamps what an idle card admits
    assert chunklift.admit(1910, 256, 64, 30 * GIB, H100[1])["decision"] == "clamped_by_memory"
    # above N_st = 2560 the table and the clamp agree: nothing to lift, whatever the memory
    assert dec(2811, H100)["decision"] == "stock" and dec(2811, H100)["chunk"] == 32
    # no probe (no CUDA device): the clamp's value
    assert chunklift.admit(1000, None, 256, None, None) == {"chunk": 256, "decision": "no_probe", "need": chunklift.need_bytes(1000, None), "free": None, "reserve": None}
    # the reserve is a fraction of the total plus a floor
    assert chunklift.reserve_bytes(80 * GIB) == int(0.15 * 80 * GIB + 1 * GIB)


def test_the_census_line_words():
    d = chunklift.admit(1000, None, 256, *H100)
    s = chunklift.line("7B9C", "residue", 1000, None, 256, d)
    assert s == "[opendde-opt] CHUNK item=7B9C site=residue n=1000 table=none clamp=256 chunk=none decision=admitted need_gib=29.80 free_gib=77.00 reserve_gib=12.87" or \
        re.fullmatch(r"\[opendde-opt\] CHUNK item=7B9C site=residue n=1000 table=none clamp=256 chunk=none decision=admitted need_gib=29\.80 free_gib=77\.00 reserve_gib=12\.8\d", s)
    d = chunklift.admit(1024, None, 256, *A100_40)
    assert " chunk=256 decision=clamped_by_memory " in chunklift.line("x", "residue", 1024, None, 256, d)
    d = chunklift.admit(600, None, None, *H100)
    assert chunklift.line("x", "residue", 600, None, None, d).endswith("chunk=none decision=stock need_gib=6.44 free_gib=77.00 reserve_gib=none")
    assert "fallback" not in chunklift.line("x", "structural", 1910, 256, 64, chunklift.admit(1910, 256, 64, None, None))   # the activation tables forbid the word
    d = chunklift.admit(1472, 512, 128, *H100, unchunked_only=True, gate=1024)
    assert chunklift.line("x", "residue", 1472, 512, 128, d) == "[opendde-opt] CHUNK item=x site=residue n=1472 table=512 clamp=128 chunk=128 decision=above_gate:1472/1024 need_gib=33.06 free_gib=77.00 reserve_gib=none"


# ---------------------------------------------------------------- the patch on a stand-in model module
MODEL_SRC = '''
class _NS:
    def __init__(self, **k): self.__dict__.update(k)

class OpenDDE:
    resolved = []
    def __init__(self, thresholds=None):
        self.configs = _NS(infer_setting=_NS(chunk_size=256, dynamic_chunk_size=True,
                                             chunk_size_thresholds=thresholds or {"1024": -1, "1536": 512, "2048": 256, "2560": 128}))
    def _get_dynamic_chunk_size(self, N_token):
        for t, c in sorted((int(k), v) for k, v in self.configs.infer_setting.chunk_size_thresholds.items()):
            if N_token <= t:
                return None if c == -1 else c
        return 32
    def _resolve_pairformer_chunk_size(self, n_token, chunk_size, *, dynamic_chunk_size):
        if not dynamic_chunk_size:
            return chunk_size
        return self._bound_pairformer_chunk_size(n_token, self._get_dynamic_chunk_size(n_token))
    @staticmethod
    def _bound_pairformer_chunk_size(n_token, chunk_size):
        requested = chunk_size or n_token
        b = max(1, 450_000_000 // max(1, n_token * n_token))
        p2 = 1 << (b.bit_length() - 1)
        bounded = min(requested, p2)
        if chunk_size is None and bounded >= n_token:
            return None
        return bounded
    def _main_inference_loop(self, n_token, n_struct, chunk_size=4):
        a = self._resolve_pairformer_chunk_size(n_token, chunk_size, dynamic_chunk_size=True)
        b = self._resolve_pairformer_chunk_size(n_struct, a, dynamic_chunk_size=True)
        OpenDDE.resolved.append((a, b))
        return a, b
'''


@pytest.fixture
def model_package(tmp_path, monkeypatch):
    """A stand-in `opendde.model.opendde` importable from tmp_path (the real stock package must not be loaded by the kit's tests)."""
    pkg = tmp_path / "opendde" / "model"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "opendde", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "opendde.py").write_text(MODEL_SRC)
    for k in [k for k in sys.modules if k == "opendde" or k.startswith("opendde.")]:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(chunklift, "STATS", {k: (False if isinstance(v, bool) else [] if isinstance(v, list) else 0) for k, v in chunklift.STATS.items() if k != "patches"})
    monkeypatch.setattr(chunklift, "_CTX", {"item": 0, "site_calls": 0})
    monkeypatch.setitem(chunklift._PLAN, "plan", None)
    from opt_core import autoload                                              # one process = one stock module; a fresh stand-in per test needs fresh per-site patches
    for attr in (f"{chunklift.CLASS}.{chunklift.RESOLVE}", f"{chunklift.CLASS}.{chunklift.LOOP}"):
        monkeypatch.delitem(autoload._PATCHES, (chunklift.TARGET, attr), raising=False)
    finders = list(sys.meta_path); before = set(sys.modules)
    yield tmp_path
    sys.meta_path[:] = finders
    for k in [k for k in sys.modules if k not in before and (k == "opendde" or k.startswith("opendde."))]:
        del sys.modules[k]


def test_the_lines_that_carry_the_lever():
    assert "chunk_lift" not in modes.LINES["S1"].levers and "chunk_lift" in modes.LINES["LSTAR2A"].levers   # tolerance class: fast (and big below the offload gate), not exact
    for name, ln in modes.LINES.items():
        if name.startswith("BIG"):
            assert "chunk_lift" not in ln.levers, name                              # the static big rows: the offload unit keeps upstream's clamp; the row-sharded line never
    assert "chunk_lift" in modes.big_line("BIG_F", ("no_dit_hoist",)).levers     # below the offload size gate big is fast's resident lever set: the lever rides it
    assert "chunk_lift" not in modes.big_line(modes.BIG_TP_LINE, ("no_dit_hoist",)).levers


def test_patch_now_admits_on_an_80_gb_card(model_package, monkeypatch, capsys):
    m = importlib.import_module(chunklift.TARGET)
    monkeypatch.setattr(chunklift, "probe", lambda: H100)
    chunklift.install()
    assert chunklift.STATS["installed"] and not chunklift.STATS["armed"]
    assert m.OpenDDE()._main_inference_loop(1000, 1910) == (None, 256)         # stock: (256, 64)
    assert m.OpenDDE()._main_inference_loop(600, 1146) == (None, 512)          # stock: (None, 256): the residue site has nothing to lift, the structural site is admitted
    assert m.OpenDDE()._main_inference_loop(1472, 2811) == (128, 32)          # = stock (128, 32): the residue site above the un-chunked gate keeps the clamp, the structural site is table == clamp
    assert m.OpenDDE()._main_inference_loop(1146, 2200) == (256, 128)         # stock: (256, 64): residue above the gate (table 512) -> the clamp; structural table 128 admitted over the clamp's 64
    st = chunklift.kit_stats()
    assert (st["calls"], st["items"], st["admitted"], st["stock"], st["above_gate"], st["clamped_by_memory"]) == (8, 4, 4, 2, 2, 0)
    out = capsys.readouterr().out.splitlines()
    assert [ln.split(" decision=")[1].split()[0] for ln in out if " CHUNK " in ln] == ["admitted", "admitted", "stock", "admitted", "above_gate:1472/1024", "stock", "above_gate:1146/1024", "admitted"]
    assert chunklift.unchunked_gate(m.OpenDDE()) == 1024 and chunklift.unchunked_gate(m.OpenDDE(thresholds={"512": 256})) == 0 and chunklift.unchunked_gate(object()) == 0
    assert re.search(r" CHUNK item=#1 site=residue n=1000 table=none clamp=256 chunk=none decision=admitted ", out[0])
    assert re.search(r" CHUNK item=#1 site=structural n=1910 table=256 clamp=64 chunk=256 decision=admitted ", out[1])
    chunklift.install()                                                        # idempotent: one wrapper, one count per call
    m.OpenDDE()._main_inference_loop(1000, 1910)
    assert chunklift.STATS["calls"] == 10 and m.OpenDDE._resolve_pairformer_chunk_size._chunk_lift


def test_patch_at_import_clamps_on_a_40_gb_card(model_package, monkeypatch, capsys):
    assert chunklift.TARGET not in sys.modules
    monkeypatch.setattr(chunklift, "probe", lambda: A100_40)
    chunklift.install()
    assert chunklift.STATS["armed"] and not chunklift.STATS["installed"]
    m = importlib.import_module(chunklift.TARGET)                              # the patches serve the import (the module's body first), then wrap the class
    chunklift._sync()
    assert chunklift.STATS["installed"] and not chunklift.STATS["armed"]
    assert m.OpenDDE()._main_inference_loop(1024, 1956) == (256, 256)         # residue: 32 GiB of scores do not fit -> the clamp's 256; structural: table 256 (27.9 GiB + reserve <= 38) admitted
    assert m.OpenDDE()._main_inference_loop(1000, 1910) == (None, 256)
    st = chunklift.kit_stats()
    assert (st["admitted"], st["clamped_by_memory"]) == (3, 1) and st["last"]["site"] == "structural"
    assert " decision=clamped_by_memory need_gib=32.00 free_gib=38.00 " in capsys.readouterr().out


def test_no_probe_keeps_the_clamp_and_a_fixed_chunk_passes_through(model_package, monkeypatch, capsys):
    m = importlib.import_module(chunklift.TARGET)
    monkeypatch.setattr(chunklift, "probe", lambda: (None, None))
    chunklift.install()
    assert m.OpenDDE()._main_inference_loop(1000, 1910) == (256, 64)          # stock values, decided `no_probe`, printed
    assert m.OpenDDE()._resolve_pairformer_chunk_size(3000, 128, dynamic_chunk_size=False) == 128
    st = chunklift.kit_stats()
    assert (st["no_probe"], st["fixed"], st["calls"]) == (2, 1, 3)
    out = capsys.readouterr().out
    assert " decision=no_probe " in out and " decision=fixed " in out


def test_a_model_class_without_the_method_fails_activation_by_name(model_package):
    m = importlib.import_module(chunklift.TARGET)
    del m.OpenDDE._resolve_pairformer_chunk_size
    with pytest.raises(chunklift.ActivationError, match="_resolve_pairformer_chunk_size"):
        chunklift.install()
    assert not chunklift.STATS["installed"]


def test_the_gate_names_an_unwrapped_model_and_refresh_accounts_the_lever(model_package, monkeypatch):
    planned = ["dit_align", "chunk_lift"]
    assert chunklift.fallbacks(planned) == []                                   # nothing imported: no event
    m = importlib.import_module(chunklift.TARGET)
    assert chunklift.fallbacks(planned) == ["chunk_lift: the model module is imported but OpenDDE._resolve_pairformer_chunk_size is not wrapped (installed after its import without the patch)"]
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "levers_planned": planned, "levers_applied": ["dit_align"], "levers_fallback": [], "partial": False})
    rep = stack.refresh()
    assert "chunk_lift" not in rep["levers_applied"] and rep["partial"] is True and rep["levers_fallback"][0].startswith("chunk_lift: the model module is imported but")
    monkeypatch.setattr(chunklift, "probe", lambda: H100)
    chunklift.install(); m.OpenDDE()._main_inference_loop(800, 1530)
    assert chunklift.fallbacks(planned) == [] and chunklift.fallbacks(["dit_align"]) == []
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "levers_planned": planned, "levers_applied": ["dit_align"], "levers_fallback": [], "partial": False})
    rep = stack.refresh()
    assert "chunk_lift" in rep["levers_applied"] and rep["partial"] is False and rep["kit_stats"]["chunk_lift"]["admitted"] == 2
    lines = [ln for ln in _report.lever_lines(rep) if " name=chunk_lift " in ln]
    assert len(lines) == 1 and re.search(r"^\[opendde-opt\] LEVER name=chunk_lift state=on impl=opendde_opt/chunklift\.py origin=kit .*\bcalls=2 items=1 admitted=2 clamped_by_memory=0 stock=0 above_gate=0 fixed=0 no_probe=0$", lines[0]), lines


def test_registry_row_and_counter():
    from opendde_opt import ran
    lv = registry.LEVERS["chunk_lift"]
    assert lv.kit == registry.HOUSE and lv.cls == "forward" and lv.switch == "-" and os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(chunklift.__file__)), lv.file))
    assert registry.PIN_STATUS["chunk_lift"][0] == "tested" and registry.validate() == []
    assert "chunk_lift" in ran.COUNTERS and "chunk_lift" not in ran.UNCOUNTED


# ---------------------------------------------------------------- the composition-time ceiling (plan / compose / gated_off / words)
def test_the_ceiling_is_the_tables_unchunked_gate():
    assert chunklift.CEILING == 1024 == chunklift.gate_of(chunklift.PIN_THRESHOLDS) and chunklift.gate_of({"512": 256}) == 0


def test_policy_within_above_mixed_unknown():
    T = lambda *ns: {f"i{k}": {"residue_tokens": n, "ligands_uncounted": 0} for k, n in enumerate(ns)}   # noqa: E731
    assert chunklift.policy(T(600, 1024))["case"] == "within" and chunklift.policy(T(600, 1024))["within"]
    assert chunklift.policy(T(1025))["case"] == "above" and not chunklift.policy(T(1025))["within"]
    p = chunklift.policy(T(200, 1472)); assert (p["case"], p["within"], p["max_tokens"], p["gate"]) == ("above", False, 1472, 1024)   # mixed: composed out (the safe side)
    p = chunklift.policy(None); assert (p["case"], p["within"], p["max_tokens"]) == ("unknown", False, None)


def test_compose_and_the_words(monkeypatch):
    s1, fast = modes.LINES["S1"], modes.LINES["LSTAR2A"]
    monkeypatch.setitem(chunklift._PLAN, "plan", chunklift.policy({"a": {"residue_tokens": 1000}}))
    assert chunklift.compose(s1) is s1 and chunklift.word() is None and chunklift.gated_off(s1) == {}
    assert modes.resolve("exact", "/no/tree", {}).line is s1                                          # within: the static row, the lever bound
    monkeypatch.setitem(chunklift._PLAN, "plan", chunklift.policy({"a": {"residue_tokens": 1472}}))
    out = chunklift.compose(fast)
    assert "chunk_lift" not in out.levers and out.exports == fast.exports and out.path_order == fast.path_order and out.fpf == fast.fpf
    assert chunklift.word() == "above_gate:1472/1024" and chunklift.gated_off(out) == {"chunk_lift": "above_gate:1472/1024"} and chunklift.gated_off(fast) == {}
    res = modes.resolve("fast", "/no/tree", {})
    assert "chunk_lift" not in res.line.levers and res.line.exports == fast.exports
    monkeypatch.setitem(chunklift._PLAN, "plan", None)                                                  # no query read (check, the env route): composed out, named `unknown`
    assert chunklift.word() == "above_gate:unknown/1024" and "chunk_lift" not in modes.resolve("exact", "/no/tree", {}).line.levers
    assert chunklift.gated_off(modes.LINES["BIG_F"]) == {"chunk_lift": "above_gate:unknown/1024"}   # big's resident composition carries it: composed out there too, named


def test_plan_reads_the_query_and_the_report_words(tmp_path, monkeypatch):
    import json as _json
    from opendde_opt import report as _rep
    q = tmp_path / "q.json"
    q.write_text(_json.dumps([{"name": "big", "sequences": [{"proteinChain": {"sequence": "A" * 1472, "count": 1}}]}]))
    class _Res: line = modes.LINES["LSTAR2A"]
    pol = chunklift.plan(_Res(), str(q))
    assert (pol["case"], pol["max_tokens"], pol["within"]) == ("above", 1472, False) and chunklift.planned()["case"] == "above"
    rep = {"levers_gated_off": {"chunk_lift": chunklift.word()}, "levers_planned": ["dit_align"], "active": True}
    assert _rep.ceiling_word(rep) == "above_gate:1472/1024" and _rep.floor_word(rep) is None
    assert _rep.lever_state("chunk_lift", rep) == ("off", "above_gate:1472/1024")
    class _Pre: line = modes.line_without(modes.LINES["LSTAR2A"], ("chunk_lift",))        # cli.pred resolves BEFORE it plans: the line arrives composed out already — the plan still decides
    assert chunklift.plan(_Pre(), str(q))["case"] == "above"
    class _Off: line = None
    assert chunklift.plan(_Off(), str(q)) == {} and chunklift.planned() is None
    q.write_text(_json.dumps([{"name": "small", "sequences": [{"proteinChain": {"sequence": "A" * 500, "count": 2}}]}]))
    assert chunklift.plan(_Pre(), str(q))["case"] == "within" and chunklift.word() is None
    assert modes.resolve("fast", "/no/tree", {}).line is modes.LINES["LSTAR2A"]                     # and the activation's resolution then carries the lever
    assert _rep.ceiling_word({"levers_gated_off": {"chunk_lift": "above_gate:unknown/1024"}}) is None   # no query read: no ceiling token on the line (the LEVER row names it)
    chunklift._reset()

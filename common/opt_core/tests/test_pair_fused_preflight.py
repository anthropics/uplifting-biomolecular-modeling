"""attn.pair_fused resolution under the core's safe-settings mechanism (opt_core.kernels.safe_settings): exact `<cc>|<triton>` row ->
`<cc>|*` row -> a key the table lists as MEASURED OFF (named_off / a candidate row not admitted: `no-cell:…+off(<why>)`, unserved, by name)
-> the lever's SAFE cell for an UNKNOWN key (ONE line, lever engaged) -> the `no-cell` word where no safe settings admit the shape (find_cell
raises; the kit's per-call route / NOT ACTIVE exit before any forward).  CPU only: the stack is named (stack=(cc, triton)); a table that does
not know a capability is a monkeypatched copy of the shipped one (its rows AND its named_off entries for that capability removed)."""
import copy

import pytest

from opt_core.attn import pair_fused as PF
from opt_core.kernels import safe_settings as SS

TRUNK = [("fpf", "prologue", (128, 4, 32)), ("fpf", "epilogue", (128, 4, 32)), ("fpf", "transition", (128, 512))]   # a c_z=128 / 4-head pair stack
H100, A100, NOROWS = ("9.0", "3.3"), ("8.0", "3.6"), ("7.0", "3.3")
LINE = "[opt_core/pair_fused:transition] safe settings served (no_cell:128x512, cc 8.0, triton 3.6)"


@pytest.fixture
def clean():
    for n in PF._NETS.values(): n.reset()
    PF.SERVED_KEYS.clear()
    yield
    for n in PF._NETS.values(): n.reset()
    PF.SERVED_KEYS.clear()


@pytest.fixture
def no_80_rows(monkeypatch):
    """The shipped table minus every cc 8.0 row and named_off entry: a capability the table does not know, with safe settings (SAFE_ROWS carries 8.0)."""
    tab = copy.deepcopy(PF.cells()); tab["rows"] = [r for r in tab["rows"] if r["cc"] != "8.0"]
    tab["named_off"] = [e for e in tab.get("named_off", []) if e.get("cc") != "8.0"]
    monkeypatch.setattr(PF, "cells", lambda path=None: tab)
    return tab


def _rows(cc, piece=None):
    return [r for r in PF.cells()["rows"] if r["cc"] == cc and r["status"] == "certified" and (piece is None or r["piece"] == piece)]



def _no_star(monkeypatch):
    """SAFE_ROWS without the any-capability '*' rows: the shape of a lever that has safe settings on named capabilities only (the case-iii branch)."""
    from opt_core.kernels import safe_settings as _SS
    for lever, rows in list(_SS.SAFE_ROWS.items()):
        if lever.startswith("pair_fused:") and "*" in rows:
            monkeypatch.setitem(_SS.SAFE_ROWS, lever, {k: v for k, v in rows.items() if k != "*"})

def test_pinned_rows_serve_by_key_and_print_nothing(monkeypatch, capfd, clean):
    assert _rows("9.0"), "the shipped table carries cc 9.0 rows"
    monkeypatch.setattr(PF, "_stack", lambda device: H100)
    for impl, piece, key in TRUNK:
        d = PF.lookup_cell(impl, piece, key, stack=H100)
        assert d.served and not d.safe and d.reason == "" and d.cc == "9.0" and d.key == tuple(key)
        assert d.served_by in ("9.0|3.3", "9.0|*") and d.served_by == PF.row_key(d.row) if d.row["triton"] != "*" else d.served_by == "9.0|*"
        assert d.row is PF.find_cell(impl, piece, key, None)                  # the very same row object find_cell serves on this stack
        assert d.word() == "served:" + d.row["variant"]
    assert capfd.readouterr().err == ""                                       # no safety net engaged: nothing printed
    assert PF.evidence_tail() == {"keys": "-"}                                # no cell CALL yet: the tail's empty form; no settings= field


def test_a100_rows_serve_the_trunk_shapes(monkeypatch):
    if not _rows("8.0", "transition"):
        pytest.skip("the table carries no cc 8.0 rows in this tree")
    for impl, piece, key in TRUNK:
        d = PF.lookup_cell(impl, piece, key, stack=A100)
        if (impl, piece, tuple(key)) == ("fpf", "epilogue", (128, 4, 32)):                     # measured slower than the lnl statements on 8.0: named off (the lnl piece serves)
            assert not d.served and d.off and PF.lookup_cell("lnl", "gate_transpose", (128,), stack=A100).served, d
            continue
        assert d.served and not d.safe and d.served_by == "8.0|*"


def test_a_measured_off_shape_keeps_the_refusal_word_qualified(monkeypatch):
    """The miss policy: a key the 9.0 table lists as a candidate only (the 384x768 transition) is MEASURED OFF — the
    refusal word the pinned card's per-call route reads, qualified `+off(not-measured)` — and find_cell raises it; nothing engages."""
    d = PF.lookup_cell("fpf", "transition", (384, 768), stack=H100)
    assert not d.served and d.off and d.served_by == "" and d.reason == "no-cell:fpf:transition:384x768:9.0|3.3+off(not-measured)" and d.word() == "off:not-measured:384x768"
    monkeypatch.setattr(PF, "_stack", lambda device: H100)
    with pytest.raises(PF.Unsupported) as e:
        PF.find_cell("fpf", "transition", (384, 768), None)
    assert e.value.reason == d.reason


def test_a_capability_without_rows_serves_the_safe_cell_with_one_line(monkeypatch, capfd, clean, no_80_rows):
    d = PF.lookup_cell("fpf", "transition", (128, 512), stack=A100)
    assert d.served and d.safe and d.served_by == "safe" and d.reason == "no_cell:128x512"
    assert d.row["cfg"] == SS.safe_row("pair_fused:transition", "8.0")["settings"] == {"BM": 16, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 0}
    assert (d.row["variant"], d.row["status"], d.row["cc"], d.row["triton"]) == ("v1", "safe", "8.0", "*")
    assert d.word("c_z=128,n=4") == "safe:no_cell:c_z=128,n=4" and d.word() == "safe:no_cell:128x512"
    assert PF.lookup_cell("fpf", "transition", (128, 512), stack=A100, variant="v1").row["variant"] == "v1"               # the pinned reference variant: the same cell
    assert PF.lookup_cell("fpf", "transition", (128, 512), stack=A100, variant="v1_fp32x").off                          # the fp32-input variant: measured off everywhere (named_off)
    assert not PF.lookup_cell("fpf", "transition", (128, 512), stack=A100, variant="transition_v3").served               # another kernel: no safe cell
    assert capfd.readouterr().err == ""                                       # a lookup / pre-flight prints nothing
    row = PF.find_cell("fpf", "transition", (128, 512), None, stack=A100)     # the CALL path: the safety net engages, ONE line
    assert row["status"] == "safe" and capfd.readouterr().err.strip() == LINE
    PF.find_cell("fpf", "transition", (128, 512), None, stack=A100)          # once per key
    assert capfd.readouterr().err == ""
    PF.find_cell("fpf", "transition", (256, 1024), None, stack=A100)         # another unknown key: its own engagement (scoped per cell) and its own line
    assert capfd.readouterr().err.strip() == LINE.replace("128x512", "256x1024")
    assert PF.safety_net("pair_fused:transition").word() == "safe:no_cell:128x512"          # the census word: the first engagement
    assert set(PF.safety_net("pair_fused:transition").cells) == {"128x512", "256x1024"} and not PF.safety_net("pair_fused:transition").wide
    PF._count_rows({"impl": "fpf", "transition": row})                       # what transition() records per call
    assert PF.evidence_tail() == {"keys": "fpf.transition:safe", "settings": "safe:no_cell:128x512"}
    # the lnl pieces carry no launch configuration: an UNKNOWN key is served the piece's default settings (lnl_fused resolves its own tiles), named ONCE
    dl = PF.lookup_cell("lnl", "ln_linear", (128,), stack=A100)
    assert dl.served and dl.served_by == "default" and dl.reason == "default:128" and dl.row["status"] == "default" and dl.word() == "default:128"
    assert PF.find_cell("lnl", "ln_linear", (128,), None, stack=A100) is not None
    assert capfd.readouterr().err.strip() == "[opt_core/pair_fused:lnl] default settings (no row for ln_linear 128 on cc 8.0, cc 8.0, triton 3.6); tuned rows exist for cc 9.0"
    assert PF.evidence_tail()["cells_note"] == "pair_fused:lnl:default:no_row"


def test_a_capability_without_rows_serves_the_any_capability_safe_settings():
    ds = PF.preflight(TRUNK, stack=NOROWS)                                   # cc 7.0: no rows; every pair_fused lever has an any-capability SAFE row
    assert [d.served for d in ds] == [True, True, True]


def test_a_capability_without_rows_and_without_safe_settings_is_the_refusal(monkeypatch):
    _no_star(monkeypatch)
    ds = PF.preflight(TRUNK, stack=NOROWS)                                   # cc 7.0: no rows, no safe settings (a lever without a '*' row)
    assert [d.served for d in ds] == [False, False, False]
    assert ds[2].reason == "no-cell:fpf:transition:128x512:7.0|3.3" and ds[2].word() == "refused:" + ds[2].reason
    with pytest.raises(PF.Unsupported) as e:
        PF.find_cell("fpf", "transition", (128, 512), None, stack=NOROWS)
    assert e.value.reason == ds[2].reason
    with pytest.raises(PF.Unsupported):
        PF.find_cell("fpf", "prologue", (128, 4, 32), None, stack=NOROWS)


class _FakeBuildError(RuntimeError):
    pass


def test_run_cell_build_failure_serves_the_safe_cell_then_refuses(capfd, clean):
    seen = []
    def build(cfg):
        seen.append(dict(cfg))
        if cfg.get("num_stages", 1) > 1:
            raise _FakeBuildError("PassManager::run failed")                 # the words of a triton MLIR failure (safe_settings.is_build_failure)
        return "ran:" + str(cfg["BM"])
    tuned = {"BM": 128, "BH": 32, "num_warps": 4, "num_stages": 2, "IL": 0}
    assert PF.run_cell("pair_fused:transition", "8.0", "3.6", build, tuned) == "ran:16"
    assert seen == [tuned, SS.safe_row("pair_fused:transition", "8.0")["settings"]]
    assert capfd.readouterr().err.strip() == "[opt_core/pair_fused:transition] safe settings served (build_failed:_FakeBuildError, cc 8.0, triton 3.6)"
    assert PF.evidence_tail()["settings"] == "safe:build_failed:_FakeBuildError"
    assert PF.run_cell("pair_fused:transition", "8.0", "3.6", build, tuned) == "ran:16" and capfd.readouterr().err == ""   # the safe cell serves the rest of the process, silently
    def never(cfg):
        raise _FakeBuildError("ptxas fatal")
    with pytest.raises(PF.Refused) as e:                                     # the safe cell cannot build either: the fail-closed case, naming the lever and --mode off
        PF.run_cell("pair_fused:transition", "8.0", "3.6", never, tuned)
    assert "pair_fused:transition" in str(e.value) and "--mode off" in str(e.value)


def test_run_cell_lets_everything_else_propagate(clean, monkeypatch):
    _no_star(monkeypatch)
    def oom(cfg):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    with pytest.raises(RuntimeError, match="out of memory"):
        PF.run_cell("pair_fused:transition", "8.0", "3.6", oom, {"BM": 128})
    assert not PF.safety_net("pair_fused:transition").on
    calls = []
    def build(cfg):
        calls.append(cfg); raise _FakeBuildError("PassManager::run failed")
    with pytest.raises(_FakeBuildError):                                    # a capability without safe settings: the tuned cell is built as given, its failure is the caller's
        PF.run_cell("pair_fused:transition", "7.0", "3.3", build, {"BM": 128})
    assert len(calls) == 1


def test_row_key_and_keys_tail(clean):
    r_exact = {"impl": "fpf", "piece": "prologue", "cc": "9.0", "triton": "3.7", "status": "certified", "variant": "prologue_v4"}
    r_def = {"impl": "fpf", "piece": "transition", "cc": "9.0", "triton": "*", "status": "certified", "variant": "v1"}
    assert (PF.row_key(r_exact), PF.row_key(r_def), PF.row_key({"status": "safe"})) == ("9.0|3.7", "9.0|*", "safe")
    PF._count_rows({"impl": "fpf", "prologue": r_exact, "epilogue": dict(r_exact, piece="epilogue"), "transition": r_def})
    assert PF.evidence_tail() == {"keys": "fpf.epilogue:9.0|3.7,fpf.prologue:9.0|3.7,fpf.transition:9.0|*"}
    ev = PF.evidence()
    assert list(ev)[-1] == "keys" and ev["served_certified"] >= 3          # the tail is the END of the evidence fields (no settings= field without a safety net)


def test_trimul_v4_cells_have_safe_rows_for_the_pinned_capabilities():
    pytest.importorskip("torch"); pytest.importorskip("triton")            # cells.py imports the kernels module (triton) at import
    from opt_core.kernels.fpf_trimul_v4 import cells as C
    assert C.SAFE_LEVER == "pair_fused:trimul"
    for cc in ("8.0", "9.0"):
        s = SS.safe_row(C.SAFE_LEVER, cc)["settings"]
        assert set(s) == {"k1", "k3"} and s["k3"] == {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}
    assert SS.safe_row(C.SAFE_LEVER, "7.0") is SS.SAFE_ROWS[C.SAFE_LEVER]["*"]   # every other capability: the any-capability SAFE cell (engaged and named by cell_for)


def test_names_exported():
    for n in ("lookup_cell", "preflight", "CellDecision", "Refused", "run_cell", "safety_net", "evidence_tail", "row_key"):
        assert n in PF.__all__


def test_every_pinned_9_0_row_resolves_to_itself_and_prints_nothing(capfd, clean):
    """Point 5: on the pinned capability every certified row's own key is served by a table row (never the safe cell, never a line) — the
    shapes the pinned cards run all have rows, so nothing of the miss policy is reachable there."""
    rows = [r for r in PF.cells()["rows"] if r["cc"] == "9.0" and r["status"] == "certified"]
    assert rows
    for r in rows:
        tt = r["triton"] if r["triton"] != "*" else "3.7"
        d = PF.lookup_cell(r["impl"], r["piece"], tuple(r["key"]), stack=("9.0", tt), variant=r.get("variant"))
        assert d.served and not d.safe and d.served_by in (f"9.0|{tt}", "9.0|*"), (r["id"] if "id" in r else r)
        assert PF.find_cell(r["impl"], r["piece"], tuple(r["key"]), None, stack=("9.0", tt), variant=r.get("variant"))["status"] == "certified"
    assert capfd.readouterr().err == "" and PF.evidence_tail().get("settings") is None

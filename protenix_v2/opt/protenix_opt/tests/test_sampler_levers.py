"""sampler_levers: cond_dedupe / dit_fused / dit_lowp / atom_fused (protenix_fpf_ditfast) + atom_attn_exact (protenix_fpf_atom_attn_exact) on the
runner seam after the attention levers — switch grammar (dit_lowp is a word), requires chains, registry rows / rows / markers, the installer's
kit lines and failure naming, state / evidence off the packages' reports, the carried files and their env reads."""
import io
import json
import os
import re
import sys
import types
from contextlib import redirect_stderr

import pytest

from protenix_opt import kits, modes, registry, report, runner_seam, sampler_levers as sl, stack, tp
from protenix_opt.registry import LEVERS, in_row


def test_tables():
    assert sl.LEVERS == ("cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact") and sl.KERNEL_LEVERS == ("cond_dedupe", "dit_fused", "atom_fused", "atom_attn_exact")
    assert sl.PRECISION == {"dit_lowp": "dit_fused"} and sl.LOWP_WORDS == ("fp16", "bf16") and sl.LOWP_OFF == "off"
    assert sl.ENVS == {"cond_dedupe": "PTX_COND_DEDUPE", "dit_fused": "PTX_DIT_FAST", "dit_lowp": "PTX_DIT_LOWP", "atom_fused": "PTX_ATOM_FAST", "atom_attn_exact": "PTX_ATOM_ATTN_EXACT"}
    for lever in sl.LEVERS:
        lv = LEVERS[lever]
        assert lv.env_keys == (sl.ENVS[lever],) and lv.probe == "marker" and lv.extra and lv.row_dependent, lever
        assert stack.MARKERS[lever] == (sl.MARKS[lever], (sl.MARKS[lever],)) and report.STRATEGY_IDS[lever] == sl.STRATEGIES[lever] and lever in report.IMPL, lever
    for lever in ("cond_dedupe", "dit_fused", "dit_lowp", "atom_fused"):
        assert LEVERS[lever].tier == registry.TOLERANCE and lever in modes.MODES["fast"] and lever in modes.MODES["big"] and lever not in modes.MODES["exact"] and lever in tp.TP_DROPPED
    assert not {"dit_fused", "dit_lowp"} & set(modes.BIG_DROPPED) and not {"PTX_DIT_FAST", "PTX_DIT_LOWP"} & set(modes.BIG_POST_DROPPED)   # protenix_opt 0.3.45: the fused DiT token stack rides big (+0.45 GiB allocated at 400-1,200 tokens, the row's peak below the stock path's at every size); cond_dedupe / atom_fused too
    assert LEVERS["atom_attn_exact"].tier == registry.EXACT and "atom_attn_exact" in modes.MODES["exact"] and LEVERS["atom_attn_exact"].replaced_by == ("atom_attn", "atom_fused")
    assert LEVERS["dit_fused"].requires == ("dit_attn",) and LEVERS["dit_lowp"].requires == ("dit_fused",) and LEVERS["atom_fused"].requires == ("atom_attn",)
    assert ("dit_attn", "calls", "0") in LEVERS["dit_fused"].subsumes and ("atom_attn", "calls", "0") in LEVERS["atom_fused"].subsumes
    assert tp.TP_PRE["PTX_DIT_FAST"] == "0" and tp.TP_PRE["PTX_DIT_LOWP"] == "off" and tp.TP_PRE["PTX_COND_DEDUPE"] == "0" and tp.TP_PRE["PTX_ATOM_FAST"] == "0"


def test_rows():
    for key, row in modes.README_ROWS.items():
        nine = key.startswith("9.0|")
        for lever, word in (("cond_dedupe", "1"), ("dit_fused", "1"), ("dit_lowp", "fp16"), ("atom_fused", "1")):
            assert row["fast"]["post"].get(sl.ENVS[lever]) == (word if nine else None), (key, lever); assert in_row(LEVERS[lever], row["fast"]) == nine
            assert sl.ENVS[lever] not in row["exact"]["post"], (key, lever)
        assert row["exact"]["post"].get("PTX_ATOM_ATTN_EXACT") == ("1" if nine else None), key; assert "PTX_ATOM_ATTN_EXACT" not in row["fast"]["post"]
    assert not any(sl.ENVS[l] in modes.OTHER_ROW["fast"]["post"] for l in sl.LEVERS)
    src = open(os.path.join(kits.kit_dir("flashpairformer"), "env.sh"), encoding="utf-8").read()
    assert not any(sl.ENVS[l] in src for l in sl.LEVERS), "the switches are row exports (cc 9.0), never env.sh defaults"


def test_switch_grammar():
    assert sl.from_env("dit_fused", {"PTX_DIT_FAST": "1"}) and not sl.from_env("dit_fused", {"PTX_DIT_FAST": "0"}) and not sl.from_env("dit_fused", {})
    with pytest.raises(ValueError, match=re.escape("PTX_DIT_FAST='2': expected 0 or 1")):
        sl.from_env("dit_fused", {"PTX_DIT_FAST": "2"})
    assert sl.from_env("dit_lowp", {"PTX_DIT_LOWP": "fp16"}) and sl.from_env("dit_lowp", {"PTX_DIT_LOWP": "bf16"})
    assert not sl.from_env("dit_lowp", {"PTX_DIT_LOWP": "off"}) and not sl.from_env("dit_lowp", {}) and not sl.from_env("dit_lowp", {"PTX_DIT_LOWP": ""})
    with pytest.raises(ValueError, match=re.escape("PTX_DIT_LOWP='fp8': expected off, fp16 or bf16")):
        sl.from_env("dit_lowp", {"PTX_DIT_LOWP": "fp8"})
    with pytest.raises(RuntimeError, match=r"dit_lowp: requires dit_fused"):
        sl.check_requires(["dit_lowp"])
    sl.check_requires(["dit_fused", "dit_lowp"]); sl.check_requires(["cond_dedupe"])


def test_installed_sites():
    assert sl.installed_sites("cond_dedupe", {"installed": True}) == 1 and sl.installed_sites("cond_dedupe", {"installed": False}) == 0 and sl.installed_sites("cond_dedupe", {}) == 0
    assert sl.installed_sites("dit_fused", {"installed": True, "blocks": 24}) == 24
    assert sl.installed_sites("atom_fused", {"installed": True, "stacks": {"enc": {"blocks": 3, "strip": 21}, "dec": {"blocks": 3, "strip": 21}}}) == 6
    assert sl.installed_sites("atom_attn_exact", {"installed": True, "loadcheck": ["a", "b", "c", "d"]}) == 1


@pytest.fixture()
def fake_pkgs(tmp_path, monkeypatch):
    """protenix_fpf_ditfast / protenix_fpf_atom_attn_exact stand-ins with the packages' faces (install fns returning their REPORT dicts, report())."""
    d = tmp_path / "protenix_fpf_ditfast"; d.mkdir()
    (d / "__init__.py").write_text(
        "__version__ = '0.9.0'\nCALLS = []\nEXITS = {}\n"
        "class LeverRefused(RuntimeError): pass\n"
        "def report(): return dict(EXITS)\n"
        "def install_cond_dedupe(model):\n    CALLS.append(('cond_dedupe', model)); EXITS['cond_dedupe'] = {'dedupe_calls': 3, 'stock_path_calls': 0, 'rows_in': 15, 'rows_computed': 3}; return {'installed': True, 'census': EXITS['cond_dedupe']}\n"
        "def install_dit_fast(model):\n    import os\n"
        "    if os.environ.get('FAKE_DITFAST_FAIL'): raise LeverRefused('dit_fused: requires lever dit_attn (PTX_DIT_ATTN=1)')\n"
        "    lowp = os.environ.get('PTX_DIT_LOWP', 'off'); CALLS.append(('dit_fused', model)); EXITS['dit_fused'] = {'stack_calls': 2, 'blocks_x_calls': 48, 'act': 'torch.float16', 'lowp': lowp, 'bias_slots': 'hoist'}\n"
        "    return {'installed': True, 'blocks': 24, 'act': 'torch.float16', 'lowp': lowp, 'attention': 'protenix_fpf_apb.dit_apb', 'bias_slots': 'hoist'}\n"
        "def install_atom_fast(model):\n    CALLS.append(('atom_fused', model)); EXITS['atom_fused'] = {'stack_calls': {'enc': 2, 'dec': 2}, 'blocks_x_calls': 12, 'cond_slots': 'hoist'}\n"
        "    return {'installed': True, 'act': 'torch.float32', 'stacks': {'enc': {'blocks': 3, 'strip': 21}, 'dec': {'blocks': 3, 'strip': 21}}, 'cond_slots': 'hoist'}\n")
    import types as _types
    X = _types.ModuleType("protenix_opt.apb_atom_exact"); X.__file__ = str(tmp_path / "apb_atom_exact_stub.py")   # the shared core's copy (protenix_opt 0.3.51), stood in by name
    exec(compile(
        "__version__ = '1.0.0'\nCALLS = []\nREP = {'installed': None, 'routes': None, 'calls': {'kernel': 0, 'original': 0}, 'original_path_reasons': {}}\n"
        "class LeverRefused(RuntimeError): pass\n"
        "def report(): return dict(REP)\n"
        "def install(model=None):\n    CALLS.append(('atom_attn_exact', model)); REP.update(installed=True, routes='qk:tf32[4..256] pv:tf32[4..256]', calls={'kernel': 24, 'original': 6}, original_path_reasons={'dtype': 6})\n"
        "    return {'installed': True, 'routes': REP['routes'], 'unreproducible_batches': [], 'loadcheck': ['c1', 'c2', 'c3', 'c4'], 'cc': 'sm_90', 'tf32': True, 'seconds': 0.4}\n", X.__file__, "exec"), X.__dict__)
    monkeypatch.setitem(sys.modules, "protenix_opt.apb_atom_exact", X)
    r = tmp_path / "runner"; r.mkdir(); (r / "__init__.py").write_text("")
    (r / "inference.py").write_text("class InferenceRunner:\n    def __init__(self, configs):\n        self.configs = configs; self.model = object()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for m in ("protenix_fpf_ditfast", "runner", "runner.inference"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setattr(sl, "_STATE", {"on": [], "patch": None, "installed": {}, "errors": {}, "reports": {}, "lowp": None, "models": 0})
    from opt_core import autoload as _autoload
    monkeypatch.setattr(_autoload, "_PATCHES", {})
    monkeypatch.setattr(runner_seam, "_INSTALLERS", []); monkeypatch.setattr(runner_seam, "_STATE", {"patch": None, "runners": 0, "ran": []})
    yield tmp_path
    for m in ("protenix_fpf_ditfast", "protenix_opt.apb_atom_exact", "runner", "runner.inference"):
        sys.modules.pop(m, None)
    p = runner_seam._STATE.get("patch")
    if p is not None:
        p._unhook()


def test_install_order_lines_state_evidence(fake_pkgs, monkeypatch):
    monkeypatch.setenv("PTX_DIT_LOWP", "fp16")
    marks = sl.install(["cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact"])
    assert marks == ["CONDDEDUPE:armed", "DITFAST:armed", "DITLOWP:requested", "ATOMFAST:armed", "ATOMATTNEXACT:armed"]
    assert runner_seam.names() == ["cond_dedupe", "dit_fused", "atom_fused", "atom_attn_exact"]
    err = io.StringIO()
    with redirect_stderr(err):
        import runner.inference as RI
        rr = RI.InferenceRunner({"n": 1})
    import protenix_fpf_ditfast as D; X = sys.modules["protenix_opt.apb_atom_exact"]
    assert D.CALLS == [("cond_dedupe", rr.model), ("dit_fused", rr.model), ("atom_fused", rr.model)] and X.CALLS == [("atom_attn_exact", rr.model)]
    e = err.getvalue()
    for line in ("[protenix-opt] CONDDEDUPE:installed(1 sites; fpf_ditfast 0.9.0)", "[protenix-opt] DITFAST:installed(24 sites; fpf_ditfast 0.9.0)", "[protenix-opt] DITLOWP:on(word=fp16)",
                 "[protenix-opt] ATOMFAST:installed(6 sites; fpf_ditfast 0.9.0)", "[protenix-opt] ATOMATTNEXACT:installed(1 sites; kernels.apb.atom_exact 1.0.0)"):
        assert line in e, (line, e)
    st = sl.state()
    assert st["patched"] and st["models"] == 1 and st["dit_fused"]["installed_on"] == 24 and st["atom_fused"]["installed_on"] == 6 and st["dit_lowp"] == {"on": True, "engaged": True, "word": "fp16", "kernel": "dit_fused"}
    assert dict(sl.evidence("cond_dedupe")) == {"models": 1, "sites": 1, "dedupe_calls": 3, "stock_path_calls": 0, "rows_in": 15, "rows_computed": 3}
    ev = dict(sl.evidence("dit_fused")); assert (ev["sites"], ev["stack_calls"], ev["blocks_x_calls"], ev["lowp"], ev["bias_slots"], ev["act"]) == (24, 2, 48, "fp16", "hoist", "torch.float16")
    ev = dict(sl.evidence("atom_fused")); assert (ev["sites"], ev["stack_calls"], ev["blocks_x_calls"], ev["cond_slots"]) == (6, "dec:2,enc:2", 12, "hoist")
    ev = dict(sl.evidence("atom_attn_exact")); assert (ev["sites"], ev["kernel_calls"], ev["original_calls"], ev["original_reasons"], ev["loadcheck"], ev["routes"]) == (1, 24, 6, "dtype:6", "4", "qk:tf324..256pv:tf324..256")
    assert dict(sl.evidence("dit_lowp")) == {"word": "fp16", "engaged": 1, "kernel": "dit_fused"}
    rep = {"active": True, "mode": "fast", "levers_applied": list(modes.MODES["fast"]), "levers_fallback": [], "reconciled": {"records": [], "moves": {}}}
    lines = {l.rsplit("lever=", 1)[1]: l for l in report.lever_lines(rep)}
    assert " sites=24 " in lines["dit_fused"] and " lowp=fp16 " in lines["dit_fused"] and " word=fp16 engaged=1 " in lines["dit_lowp"] and " dedupe_calls=3 " in lines["cond_dedupe"]
    for l in lines.values():
        assert "\n" not in l and "  " not in l


def test_dit_lowp_off_word_when_not_requested(fake_pkgs, monkeypatch):
    monkeypatch.delenv("PTX_DIT_LOWP", raising=False)
    sl.install(["dit_fused"])
    err = io.StringIO()
    with redirect_stderr(err):
        import runner.inference as RI
        RI.InferenceRunner({})
    assert "[protenix-opt] DITLOWP:off" in err.getvalue() and sl.state()["dit_lowp"]["engaged"] is False


def test_a_refusing_install_is_named_then_raised(fake_pkgs, monkeypatch):
    monkeypatch.setenv("FAKE_DITFAST_FAIL", "1")
    sl.install(["cond_dedupe", "dit_fused"])
    err = io.StringIO()
    with redirect_stderr(err), pytest.raises(RuntimeError, match="dit_fused: requires lever dit_attn"):
        import runner.inference as RI
        RI.InferenceRunner({})
    assert "[protenix-opt] DITFAST:unavailable(LeverRefused('dit_fused: requires lever dit_attn (PTX_DIT_ATTN=1)'))" in err.getvalue()
    assert sl.state()["dit_fused"]["installed_on"] == 0 and "LeverRefused" in sl.state()["dit_fused"]["error"]
    on, fb, sk, why = stack._classify("fast", ["DITFAST:unavailable(LeverRefused('x'))", "CONDDEDUPE:patched"], {}, row=modes.README_ROWS["9.0|3.7"]["fast"], row_key="9.0|3.7", sm="sm90")
    assert "dit_fused" in fb and "cond_dedupe" in on


def test_reconcile_names_a_runner_never_built(fake_pkgs):
    sl.install(["cond_dedupe", "atom_attn_exact"])
    rep = {"active": True, "mode": "exact", "levers_applied": ["cond_dedupe", "atom_attn_exact", "deadskip"], "levers_fallback": []}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner_seam, "patched", lambda: False)
        r = stack.reconcile(dict(rep), {})
    assert {"cond_dedupe", "atom_attn_exact"} <= set(r["levers_fallback"]) and "never imported" in r["fallback_reasons"]["atom_attn_exact"]


def test_carried_files_and_env_reads():
    tp3 = os.path.join(kits.kit_dir("flashpairformer"), "third_party")
    d, x = os.path.join(tp3, "protenix_fpf_ditfast"), os.path.join(tp3, "protenix_fpf_atom_attn_exact")
    assert sorted(os.listdir(d)) == ["CELLS.json", "NOTICE", "__init__.py", "_plumbing.py", "atom_fast.py", "cond_dedupe.py", "dit_fast.py", "vectors.json", "vectors.py"]
    assert not os.path.exists(x), "protenix_opt 0.3.51: the kit copy is deleted; the shared core carries the kernel as kernels.apb row atom_exact"
    rx = re.compile(r"""(?:environ(?:\.get)?\s*[\[(]\s*|getenv\(\s*)["']([A-Z][A-Z0-9_]+)["']""")
    reads = {}
    for root in (d,):
        for f in sorted(os.listdir(root)):
            if f.endswith(".py"):
                reads[os.path.basename(root) + "/" + f] = sorted(set(rx.findall(open(os.path.join(root, f), encoding="utf-8").read())))
    allowed = {"PTX_LEVER_REPORT", "PTX_COND_DEDUPE", "PTX_DIT_FAST", "PTX_DIT_LOWP", "PTX_ATOM_FAST", "PTX_ATOM_ATTN_EXACT", "PTX_DIT_ATTN", "PTX_DIT_ATTN_FP16", "PTX_ATOM_ATTN"}
    for f, names in reads.items():
        assert set(names) <= allowed, (f, names)                      # lever switch words (own + the required levers') and the lever-report path only
    assert "0.9.0" in open(os.path.join(d, "__init__.py")).read()
    assert sl.PACKAGES["atomx"][0] == "protenix_opt.apb_atom_exact"      # the exact atom attention binds kernels.apb by the word exact (row atom_exact)

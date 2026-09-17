"""``--upstream-fix ID[,ID…]`` (upstream_fix.py + upstream_issues/): the grammar, the tree's one switchable issue (EF2INV-0003), the refusal of
an unknown ID before anything launches, the arm argv (a token only when given — the flag absent leaves every arm's argv byte-identical), and
the mechanism: EF2INV-0003's ``apply()`` IS ``patches.pin_esmc_rope()`` — the same record and the same module state, on every mode,
``off`` included — while a run without the flag loads nothing from upstream_issues/ and leaves ESM-C's RoPE as imported. EF2INV-0002 (the
dtype cast every mode needs to run) stays unconditional: ``patches.apply`` / its function text are untouched by the flag."""
import ast
import glob
import os
import types

import pytest

from .. import cli, launch as L, modes as MD, patches as PT, stock_design as SD, upstream_fix as UF, attention as AT
from . import _paths as P
from .test_attention_words import fake_modules

ROPE = "EF2INV-0003"


def test_flag_spelling_is_one():
    assert UF.FLAG == cli.UPSTREAM_FIX_FLAG == "--upstream-fix"


def test_parse_grammar():
    assert UF.parse(None) == [] and UF.parse("") == [] and UF.parse(" , ") == []
    assert UF.parse(ROPE) == [ROPE] and UF.parse(f"{ROPE},{ROPE}") == [ROPE] and UF.parse(f" {ROPE} , X-1 ") == [ROPE, "X-1"]
    assert UF.parse("all") == ["all"] and UF.parse("none") == ["none"]                            # no `all` / `none` keywords: each is an (unknown) ID like any other word
    with pytest.raises(UF.UnknownUpstreamFix, match="unknown upstream fix 'all'"):
        UF.resolve(["all"], P.ROOT)
    with pytest.raises(UF.UnknownUpstreamFix, match="unknown upstream fix 'none'"):
        UF.resolve(["none"], P.ROOT)


def test_the_tree_carries_exactly_one_switchable_issue_and_names_it_by_file():
    avail = UF.available(P.ROOT)
    assert sorted(avail) == [ROPE] and os.path.basename(avail[ROPE]) == "EF2INV-0003_esmc_rope_autograd.py"
    md = sorted(os.path.basename(p) for p in glob.glob(os.path.join(P.ROOT, "upstream_issues", "*")) if not p.endswith("__pycache__"))
    assert md == ["EF2INV-0003_esmc_rope_autograd.py"]                                          # the directory holds the opt-in fixes only; the stock exception (bug 0002) is STOCK.md's, applied by patches.py
    mod = UF.load(ROPE, avail[ROPE])
    assert mod.ID == ROPE and callable(mod.apply) and mod.TARGETS == ["transformers.models.esmc.modeling_esmc._flash_attn_rotary_available"]
    for section in ("WHAT UPSTREAM DOES", "EVIDENCE", "WHAT THE FIX CHANGES", "EXPECTED EFFECT ON OUTPUTS"):
        assert section in mod.__doc__, section


def test_unknown_id_is_refused_by_name_and_nothing_is_loaded():
    with pytest.raises(UF.UnknownUpstreamFix, match=r"unknown upstream fix 'EF2INV-9999' \(--upstream-fix\); known: EF2INV-0003"):
        UF.resolve([ROPE, "EF2INV-9999"], P.ROOT)
    assert UF.resolve([], P.ROOT) == [] and UF.apply([], P.ROOT) == []


def test_apply_is_the_package_rope_pin_record_for_record():
    """EF2INV-0003 applied == patches.pin_esmc_rope(): the same record keys and values and the same module state, whether the module
    bound flash-attn's rotary kernel (as on the pinned stack) or not — the mechanism the unconditional pin used, now reached through the flag."""
    mod = UF.load(ROPE, UF.available(P.ROOT)[ROPE])
    for rotary in (True, False):
        E1 = fake_modules(True, rotary=rotary)[AT.ESMC_MODULE]; E2 = fake_modules(True, rotary=rotary)[AT.ESMC_MODULE]
        want = PT.pin_esmc_rope(module=E1)
        lines = []
        got = mod.apply(log=lines.append, module=E2)
        assert {k: got[k] for k in want} == want and got["summary"] == mod.SUMMARY and got["targets"] == mod.TARGETS
        assert want == {"name": "esmc_rope_pin", "impl": "torch", "applied": rotary, "as_imported": rotary, "source": PT.ESMC_ROPE_SOURCE,
                        "target": "transformers.models.esmc.modeling_esmc._flash_attn_rotary_available", "effective": "torch"}
        assert E1._flash_attn_rotary_available is False and E2._flash_attn_rotary_available is False


def test_upstream_fix_0002_is_untouched_by_the_flag():
    """EF2INV-0002 stays unconditional: its function text (the run.json / manifest sha) is the constant every mode has always recorded."""
    assert PT.text_sha256() == "067e1b1b5dd7cc6dba57d1abe6d44043d2332b17f6d40e47fb09a5bd2316a89e"
    assert "0002" not in "".join(UF.available(P.ROOT))


def test_argv_token_only_when_given_every_mode():
    kw = dict(cookbook_stock="/x/binder_design.py", target_name="t", target_sequence="ACDEFGHIK", binder_len=80, seed=3, out="/o", python="py")
    for name, mode in MD.MODES.items():
        base = L.argv_for(mode, P.ROOT, **kw)
        assert "--upstream-fix" not in base, name                                                   # absent: the arm argv is what it was, byte for byte
        assert L.argv_for(mode, P.ROOT, upstream_fix=[], **kw) == base
        withfix = L.argv_for(mode, P.ROOT, upstream_fix=[ROPE, "X-2"], **kw)
        assert withfix[withfix.index("--upstream-fix") + 1] == f"{ROPE},X-2" and [t for t in withfix if t not in ("--upstream-fix", f"{ROPE},X-2")] == base


def test_the_arm_parser_takes_the_flag_and_defaults_to_none(monkeypatch):
    seen = {}
    def stop(mode, kit_dir):                                                                     # prepare_arm_process is the arm's first act after parsing: stop there and read the namespace
        raise RuntimeError("stop-here")
    monkeypatch.setattr(SD, "prepare_arm_process", stop)
    real_parse = SD.argparse.ArgumentParser.parse_args
    def spy(self, argv=None):
        ns = real_parse(self, argv); seen["ns"] = ns; return ns
    monkeypatch.setattr(SD.argparse.ArgumentParser, "parse_args", spy)
    base = ["--mode", "off", "--cookbook", "x", "--pins", os.path.join(P.ROOT, "stock", "PINS.json"), "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", str(P.ROOT) + "/.never"]
    monkeypatch.setattr(SD.os, "makedirs", lambda *a, **k: None)
    assert SD.main(base) == 2 and seen["ns"].upstream_fix is None
    assert SD.main(base + ["--upstream-fix", ROPE]) == 2 and seen["ns"].upstream_fix == ROPE


def test_design_and_warm_carry_the_flag_check_does_not():
    import argparse
    seen = {}
    real = argparse.ArgumentParser.add_argument
    def spy(self, *names, **kw):
        if "--upstream-fix" in names:
            seen.setdefault(self.prog.split()[-1], True)
        return real(self, *names, **kw)
    argparse.ArgumentParser.add_argument = spy
    try:
        with pytest.raises(SystemExit):
            cli.main(["check", "--upstream-fix", ROPE])                                              # check has no such flag: argparse exits 2
    finally:
        argparse.ArgumentParser.add_argument = real
    assert seen == {"design": True, "warm": True}


def test_design_refuses_an_unknown_id_before_any_launch(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MODEL_OPT", cli._model_opt())
    monkeypatch.setattr(cli, "launch_context", lambda a, det: pytest.fail("an unknown --upstream-fix reached the launch context"))
    rc = cli.main(["design", "--mode", "off", "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", str(tmp_path / "o"), "--upstream-fix", "EF2INV-9999"])
    assert rc == 2 and "unknown upstream fix 'EF2INV-9999'" in capsys.readouterr().err


def test_design_forwards_the_resolved_ids_to_the_arm(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_OPT", cli._model_opt())
    facts = {"problems": [], "esm": "3.4.0@d0207ea3", "torch": "x", "device": "NVIDIA H100 80GB HBM3", "kit_switch_var": "EF2_FAST_KIT", "hardware": {"ok": True}, "weights": {}}
    monkeypatch.setattr(cli, "_check_facts", lambda *a, **k: dict(facts))
    monkeypatch.setattr(cli, "_cookbook_stock", lambda: __file__)
    calls = {}
    monkeypatch.setattr(cli.L, "run", lambda argv, env, log, timeout=None: calls.setdefault("argv", argv) and {"exit_code": 0, "wall_s": 1.0})
    monkeypatch.setattr(cli.M, "write", lambda *a, **k: {"missing_outputs": [], "refusals": [], "evidence": "stock"})
    for mode in ("off", "exact", "fast", "big"):
        calls.clear()
        rc = cli.main(["design", "--mode", mode, "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", str(tmp_path / f"o_{mode}"), "--upstream-fix", f"{ROPE},{ROPE}"])
        assert rc == 0 and calls["argv"][calls["argv"].index("--upstream-fix") + 1] == ROPE, mode
        calls.clear()
        rc = cli.main(["design", "--mode", mode, "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", str(tmp_path / f"p_{mode}")])
        assert rc == 0 and "--upstream-fix" not in calls["argv"], mode


def test_manifest_records_the_ids_the_arm_applied(tmp_path):
    import json
    from .. import manifest as M
    d = tmp_path / "run"; d.mkdir()
    rope = PT.pin_esmc_rope(module=types.SimpleNamespace(_flash_attn_rotary_available=True)); rope.update(id=ROPE, file="upstream_issues/EF2INV-0003_esmc_rope_autograd.py")
    (d / "run.json").write_text(json.dumps({"exit_code": 0, "patches": [{"name": "upstream_bug_0002_transition_addmm_dtype", "applied": True, "function_sha256": "ab", "source": "s"}, rope],
                                            "upstream_fix": [rope], "settings": {"deviations": []}}))
    man = M.write(str(d), MD.MODES["off"], P.ROOT, P.PINS, P.STOCK_FILE, {"exit_code": 0}, None)
    assert man["upstream_fix"] == [ROPE] and any(x.startswith("patch esmc_rope_pin: applied=True impl=torch (") for x in man["deviations"])
    (d / "run.json").write_text(json.dumps({"exit_code": 0, "patches": [], "settings": {"deviations": []}}))
    assert M.write(str(d), MD.MODES["off"], P.ROOT, P.PINS, P.STOCK_FILE, {"exit_code": 0}, None)["upstream_fix"] == []


def test_nothing_but_the_flag_route_names_upstream_issues():
    """Static: inside opt/ only upstream_fix.py (the loader), cli.py / stock_design.py (the flag) and patches.py (EF2INV-0003's mechanism, its
    docstring) name the directory; no lever module of the carried design kit and no mode does — the flag absent, nothing there is reachable."""
    hits = []
    for path in glob.glob(os.path.join(P.OPT, "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, P.OPT)
        if "/tests/" in "/" + rel or rel.startswith("ef2inv_opt/tests"):
            continue
        if "upstream_issues" in open(path, encoding="utf-8").read():
            hits.append(rel)
    assert sorted(hits) == ["ef2inv_opt/cli.py", "ef2inv_opt/patches.py", "ef2inv_opt/stock_design.py", "ef2inv_opt/upstream_fix.py"], hits
    levers = [h for h in glob.glob(os.path.join(P.FWD, "**", "*.py"), recursive=True) if not os.path.basename(h).startswith(("test_", "_ef2_test"))]   # the kit's GPU tests may apply a fix to build the arm's configuration; no LEVER module knows of it
    assert levers and not [h for h in levers if "upstream_issues" in open(h, encoding="utf-8").read() or "EF2INV-0003" in open(h, encoding="utf-8").read()]   # the carried design kit knows nothing of it
    tree = ast.parse(open(os.path.join(P.OPT, "ef2inv_opt", "modes.py")).read())
    assert "EF2INV-0003" not in ast.unparse(tree)                                                # no mode composition names the fix: a mode is a lever set, the fix rides its own flag

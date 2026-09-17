"""The ablation switch ``MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]`` (modes.ENV_LEVERS_OFF): ONE run of a mode without the named levers of its
resolved selection — kit levers, package levers, ``dtk``, big's memory levers. Per mode: ``stepgraph`` named → removed from the selection
and named (``levers_off`` on the report, ``levers_off=`` on the ACTIVE line, ``reason=levers_off`` on its LEVER line, the model process's
``--levers`` without it); a name that is no lever of the kit, a lever outside the mode's selection and any name under ``off`` are refused BY
NAME (usage, rc 2); unset or blank changes nothing anywhere (the ACTIVE line carries no ``levers_off=`` token); the variable reaches the model
process's environment untouched (stack.model_process_env strips ``AF3_TORCH_OPT*`` only)."""
import os

import pytest

from af3_torch_opt import big, cli, modes, registry, report, stack

ENV = modes.ENV_LEVERS_OFF


def test_the_switch_is_the_lines_word():
    assert ENV == "MODEL_OPT_LEVERS_OFF" and registry.LEVERS_OFF == "levers_off"
    assert modes.levers_off({}) == () and modes.levers_off({ENV: ""}) == () and modes.levers_off({ENV: " , "}) == ()
    assert modes.levers_off({ENV: " stepgraph , dtk,stepgraph"}) == ("stepgraph", "dtk")             # order kept, blanks stripped, duplicates folded
    names = modes.kit_lever_names()
    assert set(names) == set(l for v in modes.kit_lever_sets().values() for l in v) | set(registry.LEVERS) | set(big.LEVER_ORDER)
    assert {"stepgraph", "hoist", "dtk", "template_dedupe", "canonical_noise", "graph_drop", "expandable_segments"} <= set(names)


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_stepgraph_off_removes_it_and_names_it(mode, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    base = modes.resolve(mode)
    assert "stepgraph" in base["levers"] and base["levers_off"] == ()
    monkeypatch.setenv(ENV, "stepgraph")
    r = modes.resolve(mode)
    assert "stepgraph" not in r["levers"] and r["levers_off"] == ("stepgraph",)
    assert r["levers"] == tuple(l for l in base["levers"] if l != "stepgraph")                         # nothing else moves: same order, same package levers, same DTK, same padding
    assert (r["package_levers"], r["dtk"], r["padding"], r["fastnn"], r["lever_set"]) == (base["package_levers"], base["dtk"], base["padding"], base["fastnn"], base["lever_set"])
    assert modes.resolve(mode, environ={ENV: "stepgraph"}) == r                                        # an explicit environ reads the same


def test_big_drops_stepgraph(monkeypatch):
    """big builds NO step graph (graph_drop, gate 0: fast's `stepgraph` -> its `hoist`): `hoist` named -> switched off like any built lever;
    `stepgraph` named -> refused by name with the word to use (it is not built here); a memory lever named -> the selection loses it (graph_drop
    off = fast's stepgraph built at every size) and the BIG line's disabled= names it."""
    monkeypatch.delenv(ENV, raising=False)
    r = modes.resolve("big")
    assert "stepgraph" not in r["levers"] and "hoist" in r["levers"] and big.GRAPH_DROP_MIN_TOKENS == 0 and "--graph-drop-tokens" not in big.forward_argv(r["big"])
    monkeypatch.setenv(ENV, "hoist")
    r = modes.resolve("big")
    assert "stepgraph" not in r["levers"] and "hoist" not in r["levers"] and r["levers_off"] == ("hoist",) and r["big"]["levers"] == list(big.LEVER_ORDER)
    monkeypatch.setenv(ENV, "stepgraph")
    with pytest.raises(modes.LeversOffRefused, match="stepgraph — not in mode 'big''s selection .*name hoist"):
        modes.resolve("big")
    monkeypatch.setenv(ENV, "graph_drop,expandable_segments")
    r = modes.resolve("big")
    assert "stepgraph" in r["levers"] and "hoist" not in r["levers"] and r["levers_off"] == ("graph_drop", "expandable_segments")
    assert r["big"]["levers"] == [l for l in big.LEVER_ORDER if l not in ("graph_drop", "expandable_segments")] and r["big"]["disabled"] == ["graph_drop", "expandable_segments"]
    assert "--alloc" not in big.forward_argv(r["big"])                                              # the allocator word leaves with its lever


def test_package_lever_and_dtk_off(monkeypatch):
    monkeypatch.setenv(ENV, "template_dedupe")
    assert modes.resolve("exact")["package_levers"] == ("dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "castcache", "prefetch", "write_behind", "autotune_cache", "feat_par") and modes.resolve("exact")["levers"] == registry.EXACT
    monkeypatch.setenv(ENV, "dtk,canonical_noise,compile")
    r = modes.resolve("fast")
    assert r["dtk"] is False and r["package_levers"] == ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "prefetch", "write_behind", "autotune_cache", "feat_par") and "compile" not in r["levers"] and r["padding"] == "kernel_tile"   # the padding stays the mode's: the ablation names what it dropped, it does not re-derive the mode


@pytest.mark.parametrize("mode,value,words", [
    ("fast", "turbo", "turbo — not a lever of this kit"),
    ("fast", "stepgraph,turbo", "turbo — not a lever of this kit"),
    ("exact", "bf16w", "bf16w — not in mode 'exact''s selection (stepgraph+glu_proj+trimul_exact+template_dedupe+dev_scalars+tri_layout+ln_rows+attn_layout+gate_fuse+castcache+prefetch+write_behind+autotune_cache+feat_par)"),
    ("fast", "hoist", "name stepgraph"),
    ("off", "stepgraph", "not in mode 'off''s selection (no lever: off applies none)"),
])
def test_refused_by_name(mode, value, words, monkeypatch):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(modes.LeversOffRefused) as ei:
        modes.resolve(mode)
    assert words in str(ei.value) and f"{ENV}={','.join(modes.levers_off())}" in str(ei.value)
    assert isinstance(ei.value, modes.UnsupportedMode)                                                   # every entry route's usage refusal


def test_check_and_pred_lines(box, capsys, monkeypatch):
    """check / pred: the ACTIVE line gains `levers_off=<list>` (last) only when something was dropped; the model process gets the reduced --levers
    and MODEL_OPT_LEVERS_OFF itself in its environment; the dropped lever's LEVER line says reason=levers_off; the run is not partial (rc 0);
    an unknown name is rc 2 on both commands with the name in the message."""
    import json
    from .conftest import stub_calls
    monkeypatch.delenv(ENV, raising=False)
    assert cli.main(["check", "--mode", "exact"]) == 0
    err = capsys.readouterr().err
    assert "levers_off" not in err and err.rstrip().endswith(" package=template_dedupe,dev_scalars,tri_layout,ln_rows,attn_layout,gate_fuse,castcache,prefetch,write_behind,autotune_cache,feat_par padding=none compile=off:mode"), err
    monkeypatch.setenv(ENV, "stepgraph")
    assert cli.main(["check", "--mode", "exact"]) == 0
    err = capsys.readouterr().err
    assert " ACTIVE mode=exact lever_set=exact levers=glu_proj+trimul_exact dtk=0 " in err and err.rstrip().endswith(" package=template_dedupe,dev_scalars,tri_layout,ln_rows,attn_layout,gate_fuse,castcache,prefetch,write_behind,autotune_cache,feat_par padding=none compile=off:mode levers_off=stepgraph"), err
    rep = stack.check("exact")
    assert rep["levers_off"] == ["stepgraph"] and rep["levers"] == ["glu_proj", "trimul_exact"] and json.loads(json.dumps(rep))["levers_off"] == ["stepgraph"]
    inp = os.path.join(str(box["tmp"]), "in"); os.makedirs(inp, exist_ok=True)
    with open(os.path.join(inp, "tiny.json"), "w") as fh: json.dump({"name": "tiny", "sequences": [], "modelSeeds": [1]}, fh)
    for mode in ("exact", "fast"):
        assert cli.main(["pred", "--mode", mode, "--json_path", os.path.join(inp, "tiny.json"), "--output_dir", str(box["tmp"] / f"out_{mode}")]) == 0
        err = capsys.readouterr().err
        act = next(l for l in err.splitlines() if " ACTIVE mode=" in l)
        assert act.endswith(" levers_off=stepgraph") and "+stepgraph" not in act and "=stepgraph+" not in act, act
        fwd = [c for c in stub_calls(box) if c and os.path.basename(c[1]) == "forward.py"][-1]
        assert "stepgraph" not in fwd[fwd.index("--levers") + 1].split(","), fwd
        lever = next(l for l in err.splitlines() if " LEVER name=stepgraph " in l)
        assert " state=off " in lever and " reason=levers_off " in lever, lever
        assert " PARTIAL " not in err and " STOCK census=present " not in err and " partial=none " in err, err
        man = cli.last_run(); assert man["activation"]["levers_off"] == ["stepgraph"] and man["partial"] is None
    env = stack.model_process_env({"AF3_TORCH_OPT": "fast", ENV: "stepgraph", "HOME": "/x"})
    assert env[ENV] == "stepgraph" and "AF3_TORCH_OPT" not in env                                      # the switch reaches the model process; the package's own words do not
    monkeypatch.setenv(ENV, "warp_drive")
    assert cli.main(["check", "--mode", "fast"]) == cli.EXIT_USAGE == 2 and "warp_drive — not a lever of this kit" in capsys.readouterr().err
    assert cli.main(["pred", "--mode", "fast", "--json_path", os.path.join(inp, "tiny.json"), "--output_dir", str(box["tmp"] / "out_x")]) == 2
    assert "warp_drive — not a lever of this kit" in capsys.readouterr().err

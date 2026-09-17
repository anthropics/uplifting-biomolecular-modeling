"""Activation: gating on the box facts, the report contract, idempotence, late activation, the environment the children start from,
and the precondition every one of those checks stands on -- the package's own source runs on the pinned stack's interpreter."""
import glob
import io
import os
import sys
import tokenize

import pytest

import genie3_opt
from genie3_opt import cli, design, stack
from genie3_opt.tests import _stubs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = os.path.dirname(os.path.dirname(HERE))


def test_check_dry_run_on_a_full_box_would_activate(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch)
    rep = genie3_opt.check("exact")
    assert rep["dry_run"] and not rep["would_refuse"] and rep["reason"] is None
    assert rep["mode"] == "exact" and rep["tier"] == 1 and rep["attach"] == "driver" and rep["levers_planned"] == ["L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19"]
    assert rep["gpu"]["name"].startswith("NVIDIA H100") and rep["gpu"]["class"] == "H100" and rep["gpu"]["cc"] == "9.0"
    assert rep["pins"]["bad"] == [] and rep["pins"]["detail"]["checkout"]["pinned"] is True
    assert rep["genie3_root"] == os.environ["GENIE3_ROOT"] and rep["weights"] == os.environ["GENIE3_WEIGHTS"]
    assert genie3_opt.status()["active"] is False                     # a dry run arms nothing


def test_enable_arms_and_is_idempotent(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch)
    rep = genie3_opt.enable("exact")
    assert rep["active"] and rep["mode"] == "exact" and rep["levers_applied"] == [] and rep["partial"] is False
    assert genie3_opt.enable("exact") is rep
    assert genie3_opt.status() is rep
    again = genie3_opt.enable("fast")
    assert not again["active"] and "already activated as mode=exact" in again["reason"]
    assert rep["env_dropped"] == ["GENIE3_OPT_HOME"] and again["mode"] == "fast"


def test_enable_reads_the_env_variable_and_defaults(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("GENIE3_OPT", "exact")
    assert genie3_opt.enable()["mode"] == "exact"
    stack.reset_for_tests()
    monkeypatch.delenv("GENIE3_OPT")
    rep = genie3_opt.enable()
    assert rep["mode"] == "fast" and rep["mode_defaulted"] is True and rep["tier"] == 2     # the default: modes.DEFAULT_MODE (fast); GENIE3_OPT unset


def test_refusals_name_the_fact(tmp_path, monkeypatch):
    b = _stubs.box(str(tmp_path), monkeypatch, smi=False)
    rep = genie3_opt.check("exact")
    assert any("no CUDA device visible" in w for w in rep["would_refuse"])
    monkeypatch.setenv("GENIE3_WEIGHTS", str(tmp_path / "nowhere"))
    rep = genie3_opt.check("exact")
    assert any("weights missing" in w and "step=600000.ckpt" in w for w in rep["would_refuse"])
    # a patched checkout: one pinned file changed -> the pins REPORT names it (a NOTE, the report's pins.bad); neither route refuses (upstream runs any checkout)
    target = os.path.join(b["root"], "src", "genie3", "generation", "utils", "geo_utils.py")
    open(target, "a").write("\n# patched\n")
    monkeypatch.setenv("GENIE3_WEIGHTS", b["weights"])
    _stubs.make_bin(str(tmp_path))                                                       # nvidia-smi back: the checkout is the only fact left to judge
    for mode in ("exact", "off"):
        rep = genie3_opt.check(mode)
        assert rep["would_refuse"] == [] and not rep.get("reason"), (mode, rep["would_refuse"])
        assert any("differ from the pinned commit" in x and "geo_utils.py" in x for x in rep["pins"]["bad"]), rep["pins"]["bad"]
        assert any(n.startswith("stock pins:") and "geo_utils.py" in n and "reported, not gated" in n for n in rep["notes"]), rep["notes"]
    assert not hasattr(stack, "ENV_FORCE")                                                # nothing to force: no pin gate


def test_strict_enable_raises(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch, smi=False)
    with pytest.raises(genie3_opt.ActivationError):
        genie3_opt.enable("exact", strict=True)


def test_late_activation_refused_once_a_model_instance_exists(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setattr(stack, "instance_check", lambda: {"module_imported": True, "instances": 1})
    rep = genie3_opt.enable("exact")
    assert not rep["active"] and "instance(s) already exist" in rep["reason"]


def test_late_activation_refused_once_the_kit_patches_applied(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch)
    import types
    pair = types.ModuleType(stack.PAIR_MODULE)
    pair.V1PairFeatureNet = type("V1PairFeatureNet", (), {"_g3fast_patched": True})
    monkeypatch.setitem(sys.modules, stack.PAIR_MODULE, pair)
    assert stack.kit_levers_applied() == [f"{stack.PAIR_MODULE}.V1PairFeatureNet._g3fast_patched"]
    rep = genie3_opt.enable("exact")
    assert not rep["active"] and "already applied" in rep["reason"]


def test_child_environment_drops_the_switches_and_keeps_the_paths(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch)
    env = dict(os.environ, GENIE3_OPT="exact", GENIE3_OPT_FORCE="1", CUDA_MPS_PIPE_DIRECTORY="/tmp/mps", NVIDIA_TF32_OVERRIDE="1", GENIE3_FOO="x")
    child, dropped = stack.child_environment(env)
    assert dropped == ["CUDA_MPS_PIPE_DIRECTORY", "GENIE3_FOO", "GENIE3_OPT", "GENIE3_OPT_FORCE", "GENIE3_OPT_HOME", "NVIDIA_TF32_OVERRIDE"]
    assert child["GENIE3_ROOT"] == os.environ["GENIE3_ROOT"] and child["GENIE3_WEIGHTS"] == os.environ["GENIE3_WEIGHTS"]


def test_gpu_classes_and_stack_ceiling_noted_not_refused(tmp_path, monkeypatch, capsys):
    assert stack.gpu_class({"cc": "9.0", "mem_gib": 79.6}) == "H100"
    assert stack.gpu_class({"cc": "9.0", "mem_gib": 93.6}) == "H100NVL"
    assert stack.gpu_class({"cc": "9.0", "mem_gib": 140.0}) == "H200"
    assert stack.gpu_class({"cc": "10.0", "mem_gib": 179.0}) == "B200"
    assert stack.hardware_notes({"cc": "9.0"}) == [] and stack.hardware_notes({}) == []                  # the pinned stack's card and an unread capability: nothing to note
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setattr(stack, "gpu_probe", lambda: {"name": "NVIDIA B200", "mem_gib": 179.0, "cc": "10.0", "sm": "sm100", "source": "test"})
    NOTE = "compute capability 10.0 above the pinned stack's ceiling (sm_90, torch 2.7.1+cu126) — kernels untested"
    rep = genie3_opt.check("exact")                                                                      # the dry run: noted, nothing to refuse
    assert rep["would_refuse"] == [] and not rep.get("reason"), rep
    assert len(rep["notes"]) == 1 and rep["notes"][0].startswith(NOTE), rep["notes"]
    io = capsys.readouterr()
    assert f"[genie3-opt] NOTE {NOTE}" in io.err and "DRY-RUN mode=exact" in io.err and "would refuse" not in io.out
    assert cli.main(["check", "--mode", "exact"]) == 0 and f"[genie3-opt] NOTE {NOTE}" in capsys.readouterr().out   # check: rc 0, the note among its lines
    stack.reset_for_tests()
    rep = stack.activate("exact")                                                                        # the activation proper: ACTIVE, the note before the verdict line
    assert rep["active"] is True and rep["notes"][0].startswith(NOTE) and not rep.get("reason"), rep
    err = capsys.readouterr().err
    assert err.index(f"[genie3-opt] NOTE {NOTE}") < err.index("[genie3-opt] ACTIVE mode=exact"), err
    stack.reset_for_tests()
    out = str(tmp_path / "out_b200")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), out, "exact")                       # a design pass on the card: proceeds to its own outcome, the note in the run record
    assert rc == 0 and man["status"] == "ok" and man["activation"]["notes"][0].startswith(NOTE), man
    stack.reset_for_tests()
    rep = genie3_opt.check("off")                                                                        # the stock route notes it too (upstream's torch is the same stack)
    assert rep["notes"][0].startswith(NOTE) and not any("compute capability" in w for w in rep["would_refuse"]), rep


def test_driver_command_is_the_kit_line(tmp_path, monkeypatch):
    b = _stubs.box(str(tmp_path), monkeypatch)
    from genie3_opt import modes
    from genie3_opt.modes import resolve
    cmd = stack.driver_command(resolve("exact"), "/o/request.yaml", "/o", "/o/timings.json", python="/venv/bin/python")
    driver = os.path.join(b["tree"], "opt", "genie3_opt", "g3batch.py")                                     # the batched capture driver (package code; the fixture's stand-in in the test tree)
    assert cmd == ["/venv/bin/python", driver, "--config", "/o/request.yaml", "--outdir", "/o",
                   "--batch-size", "1", "--cuda-graphs", "--hoist", "--reuse-graphs", "16", "--lean-pair", "--wide-capture", "--alloc", "expandable", "--timings", "/o/timings.json"]      # the mode row's flags, then the timings file; the recipe adds no driver flag
    monkeypatch.undo()                                                                                        # outside the fixture the chain entry resolves to the package's own file
    assert stack.chain_file("genie3_opt", "g3batch.py") == os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(_stubs.__file__))), "g3batch.py")
    os.makedirs(str(tmp_path / "again"))
    b = _stubs.box(str(tmp_path / "again"), monkeypatch)
    cmd = stack.driver_command(modes.with_batch(resolve("fast"), 16), "/o/request.yaml", "/o", verbose=True)
    assert cmd[2:] == ["--config", "/o/request.yaml", "--outdir", "/o", "--batch-size", "16", "--cuda-graphs", "--hoist", "--reuse-graphs", "16", "--lean-pair", "--wide-capture", "--alloc", "expandable", "--pt-chunk", "design", "--tf32", "--trimul", "fpf", "--compile", "--verbose"]   # the fast row at batch 16; upstream's --verbose rides last


def test_jit_cache_key_grammar(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch)
    key = stack.jit_cache_key({"sm": "sm90"})
    assert key.startswith("torch") and key.endswith("-sm90") and "-cu" in key or key.endswith("-cpu-sm90")


def test_weights_gate_is_warn_and_run(tmp_path, monkeypatch, capsys):
    """The weights gate on both routes: the pinned digest prints `(pinned)`; a weight file whose sha256 differs from stock/PINS.json
    prints `NOT PINNED — …` and the pass RUNS (no flag, no force), recorded not pinned in the report and the run record; a missing file
    is refused by name. One census (manifest.weights_record) behind every route."""
    import hashlib
    from genie3_opt import design, manifest
    b = _stubs.box(str(tmp_path), monkeypatch)
    rep = stack.activate("exact", dry_run=True)                                                    # the fixture's weights are the pinned bytes
    assert rep["weights_pinned"] is True and all(f["pinned"] for f in rep["weights_files"].values()) and not rep.get("reason"), rep
    err = capsys.readouterr().err
    assert f"[genie3-opt] weights=step=600000.ckpt sha256={hashlib.sha256(_stubs.STUB_CKPT).hexdigest()[:12]} (pinned)" in err
    assert "[genie3-opt] weights=config.yaml sha256=" in err and "NOT PINNED" not in err
    other = b"model: {name: other}\n"
    open(os.path.join(b["weights"], "config.yaml"), "wb").write(other)
    line = f"[genie3-opt] weights=config.yaml sha256={hashlib.sha256(other).hexdigest()[:12]} NOT PINNED — not the digest pinned in stock/PINS.json weights; the pass runs on these files, labelled"
    assert manifest.NOT_PINNED_NOTE in line
    for mode in ("exact", "off"):                                                                  # both routes: would activate, labelled
        stack.reset_for_tests()
        rep = stack.activate(mode, dry_run=True)
        assert rep["would_refuse"] == [] and not rep.get("reason") and rep["weights_pinned"] is False, (mode, rep)
        assert rep["weights_files"]["config.yaml"] == {"sha256": hashlib.sha256(other).hexdigest(), "pinned": False} and rep["weights_files"]["step=600000.ckpt"]["pinned"] is True
        assert line in capsys.readouterr().err, mode
        assert "weights_bad" not in rep
    rec = manifest.weights_record(b["weights"])                                                    # the one census the gate read (the digest memoised by now: digest_memo)
    assert rec["pinned"] is False and rec["missing"] == [] and rec["files"]["config.yaml"]["pinned"] is False and rec["lines"][1].startswith(line[len("[genie3-opt] "):])
    assert rec["files"]["config.yaml"]["cached_utc"] is not None and os.path.isfile(rec["memo"]) and rec["memo"].startswith(str(tmp_path))   # served from the memo under XDG_CACHE_HOME
    assert manifest.weights_record(b["weights"], refresh=True)["files"]["config.yaml"]["cached_utc"] is None                                # `check` hashes afresh
    stack.reset_for_tests()
    out = str(tmp_path / "runs_not_pinned")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, seed=0), out, "exact")            # the pass runs on the unpinned config: rc 0
    assert rc == 0 and man["status"] == "ok" and man["activation"]["weights_pinned"] is False, man.get("status")
    assert man["weights"]["files"]["config.yaml"]["pinned"] is False and man["weights"]["files"]["checkpoints/step=600000.ckpt"]["pinned"] is True
    assert line in capsys.readouterr().err
    stack.reset_for_tests()
    os.remove(os.path.join(b["weights"], "checkpoints", "step=600000.ckpt"))                     # a MISSING file stays a refusal by name
    rep = stack.activate("exact", dry_run=True)
    assert any(w.startswith("weights missing (GENIE3_WEIGHTS=") and "checkpoints/step=600000.ckpt" in w for w in rep["would_refuse"]), rep
    assert rep["weights_files"]["step=600000.ckpt"] == {"missing": True} and rep["weights_pinned"] is False
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path)), str(tmp_path / "never"), "exact")
    assert rc == 3 and man["status"] == "refused" and "weights missing" in man["reason"] and not os.path.exists(str(tmp_path / "never"))


# ===== the interpreter precondition: python 3.10 (stock/PINS.json pinned_stack) compiles every module of the package =====
def _sources():
    pats = ["opt/genie3_opt/**/*.py", "opt/*.py", "stock/*.py"]
    return sorted({p for pat in pats for p in glob.glob(os.path.join(TREE, pat), recursive=True)})


def test_every_module_compiles_here():
    assert _sources(), TREE
    for p in _sources():
        compile(open(p, encoding="utf-8").read(), p, "exec")


def test_no_fstring_reuses_its_own_quote():
    if sys.version_info < (3, 12):
        return   # compile() above is the check on this interpreter
    hits = []
    for p in _sources():
        stack = []
        for t in tokenize.generate_tokens(io.StringIO(open(p, encoding="utf-8").read()).readline):
            if t.type == tokenize.FSTRING_START:
                stack.append(t.string.lstrip("fFrRbBuU"))
            elif t.type == tokenize.FSTRING_END:
                stack.pop()
            elif t.type == tokenize.STRING and stack and len(stack[-1]) == 1 and t.string.lstrip("fFrRbBuU").startswith(stack[-1]):
                hits.append(f"{os.path.relpath(p, TREE)}:{t.start[0]} {t.string[:40]}")
    assert not hits, hits

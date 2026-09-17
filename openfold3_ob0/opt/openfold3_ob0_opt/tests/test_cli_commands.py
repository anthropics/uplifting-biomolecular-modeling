"""The command layer: mode resolution and the OPENFOLD3_OB0_OPT disagreement rule, the stock argv with upstream's knobs as given, check's exit codes,
the manifest writer."""
import os
import re

import pytest

from openfold3_ob0_opt import cli, manifest, modes
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_mode_resolution_rule():
    assert cli.resolve_mode(None, {}) == modes.DEFAULT_MODE == "fast"
    assert cli.resolve_mode(None, {"OPENFOLD3_OB0_OPT": "exact"}) == "exact"
    assert cli.resolve_mode("exact", {"OPENFOLD3_OB0_OPT": "exact"}) == "exact"
    with pytest.raises(SystemExit):
        cli.resolve_mode("fast", {"OPENFOLD3_OB0_OPT": "exact"})
    with pytest.raises(ValueError):
        cli.resolve_mode("faster", {})


def _args(*extra):
    ap = cli.build_parser()
    return ap.parse_args(["pred", "--query-json", "q.json", "--output-dir", "out", *extra])


def test_stock_argv_settings_rows():
    a = _args("--mode", "off")
    argv = cli.stock_argv(a, HOME, "/w.pt", "q.json")
    assert argv[:4] == ["--query-json", "q.json", "--output-dir", "out"]
    assert argv[argv.index("--runner-yaml") + 1] == os.path.join(HOME, modes.STOCK_YAML)
    det1 = cli.stock_argv(_args("--mode", "off", "--det", "1"), HOME, "/w.pt", "q.json")                 # the stock arm under the det recipe: the DS4Sci attention off
    assert det1[det1.index("--runner-yaml") + 1] == os.path.join(HOME, modes.STOCK_DET_YAML)
    ex = cli.stock_argv(_args("--mode", "exact"), HOME, "/w.pt", "q.json"); exd = cli.stock_argv(_args("--mode", "exact", "--det", "1"), HOME, "/w.pt", "q.json")   # exact runs stock's configuration at either det level
    assert ex[ex.index("--runner-yaml") + 1] == os.path.join(HOME, modes.STOCK_YAML) and exd[exd.index("--runner-yaml") + 1] == os.path.join(HOME, modes.STOCK_DET_YAML)
    fast = cli.stock_argv(_args(), HOME, "/w.pt", "q.json")                                     # the default mode (fast): the stock runner yaml itself (the fast line's runner yaml is STOCK_YAML; the line differs from stock by its levers alone)
    assert fast[fast.index("--runner-yaml") + 1] == os.path.join(HOME, modes.LINES[("fast", None)].runner_yaml) == os.path.join(HOME, modes.STOCK_YAML)
    a = _args()
    assert argv[argv.index("--inference-ckpt-path") + 1] == "/w.pt"
    for knob in cli.UPSTREAM_KNOBS:                                                              # no knob given: none passed but the mode's runner yaml — upstream's defaults apply as shipped
        assert (knob in argv) == (knob == "--runner-yaml"), (knob, argv)
    a = _args("--use-msa-server", "false", "--use-templates", "False", "--num-model-seeds", "5", "--num-diffusion-samples", "1", "--mode", "off")
    argv = cli.stock_argv(a, HOME, "/w.pt", "q.json")                                           # given: passed through verbatim (booleans lower-cased), in upstream's option order
    assert argv[argv.index("--use-msa-server") + 1] == "false" and argv[argv.index("--use-templates") + 1] == "false"
    assert argv[argv.index("--num-model-seeds") + 1] == "5" and argv[argv.index("--num-diffusion-samples") + 1] == "1"
    with pytest.raises(SystemExit):
        _args("--settings", "kit")                                                               # no settings presets: upstream's knobs only
    a = _args("--use-templates", "true", "--num-model-seeds", "2")
    argv = cli.stock_argv(a, HOME, "/w.pt", "q.json")
    assert argv[argv.index("--use-templates") + 1] == "true" and argv[argv.index("--num-model-seeds") + 1] == "2"
    with pytest.raises(SystemExit):
        _args("--", "--inference-ckpt-name", "x")                                               # no pass-through: run_openfold's words are this CLI's named flags or nothing


def test_check_exit_codes(monkeypatch):
    """`check` exit codes on the stock and exact routes of this tree: stock's kernels (Triton) need no third-party op, so `off` and `exact`
    check clean at either det level on a CPU box; an unknown exact line is usage (2); a switch the line unsets, set by the caller, is refused (3);
    an installed openfold3 other than the pin is refused (3)."""
    d = _stubs.stub_dist()
    try:
        _stubs.reset_package()
        assert cli.main(["check", "--mode", "exact", "--det", "1"]) == 0
        assert cli.main(["check", "--mode", "off", "--det", "1"]) == 0 and cli.main(["check", "--mode", "off"]) == 0
        os.environ["OF3T_TRIATT"] = "triton"
        assert cli.main(["check", "--mode", "exact"]) == 3                      # a switch the exact line unsets (modes.LINES unset), set by the caller: refused by name
    finally:
        os.environ.pop("OF3T_TRIATT", None)
        _stubs.reset_package()
        _stubs.unstub_dist(d)
    d = _stubs.stub_dist("0.3.0")
    try:
        _stubs.reset_package()
        assert cli.main(["check", "--mode", "exact"]) == 3                      # the pin
    finally:
        _stubs.reset_package()
        _stubs.unstub_dist(d)


def test_check_exit_codes_fast(monkeypatch):
    """`check --mode fast`: clean on the pinned stub; a conflicting preset switch is refused (3)."""
    d = _stubs.stub_dist()
    try:
        _stubs.reset_package()
        assert cli.main(["check", "--mode", "fast"]) == 0
        os.environ["OF3T_TRIMUL"] = "cueq"
        assert cli.main(["check", "--mode", "fast"]) == 3                       # a conflicting preset switch (the fast line unsets OF3T_TRIMUL: the cells serve the triangle multiplication)
    finally:
        os.environ.pop("OF3T_TRIMUL", None)
        _stubs.reset_package()
        _stubs.unstub_dist(d)


def test_pred_needs_a_checkpoint_and_refuses_disagreeing_mode(monkeypatch):
    monkeypatch.delenv("OPENFOLD3_OB0_CKPT", raising=False)                                       # no --ckpt and no OPENFOLD3_OB0_CKPT: usage exit 2
    assert cli.main(["pred", "--mode", "fast", "--query-json", "q.json", "--output-dir", "o"]) == 2
    monkeypatch.setenv("OPENFOLD3_OB0_OPT", "exact")
    assert cli.main(["pred", "--mode", "fast", "--query-json", "q.json", "--output-dir", "o", "--ckpt", "/w.pt"]) == 2


def test_warm_builds_its_argv_like_pred():
    from openfold3_ob0_opt import warm
    argv = cli.warm_argv(HOME, "/w.pt", "q.json", "out")
    assert argv[:4] == ["--query-json", "q.json", "--output-dir", "out"]
    assert argv[argv.index("--runner-yaml") + 1] == os.path.join(HOME, modes.STOCK_YAML) and argv[argv.index("--inference-ckpt-path") + 1] == "/w.pt"   # no line named: the fast line's default, the stock runner yaml   # no line named, fp32: the kernels-off base
    assert cli.warm_argv(HOME, "/w.pt", "q.json", "out", modes.LINES[("exact", modes.EXACT_LINE)])[5] == os.path.join(HOME, modes.STOCK_YAML)      # the exact line warms on the stock configuration
    assert argv[argv.index("--num-model-seeds") + 1] == "1" and argv[argv.index("--num-diffusion-samples") + 1] == "1"
    assert argv[argv.index("--use-msa-server") + 1] == "false" and argv[argv.index("--use-templates") + 1] == "false"
    a = _args("--use-msa-server", "false", "--use-templates", "false", "--num-model-seeds", "1", "--num-diffusion-samples", "1")
    assert cli.stock_argv(a, HOME, "/w.pt", "q.json")[4:] == cli.warm_argv(HOME, "/w.pt", "q.json", "out")[4:]     # one builder: warm's knobs (cli.WARM_KNOBS) are these four flags
    assert cli.WARM_KNOBS == {"use_msa_server": "false", "use_templates": "false", "num_model_seeds": 1, "num_diffusion_samples": 1}
    assert not hasattr(warm, "pred_args")


def test_weights_check_verdicts(tmp_path, capsys, monkeypatch):
    """One hash against the pinned weights, never a condition: other bytes are a named WARNING and every route proceeds (rc 0,
    weights_pinned false); the opt-out is recorded; the pinned bytes print `pinned`; a missing file is usage."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))                  # the digest memo (stack.cache_root) under the test's dir
    from openfold3_ob0_opt import stack
    w = tmp_path / "w.pt"
    w.write_bytes(b"other bytes")
    for what in ("pred --mode exact --det 1", "pred --mode fast --det 0", "warm --mode exact"):
        info, rc = cli.weights_check(str(w), HOME, what=what)
        err = capsys.readouterr().err
        assert rc == 0 and info["is_pinned"] is False and re.match(r".*WARNING: WEIGHTS unknown( \(cached digest [^)]+\))? sha256=", err, re.S) and _stubs.pins()["weights"]["file"] in err and "proceeding" in err and "NOT ACTIVE" not in err, what
    info, rc = cli.weights_check(str(tmp_path / "none.pt"), HOME, what="x")
    assert rc == cli.EXIT_USAGE and info["exists"] is False
    monkeypatch.setattr(stack, "pinned_weights", lambda home=None: ("w.pt", manifest.sha256_file(str(w))))
    info, rc = cli.weights_check(str(w), HOME, what="warm")
    assert rc == 0 and info["is_pinned"] is True and "WEIGHTS pinned" in capsys.readouterr().err and "the pinned checkpoint" in "WEIGHTS pinned: the pinned checkpoint"

def test_manifest_roundtrip(tmp_path):
    rep = {"active": True, "mode": "fast", "line": None, "levers_requested": ["fast_init"], "hooks": ["trunk_kernels", "fast_inference"], "env": {"OF3_FAST_INIT": "1"},
           "carrier": os.path.join(HOME, "opt", "forward", "fast_inference", "of3_levers")}
    out = tmp_path / "pred"
    (out / "q" / "seed_42").mkdir(parents=True)
    (out / "q" / "seed_42" / "q_seed_42_sample_1_model.cif").write_text("x")
    p = manifest.dump(str(out / "rank0.json"), manifest.build(rep, out_dir=str(out), command="pred", argv=["--query-json", "q.json"], exit_code=0, checkpoint=None, runner_yaml=os.path.join(HOME, modes.STOCK_YAML), det=0))
    m = manifest.read(p)
    assert m["schema"] == "openfold3_ob0_opt/2" and m["mode"] == "fast" and m["n_cif"] == 1 and m["runner_yaml"]["sha256"] and m["exit_code"] == 0
    assert m["activation_report"]["env"] == {"OF3_FAST_INIT": "1"} and os.path.basename(p) == "rank0.json"
    assert m["levers_requested"] == ["fast_init"] and m["levers_applied"] == [] and m["levers_pending"] == ["fast_init"]     # the target never imported here
    assert m["levers_unavailable"] == [] and m["partial"] is False and m["arm_complete"] is None
    assert m["checkpoint"] == {"path": None, "exists": False, "hashed": False, "is_pinned": None} and m["weights_pinned"] is None
    assert "levers_applied" in m["activation_report"] and m["activation_report"]["lever_evidence"]["fast_init"].startswith("pending")


def test_manifest_records_the_weights_gate_and_the_stock_proof(tmp_path):
    """A stock run's manifest: arm_complete follows the environment proof, weights_pinned the gate's verdict, the opt-out is recorded."""
    from openfold3_ob0_opt import stack
    w = tmp_path / "w.pt"
    w.write_bytes(b"not the pinned bytes")
    info = stack.weights_gate(str(w), HOME)
    assert info["hashed"] and info["is_pinned"] is False and info["pinned_file"] == _stubs.pins()["weights"]["file"] and len(info["pinned_sha256"]) == 64
    assert info["sha256"] != info["pinned_sha256"] and info["bytes"] == 20
    rep = {"active": False, "mode": "off", "reason": "stock"}
    manifest.dump(str(tmp_path / "off.json"), manifest.build(rep, out_dir=str(tmp_path / "off"), command="pred", argv=[], exit_code=0, checkpoint=info, stock_proof={"ok": True}, det=0))
    m = manifest.read(str(tmp_path / "off.json"))
    assert m["arm_complete"] is True and m["weights_pinned"] is False and m["checkpoint"]["sha256"] == info["sha256"]
    manifest.dump(str(tmp_path / "off2.json"), manifest.build(rep, out_dir=str(tmp_path / "off2"), command="pred", argv=[], exit_code=0, checkpoint=info, stock_proof={"ok": False}, det=0))
    assert manifest.read(str(tmp_path / "off2.json"))["arm_complete"] is False
    unhashed = stack.weights_gate(str(w), HOME, hash_it=False)
    assert unhashed["hashed"] is False and unhashed["is_pinned"] is None and unhashed["sha256"] is None
    manifest.dump(str(tmp_path / "nh.json"), manifest.build(rep, out_dir=str(tmp_path / "nh"), command="pred", argv=[], exit_code=0, checkpoint=unhashed, det=0))
    m = manifest.read(str(tmp_path / "nh.json"))
    assert m["weights_pinned"] is None and m["checkpoint"]["hashed"] is False
    assert stack.weights_gate(str(tmp_path / "missing.pt"), HOME)["exists"] is False


def test_check_hashes_afresh_and_the_runs_read_the_memo(tmp_path, capsys, monkeypatch):
    """`check` digests a named checkpoint afresh (refresh=True: the memo entry rewritten); pred / warm read the memo (refresh=False)
    and say `(cached digest <utc>)` on a hit; the memo lives under stack.cache_root() = $XDG_CACHE_HOME/openfold3_ob0_opt."""
    from openfold3_ob0_opt import digest_memo, stack
    assert cli.WEIGHTS_REFRESH == {"check": True, "pred": False, "warm": False}
    src = open(os.path.join(HOME, "opt", "openfold3_ob0_opt", "cli.py"), encoding="utf-8").read()
    for verb in cli.WEIGHTS_REFRESH:
        assert f'refresh=WEIGHTS_REFRESH["{verb}"]' in src, verb                       # every verb's weights_check names its own refresh word
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert stack.cache_root() == str(tmp_path / "cache" / "openfold3_ob0_opt")
    calls = []
    real = digest_memo.digest
    monkeypatch.setattr(digest_memo, "digest", lambda path, memo_dir, refresh=False, hasher=None: calls.append((memo_dir, refresh)) or real(path, memo_dir, refresh=refresh, hasher=hasher))
    w = tmp_path / "w.pt"; w.write_bytes(b"not the pinned weights")
    info, rc = cli.weights_check(str(w), HOME, what="pred", refresh=cli.WEIGHTS_REFRESH["pred"])          # first sight: hashed, memo written
    assert rc == 0 and info["digest_cached_utc"] is None and calls[-1] == (stack.cache_root(), False) and os.path.isfile(info["digest_memo"])
    assert "(cached digest" not in capsys.readouterr().err
    info, rc = cli.weights_check(str(w), HOME, what="warm", refresh=cli.WEIGHTS_REFRESH["warm"])        # a memo hit: the word says so
    assert rc == 0 and info["digest_cached_utc"] and "WEIGHTS unknown (cached digest " in capsys.readouterr().err
    info, rc = cli.weights_check(str(w), HOME, what="check", refresh=cli.WEIGHTS_REFRESH["check"])        # check: afresh, entry rewritten
    assert rc == 0 and info["digest_cached_utc"] is None and calls[-1] == (stack.cache_root(), True) and "(cached digest" not in capsys.readouterr().err
    a = cli.build_parser().parse_args(["check", "--mode", "fast", "--ckpt", str(w)])                                    # the check verb takes a checkpoint
    assert a.ckpt == str(w)


def test_an_unwritable_memo_dir_is_named_and_the_checkpoint_hashed_afresh(tmp_path, capsys, monkeypatch):
    """A read-only (or uncreatable) cache root never refuses a run: the WEIGHTS line names the memo that was not written and the digest is taken afresh."""
    from openfold3_ob0_opt import stack
    w = tmp_path / "w.pt"; w.write_bytes(b"some weights")
    blocker = tmp_path / "file_not_dir"; blocker.write_text("x")                       # $XDG_CACHE_HOME is a regular file: the memo dir cannot be created (holds for root too)
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))
    for what, refresh in (("check", True), ("pred", False)):
        info, rc = cli.weights_check(str(w), HOME, what=what, refresh=refresh)
        err = capsys.readouterr().err
        assert rc == 0 and info["hashed"] is True and len(info["sha256"]) == 64 and info["digest_cached_utc"] is None and info["digest_memo_unwritable"], what
        assert "WEIGHTS digest memo not written: " in err and "hashing afresh" in err and "WARNING: WEIGHTS unknown sha256=" in err and "NOT ACTIVE" not in err, what
    ro = tmp_path / "ro"; (ro / "openfold3_ob0_opt").mkdir(parents=True); os.chmod(ro / "openfold3_ob0_opt", 0o555)   # a read-only memo dir (effective for a non-root caller; root writes through the mode bits)
    monkeypatch.setenv("XDG_CACHE_HOME", str(ro))
    try:
        info, rc = cli.weights_check(str(w), HOME, what="check", refresh=True)
        err = capsys.readouterr().err
        assert rc == 0 and info["hashed"] is True and len(info["sha256"]) == 64
        assert (info.get("digest_memo_unwritable") is None) == os.access(str(ro / "openfold3_ob0_opt"), os.W_OK)      # named exactly when the dir is not writable for this user
    finally:
        os.chmod(ro / "openfold3_ob0_opt", 0o755)
    assert stack.memo_dir_unwritable(str(tmp_path / "fresh" / "openfold3_ob0_opt")) is None and os.path.isdir(tmp_path / "fresh" / "openfold3_ob0_opt")

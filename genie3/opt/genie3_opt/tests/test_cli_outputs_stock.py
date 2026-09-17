"""The CLI verbs and exit codes; the stock arm end to end (the proof, the pins verdict copied into it, the outputs)."""
import json
import re
import os
import subprocess
import sys

from genie3_opt import cli, design, modes, report, stack
from genie3_opt.tests import _stubs


def test_stock_arm_runs_the_stock_caller_with_a_proof(tmp_path, monkeypatch, capsys):
    b = _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("GENIE3_OPT", "off")                                # stripped from the stock process, proven absent
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    out = str(tmp_path / "stock")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), out, "off", tag="s1")
    assert rc == 0 and man["status"] == "ok", man
    st = man["stock"]
    assert st["rc"] == 0 and st["cwd"] == b["root"] and st["env_stripped"] == ["GENIE3_OPT", "GENIE3_OPT_HOME", "NVIDIA_TF32_OVERRIDE"]
    assert st["cmd"][1:4] == ["-s", "-m", "genie3_opt.stock_cli"] and st["cmd"][-4:] == ["-c", os.path.join(out, "request.yaml"), "--log-dir", os.path.join(out, "logs")]
    p = st["proof"]
    assert p["ok"] and p["forbidden_env_present"] == [] and p["kit_modules_loaded"] == [] and p["kit_dirs_on_sys_path"] == [] and not p["autoload_armed"]
    assert p["torch_imported"] is False and p["genie3_imported"] is False
    assert p["pins_check"]["bad"] == [] and p["pins_check"]["detail"]["checkout"]["pinned"] is True and p["pins_check"]["detail"]["checkout"]["files_checked"] >= 150
    assert man["outputs"]["n_pdb"] == 2 and os.path.isfile(os.path.join(out, "generation_stats", "main_rank_0.json"))
    assert os.path.isfile(os.path.join(out, "stock_env_proof.json")) and os.path.isfile(os.path.join(out, "stock.log"))
    err = capsys.readouterr().err
    assert "[genie3-opt] ACTIVE mode=off attach=stock-cli" in err and "[genie3-opt] ready mode=off t=" in err
    assert "[genie3-opt] EVIDENCE" not in err                              # no lever on the stock line


def test_a_patched_checkout_runs_and_the_pins_report_names_it(tmp_path, monkeypatch, capsys):
    """The stock pins are REPORTED, never gated: a checkout file that differs from the pin is named on a NOTE line and in the run record's pins
    summary and the proof's pinned=False; the stock pass runs (upstream runs any checkout) — rc 0."""
    b = _stubs.box(str(tmp_path), monkeypatch)
    open(os.path.join(b["root"], "src", "genie3", "generation", "utils", "encode_utils.py"), "a").write("# patched\n")
    rc, man = design.run(_stubs.request(str(tmp_path)), str(tmp_path / "o"), "off")
    assert rc == 0 and man["status"] == "ok", man
    assert any("encode_utils.py" in x for x in man["activation"]["pins"]["bad"]) and man["stock"]["proof"]["ok"] and man["stock"]["proof"]["pins_check"]["detail"]["checkout"]["pinned"] is False
    err = capsys.readouterr().err
    assert "[genie3-opt] NOTE stock pins:" in err and "encode_utils.py" in err and "pinned=False" in err and "NOT STOCK" not in err
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path), name="k.yaml"), str(tmp_path / "k"), "exact")               # the kit line too: reported, runs
    assert rc == 0 and man["activation"]["pins"]["bad"], man


def test_stock_cli_module_refuses_when_forbidden(tmp_path, monkeypatch):
    b = _stubs.box(str(tmp_path), monkeypatch)
    proof = str(tmp_path / "p.json")
    env = dict({k: v for k, v in os.environ.items() if k != "GENIE3_OPT_HOME"}, GENIE3_OPT="exact")
    r = subprocess.run([sys.executable, "-s", "-m", "genie3_opt.stock_cli", "--proof-json", proof, "--env-absent", "GENIE3_OPT,GENIE3_,CUDA_MPS_", "--allowed", "GENIE3_ROOT,GENIE3_WEIGHTS",
                        "--kit-dirs", "", "--genie3-root", b["root"], "--", "generate", "-c", "x.yaml"], env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "NOT STOCK" in r.stderr
    assert json.load(open(proof))["forbidden_env_present"] == ["GENIE3_OPT"]


def test_cli_check_exit_codes(tmp_path, monkeypatch, capsys):
    _stubs.box(str(tmp_path), monkeypatch)
    assert cli.main(["check", "--mode", "exact"]) == 0
    assert "would_activate=True" in capsys.readouterr().out
    assert cli.main(["check", "--mode", "exact", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["would_activate"] and j["default_mode"] == "fast" == modes.DEFAULT_MODE and [r["mode"] for r in j["mode_table"]][:3] == ["off", "exact", "fast"]
    assert cli.main(["check", "--mode", "nope"]) == 3
    assert cli.main([]) == 2


def test_cli_design(tmp_path, monkeypatch, capsys):
    _stubs.box(str(tmp_path), monkeypatch)
    req = _stubs.request(str(tmp_path), n_sample=2)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    assert cli.main(["design", "--mode", "off", "--input", req, "--out_dir", a]) == 0
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "exact", "--input", req, "--out_dir", b]) == 0
    capsys.readouterr()
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "exact", "-c", req, "--out_dir", str(tmp_path / "c")]) == 0          # upstream's own -c spelling of the request
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "off", "--out_dir", str(tmp_path / "cc"), "--config", req, "--log-dir", str(tmp_path / "cclogs")]) == 0   # upstream's own --config spelling (what run.sh forwards verbatim after `--`)
    assert json.load(open(tmp_path / "cc" / "stub_argv.json"))[-2:] == ["--log-dir", str(tmp_path / "cclogs")]
    assert json.load(open(tmp_path / "c" / "opt_manifest.json"))["batch_size"] == 1                           # exact's capture line at upstream's own default batch (no batch_size key in the request, no --batch_size)
    stack.reset_for_tests()
    req_out = _stubs.request(str(tmp_path), n_sample=2, name="own.yaml", rootdir=str(tmp_path / "own_rootdir"))
    assert cli.main(["design", "--mode", "exact", "--input", req_out]) == 0                                     # no --out_dir: the request's own paths.rootdir, as upstream resolves it
    m = json.load(open(tmp_path / "own_rootdir" / "opt_manifest.json"))
    assert m["status"] == "ok" and m["outputs"]["n_pdb"] == 2 and "paths.rootdir" not in m["request"]["composed"]
    for bad in (["design", "--mode", "exact", "--input", req, "--out_dir", "x", "--batch", "2"],
                ["design", "--mode", "fast", "--input", req, "--out_dir", "x", "--precision", "fp32"], ["design", "--mode", "fast", "--input", req, "--out_dir", "x", "--trimul", "stock"],
                ["check", "--mode", "fast", "--trimul", "stock"],
                ["design", "--mode", "exact", "--input", req, "--out_dir", "x", "--dry-run"], ["design", "--mode", "exact", "--input", req, "--out_dir", "x", "--allow-partial"]):
        try:
            cli.main(bad)
        except SystemExit as e:                                              # argparse: no such flag / value (the lever selectors and per-lever flags are not the surface: the mode set is)
            assert e.code == 2, bad
        else:
            raise AssertionError(bad)
    stack.reset_for_tests()
    capsys.readouterr()
    assert cli.main(["design", "--mode", "exact", "--input", req, "--out_dir", str(tmp_path / "xb"), "--batch_size", "4"]) == 0    # --batch_size: generation.dataset.batch_size for this pass, every mode — the kit line runs --batch-size 4
    mb = json.load(open(tmp_path / "xb" / "opt_manifest.json"))
    assert mb["batch_size"] == 4 and mb["flags"][:2] == ["--batch-size", "4"] and mb["request"]["composed"]["generation.dataset.batch_size"] == 4
    assert "--batch-size 4 " in [l for l in capsys.readouterr().err.splitlines() if l.startswith("[genie3-opt] ACTIVE")][0]
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "off", "--input", req, "--out_dir", str(tmp_path / "xo"), "--batch_size", "4"]) == 0       # the stock route: upstream reads the key from the request copy
    assert json.load(open(tmp_path / "xo" / "opt_manifest.json"))["batch_size"] == 4 and "batch_size: 4" in open(tmp_path / "xo" / "request.yaml", encoding="utf-8").read()
    stack.reset_for_tests()
    capsys.readouterr()
    assert cli.main(["design", "--mode", "exact", "--input", req, "--out_dir", str(tmp_path / "xz"), "--batch_size", "0"]) == 3 and not os.path.exists(tmp_path / "xz")   # refused by name before anything runs
    assert "[genie3-opt] NOT ACTIVE: --batch_size 0: the batch size is an integer >= 1" in capsys.readouterr().err


def test_upstream_generate_flags_pass_through_to_the_stock_child(tmp_path, monkeypatch, capsys):
    """upstream's own generate flags (--verbose --log-dir --num-devices --shard-id --num-shards) reach `genie3 generate` verbatim on the stock
    route; --log-dir defaults to <out>/logs only when the caller gave none."""
    _stubs.box(str(tmp_path), monkeypatch)
    req = _stubs.request(str(tmp_path), n_sample=2)
    out, logs = str(tmp_path / "s"), str(tmp_path / "mylogs")
    assert cli.main(["design", "--mode", "off", "--input", req, "--out_dir", out, "--verbose", "--log-dir", logs, "--num-devices", "1", "--shard-id", "0", "--num-shards", "1"]) == 0
    argv = json.load(open(os.path.join(out, "stub_argv.json")))
    assert argv == ["generate", "-c", os.path.join(out, "request.yaml"), "--verbose", "--log-dir", logs, "--num-devices", "1", "--shard-id", "0", "--num-shards", "1"], argv
    m = json.load(open(os.path.join(out, "opt_manifest.json")))
    assert m["stock_flags"] == argv[3:] and "declined" not in m and m["shard"] is None and os.path.isdir(logs) and not os.path.exists(os.path.join(out, "logs"))
    stack.reset_for_tests()
    out2 = str(tmp_path / "s2")
    assert cli.main(["design", "--mode", "off", "--input", req, "--out_dir", out2]) == 0
    assert json.load(open(os.path.join(out2, "stub_argv.json"))) == ["generate", "-c", os.path.join(out2, "request.yaml"), "--log-dir", os.path.join(out2, "logs")]


def test_a_request_the_kit_line_does_not_compute_is_refused_by_name_before_anything_runs(tmp_path, monkeypatch, capsys):
    """upstream's beam search / sidechain prediction / more than one device on a kit mode: ONE `NOT ACTIVE: mode=<m> cannot serve
    <features> — refused by name (exit 3), nothing ran: <feature: mechanism; …>; run --mode off for the stock path` line, exit 3, and nothing ran
    (no output directory, no stock child, no driver, no mode word on any other line); `--mode off` runs the same request."""
    _stubs.box(str(tmp_path), monkeypatch)
    out = str(tmp_path / "d")
    rc = cli.main(["design", "--mode", "exact", "--input", _stubs.request(str(tmp_path)), "--out_dir", out, "--num-devices", "2"])
    err = capsys.readouterr().err
    assert rc == 3 and not os.path.exists(out), err[-1500:]
    assert "[genie3-opt] NOT ACTIVE: mode=exact cannot serve num-devices=2 — refused by name (exit 3), nothing ran: num-devices=2: " in err and "; run --mode off for the stock path" in err, err[-1500:]
    assert "--shard-id K --num-shards 2" in err and "] ACTIVE mode=" not in err and "DECLINED" not in err and "] RUN mode=" not in err and "[genie3-opt] refused: mode=exact cannot serve num-devices=2" in err, err[-1500:]
    for extra, name in (({"generation.inference.search": {"name": "beam", "width": 2}}, "inference.search"),
                        ({"generation.sampler.sampler.predict_sidechain": True}, "sampler.predict_sidechain"), ({"runtime.num_devices": 4}, "num-devices=4")):
        stack.reset_for_tests()
        req = _stubs.request(str(tmp_path), name=name.replace(".", "_") + ".yaml", extra=extra)
        o = str(tmp_path / name.replace(".", "_").replace("=", "_"))
        rc, man = design.run(req, o, "fast")
        assert rc == 3 and man["status"] == "refused" and man["refused_features"] == [name] and man["mode"] == "fast" and not os.path.exists(o), (name, man)
        e = capsys.readouterr().err
        assert f"[genie3-opt] NOT ACTIVE: mode=fast cannot serve {name} — refused by name (exit 3), nothing ran: {name}: " in e and "DECLINED" not in e and "] ACTIVE mode=" not in e, (name, e[-800:])
        stack.reset_for_tests()
        rc, man = design.run(req, o, "off")                                             # the stock route runs the same request (upstream's own generate)
        assert rc == 0 and man["mode"] == "off" and man["status"] == "ok", (name, man.get("status"), man.get("reason"))
        capsys.readouterr()
    both = design.kit_refusals({"generation": {"inference": {"search": {"name": "beam"}}, "sampler": {"sampler": {"predict_sequence": True, "predict_sidechain": True}}}}, num_devices=2)
    assert [r.split(":", 1)[0] for r in both] == ["inference.search", "sampler.predict_sidechain", "num-devices=2"]
    assert design.kit_refusals({"generation": {"dataset": {}}}) == [] and design.kit_refusals({"generation": {"dataset": {}}}, num_devices=1) == [] and design.kit_refusals({"runtime": {"num_devices": 1}}) == []
    stack.reset_for_tests()                                                                 # the sequence stage runs on a kit mode (the driver runs upstream's decode after its loop): not refused
    rc, man = design.run(_stubs.request(str(tmp_path), name="pseq.yaml", extra={"generation.sampler.sampler.predict_sequence": True}), str(tmp_path / "pseq"), "exact")
    assert rc == 0 and man["mode"] == "exact" and man["status"] == "ok" and man["request"]["sampler"]["sampler"]["predict_sequence"] is True, man
    assert not hasattr(design, "kit_declines") and not hasattr(report, "declined_line")     # no decline-to-stock route: a kit mode serves the request or refuses it by name


def test_dataset_shards_run_on_every_mode(tmp_path, monkeypatch, capsys):
    """upstream's --num-shards M --shard-id K on a kit mode: the driver line carries the flags, the pass expects exactly the shard's share of every
    problem's designs under upstream's names (<problem>_<i> over the shard's indices: ceil(n / M) per shard from K·ceil(n / M), clipped), the RUN
    line says `shard=K/M`, the manifest records it, and upstream's shard marker is written where its evaluate / status read it; a shard with no
    share runs nothing (exit 0, marker written); a shard index outside 0 <= K < M is refused by name (exit 3). The stock route passes the flags to
    upstream verbatim and expects the same share."""
    _stubs.box(str(tmp_path), monkeypatch)
    req = _stubs.request(str(tmp_path), n_sample=4)
    out = str(tmp_path / "k1")
    rc, man = design.run(req, out, "exact", num_shards=2, shard_id=1)
    err = capsys.readouterr().err
    assert rc == 0 and man["mode"] == "exact" and man["status"] == "ok" and man["shard"] == "1/2" and man["request"]["shard"] == "1/2", (man.get("status"), man.get("reason"), err[-1500:])
    assert man["request"]["designs_expected"] == 2 and man["request"]["names_expected"] == {"04_pdl1": ["04_pdl1_2", "04_pdl1_3"]} and man["request"]["design_indices"] == [2, 4]
    assert man["outputs"]["n_pdb"] == 2 and sorted(os.listdir(os.path.join(out, "04_pdl1", "pdbs"))) == ["04_pdl1_2.pdb", "04_pdl1_3.pdb"] and "incomplete" not in man
    T = json.load(open(os.path.join(out, "timings.json")))
    assert T["argv"][-4:] == ["--shard-id", "1", "--num-shards", "2"] and man["driver_pass"]["evidence"]["missing"] == []
    run_line = [l for l in err.splitlines() if l.startswith("[genie3-opt] RUN mode=exact")][0]
    assert run_line.endswith(" shard=1/2") and " designs=2 " in run_line, run_line
    assert man["shard_markers"] == [os.path.join(out, "04_pdl1", ".shard_markers", "generate_shard_1_of_2.done")] and os.path.isfile(man["shard_markers"][0])
    stack.reset_for_tests()
    out5 = str(tmp_path / "k5")
    rc, man = design.run(req, out5, "fast", num_shards=8, shard_id=5)                       # ceil(4 / 8) = 1 design per shard: shards 4..7 have no share
    assert rc == 0 and man["status"] == "ok" and man["shard"] == "5/8" and man["request"]["designs_expected"] == 0 and man["outputs"]["n_pdb"] == 0 and "driver_pass" not in man, man
    assert man["shard_markers"] == [os.path.join(out5, ".shard_markers", "generate_shard_5_of_8.done")] and os.path.isfile(man["shard_markers"][0]) and not os.path.exists(os.path.join(out5, "timings.json"))
    assert "[genie3-opt] NOTE shard 5/8 has no share of the request's designs (upstream's ceil slicing of n_sample 4 over 8 shards leaves shard 5 none): nothing runs; the shard marker is written" in capsys.readouterr().err
    stack.reset_for_tests()
    rc, man = design.run(req, str(tmp_path / "kx"), "exact", num_shards=2, shard_id=2)
    assert rc == 3 and man["status"] == "refused" and "--shard-id 2 --num-shards 2" in man["reason"] and not os.path.exists(str(tmp_path / "kx"))
    assert "[genie3-opt] NOT ACTIVE: --shard-id 2 --num-shards 2: the shard index is 0 <= K < M" in capsys.readouterr().err
    stack.reset_for_tests()                                                                 # the SAME shard run again into the same directory: upstream's resume rule — the marker says done, nothing runs, exit 0, on the kit route …
    t_first = os.path.getmtime(os.path.join(out, "timings.json"))
    rc, man = design.run(req, out, "exact", num_shards=2, shard_id=1)
    err = capsys.readouterr().err
    assert rc == 0 and man["status"] == "skipped" and man["skipped"] == "shard already complete (marker)" and "driver_pass" not in man and man["outputs"]["n_pdb"] == 2, man   # the done shard's designs as they stand
    assert man["shard_markers"] == [os.path.join(out, "04_pdl1", ".shard_markers", "generate_shard_1_of_2.done")] and os.path.getmtime(os.path.join(out, "timings.json")) == t_first
    assert "[genie3-opt] NOTE shard 1/2 already complete (marker " in err and "nothing runs, as upstream's generate skips a done shard; remove the marker to re-run it" in err and "] RUN mode=" not in err and "] EVIDENCE " not in err, err[-800:]
    stack.reset_for_tests()
    outo = str(tmp_path / "ko")
    rc, man = design.run(req, outo, "off", num_shards=2, shard_id=0)
    assert rc == 0 and man["mode"] == "off" and man["status"] == "ok" and man["shard"] == "0/2" and man["stock"]["flags"] == ["--shard-id", "0", "--num-shards", "2"], man
    assert man["request"]["names_expected"] == {"04_pdl1": ["04_pdl1_0", "04_pdl1_1"]} and man["outputs"]["n_pdb"] == 2 and "shard_markers" not in man   # the stock child writes upstream's own marker
    assert sorted(os.listdir(os.path.join(outo, "04_pdl1", "pdbs"))) == ["04_pdl1_0.pdb", "04_pdl1_1.pdb"] and os.path.isfile(os.path.join(outo, "04_pdl1", ".shard_markers", "generate_shard_0_of_2.done"))
    assert json.load(open(os.path.join(outo, "stub_argv.json")))[-4:] == ["--shard-id", "0", "--num-shards", "2"]
    stack.reset_for_tests()                                                                 # … and on the stock route: the pass names the skip (exit 0, status skipped) instead of judging upstream's own skip `incomplete`
    t_stub = os.path.getmtime(os.path.join(outo, "stub_argv.json"))
    rc, man = design.run(req, outo, "off", num_shards=2, shard_id=0)
    err = capsys.readouterr().err
    assert rc == 0 and man["status"] == "skipped" and "stock" not in man and "incomplete" not in man and os.path.getmtime(os.path.join(outo, "stub_argv.json")) == t_stub, man
    assert "[genie3-opt] NOTE shard 0/2 already complete (marker " in err and "[genie3-opt] incomplete:" not in err, err[-800:]
    assert design.shard_slice(4, 1, 2) == (2, 4) and design.shard_slice(4, 5, 8) == (4, 4) and design.shard_slice(5, 1, 3) == (2, 4) and design.shard_slice(4, None, None) == (0, 4) and design.shard_slice(4, 3, 1) == (0, 4)
    assert stack.driver_command(modes.resolve("exact"), "r.yaml", "/o", shard=(1, 2))[-4:] == ["--shard-id", "1", "--num-shards", "2"] and "--shard-id" not in stack.driver_command(modes.resolve("exact"), "r.yaml", "/o", shard=(0, 1))


def test_stock_line_reports_the_proof(tmp_path, monkeypatch, capsys):
    """After the stock pass the package prints the caller's proof as one line: STOCK proof=ok pinned=True files_checked=<n> git_head=<sha> modified=0 …"""
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=1), str(tmp_path / "s"), "off")
    out = capsys.readouterr()
    lines = [ln for ln in (out.out + out.err).splitlines() if " STOCK proof=" in ln]
    assert rc == 0 and len(lines) == 1, (rc, out.out[-800:], out.err[-800:])
    assert re.search(r"^\[genie3-opt\] STOCK proof=ok pinned=True files_checked=\d+ git_head=\S+ modified=0 forbidden_env=0 kit_modules=0 kit_dirs=0 autoload=False torch=False genie3=False$", lines[0]), lines[0]
    k = [ln for ln in (out.out + out.err).splitlines() if " KERNELS route=stock" in ln]
    assert len(k) == 1 and re.search(r"^\[genie3-opt\] KERNELS route=stock tf32_matmul=False cudnn_tf32=default float32_matmul_precision=default torch=\S+ overrides=none source=environment$", k[0]), k
    assert man["stock"]["kernels"]["tf32_matmul"] is False and man["stock"]["kernels"]["overrides"] == {}


def test_the_stock_child_never_inherits_a_tf32_override(tmp_path, monkeypatch, capsys):
    """A TF32 library override arriving from outside is stripped from the stock child (stock/PINS.json must_be_absent_prefixes) and listed; the
    KERNELS census reads the child's environment after the strip: tf32_matmul=False, overrides=none."""
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    env, stripped, absent = design.stock_environment()
    assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" not in env and "NVIDIA_TF32_OVERRIDE" not in env and {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"} <= set(stripped)
    assert design.kernels_census(env)["tf32_matmul"] is False and design.kernels_census({"NVIDIA_TF32_OVERRIDE": "1"})["tf32_matmul"] is True
    assert design.kernels_census({"NVIDIA_TF32_OVERRIDE": "1"})["overrides"] == {"NVIDIA_TF32_OVERRIDE": "1"} and design.kernels_census({"NVIDIA_TF32_OVERRIDE": "0"})["tf32_matmul"] is False
    rc, man = design.run(_stubs.request(str(tmp_path)), str(tmp_path / "o"), "off")
    assert rc == 0 and man["stock"]["kernels"]["tf32_matmul"] is False and "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in man["stock"]["env_stripped"]
    assert "overrides=none" in capsys.readouterr().err


def test_a_design_count_that_cannot_be_stated_up_front_is_named(tmp_path, monkeypatch, capsys):
    """A target request that selects no problems and whose dataset's problem list is not readable under the checkout: the pass cannot state its
    design count — `NOTE designs_expected=unknown (…)` before the pass, `RUN … designs=unknown`, `NOTE designs written=<n> (…)` after it, the
    reason in the manifest; never a silent count. Under shards a share of 0 takes the empty-shard branch whatever the problems (exit 0, marker at
    the output directory), never a driver launch judged `levers missing`."""
    _stubs.box(str(tmp_path), monkeypatch)
    req = _stubs.request(str(tmp_path), selections=None, n_sample=4, extra={"paths.dataset": "data/design/not_there"})
    out = str(tmp_path / "u")
    rc, man = design.run(req, out, "exact")
    err = capsys.readouterr().err
    assert rc == 0 and man["status"] == "ok" and man["request"]["designs_expected"] is None and man["request"]["problems"] == [], man
    assert man["designs_expected_unknown"].startswith("the request selects no problems and its dataset's problem list is not readable under the checkout (paths.dataset='data/design/not_there')")
    assert "[genie3-opt] NOTE designs_expected=unknown (the request selects no problems" in err and " designs=unknown seed=" in err and "[genie3-opt] NOTE designs written=" in err and "incomplete" not in man, err[-900:]
    stack.reset_for_tests()
    out7 = str(tmp_path / "u7")
    rc, man = design.run(req, out7, "exact", num_shards=8, shard_id=7)
    err = capsys.readouterr().err
    assert rc == 0 and man["status"] == "ok" and man["request"]["design_indices"] == [4, 4] and "driver_pass" not in man and man["shard_markers"] == [os.path.join(out7, ".shard_markers", "generate_shard_7_of_8.done")], man
    assert "[genie3-opt] NOTE shard 7/8 has no share of the request's designs" in err and "] RUN mode=" not in err, err[-600:]

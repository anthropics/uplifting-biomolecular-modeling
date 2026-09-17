"""The command line (exit codes, check's report with the pins REPORTED, Hydra's own flags riding with the overrides), the output listing
(the requested designs at each case's prefix), the stock archive and pins (PINS.json, check_pins on a pristine and on a patched checkout),
opt/forward/ kept free of compiled bytecode, and the exit rule's one home (report.py: the exit codes, the NOT ACTIVE / PARTIAL formatters with
their literal parts, the environment form of --allow-partial, and report.verdict's precedence) — locked here so every entry point (design,
design --pack, warm, check, the .pth route's _autoload, the stock caller's own process) prints the family line through the one formatter and
exits with the one code."""
import json
import argparse
import os
import re
import subprocess
import sys
import tarfile

import pytest

from rfdiffusion1_opt import _autoload, report, cli, design, modes, outputs, serve, stack, stock_cli, upstream_args
from rfdiffusion1_opt.tests._stubs import good_box, write_case_outputs, pkg_pythonpath

TREE = stack.tree_root()
P = "[rfdiffusion1-opt]"


# ------------------------------------------------------------------------------------------------------------------------- CLI
def test_cli_usage(capsys):
    assert cli.main([]) == 2
    with pytest.raises(SystemExit):
        cli.main(["serve", "--mode", "exact", "--pack", "4"])                        # no such verb: the packed line is `design --pack K`
    capsys.readouterr()
    assert cli.main(["design", "--mode", "exact"]) == 2                             # design with no target (no overrides): usage, exit 2 — nothing resolved, nothing printed as a design refusal
    io = capsys.readouterr()
    assert "no target: type upstream's overrides" in io.err and "NOT ACTIVE" not in io.err and io.out == ""
    with pytest.raises(SystemExit) as ex:                                            # the kit-invented cases-JSON form is gone: --input is no flag of design (argparse usage, exit 2)
        cli.main(["design", "--mode", "exact", "--input", os.path.join(stack.kit_dir("fast_inference"), "tests", "cases_public.json"), "--out_dir", "o"])
    assert ex.value.code == 2 and "unrecognized arguments: --input" in capsys.readouterr().err
    assert cli.main(["design", "--mode", "exact", "--pack", "2", "inference.num_designs=4"]) == 2     # the packed line: the same usage rule
    assert cli.VERBS == ("design", "check", "warm") and cli.EXIT_USAGE == 2


def test_hydra_flags_ride_with_the_overrides(monkeypatch, capsys, tmp_path):
    """Hydra's own command-line flags (--config-name / -cn, --config-path, --cfg, --resolve, --info [word], …) and upstream's KEY=VALUE / +KEY /
    ~KEY tokens wherever they stand among the kit's flags are upstream's arguments: cli.main lifts them out of argv before argparse sees them and
    puts them on the overrides (Hydra flags, then the lifted tokens, then the trailing positionals), so `--mode off` hands them to upstream
    verbatim and a kit mode refuses by name (exit 3, nothing runs) a request its line cannot serve — with or without --pack; a verb without overrides refuses them."""
    split = upstream_args.split_hydra_flags
    assert split(["design", "--mode", "off", "--config-name", "symmetry", "inference.input_pdb=x.pdb", "--cfg", "job", "--resolve"]) == (
        ["--config-name", "symmetry", "--cfg", "job", "--resolve"], ["design", "--mode", "off", "inference.input_pdb=x.pdb"])
    assert split(["-cn", "base", "--config-name=symmetry", "--info", "config", "--info", "x=1"]) == (["-cn", "base", "--config-name=symmetry", "--info", "config", "--info"], ["x=1"])   # `=` forms as typed; --info's optional word only when it is one of Hydra's
    assert split(["design", "--pack", "2", "--det", "1", "--dry-run", "warm", "--out_dir", "o"]) == ([], ["design", "--pack", "2", "--det", "1", "--dry-run", "warm", "--out_dir", "o"])   # the package's own flags are not Hydra's
    assert set(upstream_args.HYDRA_FLAGS) >= {"--config-name", "-cn", "--config-path", "-cp", "--config-dir", "-cd", "--cfg", "-c", "--package", "-p", "--resolve", "--run", "--multirun", "-m", "--shell-completion", "-sc", "--hydra-help", "--info", "-i"}
    assert "--help" not in upstream_args.HYDRA_FLAGS and "-h" not in upstream_args.HYDRA_FLAGS      # --help stays this package's
    good_box(monkeypatch)
    pdb = os.path.join(stack.kit_dir("fast_inference"), "inputs_public", "insulin_target.pdb")
    target = [f"inference.input_pdb={pdb}", "contigmap.contigs=[A1-115/0 80-80]", f"inference.output_prefix={tmp_path / 'run' / 'd'}", "inference.num_designs=2"]
    capsys.readouterr()
    assert cli.main(["design", "--mode", "off", "--config-name", "symmetry", *target, "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    cmd = out["stock_cmds"][0]
    assert cmd[cmd.index("--") + 1:cmd.index("--") + 3] == ["--config-name", "symmetry"] and cmd[cmd.index("--") + 3:cmd.index("--") + 3 + len(target)] == target   # upstream's command line, the flag first as typed
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "exact", *target, "--config-name", "symmetry", "--dry-run"]) == 3                   # a kit mode: another primary config is refused by name, exit 3, nothing runs
    io = capsys.readouterr()
    assert io.err.strip() == "[rfdiffusion1-opt] NOT ACTIVE: mode=exact cannot serve primary config symmetry.yaml [--config-name symmetry]: the resident driver composes config/inference/base.yaml (symmetry.yaml is symmetric oligomer design) — refused by name, nothing ran; `--mode off` runs this request on upstream's command line (scripts/run_inference.py)" and io.out == ""
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "exact", "--pack", "2", *target, "-cn", "symmetry", "--dry-run"]) == 3                     # the packed line refuses alike (design's one refusal)
    io = capsys.readouterr()
    assert "[rfdiffusion1-opt] NOT ACTIVE: mode=exact cannot serve primary config symmetry.yaml [-cn symmetry]: " in io.err and io.err.count("NOT ACTIVE") == 1 and io.out == ""
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "exact", *target, "--config-name", "base", "--dry-run"]) == 0                       # naming base.yaml itself is served: the resident driver composes it (the flag is left out of the driver line)
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "exact" and "--config-name" not in " ".join(out["driver_cmd"])
    for orphan in (["--config-name", "symmetry"], ["inference.num_designs=2"]):
        with pytest.raises(SystemExit):
            cli.main(orphan)                                                                                                     # no verb: upstream's arguments belong to design / check / warm
        assert "belong to design / check / warm" in capsys.readouterr().err
    # upstream's override tokens anywhere among the kit's flags (Hydra takes them anywhere): lifted onto the overrides in their typed order; a kit flag's own value stays with it
    stack.reset_for_tests()
    mixed = ["design", target[0], "--mode", "off", target[1], "--det", "0", "--dry-run", target[2], "+inference.extra=1", "~ppi.hotspot_res", target[3]]
    assert cli.main(mixed) == 0
    cmd = json.loads(capsys.readouterr().out)["stock_cmds"][0]
    assert cmd[cmd.index("--") + 1:cmd.index("--") + 7] == [target[0], target[1], target[2], "+inference.extra=1", "~ppi.hotspot_res", target[3]]   # typed order kept; --det's `0` is the flag's value, not an override
    seen = {}
    from rfdiffusion1_opt import warm as _warm
    monkeypatch.setattr(_warm, "run", lambda mode, out_dir, overrides: seen.update(mode=mode, out_dir=out_dir, overrides=overrides) or {"status": "ok", "rc": 0})
    assert cli.main(["warm", "x=1", "--mode", "exact", "--out_dir", "a=b", "y=2"]) == 0 and seen == {"mode": "exact", "out_dir": "a=b", "overrides": ["x=1", "y=2"]}; capsys.readouterr()   # a kit flag's value stays with it even when it looks like KEY=VALUE; the overrides lifted in typed order
    assert set(cli.VALUE_FLAGS) == {"--mode", "--pack", "--det", "--out_dir"} and not {"--without", "--timeout", "--input", "--tag", "--num-designs", "--startnum"} & set(cli.VALUE_FLAGS)
    assert all(upstream_args.is_override(t) for t in ("k=v", "a.b=[1,2]", "+k=1", "++k=1", "~k", "~k=2", "inference.output_prefix=out/x")) and not any(upstream_args.is_override(t) for t in ("--mode", "cases.json", "-cn", "exact"))


def test_cli_check_codes(monkeypatch, capsys):
    """check: 0 where the mode would activate, 3 (EXIT_NOT_ACTIVE, the NOT ACTIVE line) where the box would refuse or the mode is not one —
    the same code the run would exit with; check has no partial state (the levers' evidence exists only after a pass)."""
    good_box(monkeypatch)
    assert cli.main(["check", "--mode", "exact", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["report"]["dry_run"] and rep["report"]["would_refuse"] == [] and any(r["mode"] == "exact" for r in rep["mode_table"])
    assert cli.main(["check", "--mode", "fast"]) == 0                             # the fast row (K + T2 + K2 + TF32) resolves on the good box
    capsys.readouterr()
    assert cli.main(["check"]) == 0                                                 # the default mode (fast) resolves on the good box
    assert "[rfdiffusion1-opt] DRY-RUN mode=fast attach=driver tier=2 " in capsys.readouterr().err
    assert cli.main(["check", "--mode", "exact", "diffuser.partial_T=10", "potentials.guide_scale=2"]) == 0    # partial diffusion, guiding potentials: served by the kit line — its own dry run
    err = capsys.readouterr().err
    assert "[rfdiffusion1-opt] DRY-RUN mode=exact attach=driver tier=1 " in err and "NOT ACTIVE" not in err
    assert cli.main(["check", "--mode", "exact", "inference.symmetry=C3", "diffuser.partial_T=10"]) == 3    # what the kit line cannot serve: check refuses as design would (the same NOT ACTIVE line), exit 3
    err = capsys.readouterr().err
    assert err.strip() == ("[rfdiffusion1-opt] NOT ACTIVE: mode=exact cannot serve symmetric oligomers [inference.symmetry=C3]: the resident driver restates SelfConditioning.sample_step "
                           "without upstream's symmetry branches and composes config/inference/base.yaml, not symmetry.yaml — refused by name, nothing ran; `--mode off` runs this request on upstream's command line (scripts/run_inference.py)")
    assert cli.main(["check", "--mode", "off", "inference.symmetry=C3", "diffuser.partial_T=10"]) == 0 and "NOT ACTIVE" not in capsys.readouterr().err   # off never refuses a request: upstream's command line takes every key
    with pytest.raises(SystemExit):
        cli.main(["check", "--mode", "exact", "--family", "x"])                     # no such flag (argparse, exit 2)
    good_box(monkeypatch, gpu={"name": None, "mem_gib": None, "cc": None, "sm": None, "source": None})
    assert cli.main(["check", "--mode", "exact"]) == report.EXIT_NOT_ACTIVE == 3
    err = capsys.readouterr().err
    assert "[rfdiffusion1-opt] NOT ACTIVE:" in err and "no CUDA device" in err and report.NOT_ACTIVE_RE.search(err)
    assert not hasattr(cli.build_parser().parse_args(["check"]), "allow_partial")  # nothing to allow: no pass, no partial state


def test_cli_design_refusal_code(monkeypatch, tmp_path):
    good_box(monkeypatch)
    t = [f"inference.input_pdb={os.path.join(stack.kit_dir('fast_inference'), 'inputs_public', 'insulin_target.pdb')}", "contigmap.contigs=[A1-115/0 80-80]", f"inference.output_prefix={tmp_path / 'o' / 'des'}"]
    assert cli.main(["design", "--mode", "fast", *t]) == 3
    assert cli.main(["design", "--mode", "exact", *t, "--dry-run"]) == 0


def test_module_entry_point():
    env = dict(os.environ, PYTHONPATH=pkg_pythonpath())
    p = subprocess.run([sys.executable, "-m", "rfdiffusion1_opt", "--version"], env=env, capture_output=True, text=True)
    assert p.returncode == 0 and "rfdiffusion1_opt 0.7.3" in p.stdout
    p = subprocess.run([sys.executable, "-m", "rfdiffusion1_opt", "check", "--mode", "fast"], env=env, capture_output=True, text=True)
    assert p.returncode == 3 and p.stderr.startswith("[rfdiffusion1-opt] NOT ACTIVE: ") and p.stderr.count("NOT ACTIVE") == 1 and "(mode=fast)" in p.stderr   # this box's own refusal (no checkout / no CUDA device here), the mode named on the tail


def test_design_registers_the_exit_tally_in_its_own_process(tmp_path):
    """`design` prints the EXIT tally at interpreter exit of a process that RUNS a pass (the registration sits at the run line); a refusal
    before anything ran (exit 3) or a usage error (exit 2) prints none, and neither does `check`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RFD_", "RFDIFFUSION1", "WEIGHTS"))}
    env["PYTHONPATH"] = pkg_pythonpath()
    t = [f"inference.input_pdb={os.path.join(stack.kit_dir('fast_inference'), 'inputs_public', 'insulin_target.pdb')}", "contigmap.contigs=[A1-115/0 80-80]"]
    env["RFD_ROOT"] = str(tmp_path / "no_checkout")                                    # no checkout, no GPU: refused (rc 3), no tally
    r = subprocess.run([sys.executable, "-m", "rfdiffusion1_opt", "design", "--mode", "exact", *t, f"inference.output_prefix={tmp_path / 'o' / 'des'}"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "[rfdiffusion1-opt] NOT ACTIVE:" in r.stderr and "[rfdiffusion1-opt] EXIT" not in r.stderr
    r = subprocess.run([sys.executable, "-m", "rfdiffusion1_opt", "design", "--mode", "fast",
                        "inference.input_pdb=x.pdb", "contigmap.contigs=[8-8]"], env=env, capture_output=True, text=True)
    assert r.returncode == 3 and r.stderr.startswith("[rfdiffusion1-opt] NOT ACTIVE:") and "[rfdiffusion1-opt] EXIT" not in r.stderr   # refused on this box (no checkout): no tally
    root = tmp_path / "rfd"; (root / "scripts").mkdir(parents=True); (tmp_path / "w").mkdir()
    (root / "scripts" / "run_inference.py").write_text("import sys\nsys.exit(0)\n")      # a stock entry that designs nothing: the pass runs (rc 1, outputs short), the tally prints
    env.update(RFD_ROOT=str(root), WEIGHTS=str(tmp_path / "w"))
    r = subprocess.run([sys.executable, "-m", "rfdiffusion1_opt", "design", "--mode", "off", *t, f"inference.output_prefix={tmp_path / 'o2' / 'des'}", "inference.num_designs=1"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "[rfdiffusion1-opt] EXIT pid=" in r.stderr and "no driver pass ran" in r.stderr, r.stderr[-600:]
    r = subprocess.run([sys.executable, "-m", "rfdiffusion1_opt", "check", "--mode", "exact"], env=env, capture_output=True, text=True)
    assert "[rfdiffusion1-opt] EXIT" not in r.stderr                                    # the tally belongs to a design pass, not to check


def test_cli_check_json_carries_the_contract_keys(monkeypatch, capsys):
    good_box(monkeypatch)
    assert cli.main(["check", "--mode", "exact", "--json"]) == 0
    io = capsys.readouterr()
    full = json.loads(io.out)
    rep = full["report"]
    for k in ("levers_planned", "levers_applied", "levers_fallback", "levers_unavailable", "partial", "env", "line", "gpu", "stack", "tools", "det", "core", "served"):
        assert k in rep, k
    assert "pins" not in rep and "forced" not in rep                                 # the pins are check's REPORT (below), never a key or a gate of activation
    assert rep["levers_applied"] == [] and rep["partial"] is False and rep["gpu"]["class"] == "H100" and rep["det"] == 0
    pins_lines = [l for l in io.err.splitlines() if l.startswith("[rfdiffusion1-opt] PINS ")]
    assert set(full) == {"report", "pins", "mode_table"} and full["pins"]["pinned"] is True and full["pins"]["findings"] == [] and full["pins"]["lines"] == pins_lines
    assert pins_lines == ["[rfdiffusion1-opt] PINS upstream=https://github.com/RosettaCommons/RFdiffusion@86507b65 stack=" + stack.pins()["pinned_stack"]["id"] + " pinned=True"]
    assert io.err.splitlines()[0].startswith("[rfdiffusion1-opt] DRY-RUN mode=exact attach=driver tier=1 ") and io.err.splitlines()[1] == pins_lines[0]   # the activation line, then the pins report
    # a patched checkout: the finding is printed and carried, the exit code stays the activation's (0) — reported, never refused
    stack.reset_for_tests()
    good_box(monkeypatch, pins_bad=["rfdiffusion: 1 file(s) differ from the pinned commit"])
    assert cli.main(["check", "--mode", "exact", "--json"]) == 0
    io = capsys.readouterr()
    assert [l for l in io.err.splitlines() if " PINS " in l] == [f"[rfdiffusion1-opt] PINS upstream=https://github.com/RosettaCommons/RFdiffusion@86507b65 stack={stack.pins()['pinned_stack']['id']} pinned=False",
                                                                 "[rfdiffusion1-opt] PINS finding: rfdiffusion: 1 file(s) differ from the pinned commit"]
    assert json.loads(io.out)["pins"]["pinned"] is False and "NOT ACTIVE" not in io.err
    stack.reset_for_tests()
    assert cli.main(["check", "--mode", "off"]) == 0 and capsys.readouterr().err.count("[rfdiffusion1-opt] PINS ") == 2   # every check prints the report (--json or not)
    run_sh = [l for l in open(os.path.join(TREE, "run.sh"), encoding="utf-8").read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    routes = run_sh[:next(i for i, l in enumerate(run_sh) if l.startswith("exec python -m rfdiffusion1_opt ")) + 1]   # the routes end in exec; `install` is the block after them
    assert not any("check_pins" in l for l in routes)                                  # the routes run no pin gate either; stock/check_pins.py is the report's producer (stack.check_pins), the install step's check and a hand tool
    assert [l.split("||")[0].strip() for l in run_sh if "check_pins" in l] == ['python -I "$HERE/stock/check_pins.py"']   # its one caller in the script: `run.sh install`


# --------------------------------------------------------------------------------------------------------------------- outputs
def test_list_outputs_counts_the_requested_designs_at_each_prefix(tmp_path):
    """outputs.list_outputs(cases, out_dir): per case, which of the REQUESTED indices (startnum .. startnum+num_designs-1) are on disk at the
    case's prefix — `<out_dir>/<name>/des` on the cases form, the case's own `prefix` on the hydra form — with the trajectory names; the
    totals feed the exit rule (n_pdb vs expected). No digests: the kit counts its own outputs, it never judges equality between two trees."""
    out = tmp_path / "o"
    write_case_outputs(str(out), "case1", [0, 1], traj=True, seed=b"s")
    write_case_outputs(str(out), "case2", [5], traj=False)
    cases = [{"name": "case1", "num_designs": 2}, {"name": "case2", "num_designs": 2, "startnum": 5}, {"name": "missing", "num_designs": 1}]
    lo = outputs.list_outputs(cases, str(out))
    assert (lo["n_pdb"], lo["n_trb"], lo["n_traj"], lo["expected"]) == (3, 3, 4, 5) and set(lo) == {"n_pdb", "n_trb", "n_traj", "expected", "cases"}
    c1 = lo["cases"]["case1"]
    assert c1 == {"prefix": str(out / "case1" / "des"), "indices": [0, 1], "n_pdb": 2, "n_trb": 2, "n_traj": 4,
                  "designs": {"0": {"pdb": True, "trb": True, "traj": ["des_0_Xt-1_traj.pdb", "des_0_pX0_traj.pdb"]}, "1": {"pdb": True, "trb": True, "traj": ["des_1_Xt-1_traj.pdb", "des_1_pX0_traj.pdb"]}}}
    assert lo["cases"]["case2"]["designs"] == {"5": {"pdb": True, "trb": True, "traj": []}, "6": {"pdb": False, "trb": False, "traj": []}} and lo["cases"]["case2"]["n_pdb"] == 1
    assert lo["cases"]["missing"] == {"prefix": str(out / "missing" / "des"), "indices": [0], "designs": {"0": {"pdb": False, "trb": False, "traj": []}}, "n_pdb": 0, "n_trb": 0, "n_traj": 0}
    write_case_outputs(str(out), "case1", [7], traj=True)                                  # a design outside the request is not counted (another pass's, or upstream's earlier run)
    assert outputs.list_outputs(cases[:1], str(out))["n_pdb"] == 2
    h = {"name": "binder", "num_designs": 2, "startnum": 3, "prefix": str(tmp_path / "run" / "samples" / "binder")}   # the hydra form: upstream's own layout at the typed prefix
    write_case_outputs("", "", [3, 4], traj=True, prefix=h["prefix"])
    lh = outputs.list_outputs([h], str(tmp_path / "anywhere"))                              # the run directory plays no part in where a prefixed case's designs are
    assert lh["n_pdb"] == 2 and lh["n_traj"] == 4 and lh["cases"]["binder"]["prefix"] == h["prefix"] and lh["cases"]["binder"]["designs"]["3"]["traj"] == ["binder_3_Xt-1_traj.pdb", "binder_3_pX0_traj.pdb"]
    assert sorted(os.listdir(tmp_path / "run" / "samples")) == ["binder_3.pdb", "binder_3.trb", "binder_4.pdb", "binder_4.trb", "traj"]
    assert outputs.case_prefix(h, "/x") == h["prefix"] and outputs.case_prefix({"name": "c"}, str(out)) == str(out / "c" / "des") and outputs.case_indices(h) == [3, 4]
    for gone in ("compare", "compare_case", "trb_diff", "recipe_of", "TRB_EXCLUDES", "sha256_file"):
        assert not hasattr(outputs, gone), gone                                             # no equality verdict, no digest, no recipe reader in this tree


# ------------------------------------------------------------------------------------------------------------- stock & pins
def test_archive_member_count_matches_the_recipe():
    """stock/rfdiffusion-86507b65.tar.gz (the vendored archive, tracked in this repo) unpacks to the file count PINS.json archive_recipe
    declares."""
    pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
    with tarfile.open(os.path.join(TREE, "stock", "rfdiffusion-86507b65.tar.gz")) as tf:
        members = [m for m in tf if m.isfile()]
    assert len(members) == pins["archive_recipe"]["members"] == 75


def test_check_pins_pristine_and_patched(tmp_path):
    with tarfile.open(os.path.join(TREE, "stock", "rfdiffusion-86507b65.tar.gz")) as tf:
        tf.extractall(tmp_path)
    root = tmp_path / "rfdiffusion-86507b65"
    cp = os.path.join(TREE, "stock", "check_pins.py")
    p = subprocess.run([sys.executable, "-I", cp, "--no-stack", "--rfd-root", str(root)], capture_output=True, text=True)
    assert p.returncode == 0 and "pinned (checkout" in p.stdout and "75 files ==" in p.stdout
    with open(root / "rfdiffusion" / "util_module.py", "a") as fh:
        fh.write("\n# patched\n")
    p = subprocess.run([sys.executable, "-I", cp, "--no-stack", "--rfd-root", str(root)], capture_output=True, text=True)
    assert p.returncode == 3 and "a patched checkout is not stock" in p.stderr and "rfdiffusion/util_module.py" in p.stderr
    os.remove(root / "scripts" / "run_inference.py")
    p = subprocess.run([sys.executable, "-I", cp, "--no-stack", "--rfd-root", str(root)], capture_output=True, text=True)
    assert p.returncode == 3 and "missing" in p.stderr
    p = subprocess.run([sys.executable, "-I", cp, "--no-stack", "--rfd-root", str(tmp_path / "nowhere")], capture_output=True, text=True)
    assert p.returncode == 3 and "no checkout found" in p.stderr


def _check_pins_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_pins_under_test", os.path.join(TREE, "stock", "check_pins.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_pins_stack_check(monkeypatch, tmp_path):
    """The torch / dgl freeze check: the versions pass, another version is named, an absent distribution is named."""
    cp = _check_pins_module()
    with tarfile.open(os.path.join(TREE, "stock", "rfdiffusion-86507b65.tar.gz")) as tf:
        tf.extractall(tmp_path)
    root = str(tmp_path / "rfdiffusion-86507b65")
    pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
    want = {k: v for k, v in pins["pinned_stack"]["stack_check"].items() if k != "note"}
    assert set(want) == {"torch", "dgl"} and want["torch"] == pins["pinned_stack"]["torch"] and want["dgl"] == pins["pinned_stack"]["dgl"]
    monkeypatch.setattr(cp.md, "version", lambda name: {"torch": "2.4.0+cu121", "dgl": want["dgl"]}[name])      # local tags stripped on both sides
    bad, detail = cp.check(pins, root, stack=True)
    assert bad == [] and all(d["ok"] for d in detail["stack"].values())
    monkeypatch.setattr(cp.md, "version", lambda name: {"torch": "2.5.1", "dgl": want["dgl"]}[name])
    bad, detail = cp.check(pins, root, stack=True)
    assert any("torch" in b and "2.5.1" in b for b in bad) and not detail["stack"]["torch"]["ok"] and detail["stack"]["dgl"]["ok"]

    def missing(name):
        raise cp.md.PackageNotFoundError(name)
    monkeypatch.setattr(cp.md, "version", missing)
    bad, detail = cp.check(pins, root, stack=True)
    assert any("torch" in b for b in bad) and any("dgl" in b for b in bad)
    bad, _ = cp.check(pins, root, stack=False)
    assert bad == []


def test_pins_json_values():
    pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
    up = pins["upstream"]["rfdiffusion"]
    assert up["commit"] == "86507b6538f51fce57b5a72477165f03999ed7ae" and up["tag"] is None and up["version"] == "1.1.0"
    assert pins["weights"]["Complex_base_ckpt.pt"]["sha256"] == "76e4e260aefee3b582bd76b77ab95d2592e64f00c51bf344968ab9239f3250bc"
    assert "image" not in pins["pinned_stack"] and "image_note" not in pins["pinned_stack"]     # no image id is pinned: the version pins and the installer are the stack
    assert pins["pinned_stack"]["torch"] == "2.4.0" and set(pins["pinned_stack"]["stack_check"]) == {"torch", "dgl", "note"}
    assert pins["weights"]["checkpoint"] == "Complex_base_ckpt.pt" and pins["weights"]["Complex_base_ckpt.pt"]["role"].startswith("the pinned checkpoint")
    from rfdiffusion1_opt import manifest
    assert manifest.default_checkpoint() == "Complex_base_ckpt.pt"
    se = pins["stock_environment"]
    assert se["must_be_absent_prefixes"] == ["RFD_", "RFDIFFUSION1_", "ALLOW_ANY_GPU"]                                 # the stock arm is stripped of the kit namespace ONLY; torch / CUDA / caller environment (TF32 overrides, CUDA_MPS_*, …) passes as set
    assert set(stack.DROP_ENV_NAMES) >= {"NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"}                     # the kit's own driver child drops the TF32 overrides (its line declares the numerics), named in env_dropped
    assert "extra_for_exact" not in pins["pinned_stack"]
    assert se["allowed_exceptions"] == ["RFD_ROOT", "WEIGHTS"]
    for name in ("Complex_base_ckpt.pt", "Base_ckpt.pt"):                                                          # what `run.sh install --weights` fetches and checks (weights.py): an https URL, a size, a 64-hex digest per file
        w = pins["weights"][name]
        assert w["url_kit_installer"].startswith("https://files.ipd.uw.edu/pub/RFdiffusion/") and w["url_kit_installer"].endswith("/" + name)
        assert isinstance(w["size_bytes"], int) and len(w["sha256"]) == 64 and int(w["sha256"], 16) >= 0
    yaml = open(os.path.join(TREE, "stock", "src", "config", "inference", "base.yaml")).read().splitlines()
    for key, line in pins["cli_defaults"]["lines"].items():
        assert yaml[line - 1].strip().startswith(key.split(".")[-1] + ":"), (key, yaml[line - 1])


# ------------------------------------------------------------------------------------------------------------ kit bytes
def test_no_compiled_bytecode_under_opt_forward():
    """No compiled bytecode or __pycache__ directory anywhere under opt/forward/."""
    for root, dns, files in os.walk(os.path.join(TREE, "opt", "forward")):
        assert "__pycache__" not in dns, root
        assert not any(f.endswith((".pyc", ".pyo")) for f in files), (root, files)


def test_run_sh_usage_names_the_parsers_flags():
    """run.sh's usage lines carry every option of the package's parsers for the verbs they front, and no option the parsers do not define."""
    import re
    from rfdiffusion1_opt import cli
    usage = {}
    for line in open(os.path.join(TREE, "run.sh"), encoding="utf-8"):
        m = re.match(r"#   run\.sh (\w+) (.*)", line)
        if m:
            usage[m.group(1)] = set(re.findall(r"--[a-z][a-z0-9_-]*", m.group(2)))
    ap = cli.build_parser()
    subs = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction)).choices
    assert set(subs) == set(cli.VERBS) and set(usage) == set(subs) | {"install"}                        # the package's verbs + `install`, the script's own step (pip: the core, then the kit; the pin check; --weights) — the package cannot install itself
    assert usage.pop("install") == {"--weights"}
    wrapper_only = {"--config"}                                                                       # run.sh's own option (sources configs/<cfg>.env)
    for verb, sp in subs.items():
        parser_flags = {o for a in sp._actions if a.help != argparse.SUPPRESS for o in a.option_strings if o.startswith("--") and o != "--help"}   # user-facing options: a suppressed one (serve --composition: compositions are not modes) is in no usage line
        assert usage[verb] - wrapper_only == parser_flags, (verb, sorted(usage[verb] - wrapper_only - parser_flags), sorted(parser_flags - usage[verb]))


# ------------------------------------------------------------------------------------------------------------ the exit rule's one home (report.py)
def test_exit_codes_have_one_home():
    assert (report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE) == (0, 1, 2, 3)
    for mod in (cli, design, serve):                                                    # re-exports, never a second tuple
        assert (mod.EXIT_OK, mod.EXIT_FAIL, mod.EXIT_USAGE, mod.EXIT_NOT_ACTIVE) == (0, 1, 2, 3)
        assert mod.EXIT_NOT_ACTIVE is report.EXIT_NOT_ACTIVE
    assert stock_cli.EXIT_NOT_STOCK == report.EXIT_NOT_ACTIVE                            # the stock process's own code (import-light, its literal locked here)
    assert _autoload.EXIT_NOT_ACTIVE == report.EXIT_NOT_ACTIVE                           # the .pth route (imports nothing else at interpreter start)
    assert _autoload.TAG == report.TAG
    from opt_core import report as core_report, stock_proof as core_stock_proof         # the family codes and the tag grammar are the core's; the package's literals equal them
    assert (core_report.EXIT_OK, core_report.EXIT_FAIL, core_report.EXIT_USAGE, core_report.EXIT_NOT_ACTIVE) == (report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE)
    assert core_stock_proof.EXIT_NOT_STOCK == stock_cli.EXIT_NOT_STOCK
    assert core_report.prefix(report.TAG) == report.PREFIX
    for src in (open(design.__file__).read(), open(serve.__file__).read(), open(cli.__file__).read()):
        assert "EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = " not in src              # defined once, in report.py


def test_not_active_line_is_the_one_refusal_form():
    assert report.not_active_line("x") == f"{P} NOT ACTIVE: x"
    assert report.NOT_ACTIVE_RE.match(report.not_active_line("mode exact cannot attach")).group("reason") == "mode exact cannot attach"
    assert report.activation_line({"active": False, "reason": "no CUDA device", "mode": "exact"}) == f"{P} NOT ACTIVE: no CUDA device (mode=exact)"
    src = open(_autoload.__file__).read()                                                # the .pth route prints the same form with the same prefix, verbatim
    assert f"{P} NOT ACTIVE: " in src and src.count("NOT ACTIVE") >= 2


def test_partial_lines_lock_the_family_grammar():
    line = report.not_active_partial_line(["P", "E_einsum", "W1"], "pass t: levers without their evidence line ['P', 'E_einsum', 'W1']; forbidden lines 0")
    assert line == (f"{P} NOT ACTIVE: partial activation — P,E_einsum,W1: pass t: levers without their evidence line ['P', 'E_einsum', 'W1']; "
                    "forbidden lines 0; exit 3")
    assert line.startswith(f"{P} NOT ACTIVE: partial activation — ") and line.endswith("; exit 3") and "allow-partial" not in line   # no opt-out exists: a partial pass exits 3, outputs kept
    m = report.NOT_ACTIVE_PARTIAL_RE.match(line)
    assert m and m.group("code") == "3" and m.group("detail").startswith("P,E_einsum,W1: ")
    from opt_core import report as core_report                                            # the core's family grammar, minus the fleet's former opt-out tail
    detail = report.partial_detail(["P", "E_einsum", "W1"], "pass t: levers without their evidence line ['P', 'E_einsum', 'W1']; forbidden lines 0")
    assert core_report.PARTIAL_REFUSED.format(prefix=report.PREFIX, detail=detail).startswith(line)
    assert report.not_active_line("x") == core_report.not_active_line(report.TAG, "x")
    assert report.partial_detail([], "forbidden lines 2") == "none: forbidden lines 2"
    assert report.NOT_ACTIVE_RE.match(line)
    for gone in ("ENV_ALLOW_PARTIAL", "ALLOW_PARTIAL_DOC", "allow_partial_env", "partial_allowed_line", "PARTIAL_ALLOWED_FMT", "PARTIAL_ALLOWED_RE"):
        assert not hasattr(report, gone), gone


def test_no_partial_opt_out():
    """`--allow-partial` and RFDIFFUSION1_OPT_ALLOW_PARTIAL are gone from every entry point: argparse refuses the flag (exit 2), the API takes no
    such keyword, and the environment name means nothing (it is dropped from every child with the package's namespace, stack.DROP_ENV_PREFIXES)."""
    import inspect
    from rfdiffusion1_opt import design, serve, warm
    for verb, args in (("design", ["inference.input_pdb=x.pdb"]), ("design", ["--pack", "2", "inference.input_pdb=x.pdb"]), ("warm", ["--out_dir", "o"])):
        with pytest.raises(SystemExit) as ex:
            cli.build_parser().parse_args([verb] + args + ["--allow-partial"])
        assert ex.value.code == 2, verb
    for fn in (design.run, serve.run, warm.run, report.verdict):
        assert "allow_partial" not in inspect.signature(fn).parameters, fn
    assert any("RFDIFFUSION1_OPT_ALLOW_PARTIAL".startswith(p) for p in stack.DROP_ENV_PREFIXES) and not hasattr(cli, "_allow_partial")


def test_verdict_precedence():
    ok = report.verdict(n_pdb=4, expected=4)
    assert (ok["status"], ok["exit_code"], ok["reason"], ok["partial"], ok["incomplete"]) == ("ok", 0, None, [], None)
    p = report.verdict(partial=["W1"], partial_reason="worker w0: x", n_pdb=4, expected=4)
    assert (p["status"], p["exit_code"], p["reason"]) == ("partial", 3, "W1: worker w0: x") and "allow_partial" not in p
    inc = report.verdict(partial=["W1"], partial_reason="x", n_pdb=3, expected=4)                           # incomplete wins over partial, the partial record kept
    assert (inc["status"], inc["exit_code"], inc["incomplete"], inc["partial"]) == ("incomplete", 1, "3/4", ["W1"])
    f = report.verdict(failed="worker w0: rc=7", partial=["W1"], n_pdb=0, expected=4)                       # a failed process wins over incomplete and partial
    assert (f["status"], f["exit_code"], f["reason"], f["partial"], f["incomplete"]) == ("failed", 1, "worker w0: rc=7", ["W1"], "0/4")
    na = report.verdict(not_active="case c: the stock process's environment proof is not ok", failed="rc 1", n_pdb=0, expected=2)
    assert (na["status"], na["exit_code"]) == ("failed", 3)                                                # not the stock line: the activation code, above a failure
    assert report.verdict(partial=[], partial_reason="forbidden lines 1", n_pdb=1, expected=1)["exit_code"] == 3   # forbidden lines alone are partial


def test_emit_verdict_prints_the_one_line_per_status(capsys):
    report.emit_verdict(report.verdict(partial=["W1"], partial_reason="w0: x", n_pdb=1, expected=1))
    report.emit_verdict(report.verdict(partial=["W1"], partial_reason="w0: x", n_pdb=0, expected=1))
    report.emit_verdict(report.verdict(n_pdb=1, expected=1))
    lines = capsys.readouterr().err.splitlines()
    assert lines == [f"{P} NOT ACTIVE: partial activation — W1: w0: x; exit 3",
                     f"{P} PARTIAL recorded: W1: w0: x (status incomplete, exit 1)"]


def test_numerics_lines(tmp_path):
    import json as _json
    from rfdiffusion1_opt.modes import numerics
    p = tmp_path / "t_timings.json"
    prec = {"param_dtype": "torch.float32", "allow_tf32_matmul": False, "cudnn_tf32": False, "autocast_enabled": False}
    p.write_text(_json.dumps([{"case": "a", "precision": prec}, {"case": "b", "precision": prec}]))
    rec = report.numerics_from_timings(str(p))
    assert rec == prec
    assert report.numerics_line("t", rec, numerics("exact", rec)) == f"{P} NUMERICS pass=t numerics=fp32_strict numerics_source=torch matmul=highest matmul_tf32=off cudnn_tf32=off autocast=none"
    bad = dict(prec, cudnn_tf32=True)
    assert report.numerics_line("t", bad, numerics("exact", bad)).endswith("cudnn_tf32=on autocast=none numerics_mismatch=cudnn_tf32")
    p.write_text("{")
    assert "numerics_source=unreadable" in report.numerics_line("t", report.numerics_from_timings(str(p)), None)
    assert "error" in report.numerics_from_timings(str(tmp_path / "absent.json"))

"""The design surface (upstream sample_sequences.py's own arguments with its names and defaults, --input/--out, --seed / --batch_size inputs,
the kit's --mode / --det), the kit-tier words exiting 3 by name (the default mode included), the proven stock child on a box without the
model stack, check's report, the exit codes."""
import ast
import json
import os
import re
import subprocess
import sys

import pytest

from esm_if1_opt import batched, det, inputs, outputs, report, settings, stack

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))                                        # esm_if1/opt
CORE = os.path.join(os.path.dirname(os.path.dirname(OPT)), "common", "opt_core")


def run_cli(args, env=None, cwd=None):
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join([OPT, CORE] + [p for p in e.get("PYTHONPATH", "").split(os.pathsep) if p])
    e.pop("ESM_IF1_OPT", None)
    e.update(env or {})
    r = subprocess.run([sys.executable, "-m", "esm_if1_opt"] + list(args), env=e, cwd=cwd, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def no_model_stack():
    return not (stack.esm_version() and stack.gates.dist_version("torch"))


def script_defaults(script):
    """The ``add_argument`` / ``set_defaults`` defaults of a Python script's argparse block, read from its source (``ast``: no import, no
    execution): ``{dest: default}`` with ``dest`` derived as argparse does (leading dashes dropped, ``-`` -> ``_``)."""
    with open(script, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=script)
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr == "set_defaults":
            for kw in node.keywords:
                out[kw.arg] = ast.literal_eval(kw.value)
        elif node.func.attr == "add_argument" and node.args:
            names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if not names:
                continue
            kws = {kw.arg: kw.value for kw in node.keywords}
            dest = ast.literal_eval(kws["dest"]) if "dest" in kws else max(names, key=len).lstrip("-").replace("-", "_")
            if "default" in kws:
                out[dest] = ast.literal_eval(kws["default"])
            elif "action" in kws and ast.literal_eval(kws["action"]) == "store_true":
                out.setdefault(dest, False)
            else:
                out.setdefault(dest, None)
    return out


def test_sampling_options_are_the_scripts_own_and_seed_batch_are_inputs():
    d = script_defaults(os.path.join(stack.tree_home(), stack.SCRIPT_REL))          # upstream's example script in the extracted archive: read, never run
    assert {k: d[k] for k in ("chain", "temperature", "num_samples")} == {"chain": None, "temperature": 1.0, "num_samples": 1}
    ns = settings.parser().parse_args(["--input", "i", "--out", "o"])
    assert (ns.chain, ns.temperature, ns.num_samples) == (d["chain"], d["temperature"], d["num_samples"])
    assert (ns.mode, ns.det, ns.seed, ns.batch) == (None, 0, det.DEFAULT_SEED, batched.DEFAULT_BATCH) == (None, 0, 37, 64)
    assert batched.DEFAULT_SEED == det.DEFAULT_SEED
    ns = settings.parser().parse_args(["--mode", "off", "--det", "--seed", "5", "--batch", "8", "--input", "in.pdb", "--out", "o", "--chain", "C",
                                       "--temperature", "0.1", "--num-samples", "3"])
    assert (ns.mode, ns.det, ns.seed, ns.batch, ns.input, ns.out, ns.chain, ns.temperature, ns.num_samples) == ("off", 1, 5, 8, "in.pdb", "o", "C", 0.1, 3)
    P = settings.parser()
    assert [P.parse_args(["--input", "i", "--out", "o"] + x).det for x in ([], ["--det"], ["--det", "1"], ["--det", "0"])] == [0, 1, 1, 0]
    assert settings.check_values(P.parse_args(["--input", "i", "--out", "o", "--seed", "5"])) == {"det": False, "seed": 5, "batch": 64}
    with pytest.raises(settings.SettingsError):
        settings.check_values(P.parse_args(["--input", "i", "--out", "o", "--batch", "0"]))
    with pytest.raises(settings.SettingsError):
        settings.check_values(P.parse_args(["--input", "i", "--out", "o", "--num-samples", "0"]))   # the driver's own refusal (S >= 1)
    settings.check_values(P.parse_args(["--input", "i", "--out", "o", "--temperature", "0"]))          # upstream's to judge: passed through
    for gone in ("--num_samples", "--settings", "--timeout", "--json", "--quiet", "--allow-partial", "--variant"):
        with pytest.raises(SystemExit) as e:
            P.parse_args(["--input", "i", "--out", "o", gone, "1"])
        assert e.value.code == 2
    assert P.parse_args([]).input is None and P.parse_args([]).pdbfile is None            # which input / output was given is checked after parsing (inputs.input_arg, outputs.plan)
    # upstream's own spellings parse to upstream's own destinations and defaults
    ns = P.parse_args(["x.pdb"])
    assert (ns.pdbfile, ns.input, ns.out, ns.outpath, ns.multichain_backbone, ns.nogpu) == ("x.pdb", None, None, None, False, False)
    assert (d["outpath"], d["multichain_backbone"], d["nogpu"]) == (batched.DEFAULT_OUTPATH, False, False) == ("output/sampled_seqs.fasta", False, False)
    ns = P.parse_args(["x.pdb", "--chain", "B", "--outpath", "o/x.fasta", "--multichain-backbone", "--nogpu", "--batch", "5"])
    assert (ns.outpath, ns.multichain_backbone, ns.nogpu, ns.batch) == ("o/x.fasta", True, True, 5)
    assert P.parse_args(["x.pdb", "--multichain-backbone", "--singlechain-backbone"]).multichain_backbone is False
    assert P.parse_args(["x.pdb", "--batch_size", "7"]).batch == P.parse_args(["x.pdb", "--batch", "7"]).batch == 7
    with pytest.raises(settings.SettingsError):
        settings.check_values(P.parse_args(["x.pdb", "--multichain-backbone"]))                  # the multichain route designs a named chain
    settings.check_values(P.parse_args(["x.pdb", "--multichain-backbone", "--chain", "B"]))
    ns = P.parse_args(["--input", "/d/in", "--out", "o", "--chain", "A", "--num-samples", "8", "--seed", "11", "--batch", "128"])
    assert settings.driver_argv(ns, "/d/in", out_dir="/o") == ["--input", "/d/in", "--out", "/o", "--temperature", "1.0", "--num-samples", "8", "--seed", "11", "--batch_size", "128", "--chain", "A"]
    ns2 = P.parse_args(["/d/x.pdb", "--chain", "B", "--multichain-backbone", "--nogpu"])
    assert settings.driver_argv(ns2, "/d/x.pdb", outpath="/o/x.fasta") == ["--input", "/d/x.pdb", "--outpath", "/o/x.fasta", "--temperature", "1.0", "--num-samples", "1", "--seed", "37",
                                                                          "--batch_size", "64", "--chain", "B", "--multichain-backbone", "--nogpu"]
    # the driver's own parser takes exactly those tokens back
    import argparse
    back = batched.driver_arguments(argparse.ArgumentParser()).parse_args(settings.driver_argv(ns, "/d/in", out_dir="/o"))
    assert (back.input, back.out, back.chain, back.temperature, back.num_samples, back.seed, back.batch) == ("/d/in", "/o", "A", 1.0, 8, 11, 128)
    back = batched.driver_arguments(argparse.ArgumentParser()).parse_args(settings.driver_argv(ns2, "/d/x.pdb", outpath="/o/x.fasta"))
    assert (back.input, back.outpath, back.out, back.chain, back.multichain_backbone, back.nogpu) == ("/d/x.pdb", "/o/x.fasta", None, "B", True, True)


def test_one_input_one_output_layout(tmp_path):
    """Exactly one of pdbfile / --input; --out DIR (seqs/<stem>.fasta each) or --outpath FILE (one structure) — upstream's default outpath for one
    file with neither; a directory needs --out."""
    import argparse
    P = batched.driver_arguments(argparse.ArgumentParser())
    a, b = tmp_path / "a.pdb", tmp_path / "b.pdb"
    a.write_text("ATOM\n"); b.write_text("ATOM\n")
    assert inputs.input_arg(P.parse_args([str(a)])) == inputs.input_arg(P.parse_args(["--input", str(a)])) == str(a)
    for bad in ([str(a), "--input", str(b)], []):
        with pytest.raises(inputs.InputError):
            inputs.input_arg(P.parse_args(bad))
    files, stems = inputs.resolve(str(tmp_path))
    assert stems == ["a", "b"]
    out_dir, paths = outputs.plan(str(tmp_path / "o"), None, files, stems)
    assert out_dir == str(tmp_path / "o") and paths == {"a": str(tmp_path / "o" / "seqs" / "a.fasta"), "b": str(tmp_path / "o" / "seqs" / "b.fasta")}
    with pytest.raises(outputs.OutputError):
        outputs.plan(None, None, files, stems)                                               # a directory of structures needs --out
    with pytest.raises(outputs.OutputError):
        outputs.plan(None, str(tmp_path / "x.fasta"), files, stems)                          # --outpath names ONE structure's FASTA
    with pytest.raises(outputs.OutputError):
        outputs.plan(str(tmp_path / "o"), str(tmp_path / "x.fasta"), files[:1], stems[:1])   # not both
    one_dir, one = outputs.plan(None, str(tmp_path / "res" / "x.fasta"), files[:1], stems[:1])
    assert one_dir == str(tmp_path / "res") and one == {"a": str(tmp_path / "res" / "x.fasta")}
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        d_dir, d = outputs.plan(None, None, files[:1], stems[:1])                            # upstream's default --outpath, relative to the working directory
    finally:
        os.chdir(cwd)
    assert (d_dir, d) == (str(tmp_path / "output"), {"a": str(tmp_path / "output" / "sampled_seqs.fasta")})
    (tmp_path / "res").mkdir(); (tmp_path / "res" / "x.fasta").write_text(">sampled_seq_1\nAC\n>sampled_seq_2\nAD\n")
    assert outputs.census_paths(one, 2)["items_complete"] is True and outputs.census_paths(one, 3)["item_failed"] == {"a": "2/3 records"}


def test_inputs_and_outputs(tmp_path):
    d = tmp_path / "bb"; d.mkdir()
    for n in ("b.pdb", "a.cif", "c.txt"):
        (d / n).write_text("x")
    files, stems = inputs.resolve(str(d))
    assert [os.path.basename(f) for f in files] == ["a.cif", "b.pdb"] and stems == ["a", "b"]
    files, stems = inputs.resolve(str(d / "b.pdb"))
    assert stems == ["b"]
    for bad in (str(tmp_path / "nope"), str(tmp_path)):
        with pytest.raises(inputs.InputError):
            inputs.resolve(bad)
    (d / "a.pdb").write_text("x")
    with pytest.raises(inputs.InputError):
        inputs.resolve(str(d))                                                      # stems a, a
    out = tmp_path / "o"
    assert outputs.fasta_path(str(out), "b") == os.path.join(str(out), "seqs", "b.fasta")
    os.makedirs(outputs.seqs_dir(str(out)))
    with open(outputs.fasta_path(str(out), "b"), "w") as fh:
        fh.write(">sampled_seq_1\nAC\nD\n>sampled_seq_2\nEF\n")
    assert outputs.fasta_records(outputs.fasta_path(str(out), "b")) == ["ACD", "EF"]
    c = outputs.census(str(out), ["b", "zz"], 2)
    assert c == {"n_items": 2, "n_items_complete": 1, "items_complete": False, "item_failed": {"zz": "missing"}, "n_records": 2, "incomplete": "1/2"}
    assert outputs.census(str(out), ["b"], 3)["item_failed"] == {"b": "2/3 records"}


def test_usage_errors_exit_2(tmp_path):
    (tmp_path / "in.pdb").write_text("ATOM\n")
    (tmp_path / "two").mkdir(); (tmp_path / "two" / "a.pdb").write_text("ATOM\n"); (tmp_path / "two" / "b.pdb").write_text("ATOM\n")
    base = ["--input", str(tmp_path / "in.pdb"), "--out", str(tmp_path / "o")]
    for args in (["design"], ["design", "--mode", "turbo"] + base, ["design", "--mode", "default"] + base, ["design", "--mode", "off", "--batch", "0"] + base,
                 ["design", "--mode", "off"] + base + ["--settings", "upstream"], ["design", "--mode", "off", "--input", str(tmp_path / "nope"), "--out", str(tmp_path / "o")],
                 ["design", "--mode", "off"] + base + ["--outpath", str(tmp_path / "x.fasta")], ["design", "--mode", "off", str(tmp_path / "in.pdb")] + base,
                 ["design", "--mode", "off"] + base + ["--multichain-backbone"], ["design", "--mode", "off", "--input", str(tmp_path / "two"), "--outpath", str(tmp_path / "x.fasta")],
                 ["frobnicate"], ["check", "--mode", "nope"], []):
        rc, _, err = run_cli(args)
        assert rc == report.EXIT_USAGE == 2, (args, err)
    rc, _, err = run_cli(["design", "--mode", "off"] + base, env={"ESM_IF1_OPT": "fast"})
    assert rc == 2 and "disagrees" in err
    assert run_cli(["--help"])[0] == 0 and run_cli(["-h"])[0] == 0


@pytest.mark.skipif(not no_model_stack(), reason="fair-esm and torch are installed here: the stock child would run the model")
def test_off_without_the_model_stack_cannot_run_and_says_so(tmp_path):
    (tmp_path / "in.pdb").write_text("ATOM\n")
    rc, _, err = run_cli(["design", "--mode", "off", "--input", str(tmp_path / "in.pdb"), "--out", str(tmp_path / "o")])
    assert rc == 3 and re.fullmatch(r"\[esm_if1-opt\] NOT ACTIVE: mode off cannot run: (fair-esm|torch)(, torch)? not installed on \S+ \(run\.sh install; STOCK\.md\) \(mode=off\)", err.strip()), err
    assert not (tmp_path / "o").exists()


def test_the_stock_child_proves_itself_then_runs_the_script(tmp_path):
    """The proven stock child end to end on this box: the stripped environment holds by construction (ENV-CLEAN ok, proof file ok=true), then
    the script named after ``--`` runs as __main__ with its own argv — here a stand-in that records its argv (upstream's script needs the model
    stack); a forbidden name that survives into the child is NOT STOCK, exit 3, nothing runs."""
    out = tmp_path / "o"
    env, removed = stack.stock_env(str(out), base=dict(os.environ, ESM_IF1_OPT="off", ESM_IF1_KIT_X="1", NVIDIA_TF32_OVERRIDE="0", PYTHONSAFEPATH="1",
                                                  PYTHONPATH=os.pathsep.join([OPT, CORE]), TORCH_HOME=str(tmp_path / "th")))
    assert removed == ["ESM_IF1_KIT_X", "ESM_IF1_OPT", "NVIDIA_TF32_OVERRIDE", "PYTHONSAFEPATH"]
    assert env["PYTHONPATH"] == os.pathsep.join([OPT, CORE])                          # inherited unchanged: the box paths the kit (opt/ + common/opt_core), the child is a package module
    assert env["TORCH_HOME"] == str(tmp_path / "th") and env["ESM_IF1_TIMING_JSONL"] == str(out / "timing.jsonl")
    from opt_core import stock_proof
    script = tmp_path / "standin.py"
    script.write_text("import json, sys\njson.dump(sys.argv, open(sys.argv[sys.argv.index('--outpath') + 1], 'w'))\n")
    ns = settings.parser().parse_args(["--mode", "off", str(tmp_path / "in.pdb"), "--chain", "A", "--num-samples", "2", "--nogpu"])
    argv = settings.script_argv(ns, str(tmp_path / "in.pdb"), str(out / "argv.json"))
    assert argv == [str(tmp_path / "in.pdb"), "--chain", "A", "--temperature", "1.0", "--num-samples", "2", "--outpath", str(out / "argv.json"), "--nogpu"]   # the script's own flags; no --seed / --batch_size
    cmd = stock_proof.stock_command(sys.executable, report.STOCK_CHILD, proof_json=str(out / "stock_env_proof.json"), env_absent=stack.MUST_BE_ABSENT_PREFIXES,
                                    kit_dirs=[], module_prefixes=[], args=[str(script)] + argv)
    assert cmd[:4] == [sys.executable, "-s", "-m", "esm_if1_opt.stock_design"] and cmd[cmd.index("--") + 1:] == [str(script)] + argv
    os.makedirs(str(out))
    r = subprocess.run(cmd, env=env, cwd=str(out), capture_output=True, text=True, timeout=120)
    first = r.stderr.splitlines()[0]
    assert first.startswith("[esm_if1-opt] ENV-CLEAN ok (proof: absent=ESM_IF1_OPT,ESM_IF1_KIT,NVIDIA_TF32_OVERRIDE,TORCH_ALLOW_TF32_CUBLAS_OVERRIDE,CUBLAS_WORKSPACE_CONFIG,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD kit_modules=none kit_dirs=none no_user_site=True"), r.stderr
    proof = json.load(open(out / "stock_env_proof.json"))
    assert proof["ok"] is True and proof["torch_loaded_before_proof"] is False and "TORCH_HOME" in proof["present"] and "ESM_IF1_TIMING_JSONL" in proof["present"]
    assert json.load(open(out / "argv.json")) == [str(script)] + argv                # the script ran as __main__ with exactly its argv
    kinds = [l.split()[1] for l in r.stderr.splitlines() if l.startswith("[esm_if1-opt]")]
    assert kinds == ["ENV-CLEAN", "PEAK", "KERNELS"] and r.returncode == 0, r.stderr  # after the script: the two report lines (torch absent here: device=none)
    # a script that is not a file is refused by name (usage), after the proof
    r = subprocess.run(cmd[:cmd.index("--") + 1] + [str(tmp_path / "nope.py")], env=env, cwd=str(out), capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "must be upstream's script" in r.stderr
    # a forbidden name that survives into the child is NOT STOCK, exit 3, nothing runs
    r = subprocess.run(cmd, env=dict(env, ESM_IF1_OPT="off"), cwd=str(out), capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and r.stderr.splitlines()[0].startswith("[esm_if1-opt] NOT STOCK: forbidden env ['ESM_IF1_OPT']"), r.stderr


def test_the_kit_child_is_the_batched_driver(tmp_path):
    """``python -m esm_if1_opt.kit_design <driver args>``: the driver's own parser (usage errors exit 2 before any import of the model stack);
    on a box without the stack a valid line stops at the driver's first model import (rc 1)."""
    (tmp_path / "in.pdb").write_text("ATOM\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([OPT, CORE]))
    r = subprocess.run([sys.executable, "-m", report.KIT_CHILD, "--input", str(tmp_path / "in.pdb"), "--out", str(tmp_path / "o"), "--batch_size", "0"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "--batch_size must be >= 1" in r.stderr
    (tmp_path / "two").mkdir(); (tmp_path / "two" / "a.pdb").write_text("ATOM\n"); (tmp_path / "two" / "b.pdb").write_text("ATOM\n")
    r = subprocess.run([sys.executable, "-m", report.KIT_CHILD, "--input", str(tmp_path / "two"), "--outpath", str(tmp_path / "x.fasta")], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "usage:" in r.stderr                                # two structures need --out, not --outpath
    if no_model_stack():
        r = subprocess.run([sys.executable, "-m", report.KIT_CHILD, str(tmp_path / "in.pdb"), "--outpath", str(tmp_path / "x.fasta")], env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 1 and "ModuleNotFoundError" in r.stderr


@pytest.mark.skipif(not no_model_stack(), reason="fair-esm and torch are installed here")
def test_design_off_on_a_box_without_the_stack_is_not_active(tmp_path):
    """`design --mode off` refuses by name before launching anything when fair-esm / torch are absent (exit 3, nothing written)."""
    (tmp_path / "in.pdb").write_text("ATOM\n")
    rc, _, err = run_cli(["design", "--mode", "off", "--det", "--input", str(tmp_path / "in.pdb"), "--out", str(tmp_path / "o")])
    assert rc == 3 and "NOT ACTIVE: mode off cannot run" in err and not (tmp_path / "o").exists()


def test_verdict_is_the_cores_rule():
    rep = {"mode": "off", "route": "stock", "active": False, "partial": [], "gated": []}
    assert report.verdict(0, rep, None)["exit_code"] == 0
    assert report.verdict(0, rep, "1/3")["exit_code"] == 1 and report.verdict(0, rep, "1/3")["incomplete"] == "1/3"
    assert [report.verdict(rc, rep, None)["exit_code"] for rc in (1, 3, -9, 247)] == [1, 1, 1, 1]      # any non-zero child status -> 1 (NOT STOCK is answered before the verdict)
    assert report.verdict(0, rep, None)["allow_partial"] is False


def test_banner_grammars():
    assert report.not_active_stock() == "[esm_if1-opt] NOT ACTIVE: mode off (stock route) (mode=off)"
    n = report.not_applied("stock", [("seed", 37), ("batch_size", 64), ("det", 1)], "mode off runs upstream's script as shipped")
    assert re.fullmatch(report.NOT_APPLIED_RE, n) and n.startswith("[esm_if1-opt] NOT APPLIED route=stock seed=37 batch_size=64 det=1 (")
    assert re.fullmatch(report.NOT_APPLIED_RE, report.not_applied("kit", [("det", 1)], "why"))
    rep = {"mode": "fast", "route": "kit", "levers": ["batched_sampling"], "levers_applied": ["batched_sampling"], "partial": []}
    a = report.active_line(rep, batch_size=64, seed=37, det=False, multichain=False, nogpu=False)
    assert a == "[esm_if1-opt] ACTIVE mode=fast route=kit levers_requested=batched_sampling levers_applied=batched_sampling batch_size=64 seed=37 det=0 multichain=0 nogpu=0"
    assert re.fullmatch(report.ACTIVE_RE, a).group("applied") == "batched_sampling"
    ran = [{"kind": "STARTUP", "device": "cuda"}, {"kind": "ITEM", "n_seq": 64}, {"kind": "ITEM", "n_seq": 16}]      # the child's records: two batched forwards, 80 rows
    assert report.lever_lines(rep, batch_size=64, records=ran) == ["[esm_if1-opt] LEVER name=batched_sampling state=on impl=esm_if1_opt.batched origin=kit rows_per_forward=64 batches=2 rows=80"]
    assert report.lever_lines(rep, batch_size=64, records=ran[:1]) == ["[esm_if1-opt] LEVER name=batched_sampling state=off reason=no_batch_ran impl=esm_if1_opt.batched origin=kit rows_per_forward=64 batches=0 rows=0"]   # the child ended before its first forward
    m_ = report.active_line(rep, batch_size=16, seed=5, det=True, multichain=True, nogpu=True)          # the multichain route: the same lever, the same banner but its switches
    assert m_ == "[esm_if1-opt] ACTIVE mode=fast route=kit levers_requested=batched_sampling levers_applied=batched_sampling batch_size=16 seed=5 det=1 multichain=1 nogpu=1"
    assert re.fullmatch(report.ACTIVE_RE, m_).group("multichain") == "1"
    assert report.verdict(0, rep, None)["exit_code"] == 0 and report.verdict(0, rep, None)["partial"] == []
    s_ = report.invocation_line("kit", "pass", report.KIT_CHILD, 10.0, 12.5, 0, "hit")
    m = re.fullmatch(report.INVOCATION_RE, s_)
    assert m and m.group("route") == "kit" and m.group("name") == "pass" and m.group("argv0") == "esm_if1_opt.kit_design" and m.group("wall_s") == "2.500" and m.group("rc") == "0" and m.group("cache") == "hit"
    m = re.fullmatch(report.INVOCATION_RE, report.invocation_line("stock", "5YH2", "sample_sequences.py", 1.0, 2.0, 1, "na"))
    assert m and m.group("route") == "stock" and m.group("name") == "5YH2" and m.group("rc") == "1"
    with pytest.raises(ValueError):
        report.invocation_line("kit", "pass", "x", 1.0, 2.0, 0, "maybe")
    x = report.exit_line(mode="fast", route="kit", rc=0, inputs=3, n_fasta=3, designs_written=24, designs_expected=24, wall=12.34, incomplete=None, exit_code=0)
    assert re.fullmatch(r"\[esm_if1-opt\] EXIT pid=\d+ mode=fast route=kit rc=0 inputs=3 n_fasta=3 designs_written=24 designs_expected=24 wall=12.3 exit=0", x), x
    y = report.exit_line(mode="off", route="stock", rc=1, inputs=2, n_fasta=1, designs_written=8, designs_expected=16, wall=1.0, incomplete="1/2", exit_code=1)
    assert y.endswith(" mode=off route=stock rc=1 inputs=2 n_fasta=1 designs_written=8 designs_expected=16 wall=1.0 incomplete=1/2 exit=1")
    w = report.weights_line({"file": "esm_if1_gvp4_t16_142M_UR50.pt", "torch_home": "/th", "hub_entry": "linked", "source": "ESM_IF1_WEIGHTS=/weights/x.pt"})
    assert w == "[esm_if1-opt] WEIGHTS file=esm_if1_gvp4_t16_142M_UR50.pt torch_home=/th hub_entry=linked source=ESM_IF1_WEIGHTS=/weights/x.pt"


def test_check_reports_and_never_gates_on_the_checkpoint(tmp_path):
    rc, _, err = run_cli(["check", "--mode", "off"], env={"ESM_IF1_WEIGHTS": "", "TORCH_HOME": str(tmp_path / "th"), "MODEL_OPT_TARGET_GPU": ""})
    ls = err.splitlines()
    assert ls[0].startswith("[esm_if1-opt] CHECK upstream=facebookresearch/esm@2b369911 ") and " carried=2.0.1 " in ls[0], err
    assert ls[1].startswith("[esm_if1-opt] CHECK weights=esm_if1_gvp4_t16_142M_UR50.pt path=none bytes=none ") and f" expected_sha256={stack.WEIGHTS_SHA256} match=none " in ls[1], err
    assert ls[2].startswith("[esm_if1-opt] DRY-RUN mode=off route=stock levers=none "), err
    assert (rc == 0) == ls[2].endswith(" ok=True")                                   # ok=False (exit 3) exactly when fair-esm / torch are absent here
    for argv in (["check", "--mode", "fast"], ["check"]):                             # fast is the default word
        rcf, _, errf = run_cli(argv, env={"ESM_IF1_WEIGHTS": "", "TORCH_HOME": str(tmp_path / "th"), "MODEL_OPT_TARGET_GPU": ""})
        assert errf.splitlines()[2].startswith("[esm_if1-opt] DRY-RUN mode=fast route=kit levers=batched_sampling "), errf
        assert rcf == rc
    w_ = tmp_path / "w.pt"; w_.write_bytes(b"0123")
    rc2, _, err = run_cli(["check", "--mode", "off"], env={"ESM_IF1_WEIGHTS": str(w_), "TORCH_HOME": str(tmp_path / "th")})
    assert " bytes=4 " in err and " match=False " in err and rc2 == rc, err            # reported, never a gate


@pytest.mark.skipif(not no_model_stack(), reason="fair-esm and torch are installed here: the children would run the model")
def test_design_parent_tail_accounts_for_a_failed_child(tmp_path, monkeypatch, capsys):
    """The parent's whole tail on this CPU box, both modes: enable passes (distributions faked present); under off the real proven stock child
    runs upstream's script, which fails at its first model import (rc 1); under fast the kit child's driver fails the same way — banner /
    ACTIVE + LEVER, NOT APPLIED, WEIGHTS, INVOCATION, census, manifest and EXIT must name that, exit 1, no traceback of the parent."""
    from esm_if1_opt import cli, stack as _stack
    monkeypatch.setattr(_stack.gates, "dist_version", lambda name: {"fair-esm": "2.0.1", "torch": "2.4.0+cu121"}.get(name))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([OPT, CORE]))                    # the children need the package importable (the box paths the kit)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "th"))
    monkeypatch.delenv("ESM_IF1_OPT", raising=False)
    monkeypatch.delenv("ESM_IF1_WEIGHTS", raising=False)
    (tmp_path / "in.pdb").write_text("ATOM\n")
    _stack._REPORT = None
    out = tmp_path / "o"
    rc = cli.main(["design", "--mode", "off", "--input", str(tmp_path / "in.pdb"), "--out", str(out)])
    err = capsys.readouterr().err.splitlines()
    mine = [l for l in err if l.startswith("[esm_if1-opt]")]
    assert rc == 1, err
    assert mine[0] == "[esm_if1-opt] NOT ACTIVE: mode off (stock route) (mode=off)"
    assert mine[1] == "[esm_if1-opt] NOT APPLIED route=stock seed=37 batch_size=64 (mode off runs upstream's sample_sequences.py as shipped: unseeded, one sequence per model call)"
    assert mine[2].startswith("[esm_if1-opt] WEIGHTS file=esm_if1_gvp4_t16_142M_UR50.pt torch_home=") and mine[2].endswith(" hub_entry=absent source=torch.hub")
    inv = [l for l in mine if " INVOCATION " in l]
    assert len(inv) == 1 and re.fullmatch(report.INVOCATION_RE, inv[0]) and " route=stock name=in argv0=sample_sequences.py " in inv[0] and inv[0].endswith(" rc=1 cache=na")
    assert any(l.startswith("[esm_if1-opt] manifest: ") for l in mine)
    assert mine[-1].startswith("[esm_if1-opt] EXIT pid=") and " mode=off route=stock rc=1 inputs=1 n_fasta=0 designs_written=0 designs_expected=1 " in mine[-1] and mine[-1].endswith(" incomplete=1/1 exit=1")
    doc = json.load(open(out / "opt_manifest.json"))
    assert doc["exit"]["exit_code"] == 1 and doc["item_failed"] == {"in": "missing"} and doc["pass_status"] == "failed" and doc["mode"] == "off"
    assert doc["kit"]["proof"]["ok"] is True and doc["kit"]["children"][0]["rc"] == 1 and doc["kit"]["route"] == "stock"
    assert doc["kit"]["seed"] == {"value": 37, "applied": False} and doc["kit"]["batch"] == {"value": 64, "applied": False} and doc["kit"]["script"].endswith("sample_sequences.py")
    assert [r["kind"] for r in doc["kit"]["lines"]] == []                            # the script never reached the model
    assert json.load(open(out / "stock_env_proof.json"))["ok"] is True
    # fast: the kit child
    _stack._REPORT = None
    out2 = tmp_path / "o2"
    rc = cli.main(["design", str(tmp_path / "in.pdb"), "--outpath", str(out2 / "x.fasta"), "--det"])
    err = capsys.readouterr().err.splitlines()
    mine = [l for l in err if l.startswith("[esm_if1-opt]")]
    assert rc == 1, err
    assert mine[0] == "[esm_if1-opt] ACTIVE mode=fast route=kit levers_requested=batched_sampling levers_applied=batched_sampling batch_size=64 seed=37 det=1 multichain=0 nogpu=0"
    assert mine[1].startswith("[esm_if1-opt] NOT APPLIED route=kit det=1 (")
    inv = [l for l in mine if " INVOCATION " in l]
    assert len(inv) == 1 and " route=kit name=pass argv0=esm_if1_opt.kit_design " in inv[0] and inv[0].endswith(" rc=1 cache=na")
    lever = [l for l in mine if " LEVER " in l]                                       # after the child, from its records: this child ended at import, before its first forward
    assert lever == ["[esm_if1-opt] LEVER name=batched_sampling state=off reason=no_batch_ran impl=esm_if1_opt.batched origin=kit rows_per_forward=64 batches=0 rows=0"]
    assert mine.index(lever[0]) > mine.index(inv[0])
    assert mine[-1].startswith("[esm_if1-opt] EXIT pid=") and " mode=fast route=kit rc=1 inputs=1 n_fasta=0 " in mine[-1] and mine[-1].endswith(" incomplete=1/1 exit=1")
    doc = json.load(open(out2 / "opt_manifest.json"))
    assert doc["mode"] == "fast" and doc["kit"]["route"] == "kit" and doc["kit"]["levers"] == ["batched_sampling"] and doc["kit"]["proof"] is None
    assert doc["kit"]["device"] == "none" and "gated" not in doc["activation"] and doc["activation"]["levers_applied"] == ["batched_sampling"]   # no STARTUP record: the child ended before the model load
    assert doc["kit"]["seed"] == {"value": 37, "applied": True} and doc["kit"]["batch"] == {"value": 64, "applied": True} and doc["kit"]["outputs"]["files"] == ["x.fasta"]
    assert not (out2 / "stock_env_proof.json").exists()
    # fast on upstream's multichain route: the lever is on (the route is batched like the single-chain one), the banner carries multichain=1
    _stack._REPORT = None
    out3 = tmp_path / "o3"
    rc = cli.main(["design", str(tmp_path / "in.pdb"), "--outpath", str(out3 / "x.fasta"), "--chain", "B", "--multichain-backbone", "--batch_size", "16"])
    err = capsys.readouterr().err.splitlines()
    mine = [l for l in err if l.startswith("[esm_if1-opt]")]
    assert rc == 1, err
    assert mine[0] == "[esm_if1-opt] ACTIVE mode=fast route=kit levers_requested=batched_sampling levers_applied=batched_sampling batch_size=16 seed=37 det=0 multichain=1 nogpu=0"
    assert [l for l in mine if " LEVER " in l] == ["[esm_if1-opt] LEVER name=batched_sampling state=off reason=no_batch_ran impl=esm_if1_opt.batched origin=kit rows_per_forward=16 batches=0 rows=0"]
    assert not any(" NOT APPLIED " in l or "skipped" in l or "gated" in l for l in mine), mine
    doc = json.load(open(out3 / "opt_manifest.json"))
    assert doc["kit"]["levers"] == ["batched_sampling"] and doc["kit"]["settings"]["multichain_backbone"] is True and doc["kit"]["batch"] == {"value": 16, "applied": True}
    assert "--multichain-backbone" in doc["kit"]["children"][0]["argv"] and doc["kit"]["children"][0]["argv"][doc["kit"]["children"][0]["argv"].index("--batch_size") + 1] == "16"
    # a child killed by a signal (negative status) is exit 1, not the raw status
    assert report.verdict(-9, {"partial": []}, None)["exit_code"] == 1

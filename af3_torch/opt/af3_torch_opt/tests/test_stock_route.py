"""The stock route (stock_cli.py): a pass-through to xfold's own CLI — the command carries the interpreter, the launcher, the pinned CLI, the
inputs, the output directory, the parameters file and the one named non-default, nothing else; the CLI's shipped defaults are read from its
own bytes; the launcher sets the OF3 layout flag and runs the CLI as __main__; the verb's lines and exit codes."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from af3_torch_opt import cli, stock_cli, stack

from .conftest import archive_or_skip, stub_calls

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def _cli_source(tmp_path):
    return open(stock_cli.extract_cli(str(tmp_path / "w"))["path"], encoding="utf-8").read()


def test_extracted_cli_is_the_pinned_archives(tmp_path):
    archive_or_skip()
    rec = stock_cli.extract_cli(str(tmp_path / "w"))
    assert rec["member"] == "xfold-22bdeedfa309ef4ff6f9199910d8403915de69d6/run_alphafold.py"   # the upstream commit named in the member path
    assert rec["bytes"] > 10000 and re.fullmatch(r"[0-9a-f]{64}", rec["sha256"])
    src = _cli_source(tmp_path)
    assert "flags.DEFINE_bool(\n    'fastnn',\n    True," in src                      # the shipped defaults the route keeps
    assert "flags.DEFINE_integer(\n    'num_diffusion_samples',\n    5," in src
    assert "flags.DEFINE_bool(\n    'run_data_pipeline',\n    True," in src         # the one the route turns off, by name
    assert "if _USE_FASTNN.value is True:" in src and "fastnn_config.layer_norm_implementation = 'triton'" in src


def test_command_is_a_pass_through(box, tmp_path):
    """One CLI process = interpreter, launcher, the CLI, ONE input (--json_path OR --input_dir), the output dir, the parameters file, the named
    non-default — nothing else (no deterministic flag, no sample count, no recycles, no kernel switch)."""
    argv = stock_cli.argv_for(box["stock_py"], "/w/run_alphafold.py", "/in/a.json", None, "/out")
    assert argv == [box["stock_py"], stock_cli.LAUNCH, "/w/run_alphafold.py", "--json_path", "/in/a.json",
                    "--output_dir", "/out", "--model_dir", os.path.join(box["params"], stack.checkpoint()), "--run_data_pipeline=false"]
    argv = stock_cli.argv_for(box["stock_py"], "/w/run_alphafold.py", None, "/in/dir", "/out")
    assert argv[3:5] == ["--input_dir", "/in/dir"] and argv.count("--json_path") == 0
    assert stock_cli.PASS_THROUGH_FLAGS == ("--run_data_pipeline=false",)
    for bad in ((None, None), ("/in/a.json", "/in/dir")):                 # exactly one input form per process
        with pytest.raises(ValueError):
            stock_cli.argv_for(box["stock_py"], "/w/run_alphafold.py", bad[0], bad[1], "/out")

def test_launcher_sets_the_layout_flag_and_runs_the_cli_as_main(tmp_path):
    """A stand-in xfold.of3 and CLI on the tests' interpreter: the flag is True when the CLI's main runs, argv is the CLI's own."""
    (tmp_path / "xfold").mkdir(); (tmp_path / "xfold" / "__init__.py").write_text(""); (tmp_path / "xfold" / "of3.py").write_text("OF3 = False\n")
    (tmp_path / "alphafold3" / "common").mkdir(parents=True); (tmp_path / "alphafold3" / "__init__.py").write_text(""); (tmp_path / "alphafold3" / "common" / "__init__.py").write_text("")
    (tmp_path / "alphafold3" / "common" / "folding_input.py").write_text("def load_fold_inputs_from_dir(d):\n    yield from ('a', 'b')\ndef load_fold_inputs_from_path(p):\n    yield 'c'\n")
    (tmp_path / "alphafold3" / "constants").mkdir(); (tmp_path / "alphafold3" / "constants" / "__init__.py").write_text("")
    (tmp_path / "alphafold3" / "constants" / "chemical_components.py").write_text("class Ccd:\n    def __init__(self, ccd_pickle_path=None, user_ccd=None): self.user_ccd = user_ccd\n")
    cli_py = tmp_path / "run_alphafold.py"
    cli_py.write_text("import json, sys\nfrom xfold import of3\nfrom alphafold3.common import folding_input\nfrom alphafold3.constants import chemical_components\n"
                      "if __name__ == '__main__': print(json.dumps({'of3': of3.OF3, 'argv': sys.argv, 'n_dir': len(folding_input.load_fold_inputs_from_dir('d')), "
                      "'n_path': len(folding_input.load_fold_inputs_from_path('p')), 'ccd': chemical_components.cached_ccd(user_ccd='u').user_ccd}))\n")
    argv = [sys.executable, stock_cli.LAUNCH, str(cli_py), "--output_dir", "/o", "--run_data_pipeline=false"]
    r = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {"of3": True, "argv": [str(cli_py), "--output_dir", "/o", "--run_data_pipeline=false"], "n_dir": 2, "n_path": 1, "ccd": "u"}
    from af3_torch_opt import stock_launch
    assert stock_cli.LAUNCHER_COMPAT == ",".join(stock_launch.ADAPTATIONS) == "of3.OF3=True,loaders-as-lists,cached_ccd-as-Ccd"
    assert not [l for l in open(stock_launch.__file__, encoding="utf-8") if l.startswith(("from .", "import af3_torch_opt", "from af3_torch_opt"))]   # imports nothing of the package


def test_stock_verb_runs_the_cli_once_for_every_input(box, tmp_path, capsys):
    """Two --json_path = two CLI processes, each with ONE --json_path (xfold's flag is one string: a repeated flag would fold only the last
    input, exit 0 — a silent drop); both inputs' outputs exist; STOCK-CLI once, COMMAND/STEP per process, DONE once."""
    archive_or_skip()
    from .test_cli_manifest import _inputs
    ins = _inputs(box["tmp"], ("a", "b"))
    out = tmp_path / "out"
    assert cli.main(["stock", "--json_path", ins[0], "--json_path", ins[1], "--output_dir", str(out)]) == 0
    err = capsys.readouterr().err
    assert err.count("[af3-torch-opt] STOCK-CLI entry=xfold-22bdeedfa309ef4ff6f9199910d8403915de69d6/run_alphafold.py") == 1 and "n_processes=2" in err
    assert "non_default=--run_data_pipeline=false" in err and "of3_layout_flag=set-by-launcher" in err and err.count("[af3-torch-opt] COMMAND step=stock_cli") == 2
    assert f" kit_patches={stock_cli.kit_patches()} " in err and stock_cli.kit_patches() in ("none", "AF3_WEIGHT_PORT-04-equivalent")   # the carried xfold package's declared patches, by id (PINS kit_patches), on the line
    assert len(re.findall(r"^\[af3-torch-opt\] DONE step=stock_cli rc=0 n_processes=2 wall_s=", err, re.M)) == 1
    items = re.findall(r"^\[af3-torch-opt\] STOCK-ITEM name=(\S+) seed=(-?\d+) inference_s=(\d+\.\d+)$", err, re.M)   # the CLI's per-seed clock teed ONCE per (fold, seed)
    assert items == [("a", "1", "13.50"), ("b", "1", "13.50")], items                # the fold inputs' own `name` (the CLI's spelling), one line each, seed 1
    calls = [json.loads(l) for l in open(box["log"])]
    stock_calls = [c for c in calls if c[0] == box["stock_py"]]
    assert len(stock_calls) == 2 and all(c[1] == stock_cli.LAUNCH and c[2].endswith("/_stock_cli/run_alphafold.py") for c in stock_calls)
    for c, p in zip(stock_calls, ins):
        assert c[3:] == ["--json_path", p, "--output_dir", str(out), "--model_dir", os.path.join(box["params"], stack.checkpoint()), "--run_data_pipeline=false"]
    for name in ("a", "b"):                                            # the CLI's own layout (the stub's emulation of run_alphafold.py: per-sample files unprefixed)
        assert os.path.isfile(out / name / "ranking_scores.csv") and os.path.isfile(out / name / "seed-1_sample-4/model.cif") and os.path.isfile(out / name / f"{name}_model.cif")
    assert os.path.isfile(out / "_stock_cli" / "stock_cli.0.log") and os.path.isfile(out / "_stock_cli" / "stock_cli.1.log")


def test_stock_verb_refuses_mixed_inputs(box, tmp_path, capsys):
    """--json_path with --input_dir is a usage refusal (rc 2), never a process: the CLI takes one input form per process."""
    archive_or_skip()
    from .test_cli_manifest import _inputs
    d = tmp_path / "d"; d.mkdir()
    assert cli.main(["stock", "--json_path", _inputs(box["tmp"], ("a",))[0], "--input_dir", str(d), "--output_dir", str(tmp_path / "o")]) == 2
    assert "USAGE" in capsys.readouterr().err and not os.path.exists(box["log"])
    assert cli.main(["stock", "--input_dir", str(d), "--output_dir", str(tmp_path / "o2")]) == 0     # one process over a directory (empty here)
    calls = [json.loads(l) for l in open(box["log"])]; sc_ = [c for c in calls if c[0] == box["stock_py"]]
    assert len(sc_) == 1 and sc_[0][3:5] == ["--input_dir", str(d)] and "--json_path" not in sc_[0]

def test_stock_verb_refuses_without_the_stock_interpreter(box, monkeypatch, capsys, tmp_path):
    from .test_cli_manifest import _inputs
    monkeypatch.setenv("AF3_TORCH_STOCK_PY", str(tmp_path / "nowhere"))
    assert cli.main(["stock", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(tmp_path / "o")]) == 3
    assert "AF3_TORCH_STOCK_PY=" in capsys.readouterr().err
    monkeypatch.delenv("AF3_TORCH_STOCK_PY")                                                       # no default anywhere: unset is NOT ACTIVE by name, never a guessed path
    assert cli.main(["stock", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(tmp_path / "o")]) == 3
    assert "AF3_TORCH_STOCK_PY=<unset> is not an executable interpreter (no default" in capsys.readouterr().err
    assert cli.main(["stock", "--output_dir", str(tmp_path / "o")]) == 2


def test_stock_interpreter_another_account_could_have_placed_is_refused_by_name(box, capsys, tmp_path):
    """AF3_TORCH_STOCK_PY is run only when the entry, the file it resolves to, the directory holding it and the venv directory above belong to
    this user or root and neither directory is writable by others (stock_cli.interpreter_refusal); uid 0 accepts any owner."""
    from .test_cli_manifest import _inputs
    py = box["stock_py"]; bindir = os.path.dirname(py)
    assert stock_cli.interpreter_refusal(py) is None                                               # the box's own stub: fine as it is
    def owned_by(owner_of):                                                                        # lstat / stat stand-ins reporting the owner `owner_of(path)` names
        def st(path, _real=os.stat):
            r = _real(path)
            return os.stat_result((r.st_mode, r.st_ino, r.st_dev, r.st_nlink, owner_of(path, r.st_uid), r.st_gid, r.st_size, r.st_atime, r.st_mtime, r.st_ctime))
        return st
    me = 1000
    mine = owned_by(lambda p, real: me)
    assert stock_cli.interpreter_refusal(py, uid=me, lstat=mine, stat=mine) is None
    roots = owned_by(lambda p, real: 0)
    assert stock_cli.interpreter_refusal(py, uid=me, lstat=roots, stat=roots) is None              # root-owned (a system interpreter, an image's venv): fine
    for planted in (py, bindir, os.path.dirname(bindir)):                                          # the entry, its directory, the venv directory: each one policed
        other = owned_by(lambda p, real, planted=planted: 4242 if os.path.abspath(p) == os.path.abspath(planted) else me)
        why = stock_cli.interpreter_refusal(py, uid=me, lstat=other, stat=other)
        assert why and f"{planted} belongs to uid 4242, not to this user (uid {me}) or root" in why and "not run" in why, (planted, why)
        assert stock_cli.interpreter_refusal(py, uid=0, lstat=other, stat=other) is None            # uid 0: any owner (a container's root over a bind-mounted venv)
    target_other = owned_by(lambda p, real: me)
    def stat_other(path):                                                                          # the entry is this user's symbolic link, the file it resolves to is another account's
        r = target_other(path)
        return r if os.path.abspath(path) != os.path.abspath(py) else os.stat_result((r.st_mode, r.st_ino, r.st_dev, r.st_nlink, 4242, *r[5:]))
    assert "belongs to uid 4242" in stock_cli.interpreter_refusal(py, uid=me, lstat=target_other, stat=stat_other)
    mode = os.stat(bindir).st_mode & 0o7777
    os.chmod(bindir, mode | 0o002)                                                                 # a directory others can write: whoever can could replace the interpreter — refused, whatever the owner
    try:
        why = stock_cli.interpreter_refusal(py)
        assert why and f"{bindir} is writable by others" in why and "chmod o-w" in why
        assert cli.main(["stock", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(tmp_path / "o")]) == 3   # and the verb is NOT ACTIVE by that name, nothing run
        err = capsys.readouterr().err
        assert "NOT ACTIVE" in err and f"{bindir} is writable by others" in err
        assert not [c for c in stub_calls(box) if c[0] == py]                                       # the interpreter was never started
    finally:
        os.chmod(bindir, mode)


def test_compat_modules_are_re_exports_only(tmp_path):
    """The two names xfold's CLI imports that the fork's layout lacks: re-export modules of the fork's own objects, nothing else — and never
    written over a file the fork ships."""
    import ast
    from af3_torch_opt import stock_venv
    assert sorted(stock_venv.COMPAT) == ["alphafold3/model/components/base_model.py", "alphafold3/model/diffusion/__init__.py", "alphafold3/model/diffusion/model.py"]
    for rel, body in stock_venv.COMPAT.items():
        for node in ast.parse(body).body:
            assert isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign)), (rel, ast.dump(node))
            if isinstance(node, ast.ImportFrom):
                assert node.module.startswith("alphafold3.model") or node.module == "typing", rel
    venv = tmp_path / "v"; site = venv / "lib" / "python3.12" / "site-packages"; (site / "alphafold3" / "model" / "components").mkdir(parents=True)
    assert stock_venv.write_compat(str(venv)) == list(stock_venv.COMPAT)
    assert (site / "alphafold3/model/diffusion/model.py").read_text() == "from alphafold3.model.model import Model as Diffuser\n"
    with pytest.raises(SystemExit):
        stock_venv.write_compat(str(venv))                        # a second write would shadow an existing file: refused


def test_stock_cli_line_names_the_venv_record(box, tmp_path, capsys):
    archive_or_skip()
    from .test_cli_manifest import _inputs
    venv = os.path.dirname(os.path.dirname(box["stock_py"]))
    json.dump({"strategy": "overlay", "compat_modules": ["alphafold3/model/components/base_model.py"]}, open(os.path.join(venv, "stock_venv.json"), "w"))
    assert cli.main(["stock", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err
    assert "launcher_compat=of3.OF3=True,loaders-as-lists,cached_ccd-as-Ccd venv_strategy=overlay venv_compat=alphafold3/model/components/base_model.py" in err
    assert re.search(r"launcher_sha256=[0-9a-f]{64} ", err)


def test_upstream_sample_count_is_the_pinned_clis_default(box, tmp_path):
    """modes.UPSTREAM_SAMPLES is xfold's own CLI default: read from the pinned archive's run_alphafold.py (--num_diffusion_samples), never
    transcribed from memory."""
    archive_or_skip()
    from af3_torch_opt import modes
    cli_ = stock_cli.extract_cli(str(tmp_path / "cli"))
    src = open(cli_["path"], encoding="utf-8").read()
    m = re.search(r"DEFINE_integer\(\s*['\"]num_diffusion_samples['\"]\s*,\s*(\d+)", src)
    assert m, "run_alphafold.py names no --num_diffusion_samples default"
    assert modes.UPSTREAM_SAMPLES == int(m.group(1)) == 5


STOCK_ITEM_RX = re.compile(r"^\[af3-torch-opt\] STOCK-ITEM name=(?P<name>\S+) seed=(?P<seed>-?\d+) inference_s=(?P<t>\d+(?:\.\d+)?)$")   # the line's format, locked


def test_stock_item_tee_over_a_canned_upstream_log():
    """The CLI's transcript, line by line, through stock_item_teer: ONE STOCK-ITEM line per (fold input, seed) — the fold name from the CLI's
    `Processing fold input <name>` line, seed and seconds from its `Running model inference for seed <s> took  <t> seconds.` line (the
    per-seed model forward over the CLI's samples, CUDA-synchronised on both sides); featurisation / extraction / other lines tee nothing."""
    canned = [
        "Running AlphaFold 3. Please note that standard AlphaFold 3 model parameters ...\n",
        "Processing fold input 2PV7_long_name with spaces\n",
        "Featurising data for seeds (1, 2)...\n",
        "Featurising data for seeds (1, 2) took  9.51 seconds.\n",
        "Running model inference for seed 1...\n",
        "Running model inference for seed 1 took  51.20 seconds.\n",
        "Extracting output structures (one per sample) for seed 1...\n",
        "Extracting output structures (one per sample) for seed 1 took  0.61 seconds.\n",
        "Running model inference for seed 2...\n",
        "Running model inference for seed 2 took  49.87 seconds.\n",
        "Processing fold input barnase_barstar\n",
        "Running model inference for seed 7 took  3 seconds.\n",
        "Done processing 2 fold inputs.\n",
    ]
    got = []
    tee = stock_cli.stock_item_teer(lambda **kv: got.append(kv))
    for raw in canned:
        tee(raw)
    assert tee.items == [("2PV7_long_name with spaces", 1, 51.2), ("2PV7_long_name with spaces", 2, 49.87), ("barnase_barstar", 7, 3.0)]
    assert got == [{"name": "2PV7_long_name with spaces", "seed": 1, "inference_s": "51.20"}, {"name": "2PV7_long_name with spaces", "seed": 2, "inference_s": "49.87"},
                   {"name": "barnase_barstar", "seed": 7, "inference_s": "3"}]          # seconds pass through in the CLI's own spelling
    # the default emitter renders the kit line; its format is the locked regex (name keeps the CLI's own spelling; kv rendering by opt_core.report)
    lines = []
    tee2 = stock_cli.stock_item_teer(lambda **kv: lines.append(stock_cli.line("STOCK-ITEM", **kv)))
    for raw in canned[10:12]:
        tee2(raw)
    assert len(lines) == 1 and STOCK_ITEM_RX.match(lines[0]) and STOCK_ITEM_RX.match(lines[0]).group("name", "seed", "t") == ("barnase_barstar", "7", "3"), lines


def test_stock_item_patterns_are_the_pinned_clis_own_format_strings(tmp_path):
    """The two transcript patterns the tee reads are run_alphafold.py's f-strings in the PINNED archive (a re-pin whose CLI words them
    differently fails here, by name, instead of printing no STOCK-ITEM lines)."""
    archive_or_skip()
    src = _cli_source(tmp_path)
    assert "print(f'Processing fold input {fold_input.name}')" in src
    assert "f'Running model inference for seed {seed} took '" in src and "f' {time.time() - inference_start_time:.2f} seconds.'" in src
    assert stock_cli.UPSTREAM_FOLD_RX.match("Processing fold input x").group("name") == "x"
    assert stock_cli.UPSTREAM_SEED_RX.match("Running model inference for seed 1 took  51.20 seconds.").group("seed", "t") == ("1", "51.20")
    assert stock_cli.UPSTREAM_SEED_RX.match("Running model inference for seed 1...") is None

"""The CLI (own options, mode selection, the stock arguments passed through unread, the proven stock call, the exit rule) and the BORZOI_OPT hook."""
import json
import os
import subprocess
import sys

import pytest

from .. import _autoload, cli, kit, modes, stock_sad
from ._fixtures import TREE_OPT, clean_env, installed_here, make_root

DOCUMENTED_COMMAND = ["-f", "hg38.ml.fa", "-o", "out", "--rc", "--stats", "SAD,logSAD,D2,logD2", "-t", "targets.txt", "-u", "params.json", "model0_best.h5", "variants.vcf"]


def test_stock_arguments_pass_through_unread():
    """Everything after the package's own options is the stock's, untouched — every option and both invocation forms (three positionals,
    or upstream's worker form: options pickle, params, model, vcf, worker index)."""
    own, rest = cli.split_own(["--mode", "exact"] + DOCUMENTED_COMMAND, ("--mode", "--det"), ("--allow-partial",))
    assert own == {"mode": "exact"} and rest == DOCUMENTED_COMMAND
    worker_form = ["opts.pkl", "params.json", "model0_best.h5", "variants.vcf", "3"]
    own, rest = cli.split_own(["--det", "1", "--allow-partial"] + worker_form, ("--mode", "--det"), ("--allow-partial",))
    assert own == {"det": "1", "allow_partial": True} and rest == worker_form
    assert cli.det_from({}) is False and cli.det_from({"det": "1"}) is True and cli.det_from({"det": "0"}) is False
    with pytest.raises(cli.CliError):
        cli.det_from({"det": "yes"})


def test_own_options_and_mode_selection(monkeypatch):
    own, rest = cli.split_own(["--mode", "exact", "--det", "1", "-o", "x", "a", "b", "c"], ("--mode", "--det"), ())
    assert own == {"mode": "exact", "det": "1"} and rest == ["-o", "x", "a", "b", "c"]
    own, rest = cli.split_own(["--det=0", "--mode", "off", "a"], ("--mode", "--det"), ())
    assert own == {"det": "0", "mode": "off"} and rest == ["a"]
    own, rest = cli.split_own(["--mode=off", "--", "--mode", "x"], ("--mode",), ())
    assert own == {"mode": "off"} and rest == ["--mode", "x"]
    monkeypatch.delenv("BORZOI_OPT", raising=False)
    assert cli.resolve_mode(None) == "exact" and cli.resolve_mode("off") == "off"         # no --mode, no BORZOI_OPT: the package default
    with pytest.raises(modes.ActivationError):
        cli.resolve_mode("fast")                                  # no fast mode ships
    monkeypatch.setenv("BORZOI_OPT", "off")
    assert cli.resolve_mode(None) == "off" and cli.resolve_mode("off") == "off"
    with pytest.raises(cli.CliError):
        cli.resolve_mode("exact")                                 # --mode and BORZOI_OPT disagree


def test_check_exit_codes(tmp_path, monkeypatch, capsys):
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    assert cli.main(["check", "--mode", "exact"]) == 0
    assert cli.main(["check", "--mode", "off", "--json"]) == 0
    assert cli.main(["check", "--mode", "exact", "--det", "1"]) == 0 and cli.main(["check", "--det", "2"]) == cli.EXIT_USAGE
    assert cli.main(["check", "--mode", "nope"]) == cli.EXIT_NOT_ACTIVE
    assert cli.main(["nonsense"]) == cli.EXIT_USAGE
    assert cli.main([]) == cli.EXIT_USAGE and cli.main(["--help"]) == 0
    assert cli.main(["check", "--mode", "fast"]) == cli.EXIT_NOT_ACTIVE       # no fast mode ships
    monkeypatch.setenv("KIT_FWD", "0")                                       # a kit switch set by the user: refused by name
    assert cli.main(["check", "--mode", "exact"]) == cli.EXIT_NOT_ACTIVE
    err = capsys.readouterr().err
    assert "[borzoi-opt] NOT ACTIVE:" in err and "KIT_FWD" in err


def test_sad_off_runs_the_stock_in_a_proven_clean_subprocess(tmp_path, monkeypatch, capsys):
    r = make_root(tmp_path)
    # PYTHONPATH: the scratch kit dirs (cli.stock_command DROPS them — the proof's kit_dirs_on_path == []) + the package dir under test, which it
    # keeps: the stock child (`python -s -m borzoi_opt.stock_sad`) binds THIS tree's borzoi_opt whether or not one is pip-installed on the interpreter
    clean_env(monkeypatch, {**r["env"], "KIT_POST_WRITER": "t2square", "KIT_FWD": "1", "BORZOI_OPT": "off",
                            "PYTHONPATH": r["kit_dir"] + os.pathsep + os.path.join(r["kit_dir"], "v17") + os.pathsep + TREE_OPT})
    out = os.path.join(str(tmp_path), "out_off")
    args = ["sad", "--mode", "off", "-f", "g.fa", "-o", out, "--rc", "--stats", "SAD,D2", "-t", "t.txt", "-u", "p.json", "m.h5", "v.vcf"]
    rc = cli.main(args)
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "[borzoi-opt] OFF route=cli entry=" + r["stock_sad"] in err and "pinned=true" in err and "stripped=BORZOI_OPT,KIT_FWD,KIT_POST_WRITER" in err
    assert "[borzoi-opt] EXIT mode=off rc=0 stock_proof=ok kit_modules_after=none" in err
    rec = json.load(open(os.path.join(out, "fake_stock.json")))
    assert rec["args"] == ["p.json", "m.h5", "v.vcf"] and rec["env_KIT"] == [] and rec["env_BORZOI_OPT"] is None and rec["modules_kitlib"] == []
    assert os.listdir(out) == ["fake_stock.json"]                    # nothing of the kit lands beside the stock's outputs (the proof went to a private directory, removed; the child said ENV-CLEAN on its own stderr)


STAMP_ALL_APPLIED = {"kit": "pipeline_tf.v17", "onehot": {"kit_onehot": "LUT", "replaced": "baskerville.dna.dna_1hot"},
                     "forward": {"kit_fwd": "graph_copy_free", "enabled": True, "graph": "tf.function(jit_compile=False) from call 2; call 1 eager", "n_calls": 5, "n_eager_calls": 1, "n_graph_calls": 4, "n_traces": 1, "n_copy_free": 5, "n_stock_calls": 0},
                     "post_applies": True, "pool": {"threads": 4, "probe_ok": True, "probe_refusal": None},
                     "post": {"kit_post": "column_chunked_threaded_pipelined", "writer": {"writer": "exact"}, "n_variants": 5, "n_fallback_stock_path": 0}}


def _fake_job(stamp, rc=0):
    """A stand-in for the kit job subprocess: writes `stamp` (nothing when None) into KIT_STAMP_DIR, returns `rc`."""
    def call(cmd, env=None, **kw):
        if stamp is not None:
            json.dump(stamp, open(os.path.join(env["KIT_STAMP_DIR"], "kit_stamp.json"), "w"))
        return rc
    return call


def test_sad_exact_exit_rule_from_the_kit_stamp(tmp_path, monkeypatch, capsys):
    """The kit stamp judged after the run: every lever evidenced -> 0; a lever that fell back -> 3 (PARTIAL, the manifest names it),
    --allow-partial / BORZOI_OPT_ALLOW_PARTIAL=1 -> the job's own exit with allow_partial recorded; post_applies false / variants outside the
    shape class -> ROUTED post=stock:<reason>, named, the job's own exit (per-call stock routing is never a refusal); the cores probe at its
    broad default (affinity/quota) -> FALLBACK cores_probe=os_count (...), named, the job's own exit; no stamp -> 3; a failed job -> its own rc,
    the partial list still recorded."""
    import copy
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    n = [0]

    def run(stamp, extra=(), rc=0, env=None):
        n[0] += 1
        out = os.path.join(str(tmp_path), f"out_{n[0]}")
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(cli.subprocess, "call", _fake_job(stamp, rc))
        code = cli.main(["sad", "--mode", "exact", *extra, "-o", out, "-t", "t.txt", "p.json", "m.h5", "v.vcf"])
        for k in (env or {}):
            monkeypatch.delenv(k)
        assert not os.path.exists(out)                              # the kit writes nothing of its own (the fake job creates no output directory)
        return code, capsys.readouterr().err

    def judged(stamp):                                              # the same judgement the CLI prints from
        return modes.applied(modes.resolve("exact", read_gpu=False), stamp)

    code, err = run(STAMP_ALL_APPLIED)
    assert code == 0, err
    j = judged(STAMP_ALL_APPLIED)
    assert j["partial"] == [] and j["gated"] == [] and j["routed"] == [] and j["fallbacks"] == [] and sorted(j["levers_applied"]) == sorted(modes.resolve("exact", read_gpu=False)["levers"])
    assert "PARTIAL" not in err and "FALLBACK" not in err and "ROUTED" not in err and "[borzoi-opt] EXIT mode=exact rc=0 kit=pipeline_tf.v17 post_applies=true writer=exact pool=4 partial=none allow_partial=off gated=0 routed=0" in err
    # variants outside the chunked post's shape class: written by the STOCK path per call — ROUTED by name, the lever engaged for the rest, exit the job's own
    st = copy.deepcopy(STAMP_ALL_APPLIED); st["post"]["n_fallback_stock_path"] = 2
    code, err = run(st)
    assert code == 0, err
    assert ("[borzoi-opt] ROUTED post=stock:shape — pipelined_chunked_post: 2/7 variants outside the supported shape class were written by the stock path per call "
            "(per call, by name; the exit is the job's own)") in err and "NOT ACTIVE" not in err and "PARTIAL allowed" not in err and "GATED" not in err
    j = judged(st)
    assert j["partial"] == [] and j["routed_levers"] == [] and "pipelined_chunked_post" in j["levers_applied"] and len(j["routed"]) == 1
    assert "partial=none allow_partial=off gated=0 routed=1" in err
    code, err = run(st, extra=("--allow-partial",))   # --allow-partial has nothing to allow: the route is not a partial
    assert code == 0 and "allow_partial=on" in err and "NOT ACTIVE" not in err and "routed=1" in err
    # the traced forward: every call after the first runs the graph, one trace — else a named partial
    st = copy.deepcopy(STAMP_ALL_APPLIED); st["forward"].update(n_graph_calls=2, n_eager_calls=3)
    code, err = run(st)
    assert code == cli.EXIT_NOT_ACTIVE and judged(st)["partial"] == ["graph_forward"] and "2 of 4 calls after the first ran the traced graph" in err
    st = copy.deepcopy(STAMP_ALL_APPLIED); st["forward"].update(n_traces=2)
    assert judged(st)["partial"] == ["graph_forward"]
    st = copy.deepcopy(STAMP_ALL_APPLIED); st["forward"].update(n_calls=1, n_copy_free=1, n_graph_calls=0, n_traces=0); st["post"].update(n_variants=1)
    assert judged(st)["partial"] == [] and "graph_forward" in judged(st)["levers_applied"]   # a one-variant job: one eager call, nothing to trace yet — applied
    # the forward's stock calls, a writer other than the mode's: each a named partial; the cores probe at its broad default: a named FALLBACK, not a partial
    st = copy.deepcopy(STAMP_ALL_APPLIED); st["forward"].update(n_stock_calls=1); st["pool"].update(probe_ok=False, probe_refusal="join TimeoutExpired"); st["post"]["writer"]["writer"] = "other"
    code, err = run(st)
    j = judged(st)
    assert code == cli.EXIT_NOT_ACTIVE and j["partial"] == ["copy_free_forward", "post_writer"] and "cores_probe" in j["levers_applied"] and j["fallbacks"] == [("cores_probe", "os_count (join TimeoutExpired) pool=4")]
    assert "1 of 6 forward calls took the stock call" in err and "the stamp's writer is 'other'" in err and "partial=copy_free_forward,post_writer allow_partial=off" in err
    assert "[borzoi-opt] FALLBACK cores_probe=os_count (join TimeoutExpired) pool=4" in err
    code, err = run(st, extra=("--allow-partial",))
    assert code == 0 and "[borzoi-opt] PARTIAL allowed:" in err and "allow_partial=on" in err
    code, err = run(st, env={"BORZOI_OPT_ALLOW_PARTIAL": "1"})
    assert code == 0 and "allow_partial=on" in err
    # the probe fallback alone: a complete run, every lever applied, one FALLBACK line, the job's own exit
    st = copy.deepcopy(STAMP_ALL_APPLIED); st["pool"].update(probe_ok=False, probe_refusal="implausible table")
    code, err = run(st)
    assert code == 0 and judged(st)["partial"] == [] and "NOT ACTIVE" not in err and "PARTIAL" not in err, err
    assert "[borzoi-opt] FALLBACK cores_probe=os_count (implausible table) pool=4" in err and "partial=none allow_partial=off" in err
    # the option set outside the chunked post's: the STOCK post path ran for every variant — ROUTED by name (the post levers under routed_levers), never gated / partial, exit the job's own
    st = {k: v for k, v in STAMP_ALL_APPLIED.items() if k not in ("post", "pool")}; st["post_applies"] = False
    code, err = run(st)
    assert code == 0, err
    j = judged(st)
    assert j["partial"] == [] and j["gated"] == [] and j["gated_levers"] == [] and len(j["routed"]) == 1 and j["routed"][0].startswith("post=stock:options — pipelined_chunked_post, post_writer, cores_probe: the option set is outside")
    assert j["routed_levers"] == ["pipelined_chunked_post", "post_writer", "cores_probe"] and sorted(j["levers_applied"]) == ["copy_free_forward", "graph_forward", "kit_stamp", "onehot_lut"]
    assert "[borzoi-opt] ROUTED post=stock:options — " in err and "post_applies=false writer=none pool=none partial=none allow_partial=off gated=0 routed=1" in err and "GATED" not in err and "NOT ACTIVE" not in err
    code, err = run(st, extra=("--allow-partial",))
    assert code == 0 and "PARTIAL allowed" not in err and "routed=1" in err
    # no stamp at all: the hook that never fired -> every lever partial
    code, err = run(None)
    j = judged(None)
    assert code == cli.EXIT_NOT_ACTIVE and sorted(j["partial"]) == sorted(modes.resolve("exact", read_gpu=False)["levers"]) and "no kit stamp" in err and "rc=0 kit=no stamp" in err
    # a failed job: its own rc as is, the partial / routed / fallback lines still printed
    st = copy.deepcopy(STAMP_ALL_APPLIED); st["post"]["n_fallback_stock_path"] = 1; st["forward"].update(n_stock_calls=1); st["pool"].update(probe_ok=False, probe_refusal="join TimeoutExpired")
    code, err = run(st, rc=1)
    assert code == 1 and "rc=1" in err and "partial=copy_free_forward" in err and "ROUTED post=stock:shape" in err and "FALLBACK cores_probe=os_count" in err


def test_sad_refuses_an_unpinned_stock(tmp_path, monkeypatch, capsys):
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    open(r["stock_sad"], "a").write("\n# tampered\n")
    assert cli.main(["sad", "--mode", "off", "-o", str(tmp_path / "o"), "p.json", "m.h5", "v.vcf"]) == cli.EXIT_NOT_ACTIVE
    assert "not the pinned stock" in capsys.readouterr().err


def test_kit_command_env(tmp_path, monkeypatch):
    r = make_root(tmp_path)
    clean_env(monkeypatch, {**r["env"], "BORZOI_OPT": "exact", "KIT_FWD": "0"})
    assert modes.resolve("exact", read_gpu=False)["ok"] is False                  # a kit switch set by the user under the kit mode is refused (one composition)
    monkeypatch.delenv("KIT_FWD")
    rep = modes.resolve("exact", read_gpu=False)
    cmd, env = cli.kit_command(rep, ["-o", "x", "a", "b", "c"], "/tmp/stampdir")
    assert cmd == [sys.executable, rep["entry"], "-o", "x", "a", "b", "c"]
    assert "BORZOI_OPT" not in env and env["KIT_FWD"] == "1" and env["KIT_STAMP_DIR"] == "/tmp/stampdir"
    assert "KIT_POST_WRITER" not in env and "NPY_DISABLE_CPU_FEATURES" not in env   # the kit's default writer (no writer variable); production numerics
    rep = modes.resolve("exact", read_gpu=False, det=True)
    cmd, env = cli.kit_command(rep, [], "/tmp/s")
    assert env["KIT_FWD"] == "1" and all(env[k] == v for k, v in modes.DET_RECIPE.items())   # --det: the recipe exported into the job
    off = modes.resolve("off", read_gpu=False, det=True)
    cmd, env = cli.stock_command(off, [], str(tmp_path / "o"))
    env.update(off["env"])
    assert all(env[k] == v for k, v in modes.DET_RECIPE.items()) and not any(k.startswith("KIT_") for k in env)


def test_stock_env_proof(tmp_path, monkeypatch):
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    pin = kit.stock_script_pin(kit.ENTRY)["sha256"]
    ok = stock_sad.env_proof([r["kit_dir"]], r["stock_sad"], pin, environ={}, modules={"os": os}, path=["/usr/lib"], meta_path=[])
    assert ok["ok"] and ok["stock_entry_pinned"]
    bad = stock_sad.env_proof([r["kit_dir"]], r["stock_sad"], pin, environ={"KIT_FWD": "1"}, modules={"os": os}, path=["/usr/lib"], meta_path=[])
    assert not bad["ok"] and bad["forbidden_present"] == ["KIT_FWD"]
    bad2 = stock_sad.env_proof([r["kit_dir"]], r["stock_sad"], pin, environ={}, modules={"os": os}, path=[os.path.join(r["kit_dir"], "v17")], meta_path=[])
    assert not bad2["ok"] and bad2["kit_dirs_on_path"]
    bad3 = stock_sad.env_proof([r["kit_dir"]], r["stock_sad"], "f" * 64, environ={}, modules={"os": os}, path=[], meta_path=[])
    assert not bad3["ok"] and not bad3["stock_entry_pinned"]
    bad4 = stock_sad.env_proof([r["kit_dir"]], r["stock_sad"], pin, environ={}, modules={"os": os, "tensorflow": os}, path=[], meta_path=[])
    assert not bad4["ok"] and bad4["tensorflow_loaded_before_proof"]


def test_autoload_decisions(tmp_path, monkeypatch):
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    assert _autoload.decide(environ={}, argv=[r["stock_sad"]])["action"] == "none"
    assert _autoload.decide(environ={"BORZOI_OPT": "off"}, argv=[r["stock_sad"]])["action"] == "none"
    assert _autoload.decide(environ={"BORZOI_OPT": "turbo"}, argv=[r["stock_sad"]])["action"] == "refuse"
    env = {"BORZOI_OPT": "exact", "MODEL_OPT": r["root"], "BORZOI_DIR": r["env"]["BORZOI_DIR"], "PATH": os.environ.get("PATH", "")}
    d = _autoload.decide(environ=env, argv=[r["stock_sad"], "-o", "x", "a", "b", "c"])
    assert d["action"] == "swap" and d["kit_entry"] == os.path.join(r["kit_dir"], "v17", "borzoi_sad.py")
    d2 = _autoload.decide(environ=env, argv=[os.path.join(r["kit_dir"], "v17", "borzoi_sad.py")])
    assert d2["action"] == "active" and d2["report"]["writer"] == "exact" and d2["report"]["env"] == {"KIT_FWD": "1"}
    # the active action INSTALLS the composition the ACTIVE line prints (the kit's forward.install reads KIT_FWD after the model load)
    env_active = dict(env)
    d3 = _autoload.install(environ=env_active, argv=[os.path.join(r["kit_dir"], "v17", "borzoi_sad.py")], do_exec=False)
    assert d3["action"] == "active" and env_active["KIT_FWD"] == "1" and d3["env"] == {"KIT_FWD": "1"}
    env_bad = {**env, "KIT_FWD": "0"}                                  # a kit switch in the environment: refused, nothing installed (the user's value left as is)
    d4 = _autoload.install(environ=env_bad, argv=[os.path.join(r["kit_dir"], "v17", "borzoi_sad.py")], do_exec=False)
    assert d4["action"] == "active" and not d4["report"]["ok"] and env_bad["KIT_FWD"] == "0"
    assert _autoload.decide(environ={**env, "BORZOI_OPT": "turbo"}, argv=[r["stock_sad"]])["action"] == "refuse"
    assert _autoload.decide(environ={**env, "BORZOI_OPT": "fast"}, argv=[r["stock_sad"], "-o", "x", "a", "b", "c"])["action"] == "refuse"   # no fast mode ships: refused by name, never a silent stock run
    assert _autoload.decide(environ=env, argv=[r["stock_sed"]])["action"] == "engage"           # any program but the pinned borzoi_sad.py: the library route
    assert _autoload.decide(environ=env, argv=["-m"])["action"] == "engage"
    other = str(tmp_path / "other.py")
    open(other, "w").write("print(1)\n")
    assert _autoload.decide(environ=env, argv=[other])["action"] == "engage"
    tampered = str(tmp_path / "borzoi_sad.py")
    open(tampered, "w").write("print('not stock')\n")
    dd = _autoload.decide(environ=env, argv=[tampered])
    assert dd["action"] == "engage" and "not the pinned stock" in dd["reason"]       # a modified borzoi_sad.py runs as written, with the library levers
    cmd = _autoload._exec_argv("/kit/borzoi_sad.py", argv=["/stock/borzoi_sad.py", "-o", "x"], orig=["python", "-u", "/stock/borzoi_sad.py", "-o", "x"])
    assert cmd == [sys.executable, "-u", "/kit/borzoi_sad.py", "-o", "x"]
    assert _autoload._exec_argv("/kit/borzoi_sad.py", argv=["/stock/borzoi_sad.py", "-o", "x"], orig=None) == [sys.executable, "/kit/borzoi_sad.py", "-o", "x"]


def test_the_shipped_pth_is_the_generated_guard():
    """opt/borzoi_opt_autoload.pth is exactly what opt/_build_backend.py generates for this package (python opt/_build_backend.py
    borzoi_opt BORZOI_OPT borzoi-opt): the guard header + the one guarded import of borzoi_opt._autoload, whose import runs install()."""
    import glob, importlib.util
    opt = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location("_borzoi_build_backend", os.path.join(opt, "_build_backend.py"))
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    text = open(os.path.join(opt, "borzoi_opt_autoload.pth"), encoding="utf-8").read()
    assert text == backend.pth_text("borzoi_opt", "BORZOI_OPT", "borzoi-opt", 3)
    assert backend.pth_fields(text) == ("borzoi_opt", "BORZOI_OPT", "borzoi-opt", 3) and "import borzoi_opt._autoload" in text
    assert len(glob.glob(os.path.join(opt, "*_autoload.pth"))) == 1
    assert isinstance(_autoload.STARTUP, dict) and _autoload.STARTUP["action"] in ("none", "engage")   # importing the module ran install(): this process is no borzoi script, so nothing or the finder


def test_autoload_hook_end_to_end_swaps_the_stock_for_the_kit_entry(tmp_path, monkeypatch):
    """The hook's DECISIONS in a child interpreter bound to THIS tree (PYTHONPATH=TREE_OPT; no install needed): under BORZOI_OPT=exact with the
    stock borzoi_sad.py as argv[0], `borzoi_opt._autoload.install(do_exec=False)` computes the swap to the kit entry (SWAP + ACTIVE lines, the
    kit env) — the real kit entry imports TensorFlow, so the exec itself is not run here; with borzoi_sed.py as argv[0] it takes the library
    route (engage: the levers land when the stock library is imported). The INSTALLED .pth firing at interpreter start is the next test's (it needs `pip install -e opt` on the interpreter)."""
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    env = {**os.environ, "BORZOI_OPT": "exact", "MODEL_OPT": r["root"], "BORZOI_DIR": r["env"]["BORZOI_DIR"], "PYTHONPATH": TREE_OPT}
    code = ("d = A.install(do_exec=False); "
            "print('ACTION', d['action'], d.get('kit_entry'), d.get('env'), d.get('exec'))")
    # the hook decides on sys.argv[0]: the module is imported FIRST (as the site .pth imports it at interpreter start, argv still `-c`: inert), then
    # the stand-in stock script's path is patched into argv[0] and install(do_exec=False) computes the decision without the exec
    pre = "import os, sys, borzoi_opt._autoload as A; "
    p = subprocess.run([sys.executable, "-c", pre + "sys.argv = [%r, '-o', 'x', 'a', 'b', 'c']; %s" % (r["stock_sad"], code)],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "ACTION swap" in p.stdout and "v17/borzoi_sad.py" in p.stdout and "'KIT_FWD': '1'" in p.stdout and "KIT_POST_WRITER" not in p.stdout
    assert "[borzoi-opt] SWAP mode=exact stock=" + r["stock_sad"] in p.stderr and p.stderr.count("[borzoi-opt] ACTIVE mode=exact route=env") == 1
    assert "kit_env=KIT_FWD=1 det=off" in p.stderr
    # sed under the switch: the library route (nothing printed until the stock library is imported)
    p2 = subprocess.run([sys.executable, "-c", pre + "sys.argv = [%r]; %s" % (r["stock_sed"], code)],
                        env=env, capture_output=True, text=True)
    assert p2.returncode == 0 and "ACTION engage" in p2.stdout and "NOT ACTIVE" not in p2.stderr, p2.stderr


INSTALLED_HERE = installed_here()


@pytest.mark.skipif(INSTALLED_HERE is None, reason="the autoload .pth fires at interpreter start only where THIS tree's borzoi_opt is pip-installed on the interpreter (`pip install -e opt`, run.sh install); a PYTHONPATH-only interpreter has no .pth — skipped by name")
def test_installed_pth_hook_swaps_or_refuses_at_interpreter_start(tmp_path, monkeypatch):
    """The INSTALLED state end to end: `BORZOI_OPT=exact python <stock borzoi_sad.py> ...` as a plain script — the site .pth imports
    borzoi_opt._autoload at interpreter start, prints SWAP and execs the kit entry (which imports TensorFlow: only the SWAP line before anything
    ran is asserted); any other program importing the stock library under the switch gets the library levers installed on it (ACTIVE …
    route=hook, the program runs, EXIT … route=hook at exit); with the switch unset nothing of the kit is imported."""
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    env = {**os.environ, "BORZOI_OPT": "exact", "MODEL_OPT": r["root"], "BORZOI_DIR": r["env"]["BORZOI_DIR"]}
    # the real .pth path: the stock stand-in run as a script under BORZOI_OPT=exact -> the hook swaps to the real kit entry, which then
    # imports TensorFlow; we only assert the SWAP line was printed before anything ran (the kit entry's import may fail here).
    p3 = subprocess.run([sys.executable, r["stock_sad"], "-o", str(tmp_path / "o3"), "a", "b", "c"], env={**env, "BORZOI_OPT": "exact"}, capture_output=True, text=True, timeout=600)
    assert "[borzoi-opt] SWAP mode=exact stock=" + r["stock_sad"] in p3.stderr, p3.stderr[-2000:]
    assert "fake stock ran" not in p3.stdout                     # the stock never ran under the switch
    # the library route: another program importing the stock library under the switch gets the levers installed on the library
    fake_lib = tmp_path / "baskerville"
    fake_lib.mkdir()
    (fake_lib / "__init__.py").write_text("")
    (fake_lib / "dna.py").write_text("def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):\n    return 'stock one-hot'\n")
    (fake_lib / "seqnn.py").write_text("class SeqNN:\n    ensemble = None\n    model = None\n    def __call__(self, x, head_i=None, dtype='float32'):\n        return 'stock call'\n")
    other = tmp_path / "other.py"
    other.write_text("from baskerville import dna, seqnn\nimport sys\n"
                     "print('ran stock lib', dna.dna_1hot.__module__, seqnn.SeqNN.__call__.__module__, sorted(m for m in sys.modules if m.startswith('kitlib')))\n")
    p4 = subprocess.run([sys.executable, str(other)], env={**env, "BORZOI_OPT": "exact", "PYTHONPATH": str(tmp_path)}, capture_output=True, text=True)
    assert p4.returncode == 0, p4.stderr[-2000:]
    assert "ran stock lib kitlib.onehot kitlib.forward ['kitlib', 'kitlib.forward', 'kitlib.onehot']" in p4.stdout, p4.stdout
    assert "[borzoi-opt] ACTIVE mode=exact route=hook program=" + str(other) + " levers=graph_forward,copy_free_forward,onehot_lut kit=" in p4.stderr, p4.stderr
    assert "[borzoi-opt] EXIT mode=exact route=hook program=" + str(other) + " forward: n_calls=0 " in p4.stderr and "onehot=LUT" in p4.stderr, p4.stderr
    assert "KIT_FWD" not in p4.stdout                            # nothing exported into the program's environment
    # with the switch unset the same program runs with nothing of the kit
    p5 = subprocess.run([sys.executable, str(other)], env={k: v for k, v in {**env, "PYTHONPATH": str(tmp_path)}.items() if k != "BORZOI_OPT"}, capture_output=True, text=True)
    assert p5.returncode == 0 and "ran stock lib baskerville.dna baskerville.seqnn []" in p5.stdout and "[borzoi-opt]" not in p5.stderr

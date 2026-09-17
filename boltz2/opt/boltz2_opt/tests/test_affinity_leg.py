"""boltz2_opt.affinity_leg: upstream's affinity leg (boltz main.py predict, after the structure pass) for one processed input, verbatim, in a
clean interpreter the worker starts right after that input's structure pass with the four RNG streams handed over."""
import json
import os
import pickle
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from .. import manifest as mf
from .. import writer as _writer_mod   # the kit's writer module (not installed in these tests: WR.wait_item is the no-op stock has)
from .. import affinity_leg as AL, report as rep, stack, worker


@pytest.fixture(autouse=True)
def one_shot_form_unless_the_test_says_otherwise(monkeypatch):
    """The legs' stand-ins in this file are one-shot interpreters handed to AL.command (form `item`, by name); the served form — the default,
    one clean interpreter per pass (0.3.24) — is driven by test_the_pass_s_legs_run_in_one_served_interpreter_… which lifts this word."""
    monkeypatch.setenv(AL.FORM_ENV, "item")
    AL.reset_for_tests()
    yield
    AL.reset_for_tests()

from . import _stubs
from .test_affinity_inputs import AFFINITY_YAML

MAIN_PY = os.path.join(stack.tree_dir(), "stock", "src", "boltz", "main.py")
TRUNK_SRC = os.path.join(stack.tree_dir(), "opt", "forward", "trunk_levers", "src")
BASES = ["bz_worker_lev.py", "bz_worker_levf2.py"]


def test_the_affinity_options_are_the_stock_cli_defaults():
    src = open(MAIN_PY).read()
    def option(name):
        m = re.search(r'@click\.option\(\s*"--%s",(.*?)\)\s*\n(?=@click|def )' % name, src, re.S); assert m, name
        return m.group(1)
    assert re.search(r"default=200\b", option("sampling_steps_affinity")) and AL.DEFAULTS["sampling_steps_affinity"] == 200
    assert re.search(r"default=5\b", option("diffusion_samples_affinity")) and AL.DEFAULTS["diffusion_samples_affinity"] == 5
    assert "is_flag=True" in option("affinity_mw_correction") and "default" not in option("affinity_mw_correction") and AL.DEFAULTS["affinity_mw_correction"] is False
    assert re.search(r"default=None\b", option("affinity_checkpoint")) and AL.DEFAULTS["affinity_checkpoint"] is None and AL.AFFINITY_CKPT == "boltz2_aff.ckpt"
    assert 'affinity_checkpoint = cache / "boltz2_aff.ckpt"' in src
    assert AL.AFFINITY_CKPT in stack.CACHE_FILES, "the frozen cache the kit gates on carries the affinity checkpoint"


def test_the_leg_is_main_py_s_affinity_block_verbatim_but_for_its_three_named_substitutions():
    """affinity_leg.run's block == boltz/main.py predict()'s from '# Check if affinity predictions are needed' to the end of its trainer.predict,
    once the three substitutions its docstring names are applied to main.py's text (names from the request; the asdict'ed dicts; the Trainer
    constructed instead of reused) — statement by statement."""
    main = open(MAIN_PY).read()
    i = main.index("    # Check if affinity predictions are needed"); j = main.index("return_predictions=False,\n        )\n", i) + len("return_predictions=False,\n        )\n")
    stock = textwrap.dedent(main[i:j])
    stock = (stock.replace("            return\n", "            return 0\n").replace("        return\n", "        return 0\n")
                  .replace("asdict(diffusion_params)", "asdict_diffusion_params").replace("asdict(pairformer_args)", "asdict_pairformer_args").replace("asdict(msa_args)", "asdict_msa_args")
                  .replace("    trainer.callbacks[0] = pred_writer\n",
                           '    trainer = Trainer(default_root_dir=out_dir, strategy="auto", callbacks=[pred_writer], accelerator="gpu", devices=1, precision="bf16-mixed")   # (3)\n'))
    leg = open(AL.__file__).read()
    a = leg.index("    # Check if affinity predictions are needed"); b = leg.index("return_predictions=False,\n        )\n", a) + len("return_predictions=False,\n        )\n")
    mine = textwrap.dedent(leg[a:b])
    norm = lambda t: [l.rstrip() for l in t.splitlines() if l.strip()]   # noqa: E731
    assert norm(mine) == norm(stock), "\n".join(f"{x!r}\n{y!r}" for x, y in zip(norm(mine), norm(stock)) if x != y)
    # and main.py constructs the Trainer it reuses there with exactly these arguments (strategy auto at one device, bf16-mixed for boltz2)
    assert re.search(r'trainer = Trainer\(\s*default_root_dir=out_dir,\s*strategy=strategy,\s*callbacks=\[pred_writer\],\s*accelerator=accelerator,\s*devices=devices,\s*precision=32 if model == "boltz1" else "bf16-mixed",\s*\)', main)
    pre = leg[leg.index("def preamble("):leg.index("def set_rng(")]
    for line in ('warnings.filterwarnings("ignore", ".*that has Tensor Cores. To properly utilize them.*")', "torch.set_grad_enabled(False)",
                 'torch.set_float32_matmul_precision("highest")', "Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)",
                 'for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:', 'os.environ[key] = os.environ.get(key, "1")'):
        assert line in pre and line.split("(")[0].strip() in main, line


def test_request_write_request_and_command(tmp_path):
    req = AL.request(out_dir=Path("/o/boltz_results_x"), targets_dir=Path("/o/boltz_results_x/processed/structures"), msa_dir=Path("/o/p/msa"), constraints_dir=None,
                     template_dir=None, extra_mols_dir=Path("/o/p/mols"), cache=Path("/cache"), mol_dir=Path("/cache/mols"), num_workers=1,
                     diffusion_process_args={"step_scale": 1.5}, pairformer_args={"v2": True}, msa_args={"subsample_msa": False})
    assert req["out_dir"] == "/o/boltz_results_x" and req["constraints_dir"] is None and req["num_workers"] == 1 and req["diffusion_process_args"] == {"step_scale": 1.5}
    assert {k: req[k] for k in AL.DEFAULTS} == AL.DEFAULTS
    assert AL.request(**{**{k: None for k in ("out_dir", "targets_dir", "msa_dir", "constraints_dir", "template_dir", "extra_mols_dir", "cache", "mol_dir")}, "num_workers": 2,
                         "diffusion_process_args": {}, "pairformer_args": {}, "msa_args": {}}, diffusion_samples_affinity=3)["diffusion_samples_affinity"] == 3
    with pytest.raises(ValueError):
        AL.request(out_dir=None, targets_dir=None, msa_dir=None, constraints_dir=None, template_dir=None, extra_mols_dir=None, cache=None, mol_dir=None, num_workers=1,
                   diffusion_process_args={}, pairformer_args={}, msa_args={}, sampling_steps=10)
    json.dumps(req)                                                          # JSON-able
    rng = {"torch": b"\\x01\\x02", "np": ("MT19937", [1, 2, 3], 624, 0, 0.0), "py": (3, (1, 2, 3), None), "cuda": b"\\x00" * 16}
    path = AL.write_request(tmp_path / "leg" / "x_s0", req, rng)
    assert path == str(tmp_path / "leg" / "x_s0" / "request.json")
    back = json.load(open(path)); assert back["rng"] == str(tmp_path / "leg" / "x_s0" / "rng.pkl") and {k: v for k, v in back.items() if k != "rng"} == req
    assert pickle.load(open(back["rng"], "rb")) == rng
    assert AL.command(path) == [sys.executable, "-s", "-m", "boltz2_opt.affinity_leg", path] and AL.command(path, python="/x/py")[0] == "/x/py"
    assert AL.main([]) == 2


def _leg_region(base):
    src = open(os.path.join(TRUNK_SRC, base)).read()
    i = src.index("# ---------------- upstream's affinity leg"); j = src.index('\nITEMS = B["items"]\n')
    return src[i:j]


@pytest.mark.parametrize("base", BASES)
def test_the_worker_s_run_affinity_leg_hands_over_the_streams_and_names_a_failed_leg(tmp_path, monkeypatch, base):
    """bz_worker_lev*.py run_affinity_leg, verbatim under stand-ins: no property → None, nothing written; the property → request.json + rng.pkl
    under KD/_affinity_leg/<name>_s<seed>/ (the launch's kit directory), the clean interpreter started on them (AL.command), affinity_<id>.json required, else a named error."""
    out = tmp_path / "out"; item_dir = out / "boltz_results_b"; (item_dir / "predictions" / "b").mkdir(parents=True)
    events = []
    processed = types.SimpleNamespace(manifest=types.SimpleNamespace(records=[types.SimpleNamespace(id="b", affinity=None)]), targets_dir=item_dir / "processed" / "structures",
                                      msa_dir=item_dir / "processed" / "msa", constraints_dir=None, template_dir=None, extra_mols_dir=item_dir / "processed" / "mols")
    ns = dict(os=os, sys=sys, time=__import__("time"), subprocess=subprocess, OUT=out, KD=out / "_kit", CACHE=Path("/cache"), mol_dir=Path("/cache/mols"), args=types.SimpleNamespace(num_workers=1), OPTS={"affinity": {"sampling_steps_affinity": 9}},
              ev=lambda name, **kw: events.append((name, kw)), rng_state=lambda: {"torch": b"t", "np": ("MT19937",), "py": (3,), "cuda": b"c"}, WR=_writer_mod,
              stock_model_kwargs=lambda: {"diffusion_process_args": {"step_scale": 1.5}, "pairformer_args": {"v2": True}, "msa_args": {"use_paired_feature": True}, "predict_args": {}})
    exec(_leg_region(base), ns)
    run_affinity_leg = ns["run_affinity_leg"]
    assert run_affinity_leg(item_dir, processed, {"name": "b"}, 0) is None and not (out / "_kit" / "_affinity_leg").exists() and events == []
    processed.manifest.records[0].affinity = types.SimpleNamespace(binder="L")
    writes_json = ("import json, os, pickle, sys; r = json.load(open(sys.argv[-1])); pickle.load(open(r['rng'], 'rb')); "
                   "d = os.path.join(r['out_dir'], 'predictions', 'b'); open(os.path.join(d, 'affinity_b.json'), 'w').write(json.dumps({'num_workers': r['num_workers'], 'steps': r['sampling_steps_affinity']}))")
    monkeypatch.setattr(AL, "command", lambda path, python=None: [sys.executable, "-c", writes_json, path])
    dt = run_affinity_leg(item_dir, processed, {"name": "b"}, 7)
    assert isinstance(dt, float) and [e[0] for e in events] == ["affinity_leg_start", "affinity_leg_done"]
    req = json.load(open(out / "_kit" / "_affinity_leg" / "b_s7" / "request.json"))
    assert req["out_dir"] == str(item_dir) and req["cache"] == "/cache" and req["msa_args"] == {"use_paired_feature": True} and req["sampling_steps_affinity"] == 9 and req["diffusion_samples_affinity"] == 5   # the caller's --sampling_steps_affinity (batch options.affinity) over upstream's defaults
    assert pickle.load(open(req["rng"], "rb")) == {"torch": b"t", "np": ("MT19937",), "py": (3,), "cuda": b"c"}
    assert json.load(open(item_dir / "predictions" / "b" / "affinity_b.json")) == {"num_workers": 1, "steps": 9}
    os.remove(item_dir / "predictions" / "b" / "affinity_b.json"); events.clear()
    monkeypatch.setattr(AL, "command", lambda path, python=None: [sys.executable, "-c", "import sys; sys.exit(4)", path])   # a failing leg: named (event), no raise — the pass goes on, the parent counts the unit failed by the missing json
    assert isinstance(run_affinity_leg(item_dir, processed, {"name": "b"}, 7), float)
    assert [e[0] for e in events] == ["affinity_leg_start", "affinity_leg_failed"] and events[1][1]["rc"] == 4 and "affinity_b.json absent" in events[1][1]["reason"]
    writes_then_dies = writes_json + "; sys.exit(5)"
    monkeypatch.setattr(AL, "command", lambda path, python=None: [sys.executable, "-c", writes_then_dies, path]); events.clear()
    run_affinity_leg(item_dir, processed, {"name": "b"}, 7)
    assert events[1][0] == "affinity_leg_failed" and not (item_dir / "predictions" / "b" / "affinity_b.json").exists(), "a non-zero leg's json is not kept"
    monkeypatch.setattr(AL, "command", lambda path, python=None: [sys.executable, "-c", "pass", path]); events.clear()            # exit 0 but no json: named too
    run_affinity_leg(item_dir, processed, {"name": "b"}, 7)
    assert events[1][0] == "affinity_leg_failed" and events[1][1]["rc"] == 0


@pytest.mark.parametrize("base", BASES)
def test_the_loops_call_the_leg_between_predict_and_the_outputs_move(base):
    src = open(os.path.join(TRUNK_SRC, base)).read()
    pipelined = src[src.index("\nif args.pipeline:\n"):src.index("\nfor it in ITEMS:\n")]; plain = src[src.index("\nfor it in ITEMS:\n"):]
    for body, pred in ((pipelined, "predict_pipelined(dm, it_, st, out_dir, processed)"), (plain, "trainer.predict(MODEL, datamodule=data_module, return_predictions=False)")):
        a, b, c = body.index(pred), body.index("t_aff = run_affinity_leg(out_dir, processed, it, s)"), body.index('dst = OUT / "by_seed"')
        assert a < b < c and '"affinity_s": t_aff' in body


def test_an_affinity_input_s_unit_is_complete_only_with_its_affinity_json(tmp_path, monkeypatch, capsys):
    assert worker.expected_files("x", False) == ["x_model_0.cif"] and worker.expected_files("x", True) == ["x_model_0.cif", "affinity_x.json"]
    _stubs.gated_ok(monkeypatch); _stubs.run_staged_worker_here(monkeypatch)
    plain = _stubs.write_yamls(str(tmp_path), ("a",))[0]; aff = os.path.join(str(tmp_path), "withaff.yaml"); open(aff, "w").write(AFFINITY_YAML)
    _stubs.install_fake_stage(monkeypatch)                                  # the leg wrote affinity_withaff.json beside the structure
    rep.reset_tally()
    rc = worker.run("fast", [plain, aff], str(tmp_path / "o1"), [0])
    text = capsys.readouterr().out
    assert rc == 0, text
    m = mf.LAST
    assert m["report"]["outputs"]["status"] == "complete" and m["report"]["outputs"]["ok"] == 2 and os.path.isfile(tmp_path / "o1" / "by_seed" / "withaff" / "s0" / "affinity_withaff.json")
    _stubs.install_fake_stage(monkeypatch, affinity_json=False)             # the structure is there, the affinity json is not: the unit is failed, named
    rep.reset_tally()
    rc = worker.run("fast", [plain, aff], str(tmp_path / "o2"), [0])
    text = capsys.readouterr().out
    assert rc == rep.EXIT_FAILED == 1, text
    assert "[boltz2-opt] FAILED item=withaff seed=0 reason=outputs short (no affinity_withaff.json under by_seed/withaff/s0/)" in text
    m = mf.LAST
    assert m["report"]["outputs"]["ok"] == 1 and m["report"]["outputs"]["failed"] == 1
    assert m["report"]["outputs"]["failed_units"] == [{"name": "withaff", "seed": 0, "reason": "outputs short (no affinity_withaff.json under by_seed/withaff/s0/)"}]


def test_affinity_inputs_reads_the_property_key_case_insensitively_as_upstream_does(tmp_path):
    p = os.path.join(str(tmp_path), "cap.yaml"); open(p, "w").write(AFFINITY_YAML.replace("- affinity:", "- Affinity:"))
    q = os.path.join(str(tmp_path), "other.yaml"); open(q, "w").write(AFFINITY_YAML.replace("- affinity:", "- affinity_like_thing:"))
    assert worker.affinity_inputs(worker.items_from_yamls([p, q])) == ["cap"]
    schema = os.path.join(stack.tree_dir(), "stock", "src", "boltz", "data", "parse", "schema.py")   # carried in the wheel; the tree's src copy holds main.py and the model
    src = open(schema).read() if os.path.isfile(schema) else ""
    assert not src or "next(iter(prop.keys())).lower()" in src


# ---------------------------------------------------------------- n_gpu > 1: the leg on rank 0 only (affinity_leg=rank0_only) ----------------------
WRITES_JSON = ("import json, os, sys; r = json.load(open(sys.argv[-1])); d = os.path.join(r['out_dir'], 'predictions', 'b'); os.makedirs(d, exist_ok=True); "
               "open(os.path.join(d, 'affinity_b.json'), 'w').write(json.dumps({'by': os.environ.get('ROWPAIR_RANK')})); open(os.path.join(r['out_dir'], 'leg_ran'), 'w').write('x')")


def _xp_entry(tmp, leg_rc, how="run"):
    """Rank body: every rank writes its own request (its own out_dir, as the worker does) and calls execute(); the clean interpreter is a stand-in
    that writes affinity_b.json + a `leg_ran` marker into the request's out_dir and exits ``leg_rc``. ``how``: ``run`` (that), ``no_interpreter``
    (rank 0's interpreter path does not exist: subprocess.run raises before anything is sent), ``rank1_cannot_copy`` (rank 1's own
    ``affinity_b.json`` path is a non-empty directory: its copy raises)."""
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    cm = D.comm()
    out_dir = os.path.join(tmp, f"rank{cm.rank}", "boltz_results_b"); os.makedirs(os.path.join(out_dir, "predictions", "b"))
    own = os.path.join(out_dir, "predictions", "b", "affinity_b.json")
    if how == "rank1_cannot_copy" and cm.rank == 1:
        os.makedirs(os.path.join(own, "blocker"))                                                    # unlink(own) raises: a directory
    path = AL.write_request(os.path.join(tmp, f"kit{cm.rank}", "_affinity_leg", "b_s0"), {"out_dir": out_dir, **AL.DEFAULTS}, {"torch": b"t"})
    exe = "/nonexistent/python" if how == "no_interpreter" else sys.executable
    AL.command = lambda p, python=None: [exe, "-c", WRITES_JSON + f"; sys.exit({int(leg_rc)})", p]      # this rank process's stand-in for the clean interpreter
    try:
        rc, refused = AL.execute(path), None
    except RowpairRefused as e:
        rc, refused = None, str(e)
    mine = {"rank": cm.rank, "rc": rc, "refused": refused, "leg_ran": os.path.exists(os.path.join(out_dir, "leg_ran")), "json": os.path.isfile(own), "islink": os.path.islink(own),
            "content": json.load(open(own)) if os.path.isfile(own) else None, "out_dir": out_dir, "calls": dict(AL._CALLS)}
    return cm.allgather_obj(mine)


@pytest.mark.parametrize("leg_rc", [0, 4])
def test_at_n_gpu_2_the_leg_runs_on_rank0_only_and_its_exit_code_and_files_reach_the_other_rank(tmp_path, monkeypatch, leg_rc):
    pytest.importorskip("torch")
    RD = pytest.importorskip("opt_core.mem.rowpair.rankdata", reason="the core's rank-data contract (rankdata.broadcast_features) is required")
    if not hasattr(RD, "broadcast_features"):
        pytest.skip("the core's rank-data contract (rankdata.broadcast_features) is required")
    from opt_core.mem.rowpair import launch
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    r0, r1 = launch.run_sharded(2, _xp_entry, str(tmp_path), leg_rc, mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=300, log_dir=str(tmp_path))
    assert (r0["rank"], r1["rank"]) == (0, 1) and r0["rc"] == r1["rc"] == leg_rc and r0["refused"] is r1["refused"] is None   # rank 0's exit code is every rank's
    assert r0["leg_ran"] is True and r1["leg_ran"] is False                                        # no leg process on rank 1
    assert r0["calls"] == r1["calls"] == {"affinity": 1}                                           # one store key per leg, the same on every rank
    if leg_rc == 0:
        assert r0["json"] and not r0["islink"] and r0["content"] == {"by": "0"}
        assert r1["json"] and not r1["islink"] and r1["content"] == {"by": "0"}                 # rank 1 holds a copy of rank 0's result (rank 0's leg wrote it: by=0)
    else:
        assert not r1["json"]                                                                      # a failed leg: nothing copied; the worker names the unit failed on every rank alike
    log1 = open(os.path.join(str(tmp_path), "rank1.log"), errors="replace").read()
    assert f"[boltz2-opt] affinity_leg=rank0_only rank=1 ranks=2 rc={leg_rc} files={1 if leg_rc == 0 else 0} " in log1, log1[-1500:]


def test_solo_execute_is_the_clean_interpreter_and_the_words_are_one(tmp_path, monkeypatch):
    """n_gpu = 1: execute(path) == subprocess.run(command(path)).returncode, nothing sent anywhere; the ×P word and the line tag have one spelling."""
    from boltz2_opt import report
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False); monkeypatch.delenv("ROWPAIR_RANK", raising=False)
    monkeypatch.setattr(AL, "command", lambda p, python=None: [sys.executable, "-c", "import sys; sys.exit(6)", p])
    assert AL.execute(str(tmp_path / "request.json")) == 6
    seen = {}
    monkeypatch.setattr(AL, "command", lambda p, python=None: (seen.__setitem__("python", python), [sys.executable, "-c", "pass", p])[1])
    assert AL.execute("r.json", python="/x/py") == 0 and seen["python"] == "/x/py"                 # a caller's interpreter reaches command()
    assert AL.tp_word() == "affinity_leg=rank0_only" and AL.TP_FORM == "rank0_only" and AL._prefix() == report.PREFIX == "[boltz2-opt]"
    from opt_core.mem.rowpair import RowpairRefused
    monkeypatch.setenv("ROWPAIR_WORLD", "2"); monkeypatch.setenv("ROWPAIR_RANK", "1")                   # a rank of a ×P launch without its group: refused by name, no leg run per rank in silence
    with pytest.raises(RowpairRefused, match="has no rank group for the affinity leg"):
        AL.execute(str(tmp_path / "request.json"))


@pytest.mark.parametrize("how, word", [("no_interpreter", "affinity_rank0_failed: FileNotFoundError"), ("rank1_cannot_copy", "affinity_result_not_taken: rank(s) [1] of 2")])
def test_at_n_gpu_2_a_leg_rank0_cannot_start_or_a_copy_a_rank_cannot_make_is_refused_by_name_on_every_rank(tmp_path, monkeypatch, how, word):
    pytest.importorskip("torch")
    RD = pytest.importorskip("opt_core.mem.rowpair.rankdata", reason="the core's rank-data contract (rankdata.broadcast_features) is required")
    if not hasattr(RD, "broadcast_features"):
        pytest.skip("the core's rank-data contract (rankdata.broadcast_features) is required")
    from opt_core.mem.rowpair import launch
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    r0, r1 = launch.run_sharded(2, _xp_entry, str(tmp_path), 0, how, mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=300, log_dir=str(tmp_path))
    assert r0["rc"] is None and r1["rc"] is None                                                   # no rank returned an exit code: both raised
    assert word in r0["refused"] and word in r1["refused"], (r0["refused"], r1["refused"])       # the SAME name on every rank
    assert r0["calls"] == r1["calls"] == {"affinity": 1}                                           # the store key was numbered on both ranks before anything else
    if how == "no_interpreter":
        assert r0["leg_ran"] is False and r1["leg_ran"] is False and not r1["json"]
    else:
        assert r0["leg_ran"] is True and r0["json"] and not r1["json"]                             # rank 0's leg ran and wrote; rank 1 holds nothing
        log1 = open(os.path.join(str(tmp_path), "rank1.log"), errors="replace").read()
        assert "[boltz2-opt] affinity_leg=rank0_only rank=1 ranks=2 rc=0 files=1 " in log1        # the line is printed before the refusal


SERVED_STANDIN = textwrap.dedent("""
    import json, os, sys
    sys.path[:0] = %r
    from boltz2_opt import affinity_leg as AL
    N = {"n": 0}
    def handle(path):
        r = json.load(open(path)); N["n"] += 1
        print(f"standin leg {N['n']} for {path} pid={os.getpid()}", flush=True)
        if r.get("die"): os._exit(7)
        if r.get("fail"): raise ValueError("boom " + path)
        d = os.path.join(r["out_dir"], "predictions", "b"); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "affinity_b.json"), "w").write(json.dumps({"pid": os.getpid(), "n": N["n"]}))
        return 0
    sys.exit(AL.serve(handle=handle))
""")


def _req(tmp_path, name, **extra):
    d = tmp_path / name; (d / "predictions" / "b").mkdir(parents=True)
    p = d / "request.json"; p.write_text(json.dumps({"out_dir": str(d), "rng": str(d / "rng.pkl"), **extra})); (d / "rng.pkl").write_bytes(pickle.dumps({}))
    return str(p), d / "predictions" / "b" / "affinity_b.json"


def test_the_pass_s_legs_run_in_one_served_interpreter_and_each_leg_s_exit_code_and_transcript_come_back(tmp_path, monkeypatch, capsys):
    """0.3.24, form `process` (the default): execute() hands every request of the pass to ONE clean interpreter (`--serve`): one process for
    all legs (same pid in what the stand-in wrote), each leg's rc from its RESULT line, its transcript echoed into this one (the RESULT lines
    are protocol, not transcript); a leg that raises is rc 1 and the interpreter serves on; one that kills the interpreter is failed by its exit
    code and the next leg starts a fresh interpreter; stdin closed = it exits 0. Form `item` (by name): one interpreter per leg, AL.command."""
    monkeypatch.delenv(AL.FORM_ENV, raising=False); AL.reset_for_tests()
    paths = [os.path.dirname(os.path.dirname(AL.__file__))] + [p for p in sys.path if p]
    monkeypatch.setattr(AL, "server_command", lambda python=None: [sys.executable, "-c", SERVED_STANDIN % paths])
    p1, j1 = _req(tmp_path, "a_s0"); p2, j2 = _req(tmp_path, "b_s0"); p3, j3 = _req(tmp_path, "c_s0", fail=True); p4, j4 = _req(tmp_path, "d_s0", die=True); p5, j5 = _req(tmp_path, "e_s0")
    assert AL.leg_form() == "process" and AL.execute(p1) == 0 and AL.execute(p2) == 0
    a, b = json.load(open(j1)), json.load(open(j2))
    assert a["pid"] == b["pid"] and (a["n"], b["n"]) == (1, 2) and AL._SERVER["starts"] == 1, "one interpreter served both legs"
    out = capsys.readouterr().out
    assert f"standin leg 1 for {p1}" in out and f"standin leg 2 for {p2}" in out and AL.RESULT not in out, "the legs' words are in this transcript; the protocol line is not"
    assert AL.execute(p3) == 1 and not j3.exists() and AL._SERVER["proc"] is not None and AL._SERVER["proc"].poll() is None, "a leg that raised: rc 1, the interpreter serves on"
    assert AL.execute(p4) == 7 and AL._SERVER["proc"] is None, "a leg that killed the interpreter is failed by its exit code"
    assert AL.execute(p5) == 0 and json.load(open(j5))["pid"] != a["pid"] and AL._SERVER["starts"] == 2, "the next leg started a fresh interpreter"
    assert AL.close_server(timeout=30) == 0 and AL._SERVER["proc"] is None
    monkeypatch.setenv(AL.FORM_ENV, "item")                                          # the one-shot form by name: AL.command per leg, no served interpreter
    monkeypatch.setattr(AL, "command", lambda path, python=None: [sys.executable, "-c", "import sys; sys.exit(3)", path])
    assert AL.leg_form() == "item" and AL.execute(p1) == 3 and AL._SERVER["proc"] is None
    AL.reset_for_tests()


def test_serve_answers_one_result_line_per_request_and_survives_a_failing_leg(capsys):
    import io
    seen = []
    def handle(path):
        seen.append(path)
        if path == "r2": raise RuntimeError("x")
        if path == "r3": raise SystemExit(5)
        return 0
    assert AL.serve(stdin=io.StringIO("r1\n\nr2\nr3\nr4\n"), handle=handle) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith(AL.RESULT)]
    assert seen == ["r1", "r2", "r3", "r4"] and lines == [f"{AL.RESULT} rc=0 request=r1", f"{AL.RESULT} rc=1 request=r2", f"{AL.RESULT} rc=5 request=r3", f"{AL.RESULT} rc=0 request=r4"]
    assert AL.main([AL.SERVE_FLAG, "x"]) == 2 and AL.server_command("/p/y") == ["/p/y", "-s", "-m", "boltz2_opt.affinity_leg", AL.SERVE_FLAG]


class _restricted_classmethod:
    """pytorch_lightning's descriptor for LightningModule.load_from_checkpoint: read on the class it hands back a plain FUNCTION with the class
    bound inside (no __self__ / __func__; __wrapped__ = the undecorated method) and refuses instance calls."""
    def __init__(self, method): self.method = method
    def __get__(self, instance, cls):
        import functools
        @functools.wraps(self.method)
        def wrapper(*args, **kwargs):
            if instance is not None: raise TypeError("load_from_checkpoint cannot be called on an instance")
            return self.method(cls, *args, **kwargs)
        return wrapper


def _fake_module_class(form):
    torch = pytest.importorskip("torch"); import random; import numpy as np
    def _load(cls, checkpoint_path, strict=True, **kwargs):
        m = cls(); m.kwargs = kwargs; return m
    class Fake:
        inits = 0
        def __init__(self): type(self).inits += 1; self.w = torch.randn(4); self.u = np.random.rand(2); self.r = random.random()   # a constructor drawing on all three streams
        def cpu(self): return self
    Fake.load_from_checkpoint = classmethod(_load) if form == "classmethod" else _restricted_classmethod(_load)
    return Fake


@pytest.mark.parametrize("form", ["classmethod", "lightning_restricted_classmethod"])
def test_the_memoised_constructor_replays_its_draws_so_a_held_model_leaves_the_streams_where_a_fresh_build_does(form):
    """memoize_constructor (the served interpreter's one substitution): same (checkpoint, arguments, pre-construction stream states) → the
    module built before is returned AND the torch-CPU / numpy / python streams are set to what that construction left, so the draws after it
    (the DataLoader's base seed, …) read as after a fresh build; other arguments or other pre-states → a real construction. Torch CPU here;
    both the plain classmethod form and pytorch_lightning's restricted-classmethod descriptor (what Boltz2.load_from_checkpoint is)."""
    torch = pytest.importorskip("torch"); import random; import numpy as np
    AL.reset_for_tests()
    Fake = _fake_module_class(form)
    def seed(t, n, p): torch.manual_seed(t); np.random.seed(n); random.seed(p)
    def after(): return (float(torch.rand(1)), float(np.random.rand()), random.random())
    w = AL.memoize_constructor(Fake); assert AL.memoize_constructor(Fake) is w, "idempotent"
    seed(1, 2, 3); m1 = Fake.load_from_checkpoint("ck.pt", predict_args={"recycling_steps": 5}, ema=False); x1 = after()
    seed(1, 2, 3); m2 = Fake.load_from_checkpoint("ck.pt", predict_args={"recycling_steps": 5}, ema=False); x2 = after()
    assert m2 is m1 and Fake.inits == 1 and x2 == x1 and (AL._MEMO["builds"], AL._MEMO["hits"]) == (1, 1), "held model, constructor draws replayed exactly"
    seed(1, 2, 3); m3 = Fake.load_from_checkpoint("ck.pt", predict_args={"recycling_steps": 6}, ema=False)
    assert m3 is not m1 and Fake.inits == 2, "other arguments: a real construction"
    seed(9, 2, 3); m4 = Fake.load_from_checkpoint("ck.pt", predict_args={"recycling_steps": 5}, ema=False); x4 = after()
    seed(9, 2, 3); Fake.inits_before = Fake.inits; m5 = Fake.load_from_checkpoint("ck.pt", predict_args={"recycling_steps": 5}, ema=False); x5 = after()
    assert m4 is not m1 and m5 is m4 and x5 == x4 and Fake.inits == 3, "other pre-construction states: their own construction, then held likewise"
    AL.reset_for_tests()


@pytest.mark.parametrize("base", BASES)
def test_the_worker_starts_the_served_interpreter_when_it_parses_an_affinity_input_and_logs_the_form(base):
    src = open(os.path.join(TRUNK_SRC, base)).read()
    helper = src[src.index("def warm_affinity_leg(processed):"):src.index("def run_affinity_leg(")]
    assert 'if any(getattr(r, "affinity", None) for r in processed.manifest.records):' in helper and "AL.warm()" in helper, "only for inputs declaring the property"
    serial = src[src.index("\nfor it in ITEMS:\n"):]; piped = src[src.index("    def get_proc(it):"):src.index("    def ready(k):")]
    assert serial.index("warm_affinity_leg(processed)") < serial.index("rng_set(POST_CTOR[s])") and "warm_affinity_leg(PROC[it[\"name\"]][1])" in piped, "ahead of the structure pass, on both loops"
    assert 'LOG["env"]["affinity_leg"] = AL.leg_form()' in src and src.index("from boltz2_opt import affinity_leg as AL") < src.index('\nITEMS = B["items"]\n') < src.index('LOG["env"]["affinity_leg"]')


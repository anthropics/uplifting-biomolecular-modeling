"""The packed line (`design --pack K`): the served resolution (the row on the same driver, flag for flag, on the exact and fast rows; the
design route's exact line literally unchanged), the facts that refuse it (off, a request with no kit row), the request slicing rule,
the launcher template (`{W}` in the per-worker slots, shell-quoted; every worker
writes at the target's own prefix, the run record in the prefix's directory), the gates (the launcher file, the MPS control binary), and a run through
the packing launcher with the GPU tools and the driver stubbed (the memory-estimate arithmetic, the per-worker logs, the rc relay, the evidence
read-back, the manifest)."""
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys

import pytest

from rfdiffusion1_opt import report, cli, design, modes, registry, serve, stack
from rfdiffusion1_opt.registry import BASE_DRIVER, KIT_BASE
from rfdiffusion1_opt.tests._stubs import good_box

DESIGN_EXACT_LINE = "RFD_PDBIO=1 <fast_inference>/drivers/rfd_bench.py --fastpath chain_breaks,full_graph,rbf,msa_index --prep 1 --einsum-route 1 --fullgraph 1"
SERVED_EXACT_LINE = DESIGN_EXACT_LINE                                        # the served line = the design route's row on the same driver, flag for flag
LAUNCHER_SHA256 = "f141955dabc978a01ef04fdf43a7bf5b95405512f7163bf306455fa9788012dd"          # common/mps_packing/mps_workers.sh as carried (the memory estimate a NOTE line, never a refusal)
PDB = os.path.join(stack.kit_dir(KIT_BASE), "inputs_public", "insulin_target.pdb")   # the bundled insulin-receptor example target
REMOVED_FLAGS = ("--input", "--out_dir", "--num-designs", "--startnum", "--tag")       # the kit-invented cases-JSON form is gone: `design` takes upstream's overrides and nothing else


def target(out, n=4, start=0, *more):
    """Upstream's own overrides for ONE target (the bundled example) at prefix <out>/binder — the request of `design [--pack K]`; the run record
    (opt_manifest.json, logs, timings) lands in <out>, the prefix's directory."""
    return [f"inference.input_pdb={PDB}", "contigmap.contigs=[A1-115/0 80-80]", "ppi.hotspot_res=[A59,A83,A91]", f"inference.output_prefix={os.path.join(str(out), 'binder')}",
            f"inference.num_designs={n}", f"inference.design_startnum={start}", *more]


# --------------------------------------------------------------------------------------------------------------------- the table
def test_design_exact_line_is_unchanged_by_the_served_line():
    assert modes.resolve("exact").line == DESIGN_EXACT_LINE
    assert modes.resolve("exact").levers == ("U1", "C1", "P", "E_einsum", "W1", "IO1") and not modes.resolve("exact").served
    assert "--gpu-poll" not in modes.resolve("exact").flags


def test_served_row():
    r = modes.resolve("exact", served=True)
    assert r.served and r.line == SERVED_EXACT_LINE
    assert r.levers == ("U1", "C1", "P", "E_einsum", "W1", "IO1") == modes.resolve("exact").levers and r.tier == 1 and r.env == {"RFD_PDBIO": "1"}   # the row's own levers: the served line adds none
    assert r.driver_chain == ((KIT_BASE, BASE_DRIVER),) == modes.resolve("exact").driver_chain                # the same driver as the design route
    assert r.flags == tuple(modes.K_FLAGS) and modes.serve_flags("exact") == () == modes.serve_flags("fast")
    f, fd = modes.resolve("fast", served=True), modes.resolve("fast")                                     # the fast row packs too: the same driver, fast's flags and environment row
    assert f.served and f.driver_chain == fd.driver_chain and f.levers == fd.levers and f.env == fd.env and f.tier == fd.tier == 2
    assert f.flags == fd.flags and f.line == fd.line == SERVED_EXACT_LINE.replace("RFD_PDBIO=1 ", "RFD_PDBIO=1 RFD_SE3FAST=t2 ") + " --triton-ln 1 --tf32 1"
    assert modes.resolve(None, served=True) == f                                                          # no --mode: fast, on the packed line too
    assert "served: K workers under CUDA MPS" in r.note and "--gpu-poll" not in r.note
    for gone in ("SERVE_JIT_WARMUP", "SERVE_WARMUP_FLAG", "SERVE_DRIVER", "SERVE_LEVERS", "SERVE_POLL"):  # no warm-up lever, no driver copy, no sampler flag: the served line is the row
        assert not hasattr(modes, gone), gone
    assert "G1" not in registry.LEVERS and not hasattr(registry, "KIT_GK")
    # the driver times nothing and samples nothing (no NVML poller, no phase timers): its run record is the evidence file alone
    drv = open(os.path.join(stack.kit_dir(KIT_BASE), BASE_DRIVER), encoding="utf-8").read()
    assert "--gpu-poll" not in drv and "GpuPoller" not in drv and "PhaseTimer" not in drv and "perf_counter" not in drv and "--jit-warmup" not in drv


def test_served_line_refused_by_name_where_not_served():
    with pytest.raises(modes.ModeError, match=re.escape("--pack has no stock row: the stock command line is one process per invocation (stock/src/scripts/run_inference.py), no resident worker to pack; run --mode off without --pack") + "$"):
        modes.resolve("off", served=True)
    assert set(modes.NOT_SERVED) == {"off"} and all(t.startswith("--pack ") for t in modes.NOT_SERVED.values())   # the packed line's own fact, in the flag's name (no `serve` verb exists)
    rows = [r for r in modes.table() if r.get("served")]
    assert {(r["mode"], r["defined"]) for r in rows} == {("off", False), ("exact", True), ("fast", True)}   # a served row per mode; the kit packs exact and fast
    assert all(r["driver_chain"] == ["fast_inference/drivers/rfd_bench.py"] for r in rows if r["defined"])   # the served rows run the base driver itself


def test_no_composition_vocabulary(monkeypatch, tmp_path, capsys):
    """K is the packed line's only axis: no composition word exists (table, resolutions, --help, serve.run), a lever-set name typed as --mode is
    unknown by name, and no lever-subtraction flag exists (`--without` is gone: a mode runs its row whole)."""
    for gone in ("COMPOSITION_POINTER", "check_composition", "DEFAULT_COMPOSITION", "SERVE_COMPOSITIONS"):
        assert not hasattr(modes, gone), gone
    import inspect
    assert "composition" not in inspect.signature(modes.resolve).parameters and "composition" not in inspect.signature(serve.run).parameters
    with pytest.raises(modes.ModeError, match=re.escape("unknown mode 'exact_w1' (expected off|exact|fast)")):
        modes.resolve("exact_w1")                                                                   # a lever-set name typed as --mode: unknown by name
    assert "without" not in inspect.signature(modes.resolve).parameters and "without" not in inspect.signature(serve.run).parameters and "allow_partial" not in inspect.signature(serve.run).parameters
    assert not any("composition" in json.dumps(r, default=str) for r in modes.table())
    for verb in cli.VERBS:                                                                          # the word is in no --help; nor are the retired estimate flags
        sp = next(a for a in cli.build_parser()._actions if isinstance(a, cli.argparse._SubParsersAction)).choices[verb]
        h = sp.format_help()
        assert "composition" not in h and "--worker-gb" not in h and "--headroom-gb" not in h and "exact_w1" not in h and "--without" not in h and "--allow-partial" not in h, verb
    assert "--pack" in next(a for a in cli.build_parser()._actions if isinstance(a, cli.argparse._SubParsersAction)).choices["design"].format_help()


def test_slice_cases_rule():
    cases = [{"name": "a", "num_designs": 16, "startnum": 0}, {"name": "b", "num_designs": 16, "startnum": 100}]
    s = serve.slice_cases(cases, 4)
    assert [[(c["name"], c["startnum"], c["num_designs"]) for c in w] for w in s] == [
        [("a", 0, 4), ("b", 100, 4)], [("a", 4, 4), ("b", 104, 4)], [("a", 8, 4), ("b", 108, 4)], [("a", 12, 4), ("b", 112, 4)]]
    assert serve.slice_cases(cases, 1)[0] == cases                                     # K = 1: the request itself
    with pytest.raises(serve.ServeError, match="does not slice into K=3 workers"):
        serve.slice_cases(cases, 3)
    with pytest.raises(serve.ServeError, match="K must be >= 1"):
        serve.slice_cases(cases, 0)
    # the union over workers is the K = 1 design-index set
    assert sorted(i for w in s for c in w if c["name"] == "a" for i in range(c["startnum"], c["startnum"] + c["num_designs"])) == list(range(16))


# ------------------------------------------------------------------------------------------------------------------ the template
def test_worker_template_and_environment(monkeypatch, tmp_path):
    good_box(monkeypatch)
    r = modes.resolve("exact", served=True)
    tpl, argv = serve.worker_template(r, str(tmp_path / "o"), str(tmp_path / "work"), "/opt/rfd", "/models/rfdiffusion", python="python")
    assert argv[:9] == ["python", "-m", "rfdiffusion1_opt.driver_run", "--fixed", "inference.write_trajectory=True", "--fixed", "inference.cautious=True", "--fixed", "inference.deterministic=False"]
    assert argv[argv.index("--cases") + 1] == str(tmp_path / "work" / "w{W}" / "cases.json")           # the worker's cases file: in the pass's scratch directory, never among the outputs
    assert argv[argv.index("--out") + 1] == str(tmp_path / "o") and argv[argv.index("--tag") + 1] == "w{W}"   # one output directory for every worker (the cases' own prefixes; disjoint design indices), the run record tagged w<W>
    assert argv[len(argv) - 1 - argv[::-1].index("--cases") + 1] == str(tmp_path / "work" / "w{W}" / "cases.json")   # driver_run's --cases and the driver's
    assert os.path.join(stack.kit_dir(KIT_BASE), BASE_DRIVER) in argv and not any("se3k" in a for a in argv)
    assert "--fullgraph 1 --no-traj 0" in " ".join(argv) and "--gpu-poll" not in argv and tpl.count("{W}") == 3 and shlex.split(tpl.replace("{W}", "3")) == [a.replace("{W}", "3") for a in argv]
    _, dargv = serve.worker_template(modes.resolve("exact", served=True, det=True), str(tmp_path / "o"), str(tmp_path / "work"), "/opt/rfd", "/models/rfdiffusion", python="python")
    assert dargv[7:9] == ["--fixed", "inference.deterministic=True"]                                      # --det 1 reaches every worker's composition
    _, fargv = serve.worker_template(modes.resolve("fast", served=True), str(tmp_path / "o"), str(tmp_path / "work"), "/opt/rfd", "/models/rfdiffusion", python="python")
    assert "--triton-ln 1 --tf32 1 --no-traj 0" in " ".join(fargv)                                          # fast's flags on the same driver
    env, dropped = stack.driver_environment(r, {"name": "NVIDIA H100 80GB HBM3"}, environ={"PATH": "/usr/bin", "RFD_PREP": "0", "PYTHONPATH": "/x"})
    assert env.get("PYTHONPATH") in (None, "/x") and dropped == ["RFD_PREP"] and env["DGLBACKEND"] == "pytorch"   # the served chain imports from its own directory like the design route's: no kit path is laid on PYTHONPATH
    env2, _ = stack.driver_environment(modes.resolve("exact"), {"name": "NVIDIA H100 80GB HBM3"}, environ={"PATH": "/usr/bin"})
    assert "PYTHONPATH" not in env2                                                    # the design route's chain imports from its own directory
    fenv, _ = stack.driver_environment(modes.resolve("fast", served=True), {"name": "NVIDIA H100 80GB HBM3"}, environ={"PATH": "/usr/bin"})
    assert fenv["RFD_SE3FAST"] == "t2" and "PYTHONPATH" not in fenv                          # the fast row's environment reaches the packed workers; no kit path on PYTHONPATH
    lenv = serve.launcher_environment(env, 4, 6.7, str(tmp_path / "o"))
    assert (lenv["K"], lenv["WORKER_GB"], lenv["LOG_DIR"]) == ("4", "6.7", str(tmp_path / "o")) and "HEADROOM_GB" not in lenv
    assert serve.launcher_environment(env, 2, 10.0, str(tmp_path / "o"))["WORKER_GB"] == "10" and cli.PACK_WORKER_GB_DEFAULT == 10.0 and cli.PACK_WORKER_GB_ENV == "MODEL_OPT_PACK_WORKER_GB"
    monkeypatch.delenv(cli.PACK_WORKER_GB_ENV, raising=False)
    assert cli._worker_gb() == 10.0                                            # the per-worker device-memory estimate the launcher's memory-estimate note uses: 10 GB unless the environment names another (configs/*.env)
    monkeypatch.setenv(cli.PACK_WORKER_GB_ENV, "6.7"); assert cli._worker_gb() == 6.7
    monkeypatch.setenv(cli.PACK_WORKER_GB_ENV, "lots"); assert cli._worker_gb() == 10.0      # an unreadable value falls back to the default (the estimate is a NOTE, never a gate)
    mps_root = lenv["MPS_DIR_ROOT"]                                           # the daemon's pipe/log root: outside the output directory (a socket is not an output), fresh per pass
    assert os.path.isdir(mps_root) and not os.path.abspath(mps_root).startswith(os.path.abspath(str(tmp_path / "o")) + os.sep) and os.path.basename(mps_root).startswith("rfdiffusion1_mps_")


# --------------------------------------------------------------------------------------------------------------------- gates
def _served_box(monkeypatch, launcher=None, mps_control="/usr/bin/nvidia-cuda-mps-control"):
    from rfdiffusion1_opt.tests import _stubs
    g = good_box(monkeypatch, tools=dict(_stubs.TOOLS, mps_control=mps_control))
    if launcher is not None:
        monkeypatch.setattr(stack, "pack_launcher", lambda: launcher)
    return g


def test_serve_refusals_before_anything_runs(monkeypatch, tmp_path, capsys):
    stack.reset_for_tests()
    _served_box(monkeypatch, launcher=str(tmp_path / "absent" / "mps_workers.sh"))
    rc, man = serve.run("exact", target(tmp_path / "o"), k=4, worker_gb=6.7)
    assert rc == 3 and man["status"] == "refused" and "the packing launcher is not in this tree: common/mps_packing/mps_workers.sh" in man["reason"] and not (tmp_path / "o").exists()
    launcher = tmp_path / "mps_workers.sh"; launcher.write_text("#!/bin/bash\nexit 0\n")
    _served_box(monkeypatch, launcher=str(launcher), mps_control=None)
    rc, man = serve.run("exact", target(tmp_path / "o"), k=4, worker_gb=6.7)
    assert rc == 3 and "nvidia-cuda-mps-control is not on PATH" in man["reason"]
    _served_box(monkeypatch, launcher=str(launcher))
    rc, man = serve.run("off", target(tmp_path / "o"), k=1)
    assert rc == 3 and "--pack has no stock row" in man["reason"]
    capsys.readouterr()
    rc, man = serve.run("exact", target(tmp_path / "o", 4, 0, "potentials.guide_scale=2"), k=2, dry_run=True, python="python")   # guiding potentials are served: packed like any request of the kit line
    assert rc == 0 and man["mode"] == "exact" and man["status"] == "dry-run" and "declined" not in man and man["serve"]["k"] == 2 and "NOT ACTIVE" not in capsys.readouterr().err
    assert [w[0]["num_designs"] for w in man["serve"]["slices"]] == [2, 2] and "--compose potentials.guide_scale=2" in man["serve"]["template"]
    rc, man = serve.run("exact", target(tmp_path / "o", 4, 0, "inference.symmetry=C3"), k=2, dry_run=True, python="python")      # what the kit line cannot serve: refused by name before any slicing, exit 3 (design's one refusal)
    assert rc == 3 and man["status"] == "refused" and "serve" not in man and "stock_cmds" not in man
    assert "[rfdiffusion1-opt] NOT ACTIVE: mode=exact cannot serve symmetric oligomers [inference.symmetry=C3]: " in capsys.readouterr().err
    rc, man = serve.run("exact", target(tmp_path / "o") + ["--config-name", "symmetry"], k=2, dry_run=True, python="python")   # another primary config: refused by name alike
    assert rc == 3 and man["status"] == "refused" and man["refused"][0].startswith("primary config symmetry.yaml [--config-name symmetry]") and "declined" not in man
    capsys.readouterr()
    rc, man = serve.run("off", target(tmp_path / "o"), k=2, dry_run=True, python="python")   # off has no served line: refused by name, exit 3, nothing runs
    assert rc == 3 and man["status"] == "refused" and man["reason"] == modes.NOT_SERVED["off"]
    assert "[rfdiffusion1-opt] NOT ACTIVE: --pack has no stock row" in capsys.readouterr().err
    rc, man = serve.run("exact", ["inference.num_designs=4"], k=2)                             # no target: a usage error (2), as on the design route
    assert rc == design.EXIT_USAGE == 2 and man["status"] == "usage"
    import inspect
    assert list(inspect.signature(serve.run).parameters)[:2] == ["mode", "overrides"] and not {"cases_path", "out_dir", "num_designs", "startnum"} & set(inspect.signature(serve.run).parameters)   # the packed line's request is the command line's one target, nothing else
    rc, man = serve.run("exact", target(tmp_path / "o", 16), k=3)                              # the box is armed for exact served from here on
    assert rc == 3 and man["reason"] == ("case binder: num_designs 16 does not slice into K=3 workers (the packing rule gives every worker the same designs-per-worker count); "
                                        "make inference.num_designs a multiple of 3")
    assert not (tmp_path / "o").exists()                                                      # every refusal above happened before anything was made
    import inspect
    assert "composition" not in inspect.signature(serve.run).parameters and inspect.signature(serve.run).parameters["worker_gb"].default == 10.0 == cli.PACK_WORKER_GB_DEFAULT
    stack.reset_for_tests()
    rc, man = design.run("exact", target(tmp_path / "d"), dry_run=True, python="python")     # the design route after a served activation: a fresh process arms its own
    assert rc == 0 and man["line"] == DESIGN_EXACT_LINE
    stack.reset_for_tests()
    assert stack.activate("exact", served=True)["active"] is True                             # armed for the served line ...
    rc, man = design.run("exact", target(tmp_path / "d2"), python="python")
    assert rc == 3 and "a different mode or route is refused" in man["reason"]              # ... one process, one line: the design route is refused by name


def test_serve_dry_run(monkeypatch, tmp_path):
    stack.reset_for_tests()
    launcher = tmp_path / "mps_workers.sh"; launcher.write_text("#!/bin/bash\nexit 0\n")
    _served_box(monkeypatch, launcher=str(launcher))
    rc, man = serve.run("exact", target(tmp_path / "o", 8), k=4, worker_gb=6.7, dry_run=True, python="python")
    assert rc == 0 and man["status"] == "dry-run" and man["line"] == SERVED_EXACT_LINE and "jit_warmup" not in man and "without" not in man
    sv = man["serve"]
    assert sv["k"] == 4 and sv["worker_gb"] == 6.7 and sv["launcher"] == str(launcher) and sv["launcher_cmd"] == ["bash", str(launcher), sv["template"]]
    assert [c["startnum"] for c in [s[0] for s in sv["slices"]]] == [0, 2, 4, 6] and all(c["num_designs"] == 2 for s in sv["slices"] for c in s)
    assert sv["template_argv"][sv["template_argv"].index("--out") + 1] == str(tmp_path / "o") and sv["worker_slot"] == "w{W}" and sv["worker_log"] == "mps_worker_{W}.log"   # the workers' run record: the prefix's directory
    assert not (tmp_path / "o").exists()
    rc, man1 = serve.run("exact", target(tmp_path / "o1"), k=1, dry_run=True, python="python")
    assert rc == 0 and man1["serve"]["worker_gb"] == 10.0 and "worker_gb_passed" not in man1["serve"] and "composition" not in man1["serve"]   # the estimate defaults (cli.PACK_WORKER_GB_DEFAULT); K = 1 runs through the same launcher
    rc, manf = serve.run("fast", target(tmp_path / "of"), k=2, dry_run=True, python="python")   # base's fast row packs: fast's flags on the same driver
    assert rc == 0 and manf["mode"] == "fast" and manf["served"] is True and manf["line"].endswith("--triton-ln 1 --tf32 1") and manf["env_row"] == {"RFD_PDBIO": "1", "RFD_SE3FAST": "t2"}
    stack.reset_for_tests()
    rc, mand = serve.run(None, target(tmp_path / "od"), k=2, dry_run=True, python="python")     # no --mode: fast here too
    assert rc == 0 and mand["mode"] == "fast" and mand["activation"]["mode_defaulted"] is True
    stack.reset_for_tests()
    # a typed start number: the one target sliced K ways from it, at its own prefix
    ov = [f"inference.input_pdb={PDB}", "contigmap.contigs=[A1-115/0 80-80]", f"inference.output_prefix={tmp_path / 'run' / 'binder'}", "inference.num_designs=4", "inference.design_startnum=10"]
    rc, manh = serve.run("exact", ov, k=2, dry_run=True, python="python")
    assert rc == 0 and manh["out_dir"] == str(tmp_path / "run") and manh["serve"]["slices"] == [[{"name": "binder", "startnum": 10, "num_designs": 2}], [{"name": "binder", "startnum": 12, "num_designs": 2}]]
    assert manh["cases"][0]["prefix"] == str(tmp_path / "run" / "binder") and manh["serve"]["template_argv"][manh["serve"]["template_argv"].index("--out") + 1] == str(tmp_path / "run")


# -------------------------------------------------------------------------------------------------------- through the launcher
def _fake_gpu_tools(bin_dir, total_mib="81559"):
    """nvidia-smi (the card's memory.total, mps_workers.sh:16) and nvidia-cuda-mps-control (-d, quit) as the launcher calls them."""
    os.makedirs(bin_dir, exist_ok=True)
    smi = os.path.join(bin_dir, "nvidia-smi"); ctl = os.path.join(bin_dir, "nvidia-cuda-mps-control")
    open(smi, "w").write(f"#!/bin/bash\necho {total_mib}\n")
    open(ctl, "w").write("#!/bin/bash\nif [ \"${1:-}\" = -d ]; then exit 0; fi\ncat > /dev/null; exit 0\n")
    for p in (smi, ctl):
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)


FAKE_DRIVER = (
    "import sys, os, json, time\n"
    "a = sys.argv[1:]; out = a[a.index('--out')+1]; tag = a[a.index('--tag')+1]; cases = json.load(open(a[a.index('--cases')+1]))\n"
    "print('fastpath levers:', ['chain_breaks']); print('prep lever: active = True'); print('einsum lever E: {}'); print('fullgraph mode: applied = True'); print('pdb writer lever IO1: armed (rfdiffusion.util.writepdb, writepdb_multi)'); print('[rfdiffusion1-opt.driver_run] PDBIO_FINAL ' + json.dumps(dict(armed=True, n_calls=3, n_verified=2, n_mismatch=0, mismatches=[], seconds=0.1)))\n"

    "sys.path.insert(0, os.environ['STUBS']); from _stubs import write_case_outputs\n"
    "t0 = time.time(); rows = []\n"
    "for c in cases:\n"
    "    idx = list(range(c.get('startnum', 0), c.get('startnum', 0) + c['num_designs']))\n"
    "    write_case_outputs(out, c['name'], idx, traj=('--no-traj' in a and a[a.index('--no-traj')+1] == '0'), prefix=c.get('prefix'))\n"   # rfd_bench_gk.py:304: the case's own prefix when it names one
    "    rows.append({'case': c['name'], 'precision': {'param_dtype': 'torch.float32', 'allow_tf32_matmul': False, 'cudnn_tf32': False, 'autocast_enabled': False},\n"
    "                 'designs': [{'i_des': i, 'pdb': c['name'] + '/des_%d.pdb' % i} for i in idx]})\n"
    "json.dump({'cases': rows, 'fullgraph_final': {'n_replay': 3, 'n_capture': 2, 'capture_errors': [], 'eager_fallback_calls': 0}}, open(os.path.join(out, tag + '_timings.json'), 'w'))\n"
    "print('MPS pipe:', os.environ.get('CUDA_MPS_PIPE_DIRECTORY'))\n"
    "sys.exit(int(os.environ.get('FAKE_RC', '0')))\n")


def _packing_launcher():
    p = stack.pack_launcher()
    if not os.path.isfile(p):
        pytest.skip(f"the packing launcher is not in this tree ({p})")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == LAUNCHER_SHA256
    return p


def test_the_oom_scan_reads_only_the_logs_this_launch_wrote(tmp_path):
    """A smaller-K launch into an output directory that still holds a LARGER launch's worker logs is judged on the K logs it wrote itself:
    a stale mps_worker_<W>.log (W >= K) carrying an out-of-memory text from the earlier launch does not mark this one (rc 6, the OOM
    WARNING line); an out-of-memory text in one of its own K logs still does."""
    launcher = _packing_launcher()
    _fake_gpu_tools(str(tmp_path / "bin"))
    os.makedirs(tmp_path / "o")
    (tmp_path / "o" / "mps_worker_3.log").write_text("torch.OutOfMemoryError: CUDA out of memory. (an earlier K=4 launch)\n")
    env = dict(os.environ, PATH=str(tmp_path / "bin") + os.pathsep + os.environ.get("PATH", ""), K="2", WORKER_GB="6.7",
               LOG_DIR=str(tmp_path / "o"), MPS_DIR_ROOT=str(tmp_path / "mps"))
    r = subprocess.run(["bash", launcher, "echo worker {W} ran"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "WARNING: OOM" not in r.stdout, (r.returncode, r.stdout[-600:], r.stderr[-300:])
    assert [(tmp_path / "o" / f"mps_worker_{w}.log").read_text() for w in (0, 1)] == ["worker 0 ran\n", "worker 1 ran\n"] and (tmp_path / "o" / "mps_worker_3.log").exists()
    r = subprocess.run(["bash", launcher, "if [ {W} = 1 ]; then echo 'RuntimeError: CUDA out of memory.'; fi"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 6 and "[mps_workers] WARNING: OOM text found in a worker log" in r.stdout, (r.returncode, r.stdout[-600:])
    # no MPS_DIR_ROOT (the launcher run by hand): the control daemon's pipe/log directories live under a private mktemp root (mode 0700) below TMPDIR
    os.makedirs(tmp_path / "t"); env.pop("MPS_DIR_ROOT"); env["TMPDIR"] = str(tmp_path / "t")
    r = subprocess.run(["bash", launcher, "echo pipe=$CUDA_MPS_PIPE_DIRECTORY"], env=env, capture_output=True, text=True, timeout=120)
    roots = [d for d in os.listdir(tmp_path / "t") if d.startswith("mps_")]
    assert r.returncode == 0 and len(roots) == 1 and stat.S_IMODE(os.stat(tmp_path / "t" / roots[0]).st_mode) == 0o700, (r.returncode, roots, r.stdout[-400:])
    pipe = re.search(r"^pipe=(.+)$", (tmp_path / "o" / "mps_worker_0.log").read_text(), re.M).group(1)
    assert pipe.startswith(str(tmp_path / "t" / roots[0]) + os.sep + "run_") and pipe.endswith(os.sep + "pipe") and os.path.isdir(pipe)


def test_serve_through_the_packing_launcher(monkeypatch, tmp_path, capsys):
    """K = 2 through common/mps_packing/mps_workers.sh with the GPU tools and the driver stubbed: the launcher's memory estimate stays under
    the card (2 x 6.7 + 4.0 <= 79.6: no NOTE line), the workers run the substituted template under the launcher's MPS variables, their logs land as
    mps_worker_<W>.log, every lever of the row is evidenced per worker, every worker's
    NUMERICS line is on the transcript, the manifest counts the designs of both slices."""
    stack.reset_for_tests()
    launcher = _packing_launcher()
    _fake_gpu_tools(str(tmp_path / "bin"))
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("STUBS", os.path.dirname(__file__))
    _served_box(monkeypatch, launcher=launcher)
    fake = tmp_path / "fake_driver.py"; fake.write_text(FAKE_DRIVER)
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None:
                        [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag, "--no-traj", res.settings.driver_flags[1]] + list(res.flags))
    rc, man = serve.run("exact", target(tmp_path / "o"), k=2, worker_gb=6.7, tag="p")
    assert rc == 0 and man["status"] == "ok", (man.get("serve"), man.get("driver_passes"))
    sv = man["serve"]
    assert sv["rc"] == 0 and any("launching K=2 workers; card 79.6 GB, budget 17.4 GB" in ln for ln in sv["lines"]) and sv["launcher_env"]["WORKER_GB"] == "6.7"
    assert [p["tag"] for p in man["driver_passes"]] == ["w0", "w1"] and all(p["rc"] == 0 for p in man["driver_passes"])
    assert man["levers_evidenced"] == ["C1", "E_einsum", "IO1", "P", "W1"] and man["levers_missing"] == [] and "jit_warmup" not in man
    assert man["outputs"]["n_pdb"] == 4 * len(man["cases"]) == man["outputs"]["expected"]
    for w in (0, 1):
        log = open(tmp_path / "o" / f"mps_worker_{w}.log", encoding="utf-8").read()
        pipe = re.search(r"^MPS pipe: (.+)$", log, re.M).group(1)                                                     # the launcher's fresh pipe directory reached the worker …
        assert "--jit-warmup" not in man["driver_passes"][w]["cmd"] and "--gpu-poll" not in man["driver_passes"][w]["cmd"] and "/rfdiffusion1_mps_" in pipe and pipe.endswith("/pipe")   # the worker's process history is the stock command line's: no warm-up flag, no sampler flag
        assert man["driver_passes"][w]["numerics"]["source"] == "torch" and man["driver_passes"][w]["numerics"]["mismatch"] == []
        assert not os.path.abspath(pipe).startswith(os.path.abspath(str(tmp_path / "o")) + os.sep)                   # … outside the output directory (a socket is not an output)
        assert not os.path.exists(tmp_path / "o" / "mps")
        wcases = shlex.split(man["driver_passes"][w]["cmd"])[shlex.split(man["driver_passes"][w]["cmd"]).index("--cases") + 1]
        assert not wcases.startswith(str(tmp_path / "o")) and os.path.basename(os.path.dirname(wcases)) == f"w{w}"           # the worker's cases file: the pass's scratch directory, <work>/w<W>/cases.json
        sl = json.load(open(wcases))
        assert all(c["num_designs"] == 2 and c["startnum"] == 2 * w for c in sl)
        assert not (tmp_path / "o" / f"w{w}").exists() and not (tmp_path / "o" / "_cases").exists()                          # no per-worker output directory, no cases among the outputs
        assert man["driver_passes"][w]["outputs"]["n_pdb"] == 2 * len(man["cases"]) and man["driver_passes"][w]["outputs"]["cases"]["binder"]["indices"] == [2 * w, 2 * w + 1]
    assert man["cases"][0]["name"] == "binder" and man["cases"][0]["prefix"] == str(tmp_path / "o" / "binder")   # every worker wrote its slice at the target's own prefix: <prefix>_<i>, the K = 1 layout, upstream's layout
    assert sorted(f for f in os.listdir(tmp_path / "o") if f.endswith(".pdb")) == ["binder_0.pdb", "binder_1.pdb", "binder_2.pdb", "binder_3.pdb"] and sorted(os.listdir(tmp_path / "o" / "traj"))[0] == "binder_0_Xt-1_traj.pdb"
    assert sorted(f for f in os.listdir(tmp_path / "o") if os.path.isfile(tmp_path / "o" / f) and not f.startswith("binder_")) == ["mps_worker_0.log", "mps_worker_1.log", "mps_workers.log", "opt_manifest.json", "w0_timings.json", "w1_timings.json"]   # the run record beside the designs
    assert man["outputs"]["per_worker"]["w1"]["n_pdb"] == 2 * len(man["cases"]) and set(man["outputs"]) == {"n_pdb", "expected", "per_worker"}
    for w, p_ in enumerate(man["driver_passes"]):                                                    # the manifest carries no kit timing: no wall, no ready figure, no design rows (the run record file is cited by path)
        assert not ({"wall_s", "ready_s", "first_design", "timings"} & set(p_)) and p_["timings_json"].endswith(f"w{w}_timings.json")
    err = capsys.readouterr().err
    assert "[rfdiffusion1-opt] PLAN mode=exact route=served attach=driver" in err and "jit_warmup" not in err and "[rfdiffusion1-opt] RUN mode=exact cases=" in err and "process=served K=2" in err
    assert not any(l.startswith("[rfdiffusion1-opt] ready ") for l in err.splitlines())               # the kit prints no cold-start / timing line
    record = {"param_dtype": "torch.float32", "allow_tf32_matmul": False, "cudnn_tf32": False, "autocast_enabled": False}   # FAKE_DRIVER's precision record
    want = [report.numerics_line(f"w{w}", record, modes.numerics("exact", record, levers=modes.resolve("exact", served=True).levers)) for w in (0, 1)]
    assert all(l in err.splitlines() for l in want) and all(" numerics_source=torch " in l + " " for l in want)   # every worker's torch read-back republished on the transcript, in the design route's words (report.numerics_line)
    m = json.load(open(tmp_path / "o" / "opt_manifest.json"))
    assert m["serve"]["k"] == 2 and m["line"] == SERVED_EXACT_LINE and m["activation"]["served"] is True


def test_serve_gm1_note_and_worker_failure_through_the_launcher(monkeypatch, tmp_path, capsys):
    stack.reset_for_tests()
    launcher = _packing_launcher()
    _fake_gpu_tools(str(tmp_path / "bin"))
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("STUBS", os.path.dirname(__file__))
    _served_box(monkeypatch, launcher=launcher)
    fake = tmp_path / "fake_driver.py"; fake.write_text(FAKE_DRIVER)
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None:
                        [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag, "--no-traj", res.settings.driver_flags[1]] + list(res.flags))
    # memory estimate: 4 x 24.2 + 4.0 > 79.6 → the launcher NOTES the estimate and proceeds (no memory estimate refuses a run): every worker starts, the
    # package relays the note in its own grammar (report.note_line) and records it (serve.notes); the run is judged like any other
    capsys.readouterr()
    rc, man = serve.run("exact", target(tmp_path / "o"), k=4, worker_gb=24.2)
    err = capsys.readouterr().err
    note = "[mps_workers] NOTE: K=4 x 24.2 GB + 4.0 GB headroom = 100.8 GB > card 79.6 GB — requested worker footprint exceeds the card by estimate; proceeding — may OOM"
    assert rc == 0 and man["status"] == "ok" and man["serve"]["rc"] == 0 and man["serve"]["refusal_line"] is None and "reason" not in man, (man.get("status"), man.get("reason"))
    assert [ln for ln in man["serve"]["lines"] if ln.startswith("[mps_workers] NOTE: ")] == man["serve"]["notes"] and len(man["serve"]["notes"]) == 1 and man["serve"]["notes"][0].startswith(note)
    assert err.count("[rfdiffusion1-opt] NOTE: ") == 1 and "[rfdiffusion1-opt] NOTE: requested worker footprint exceeds the card by estimate; proceeding — may OOM (launcher: K=4 x 24.2 GB + 4.0 GB headroom = 100.8 GB > card 79.6 GB)" in err
    assert "NOT ACTIVE" not in err and [p["tag"] for p in man["driver_passes"]] == ["w0", "w1", "w2", "w3"] and all(p["rc"] == 0 for p in man["driver_passes"]) and os.path.exists(tmp_path / "o" / "mps_worker_3.log")
    # a failing worker: its rc relayed by the launcher (mps_workers.sh:32), the pass `failed`
    monkeypatch.setenv("FAKE_RC", "7")
    rc, man = serve.run("exact", target(tmp_path / "o2", 2), k=1)
    assert rc == 1 and man["status"] == "failed" and man["serve"]["rc"] == 7 and man["driver_passes"][0]["rc"] == 7


def test_serve_exit_rule_through_the_launcher(monkeypatch, tmp_path, capsys):
    """The exit rule on the served line (report.verdict): a worker with a lever on the stock path is a partial activation — exit 3 with the
    family line, the partial levers by name, the outputs kept; a worker's own rc 3 is the worker's (mps_workers.sh:32) — the memory estimate never
    refuses; the launcher's OOM line is `failed` above every partial worker; outputs short of the request are `incomplete` (exit 1)."""
    stack.reset_for_tests()
    launcher = _packing_launcher()
    _fake_gpu_tools(str(tmp_path / "bin"))
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("STUBS", os.path.dirname(__file__))
    _served_box(monkeypatch, launcher=launcher)
    fake = tmp_path / "fake_driver.py"
    fake.write_text(FAKE_DRIVER.replace("print('fullgraph mode: applied = True')", "print('fullgraph mode: applied = True') if not os.environ.get('NO_W1') else None")
                    .replace("idx = list(range(c.get('startnum', 0), c.get('startnum', 0) + c['num_designs']))",
                             "idx = list(range(c.get('startnum', 0), c.get('startnum', 0) + c['num_designs'] - int(os.environ.get('SHORT', '0'))))")
                    .replace("print('MPS pipe:', os.environ.get('CUDA_MPS_PIPE_DIRECTORY'))",
                             "print('MPS pipe:', os.environ.get('CUDA_MPS_PIPE_DIRECTORY')); print('CUDA out of memory' if os.environ.get('OOM') else 'fine')"))
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None:
                        [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag] + list(res.flags))
    monkeypatch.setenv("NO_W1", "1")
    rc, man = serve.run("exact", target(tmp_path / "o", 2), k=1)
    assert rc == report.EXIT_NOT_ACTIVE == 3 and man["status"] == "partial" and man["partial"] == ["W1"] and "allow_partial" not in man and man["exit_code"] == 3
    assert man["partial_reason"] == "worker w0: levers without their evidence line ['W1']; forbidden lines 0" and man["serve"]["refusal_line"] is None
    err = capsys.readouterr().err
    assert "[rfdiffusion1-opt] NOT ACTIVE: partial activation — W1: worker w0: levers without their evidence line ['W1']; forbidden lines 0; exit 3" in err.splitlines()
    assert (tmp_path / "o" / "opt_manifest.json").is_file() and "allow-partial" not in err                # the outputs and the manifest are kept; no opt-out exists
    # a worker exiting 3 (its own refusal) with every lever evidenced: `failed` by the worker's rc (mps_workers.sh:32 relays a worker's rc; the memory estimate has no code)
    monkeypatch.delenv("NO_W1"); monkeypatch.setenv("FAKE_RC", "3")
    rc, man = serve.run("exact", target(tmp_path / "o2", 2), k=1)
    assert rc == 1 and man["status"] == "failed" and man["reason"] == "worker w0: rc=3" and man["serve"]["rc"] == 3 and "GM-1" not in man["reason"] and man["serve"]["refusal_line"] is None
    monkeypatch.delenv("FAKE_RC")
    # the launcher's OOM scan (rc 6, its WARNING line) above a partial worker: `failed`, the partial record kept beside it
    monkeypatch.setenv("NO_W1", "1"); monkeypatch.setenv("OOM", "1")
    rc, man = serve.run("exact", target(tmp_path / "o3", 2), k=1)
    assert rc == 1 and man["status"] == "failed" and man["reason"].startswith("out-of-memory text in a worker log") and man["serve"]["rc"] == 6 and man["partial"] == ["W1"]
    assert "[rfdiffusion1-opt] PARTIAL recorded: W1: " in capsys.readouterr().err
    monkeypatch.delenv("OOM")
    # outputs short of the request above a partial worker: `incomplete`, exit 1, the partial record kept beside it
    monkeypatch.delenv("OOM", raising=False); monkeypatch.setenv("SHORT", "1")
    rc, man = serve.run("exact", target(tmp_path / "o4", 2), k=1)
    assert rc == 1 and man["status"] == "incomplete" and man["partial"] == ["W1"] and man["incomplete"] == f"{len(man['cases'])}/{2 * len(man['cases'])}"


def test_launcher_refusal_is_read_from_its_lines(tmp_path):
    """serve.launcher_refusal: the tools' refusal lines (mps_workers.sh:20,24) name the refusal only when no worker ran; a `worker <W> rc=`
    line means the workers ran and the launcher's rc is a worker's own (mps_workers.sh:32); the memory-estimate line is a NOTE (launcher_notes)."""
    log = tmp_path / "mps_workers.log"
    log.write_text("[mps_workers] NOTE: K=4 x 24.2 GB + 4.0 GB headroom = 100.8 GB > card 79.6 GB — requested worker footprint exceeds the card by estimate; proceeding — may OOM\n")
    assert serve.launcher_refusal(str(log)) is None and serve.launcher_notes(str(log)) == [log.read_text().strip()]   # the memory estimate is a NOTE, never a refusal
    assert 3 not in serve.LAUNCHER_RC and all(code != 3 for code, _ in serve.LAUNCHER_REFUSALS)
    log.write_text("[mps_workers] nvidia-cuda-mps-control not found on PATH\n")
    assert serve.launcher_refusal(str(log))[0] == 4
    log.write_text("[mps_workers] launching K=1 workers; card 79.6 GB, budget 4.0 GB\n[mps_workers] worker 0 rc=3\n")
    assert serve.launcher_refusal(str(log)) is None and serve.launcher_oom(str(log)) is False
    log.write_text("[mps_workers] worker 0 rc=0\n[mps_workers] WARNING: OOM text found in a worker log\n")
    assert serve.launcher_oom(str(log)) is True
    assert serve.launcher_refusal(str(tmp_path / "absent.log")) is None


# ---------------------------------------------------------------------------------------------------------------- the CLI
def test_cli_pack_codes(monkeypatch, tmp_path, capsys):
    """`design --pack K` is the packed line's whole command surface: no `serve` verb, no --worker-gb / --headroom-gb / --composition flags."""
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "exact", "--pack", "4", *target(tmp_path / "o"), "--dry-run"]) == 3   # no GPU / MPS control on the CPU box: refused by name
    err = capsys.readouterr().err
    assert "[rfdiffusion1-opt] NOT ACTIVE:" in err and "route=served" in err
    assert cli.main(["design", "--mode", "exact", "--pack", "4", *target(tmp_path / "o", 3), "--dry-run"]) == 3   # 3 designs do not slice into 4 workers: refused by name before any gate
    assert "does not slice into K=4 workers" in capsys.readouterr().err
    cases_json = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    for gone in (["serve", "--mode", "exact", "--pack", "2", *target(tmp_path / "o")],
                 ["design", "--pack", "2", "--worker-gb", "6.7", *target(tmp_path / "o")],
                 ["design", "--pack", "2", "--composition", "house", *target(tmp_path / "o")],
                 ["design", "--pack", "2", "--headroom-gb", "4", *target(tmp_path / "o")],
                 ["design", "--mode", "exact", "--input", cases_json, "--out_dir", str(tmp_path / "o")],           # the kit-invented cases-JSON form and its flags are gone: argparse usage errors, exit 2
                 ["design", "--mode", "exact", *target(tmp_path / "o"), "--out_dir", str(tmp_path / "o")],
                 ["design", "--mode", "exact", *target(tmp_path / "o"), "--num-designs", "4"],
                 ["design", "--mode", "exact", *target(tmp_path / "o"), "--startnum", "2"],
                 ["design", "--mode", "exact", *target(tmp_path / "o"), "--tag", "t"],
                 ["warm", "--mode", "exact", "--input", cases_json, "--out_dir", str(tmp_path / "w")]):
        with pytest.raises(SystemExit) as ex:
            cli.main(gone)
        assert ex.value.code == 2, gone                                                                    # argparse: no such verb / flag
    opts = {o for a in next(a for a in cli.build_parser()._actions if isinstance(a, cli.argparse._SubParsersAction)).choices["design"]._actions for o in a.option_strings}
    assert not set(REMOVED_FLAGS) & opts and opts == {"-h", "--help", "--mode", "--pack", "--det", "--dry-run"}, opts   # design's whole flag surface: the mode, the packed line's K, the recipe, the dry run — the input is upstream's positional overrides
    stack.reset_for_tests(); capsys.readouterr()
    launcher = tmp_path / "mps_workers.sh"; launcher.write_text("#!/bin/bash\nexit 0\n")
    _served_box(monkeypatch, launcher=str(launcher))
    assert cli.main(["design", "--mode", "exact", "--pack", "2", *target(tmp_path / "d"), "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "dry-run" and out["serve"]["k"] == 2 and out["serve"]["worker_gb"] == 10.0
    stack.reset_for_tests()
    monkeypatch.setenv(cli.PACK_WORKER_GB_ENV, "6.7")                                                    # the estimate's environment form reaches the launcher record
    assert cli.main(["design", "--mode", "exact", "--pack", "2", *target(tmp_path / "d2"), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["serve"]["worker_gb"] == 6.7
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "exact", "--pack", "2", *target(tmp_path / "d3"), "--dry-run", "potentials.guide_scale=2"]) == 0   # guiding potentials: served, packed
    io = capsys.readouterr()
    assert "NOT ACTIVE" not in io.err and json.loads(io.out)["mode"] == "exact"
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "exact", "--pack", "2", *target(tmp_path / "d3"), "--dry-run", "inference.symmetry=C3"]) == 3      # what the kit line cannot serve: refused by name, exit 3, nothing printed on stdout
    io = capsys.readouterr()
    assert "[rfdiffusion1-opt] NOT ACTIVE: mode=exact cannot serve symmetric oligomers [inference.symmetry=C3]: " in io.err and io.out == ""
    stack.reset_for_tests()
    assert cli.main(["design", "--mode", "off", "--pack", "2", *target(tmp_path / "d4"), "--dry-run"]) == 3   # --mode off --pack: no stock row to pack, refused by name
    assert "--pack has no stock row" in capsys.readouterr().err

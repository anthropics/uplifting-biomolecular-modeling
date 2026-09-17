"""`design` — one BoltzGen design job from a spec YAML: upstream's ``configure`` and then the mode's launch.

Every mode configures the same way: ``python -m boltzgen.cli.boltzgen configure <spec> --output <run_dir> [--cache $BOLTZGEN_CACHE]
<caller args>`` in a clean stock environment (design.stock_env: every must-be-absent name of stock/PINS.json stripped; the caller's
arguments are upstream's own ``configure`` arguments, passed through as given — ``--cache`` is added from BOLTZGEN_CACHE only when the
caller names none). Then:

* ``off``      — per step of ``<run_dir>/steps.yaml`` one clean subprocess of ``stock_design.py`` (the environment proof, then upstream's
  step; seeded ``seed + i`` when the caller passed ``--seed``, unseeded — upstream's own form — when not);
* a kit mode — the runner in one child: ``python forward/xattempt_addon/src/xa_run.py <run_dir> <seed>`` under the environment
  ``stack.mode_env`` builds from the mode table (PYTHONPATH order, the runner's own switch defaults and the mode's overrides,
  ``BG_TIMING_FILE`` for the hook's timing / seed records), cwd = the output's parent (as for ``configure``).

The child's stderr is passed through and kept at ``<run_dir>/opt_run.log``; the levers are read off the lever modules' own lines
(stack.levers_from_lines) and ``inproc_times.json`` (written by ``bg_inproc.py``); ``opt_manifest.json`` is written beside the outputs with the activation line's report.

Every mode ends with the design census (``account_designs``, one ``DESIGNS`` line and the manifest's ``designs``): how many design
files the run's design-generating step asked upstream to write (its resolved ``config/<step>.yaml``: specs × multiplicity ×
diffusion_samples — ``--num_designs`` rounded up to whole diffusion batches), how many it wrote, and how many batches upstream's own
handlers skipped (``| WARNING: ran out of memory, skipping batch`` / ``WARNING: Skipping batch. Exception for …`` in the run log, every
model step) — upstream skips such a batch and its step still exits 0, so the census is where a shortfall is stated.

Every model process of a KIT mode is armed with the accelerator census (census.py) by ``BOLTZGEN_OPT_KERNELS=<payload>`` (kit_env) —
the stock child (mode ``off``) runs upstream alone after its environment proof and carries no census; the payload carries the run's base ``--seed`` for the
line's ``seed=`` word (``none`` unseeded). Each prints one ``[boltzgen-opt <mode>] KERNELS route=… …`` line and one ``PEAK`` line at exit;
``evidence`` / ``run_stock`` read the KERNELS lines back and record the account (``kernels.verdict``,
against what the mode table and upstream's own ``Using kernels:`` line make the route expect: ``kernel_expectation``) in
``opt_manifest.json`` — a record, never an exit.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from opt_core import stock_proof

from . import census, kernels, manifest, modes, report, stack, stock_design
from .codes import EXIT_NOT_ACTIVE

TIMING_FILE = "opt_timing.jsonl"
RUN_LOG = "opt_run.log"
STEPS_FILE = "steps.yaml"
STOCK_ENV_KEEP = ("HF_HUB_OFFLINE", "PYTHONUNBUFFERED")


class DesignError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------ environment
def stock_env(base: Optional[dict] = None) -> Tuple[dict, List[str]]:
    """The pristine environment of the stock arm and of ``configure``: every must-be-absent name stripped (stock/PINS.json: the prefixes
    and the names, each read as a prefix exactly as the child's proof reads its ``--env-absent`` list — opt_core.stock_proof.forbidden), no
    PYTHONPATH, no BOLTZGEN_OPT*, ``HF_HUB_OFFLINE=1`` and ``PYTHONUNBUFFERED=1`` set."""
    se = stack.pins()["stock_environment"]
    base, dropped = stock_proof.strip_env(os.environ if base is None else base, list(se["must_be_absent_prefixes"]) + list(se["must_be_absent_names"]))
    base["HF_HUB_OFFLINE"] = base.get("HF_HUB_OFFLINE", "1")
    base["PYTHONUNBUFFERED"] = "1"
    return base, dropped


def item_id(run_dir: str) -> str:
    """The design job's name for the PEAK line: the run directory's base name (a caller that stages one directory per item gets one name per item)."""
    return os.path.basename(os.path.normpath(run_dir)) or "?"


def kit_env(res: modes.Resolution, run_dir: str, base: Optional[dict] = None, exp: Optional[dict] = None, arm: bool = True,
            seed: Optional[int] = None) -> Tuple[dict, List[str]]:
    env, dropped = stack.mode_env(res, base)
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["BG_TIMING_FILE"] = os.path.join(run_dir, TIMING_FILE)          # the hook's own timing records
    expect = exp["expect"] if exp else "on"                              # the running verbs pass the run's expectation (kernel_expectation); a bare caller arms the child for kernels on
    if arm:                                                              # a run without a GPU step launches no model process: nothing to arm (run_kit passes arm=False then)
        env[census.ENV_KERNELS] = census.payload(res.mode, expect, item=item_id(run_dir), seed=seed)   # arms the census in the child at interpreter start (_autoload, early: in a kit mode an absent library means the mode cannot activate — refused before the step)
    return env, dropped


# --------------------------------------------------------------------------------------------------------------- configure
def configure_args(cache: Optional[str], user_args: List[str]) -> List[str]:
    """The caller's own ``configure`` arguments, passed through as given, behind ``--cache <BOLTZGEN_CACHE>`` when the caller names no
    ``--cache`` (argparse: a later occurrence wins, so the caller's always does)."""
    return (["--cache", cache] if cache and "--cache" not in user_args else []) + list(user_args)


def configure_cmd(spec: str, run_dir: str, user_args: List[str], cache: Optional[str]) -> List[str]:
    return [sys.executable, "-m", "boltzgen.cli.boltzgen", "configure", spec, "--output", run_dir] + configure_args(cache, user_args)


def configure(spec: str, run_dir: str, user_args: List[str], cache: Optional[str], log_path: Optional[str] = None) -> dict:
    cmd = configure_cmd(spec, run_dir, user_args, cache)
    env, dropped = stock_env()
    os.makedirs(run_dir, exist_ok=True)
    log_path = log_path or os.path.join(run_dir, "opt_configure.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(run_dir)) or None)
    if r.returncode != 0:
        tail = open(log_path, encoding="utf-8", errors="replace").read()[-1500:]
        raise DesignError(f"configure failed rc={r.returncode} ({shlex.join(cmd)}):\n{tail}")
    steps = read_steps(run_dir)
    return {"cmd": cmd, "wall_s": round(time.time() - t0, 3), "steps": [s["name"] for s in steps], "dropped_env": dropped, "log": log_path,
            "kernels_resolution": kernels.upstream_resolution(log_path)}


CONFIGURE_LOG = "opt_configure.log"


def kernel_expectation(mode: str, run_dir: str, conf: Optional[dict] = None) -> dict:
    """What the route expects of the accelerators (kernels.expectation): the mode table and upstream's own resolution line, read from
    ``configure``'s result or from ``<run_dir>/opt_configure.log`` (``--use_kernels false``, upstream's own switch, makes the route expect
    no library call — recorded, never refused)."""
    res = (conf or {}).get("kernels_resolution") or kernels.upstream_resolution(os.path.join(run_dir, CONFIGURE_LOG))
    exp = kernels.expectation(mode, res)
    exp["resolution"] = res
    return exp


def gpu_steps_of(run_dir: str, steps: Optional[List[str]] = None) -> List[str]:
    """The GPU (model) steps of a configured run directory, restricted to ``steps`` when given (stack.GPU_STEPS)."""
    names = [s["name"] for s in read_steps(run_dir)]
    if steps:
        names = [n for n in names if n in steps]
    return [n for n in names if n in stack.GPU_STEPS]


def read_steps(run_dir: str) -> List[dict]:
    import yaml
    with open(os.path.join(run_dir, STEPS_FILE), encoding="utf-8") as fh:
        return list(yaml.safe_load(fh)["steps"])


# ------------------------------------------------------------------------------------------------------------- stock arm
def step_seed(seed: Optional[int], i: int) -> Optional[int]:
    """Step i's seed under the recipe (``seed + i``); ``None`` when the run is unseeded."""
    return None if seed is None else seed + i


STOCK_MODULE = "boltzgen_opt.stock_design"
# The stock child's first statements (``python -S -s -c STOCK_PREAMBLE % roots …``): THIS tree's package and core directories
# (stack.package_roots() — the copies the caller itself runs, what mode_env hands every kit child on PYTHONPATH) first on sys.path and the
# package + the core's proof module imported from them (stdlib-only imports, the two modules the proof allows) BEFORE the interpreter's site
# runs, then site (site.main(): the site directories and their .pth files, as at any start), then the stock caller module as __main__ (a
# submodule of the package already loaded: this tree's). PYTHONPATH stays absent (the proof's contract) and nothing else of the environment
# changes. Why: an interpreter whose site directory carries a second install of the package — an editable install of another checkout of
# this tree, say — resolves ``-m boltzgen_opt.stock_design`` to THAT copy (its autoload .pth imports the package
# while site initialises, before any path entry of this tree exists), and the stock step would run another version's caller: its seed recipe,
# its proof. `-S` defers site to the preamble; `-s` keeps the user site out as before (the proof's no_user_site).
STOCK_PREAMBLE = ("import sys; sys.path[:0] = [p for p in %r if p not in sys.path]; import boltzgen_opt, opt_core.stock_proof; import site; site.main(); "
                  "import runpy; runpy.run_module(%r, run_name='__main__', alter_sys=True)")


def stock_preamble() -> str:
    return STOCK_PREAMBLE % (stack.package_roots(), STOCK_MODULE)


def stock_step_cmd(seed: Optional[int], config: str, proof_json: str) -> List[str]:
    """The stock child's command, in the shared core's stock-proof contract (opt_core.stock_proof.stock_command: the own options, ``--``, the
    stock arguments ``<seed|none> <config>``), its module started by ``-S -s -c <stock_preamble()>`` in place of ``-s -m <module>`` so that the
    child runs THIS tree's caller (STOCK_PREAMBLE). Nothing of the kit rides the line: the child checks its environment is clean of the kit
    and runs upstream's step alone."""
    se = stack.pins()["stock_environment"]
    cmd = stock_proof.stock_command(sys.executable, STOCK_MODULE, proof_json=proof_json,
                                    env_absent=list(se["must_be_absent_prefixes"]) + list(se["must_be_absent_names"]),
                                    kit_dirs=[stack.kit_dir(k) for k in modes.KITS],
                                    args=[stock_design.NO_SEED if seed is None else str(seed), config],
                                    module_prefixes=stack.KIT_MODULE_PREFIXES)
    i = cmd.index("-m")
    if cmd[i + 1] != STOCK_MODULE:
        raise AssertionError(f"stock_command's form changed: {cmd[:i + 2]}")
    return cmd[:1] + ["-S", "-s", "-c", stock_preamble()] + cmd[i + 2:]


def run_stock(run_dir: str, seed: Optional[int], echo: bool = True) -> dict:
    """Per step i: a clean subprocess (seeded ``seed + i`` when ``seed`` is given, unseeded when not), BOLTZGEN_PIPELINE_STEP set as
    upstream sets it; nothing of the kit runs inside it (no census — a stock process is upstream alone after its environment proof)."""
    steps = read_steps(run_dir)
    env, dropped = stock_env()
    log_path = os.path.join(run_dir, RUN_LOG)
    per_step = []
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for i, s in enumerate(steps):
            name, cfg = s["name"], os.path.join(run_dir, s["config_file"])
            proof = os.path.join(run_dir, f"stock_env_proof_{name}.json")
            cmd = stock_step_cmd(step_seed(seed, i), cfg, proof)
            e = dict(env); e[stack.ENV_STEP] = name; e["BOLTZGEN_PIPELINE_PROGRESS"] = f"Step {i + 1}/{len(steps)}"
            ts = time.time()
            rc = _run_tee(cmd, e, log, echo, cwd=os.path.dirname(os.path.abspath(run_dir)))
            pr = _read_json(proof)
            per_step.append({"step": name, "seed": step_seed(seed, i), "rc": rc, "wall_s": round(time.time() - ts, 3), "proof_ok": bool(pr and pr.get("ok")), "proof": proof})
            if rc != 0:
                break
    rc = per_step[-1]["rc"] if per_step else 1
    return {"form": "process", "arm": "stock", "steps": per_step, "rc": rc, "wall_s": round(time.time() - t0, 3), "dropped_env": dropped, "log": log_path,
            "stock_env_proof": [p["proof"] for p in per_step], "kernels": None, "route": census.route_of("off")}   # kernels: no census in a stock process (upstream alone after the proof); upstream's own `Using kernels:` resolution is the configure record's


# ------------------------------------------------------------------------------------------------------------ the runner
def runner_cmd(res: modes.Resolution, run_dir: str, seed: int, steps: Optional[List[str]] = None) -> List[str]:
    cmd = [sys.executable, os.path.join(stack.opt_home(), res.runner), run_dir, str(seed)]
    if steps:
        cmd.append(",".join(steps))
    return cmd


def run_kit(res: modes.Resolution, run_dir: str, seed: int, echo: bool = True, steps: Optional[List[str]] = None,
            exp: Optional[dict] = None) -> dict:
    exp = exp or kernel_expectation(res.mode, run_dir)
    model_processes = 1 if gpu_steps_of(run_dir, steps) else 0            # the runner is the one model process of the run (GPU steps in-process); a CPU-steps-only run has none
    env, dropped = kit_env(res, run_dir, exp=exp, arm=bool(model_processes), seed=seed)
    cmd = runner_cmd(res, run_dir, seed, steps)
    log_path = os.path.join(run_dir, RUN_LOG)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        rc = _run_tee(cmd, env, log, echo, cwd=os.path.dirname(os.path.abspath(run_dir)))
    out = {"form": "process", "arm": "kit", "cmd": cmd, "rc": rc, "wall_s": round(time.time() - t0, 3), "dropped_env": dropped, "log": log_path,
           "pythonpath": env.get("PYTHONPATH"), "exported": dict(res.env), "route": census.route_of(res.mode)}
    out.update(evidence(res, run_dir, log_path, exp=exp, model_processes=model_processes))
    return out


# ------------------------------------------------------------------------------------------------------------ evidence
def evidence(res: modes.Resolution, run_dir: str, log_path: str, exp: Optional[dict] = None, model_processes: int = 1) -> dict:
    lines = _lines(log_path)
    ev = stack.levers_from_lines(lines, res)
    exp = exp or kernel_expectation(res.mode, run_dir)
    ev["kernels"] = kernels.census_of(lines, exp, model_processes, where=f" ({log_path})")
    times = _read_json(os.path.join(run_dir, "inproc_times.json"))
    if times:
        ev["levers_applied"].append("inproc"); ev["inproc_times"] = times
    else:
        ev["levers_fallback"].append("inproc"); ev["fallback_reasons"]["inproc"] = "no inproc_times.json (the partner runner did not complete)"
    ev["partial"] = bool(ev["levers_fallback"])
    return ev


# --------------------------------------------------------------------------------------------------------- design census
DESIGN_SOURCE_TASK = "data_from_yaml.FromYamlDataModule"        # the data module of a design-GENERATING step (upstream's `design`; `inverse_folding` under --only_inverse_fold): it reads the spec; every later step reads a design directory (FromGeneratedDataModule)
UPSTREAM_OOM_SKIP = re.compile(r"\| WARNING: ran out of memory, skipping batch")     # upstream's boltz.py predict_step: the batch is dropped, the step goes on and exits 0
UPSTREAM_FEATURIZER_SKIP = re.compile(r"WARNING: Skipping batch\. Exception for ")     # upstream's boltz.py predict_step: a featurizer exception marked the batch; dropped the same way
DESIGN_SUFFIX, NATIVE_SUFFIX = ".cif", "_native.cif"           # upstream's DesignWriter: `<stem>_<idx>.cif` per design, `<stem>_<idx>_native.cif` the input structure it may copy beside it


def design_request(run_dir: str) -> Optional[dict]:
    """The design-generating step of the configured run (``steps.yaml`` lists the steps ``configure`` enabled: upstream's ``--steps`` is
    already applied there) and what its resolved config — the one upstream's ``configure`` wrote, ``<run_dir>/config/<step>.yaml`` — asks
    the ``DesignWriter`` to write: ``requested`` =
    specs × ``data.cfg.multiplicity`` × ``diffusion_samples`` (the writer's own ``total_files`` per spec;
    ``configure`` sets multiplicity = ceil(num_designs / diffusion_batch_size) and diffusion_samples = the batch). None when the run
    has no such step (or no readable ``steps.yaml`` / step config: a run that failed before writing them). A relative ``output`` is
    configure's: resolved against the run directory's parent (its working directory)."""
    import yaml
    try:
        configured = read_steps(run_dir)
    except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError):
        return None
    for s in configured:
        path = os.path.join(run_dir, s["config_file"])
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError):
            continue
        data = cfg.get("data") or {}
        if not str(data.get("_target_", "")).endswith(DESIGN_SOURCE_TASK):
            continue
        dc = data.get("cfg") or {}
        specs = dc.get("yaml_path")
        n_specs = len(specs) if isinstance(specs, (list, tuple)) else 1
        mult, samples = _count(dc.get("multiplicity", 1)), _count(cfg.get("diffusion_samples", 1))
        out = str(cfg.get("output") or "")
        if out and not os.path.isabs(out):
            out = os.path.join(os.path.dirname(os.path.abspath(run_dir)), out)
        requested = n_specs * mult * samples if mult is not None and samples is not None else None
        return {"step": s["name"], "config": path, "output_dir": out, "specs": n_specs, "multiplicity": mult, "diffusion_samples": samples,
                "requested": requested, "reuse": bool(dc.get("skip_existing"))}
    return None


def _count(v) -> Optional[int]:
    """An integer count from a resolved config value; None when it is not one (an unresolved ``${...}`` interpolation)."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def designs_written(out_dir: str, since: Optional[float] = None) -> Tuple[int, int]:
    """(written, stale): the design files at the top of ``out_dir`` modified at or after ``since`` (epoch seconds; every one when None)
    and the older ones — files an earlier run into the same directory left and this run did not rewrite."""
    try:
        names = os.listdir(out_dir)
    except OSError:
        return 0, 0
    written = stale = 0
    for n in names:
        if not n.endswith(DESIGN_SUFFIX) or n.endswith(NATIVE_SUFFIX):
            continue
        if since is None or (_mtime(os.path.join(out_dir, n)) or 0.0) >= since - 1.0:      # 1 s: file timestamp granularity
            written += 1
        else:
            stale += 1
    return written, stale


def designs_census(run_dir: str, log_lines: List[str]) -> dict:
    """The run's design account: ``requested`` / ``produced`` for its design-generating step (None / None when the run has none),
    ``oom_skipped`` / ``featurizer_skipped`` = the batches upstream's own handlers dropped, counted from its two skip lines over every
    model step in ``log_lines`` (unanchored: a progress-bar fragment can precede them on a physical line), ``stale`` = design files of
    an earlier run this run did not rewrite (this run's are the ones written since its ``steps.yaml``, which ``configure`` rewrites
    every run), ``reuse`` = upstream's ``--reuse`` was on (it keeps existing designs and writes the rest: produced then counts them all)."""
    c = {"step": None, "requested": None, "produced": None,
         "oom_skipped": sum(1 for ln in log_lines if UPSTREAM_OOM_SKIP.search(ln)),
         "featurizer_skipped": sum(1 for ln in log_lines if UPSTREAM_FEATURIZER_SKIP.search(ln)),
         "stale": 0, "reuse": False, "output_dir": None, "config": None}
    req = design_request(run_dir)
    if req is not None:
        since = None if req["reuse"] else _mtime(os.path.join(run_dir, STEPS_FILE))
        produced, stale = designs_written(req["output_dir"], since)
        c.update(req, produced=produced, stale=stale)
    return c


def account_designs(run_dir: str, run: dict) -> dict:
    """The census of a finished run (kit or stock arm alike: the parent counts, nothing enters the child) and its one ``DESIGNS`` line."""
    c = designs_census(run_dir, _lines(run.get("log") or os.path.join(run_dir, RUN_LOG)))
    report.say(report.designs_line(c))
    return c


# ------------------------------------------------------------------------------------------------------------- helpers
def _run_tee(cmd: List[str], env: dict, log, echo: bool, cwd: Optional[str] = None) -> int:
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd, errors="replace")
    assert p.stdout is not None
    for line in p.stdout:
        log.write(line)
        if echo:
            sys.stderr.write(line); sys.stderr.flush()
    return p.wait()


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _lines(log_path: str) -> List[str]:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def _mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def process_report(res: modes.Resolution, run: dict, gpu: Optional[dict]) -> dict:
    """The activation report of a process form, from the child's evidence — the same shape enable() returns."""
    rep = {"active": res.active and run.get("rc") == 0, "mode": res.mode, "form": "process", "switches": modes.describe_line(res),
           "levers_applied": run.get("levers_applied", []), "levers_fallback": run.get("levers_fallback", []),
           "fallback_reasons": run.get("fallback_reasons", {}), "levers_unavailable": [], "partial": run.get("partial", False),
           "dropped_env": run.get("dropped_env", []), "boltzgen_version": stack.boltzgen_version(), "package_version": stack.package_version(),
           "gpu": gpu, "gpu_gate": run.get("gpu_gate"), "stack_key": stack.stack_key(gpu) if gpu else None,
           "python": sys.version.split()[0], "torch": None, "pythonpath": run.get("pythonpath"), "exported": run.get("exported"),
           "kit_stats_lines": run.get("kit_stats_lines"), "rc": run.get("rc")}
    if not res.active:
        rep["reason"] = "mode off: stock — upstream alone in a pristine process" if run.get("rc") == 0 else f"stock run failed rc={run.get('rc')}"
        rep["stock_env_proof"] = run.get("stock_env_proof")
    elif run.get("rc") == EXIT_NOT_ACTIVE:
        rep["reason"] = f"the kit's child refused by name (exit {EXIT_NOT_ACTIVE}; its NOT ACTIVE line above, kept in {run.get('log')})"   # an activation gate, or a model step a planned lever could not serve (stack.arm_step_gate)
    elif run.get("rc") != 0:
        rep["reason"] = f"kit run failed rc={run.get('rc')} (see {run.get('log')})"
    if run.get("oom") or (run.get("designs") or {}).get("oom_skipped"):   # an OOM the kit's trace saw inside Boltz.forward (its `[sz]` event line) or upstream's handler skipped (the design census, run["designs"]); a run without one carries no such key
        rep["oom"] = True
    if run.get("kernels") is not None:
        rep["kernels"] = {**run["kernels"], "route": run.get("route")}
    return rep

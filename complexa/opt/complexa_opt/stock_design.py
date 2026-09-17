"""The generation run of the stock route — the tree's only caller of upstream code — in a proven-clean child process.

``run()``: read the target entry when one is given (``inputs.py``); check the pinned weights directory holds both checkpoints at their
pinned byte counts (``stack.weights_gate``: presence and size, the sha256 digests of ``stock/PINS.json`` cited on the WEIGHTS line, not
recomputed), upstream's console script and checkout (``stack.console_script`` / ``pipeline_config``); compose the clean environment
(``clean_env``: this package's variables, every variable under the must-be-absent prefixes of ``stock/PINS.json`` and the interpreter
switch ``PYTHONSAFEPATH`` removed; kit ``PYTHONPATH`` entries removed; ``PYTHONDONTWRITEBYTECODE=1`` set), take the proof in a probe child
of exactly that environment (``env_proof``: the core's clean-process proof ``opt_core.stock_proof.env_proof`` — no variable under the
must-be-absent prefixes, no kit directory on ``sys.path``, no autoload finder armed, no torch and nothing of the core loaded — plus this
kit's own census of what the install's ``complexa_opt_autoload.pth`` loads in every interpreter: exactly the declared inert pair
``stack.PTH_MODULES`` (the package ``__init__`` and ``_autoload``, which under an absent COMPLEXA_OPT import nothing else) and no other
``complexa_opt`` module; plus which data-path variables are present and the child's ``sys.path``) — written to ``<out>/stock_env_proof.json``
and printed as one ``ENV-CLEAN`` line; a proof that is not ok is ``NOT STOCK`` (exit 3) and nothing is launched. Then launch ONE child with the run directory as its working directory (upstream roots its outputs and
logs there: generate.py setup(), cli_runner.py LOG_DIR)::

    complexa generate <checkout>/configs/search_binder_local_pipeline.yaml --verbose
        [++generation.task_name=<item> ++generation.target_dict_cfg={<item>:{…the entry…}}]
        ++ckpt_path=<CKPT_PATH> ++autoencoder_ckpt_path=<CKPT_PATH>/complexa_ae.ckpt  <the caller's Hydra overrides and generate options, verbatim>

(``compose``: upstream's own command on the shipped pipeline configuration — the target entry when one is given, the pinned checkpoints'
location — and nothing else: what the run computes (search algorithm, reward model, designs, seed, batch, run name) is the shipped
configuration's unless the caller's own tokens say otherwise; those come LAST, so a caller's value for any key the package spells is the
one Hydra applies: nothing the caller gives is refused or rewritten here, upstream answers for it). The child is upstream's console script running upstream's installed code:
nothing is staged, copied, patched or injected. Its log output is relayed to stderr and kept in ``<out>/design.log``. Then the census
(``outputs.census``: the PDBs this run wrote under the run's root against the designs the tokens ask for), the verdict (``opt_core.report.verdict``:
exit 0 complete, 1 the child failed or wrote short), ``<out>/opt_manifest.json`` (``manifest.py``), the OUTPUTS and EXIT lines.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from opt_core import manifest as core_manifest
from opt_core import process as core_process
from opt_core import report as core_report

from . import ActivationError, inputs as _inputs, manifest as _manifest, modes, outputs as _outputs, report as _report, settings as _settings, stack

PROOF_FILE = "stock_env_proof.json"
DESIGN_LOG = "design.log"
DATA_PATH_ENV = ("MODEL_OPT", "MODEL_OPT_STATE", stack.ENV_UPSTREAM, stack.ENV_WEIGHTS, "COMPLEXA_INIT", "COMMUNITY_MODELS_PATH",
                 "DATA_PATH", "AF2_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "USE_V2_COMPLEXA_ARCH", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF",
                 "PYTHONPATH")                          # recorded as present, never judged
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "CUDA error: out of memory")


class StockError(RuntimeError):
    """The run could not start (no checkout, no weights, a proof child that cannot report)."""


def _is_kit_path(p: str) -> bool:
    ap = os.path.realpath(p) if p else ""
    roots = [os.path.realpath(os.path.join(stack.tree_home(), "opt")), os.path.realpath(stack.package_dir())]
    return bool(p) and any(ap == r or ap.startswith(r + os.sep) for r in roots)


def clean_env(base: Optional[dict] = None, export: Optional[dict] = None, keep_kit_paths: bool = False) -> Tuple[dict, dict]:
    """(env, removed): ``base`` (the caller's environment) with the package variables, the must-be-absent names, ``PYTHONSAFEPATH`` and — on
    the stock route — kit PYTHONPATH entries removed, ``PYTHONDONTWRITEBYTECODE=1`` and ``export`` set. The kit route keeps the kit's
    PYTHONPATH entries (``keep_kit_paths``): they are how this tree's package reaches upstream's interpreter ahead of any older copy an
    image may hold."""
    base = dict(os.environ if base is None else base)
    p = stack.pins()["stock_environment"]
    prefixes, keep = tuple(p["must_be_absent_prefixes"]), set(p.get("must_be_absent_except") or [])
    removed: Dict[str, List[str]] = {"variables": [], "pythonpath": []}
    env = {}
    for k, v in base.items():
        if k in stack.PACKAGE_ENV or k in stack.INTERPRETER_ENV or (k.startswith(prefixes) and k not in keep):
            removed["variables"].append(k)
            continue
        env[k] = v
    pp = [e for e in env.get("PYTHONPATH", "").split(os.pathsep) if e]
    kept = [e for e in pp if keep_kit_paths or not _is_kit_path(e)]
    removed["pythonpath"] = [e for e in pp if e not in kept]
    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for k, v in (export or {}).items():
        env[k] = str(v)
    removed["variables"].sort()
    return env, removed


def core_root() -> str:
    """The directory holding the ``opt_core`` package this process imports (handed to the proof child explicitly)."""
    import opt_core
    return os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))


def env_proof(env: dict, python: Optional[str] = None, cwd: Optional[str] = None) -> dict:
    """Run a probe child in ``env`` (default interpreter: the one upstream's console script runs, else this one) and return the core's
    clean-process proof plus ``pth_modules`` (the ``complexa_opt`` modules the interpreter holds at start: the declared inert pair or fewer,
    else not ok), ``present`` (the data-path variables set) and ``sys_path``. StockError when the child cannot report."""
    if python is None:
        try:
            python = stack.console_script().get("interpreter") or sys.executable
        except ActivationError:
            python = sys.executable
    prefixes = list(stack.pins()["stock_environment"]["must_be_absent_prefixes"])
    code = ("import json, os, sys\n"
            "held = sorted(m for m in sys.modules if m == 'complexa_opt' or m.startswith('complexa_opt.'))\n"
            f"sys.path.insert(0, {core_root()!r})\n"
            "from opt_core import stock_proof as sp\n"
            f"p = sp.env_proof(env_absent={prefixes!r}, kit_dirs=[], module_prefixes=[m for m in held if m not in {list(stack.PTH_MODULES)!r}])\n"
            f"pkg = os.path.realpath({stack.package_dir()!r})\n"
            "p['kit_dirs_on_path'] = sorted(e for e in sys.path if e and os.path.exists(e) and (os.path.realpath(e) == pkg or os.path.realpath(e).startswith(pkg + os.sep)))\n"
            "p['pth_modules'] = held\n"
            f"p['pth_modules_declared'] = {list(stack.PTH_MODULES)!r}\n"
            "p['ok'] = bool(p['ok']) and not p['kit_dirs_on_path'] and set(held) <= set(p['pth_modules_declared'])\n"
            f"p['present'] = sorted(k for k in os.environ if k in {list(DATA_PATH_ENV)!r})\n"
            "p['sys_path'] = [e for e in sys.path[1:] if e]\n"
            "print(json.dumps(p))\n")
    r = subprocess.run([python, "-c", code], env=env, cwd=cwd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise StockError(f"the env-proof child exited {r.returncode}: {r.stderr.strip()[-800:]}")
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise StockError(f"the env-proof child printed no proof: {r.stdout[-300:]!r} {r.stderr[-300:]!r}") from None


def env_clean_line(proof: dict, removed: dict) -> str:
    """``[complexa-opt] ENV-CLEAN ok (<n> variables removed: <names|none>; proof: <core clean sentence> autoload=<armed finders|none>
    pth_modules=<held|none> torch_preloaded=<bool>; present: <names|none>)`` — or ``NOT STOCK: <violations>`` when the proof is not ok. The
    line lists, never judges."""
    from opt_core import stock_proof as sp
    if not proof.get("ok"):
        extra = sorted(set(proof.get("pth_modules") or []) - set(proof.get("pth_modules_declared") or []))
        return f"{_report.PREFIX} NOT STOCK: {sp.violations_sentence(proof)}" + (f", package modules beyond the inert pair {extra}" if extra else "")
    names = removed["variables"] + [f"PYTHONPATH:{e}" for e in removed["pythonpath"]]
    return (f"{_report.PREFIX} ENV-CLEAN ok ({len(names)} variables removed: {','.join(names) or 'none'}; proof: {sp.clean_sentence(proof)}"
            f" autoload={','.join(proof.get('autoload_armed') or []) or 'none'} pth_modules={','.join(proof.get('pth_modules') or []) or 'none'}"
            f" torch_preloaded={proof['torch_loaded_before_proof']}; present: {','.join(proof.get('present') or []) or 'none'})")


def _oom_in(log_path: str) -> bool:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-20000:]
    except OSError:
        return False
    return any(m in tail for m in OOM_MARKERS)


def compose(*, console_script: str, config_path: str, weights_dir: str, item: Optional[str] = None, entry: Optional[dict] = None,
            overrides: Sequence[str] = ()) -> List[str]:
    """The child's argv: upstream's command on the shipped pipeline configuration — the target entry when given, the checkpoints' location —
    then the caller's Hydra overrides and generate options verbatim and LAST (a later Hydra token wins)."""
    argv = [console_script, "generate", config_path, "--verbose"]      # --verbose: the wrapper lets the child's streams through (else it filters them into ./logs/*.log, cli_runner.py); output routing only
    if item is not None:
        argv += _inputs.overrides(item, entry or {})
    argv += [f"++ckpt_path={weights_dir}", f"++autoencoder_ckpt_path={os.path.join(weights_dir, 'complexa_ae.ckpt')}"]
    argv += [str(t) for t in overrides]
    return argv


def _launch(route: str, item: Optional[str], run_name: Optional[str], argv: List[str], env: dict, cwd: str, log_path: str) -> dict:
    """Run the child; relay its lines to stderr and ``log_path``; print its INVOCATION line; return its record."""
    def on_line(ln: str):
        sys.stderr.write(ln)
        sys.stderr.flush()
    t_launch = time.time()
    try:
        r = core_process.run_logged(argv, timeout_s=None, on_line=on_line, log_path=log_path, what=item or "run", env=env, cwd=cwd)
        rc, err = r.rc, None
    except OSError as e:
        rc, err = core_process.RC_LAUNCH, f"could not launch: {e}"
    t_exit = time.time()
    _report.emit(_report.invocation_line(route, item, run_name, cwd, t_launch, t_exit, rc, argv))
    rec = {"name": item, "run_name": run_name, "route": route, "argv": argv, "command_line": _report.shell(argv), "cwd": cwd,
           "t_launch": round(t_launch, 3), "t_exit": round(t_exit, 3), "wall_s": round(t_exit - t_launch, 3), "rc": rc, "log": log_path}
    if err:
        rec["error"] = err
    if rc != 0 and _oom_in(log_path):
        rec["oom"] = True
    return rec


def effective_names(item: Optional[str], overrides: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """(item, run name) as upstream will read them: the caller's LAST ``++generation.task_name=`` token wins over the ``--input`` item (it
    comes later on the command line); the run name is the caller's LAST ``++run_name=`` token, else the shipped ``run_name``."""
    eff_item = _settings.last_value(overrides, _settings.TASK_KEY)
    eff_run = _settings.last_value(overrides, _settings.RUN_NAME_KEY)
    return (eff_item if eff_item is not None else item), (eff_run if eff_run is not None else _settings.SHIPPED_RUN_NAME)


def run(*, mode: str, out_dir: str, input_path: Optional[str] = None, overrides: Sequence[str] = ()) -> dict:
    """One generation run in ``mode`` (``off``). Returns the run record (``exit_code``, census, manifest path, invocation)."""
    t0 = time.time()
    route = modes.route_of(mode)
    values = _settings.values_of(overrides)                               # designs / seed / batch as the run will use them (tokens, else the shipped values) — read, never judged
    item, entry = (_inputs.load_entry(input_path) if input_path else (None, None))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rep = stack.stock_report(mode)
    _report.emit(_report.not_active(mode, route))
    # the pinned weights (presence and byte count), upstream's console script and checkout
    w = stack.weights_gate()
    _report.emit(w["line"])
    if not w["pinned"]:
        _report.emit(_report.refused("weights: " + "; ".join(w["bad"]), mode))
        return {"exit_code": _report.EXIT_NOT_ACTIVE, "reason": "weights"}
    cs = stack.console_script()
    cfg_path = stack.pipeline_config()
    env, removed = clean_env()
    proof = env_proof(env, cwd=out_dir)
    proof_path = os.path.join(out_dir, PROOF_FILE)
    with open(proof_path, "w", encoding="utf-8") as fh:
        json.dump(dict(proof, removed=removed), fh, indent=1, sort_keys=True)
        fh.write("\n")
    _report.emit(env_clean_line(proof, removed))
    if not proof.get("ok"):
        return {"exit_code": _report.EXIT_NOT_ACTIVE, "reason": "not stock", "proof": proof_path}
    # compose and launch
    argv = compose(console_script=cs["path"], config_path=cfg_path, weights_dir=w["dir"], item=item, entry=entry, overrides=overrides)
    eff_item, eff_run, root, before = prelaunch(out_dir, item, overrides, values)
    inv = _launch(route, item, eff_run, argv, env, out_dir, os.path.join(out_dir, DESIGN_LOG))
    return finish(mode=mode, route=route, rep=rep, t0=t0, out_dir=out_dir, argv=argv, inv=inv, values=values, item=item, eff_item=eff_item, eff_run=eff_run,
                  root=root, before=before, entry=entry, input_path=input_path, overrides=overrides, w=w, cs=cs, cfg_path=cfg_path,
                  env_block={"path": proof_path, "ok": proof["ok"], "removed": removed})


def prelaunch(out_dir: str, item: Optional[str], overrides: Sequence[str], values: dict):
    """(effective item, run name, the run's composed root, the designs already under it) — with a NOTE line for a results file or earlier
    designs already in ``out_dir`` (either changes what upstream will do) and for a design count the package cannot read from the tokens
    (the census then holds the run to at least one design written and prints ``designs_expected=unknown``); named, never refused."""
    eff_item, eff_run = effective_names(item, overrides)
    root = _outputs.run_root(out_dir, eff_item, eff_run)
    if not values["counted"]:                                                # ``values`` = settings.values_of(overrides), read once by the caller
        _report.emit(_report.note_line("designs_expected", "unknown", f"{_settings.NSAMPLES_KEY}={values['nsamples']!s} × {_settings.NREPEAT_KEY}={values['nrepeat']!s} is not a pair of "
                                       "integer literals the package can count with (the tokens reach upstream as given): the census holds this run to at least one design written"))
    csv_path = _outputs.results_csv(out_dir)
    if os.path.exists(csv_path):
        _report.emit(_report.note_line("results_csv", csv_path, "present: upstream logs 'Results already exist' and exits 0 without generating (generate.py main)"))
    before = set(_outputs.design_files(root))
    if before:
        _report.emit(_report.note_line("preexisting_designs", len(before), f"{root} already holds designs; this run's are counted apart from them"))
    return eff_item, eff_run, root, before


def finish(*, mode: str, route: str, rep: dict, t0: float, out_dir: str, argv: List[str], inv: dict, values: dict, item, eff_item, eff_run, root: str, before: set,
           entry, input_path, overrides, w: dict, cs: dict, cfg_path: str, env_block: dict, kit_record: Optional[dict] = None) -> dict:
    """After the child: the census of the designs it wrote, the verdict (exit 0 complete · 1 failed or short · 3 — kit route — a partial or
    absent activation, ``rep['partial']``; a lever idle by its declared gate, ``rep['gated']``, is named on the EXIT line and prices nothing),
    the manifest, the OUTPUTS and EXIT lines. Shared by the stock and the kit route."""
    root_source = "composed"
    if eff_item is None or not os.path.isdir(root):                       # no task name of ours to compose the root from (the configuration's own), or upstream rooted elsewhere: the root it touched, named as such
        found = _outputs.newest_root(out_dir, inv["t_launch"])
        if found is not None and os.path.realpath(found) != os.path.realpath(root):
            root, root_source, before = found, "newest", set()
    census = _outputs.census(out_dir, root, eff_item, values["designs"], preexisting=before, root_source=root_source)
    for it, why in list(census["item_failed"].items()):
        if inv["rc"] != 0:
            census["item_failed"][it] = f"{why} (rc={inv['rc']}{', oom' if inv.get('oom') else ''}{', ' + inv['error'] if inv.get('error') else ''})"
    _report.emit(_report.outputs_line(route, eff_item, census["designs_written"], census["designs_expected"], census["root"], root_source))
    rc = 0 if inv["rc"] == 0 else 1
    v = core_report.verdict(rc, rep, allow_partial=False, incomplete=census["incomplete"])
    kit = {"route": route, "route_words": modes.ROUTE_WORDS[route], "designs": values["designs"], "nsamples": values["nsamples"], "nrepeat": values["nrepeat"],
           "seed": values["seed"], "seed_source": ("override" if values["seed_given"] else "shipped config (search_binder_local_pipeline.yaml seed)"), "seed_of_config": _settings.SHIPPED_SEED,
           "batch": values["batch"], "batch_source": ("override" if values["batch_given"] else "shipped config (binder_generate.yaml dataloader.batch_size)"),
           "overrides": [str(t) for t in overrides], "item": eff_item, "run_name": eff_run, "entry": entry, "input": (os.path.abspath(input_path) if input_path else None),
           "binder_length": (_inputs.binder_length(entry) if entry else None), "designs_written": census["designs_written"], "designs_expected": census["designs_expected"],
           "designs_preexisting": census["designs_preexisting"], "design_files": census["files"], "run_root": census["root"], "run_root_source": root_source,
           "weights": {"dir": w["dir"], "files": w["files"], "checked": "bytes"}, "console_script": cs, "pipeline_config": cfg_path,
           "env_proof": env_block, "invocation": inv, "wall_s": round(time.time() - t0, 3)}
    if kit_record is not None:
        kit["kit_record"] = kit_record
    doc = _manifest.build(mode=mode, route=route, report=rep, command=argv, stack_block=core_manifest.stack_block(gpu=stack.gpu(), stack_key=stack.stack_key()),
                          exit_=v, census=census, kit=kit)
    mpath = _manifest.write(out_dir, doc)
    _report.emit(_manifest.line(mpath))
    if v["partial"] and v["exit_code"] == core_report.EXIT_NOT_ACTIVE:
        _report.emit(_report.refused(f"partial activation in the generation process: {', '.join(v['partial'])} (KIT-RECORD above; a mode is all of its levers)", mode))
    _report.emit(_report.exit_line(mode=mode, route=route, rc=inv["rc"], written=census["designs_written"], expected=census["designs_expected"],
                                   wall=time.time() - t0, incomplete=census["incomplete"], exit_code=v["exit_code"], partial=v["partial"], gated=v["gated"]))
    return {"exit_code": v["exit_code"], "rc": inv["rc"], "census": census, "manifest": mpath, "invocation": inv, "proof": env_block.get("path"), "out": out_dir}

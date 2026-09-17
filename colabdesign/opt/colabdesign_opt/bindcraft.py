"""The driver of both arms: BindCraft's own design step, `binder_hallucination` (stock/src/bindcraft/functions/colabdesign_utils.py),
imported UNMODIFIED from the vendored, pinned tree and called with its arguments derived exactly as `bindcraft.py` derives them. The stock
arm (mode off: stock_launch.py → stock_design.py) runs it in the clean subprocess; the kit arm (exact | fast: kit_launch.py)
runs the same bytes after `levers.install(mode)` patched the AF2 model classes — the levers live inside the model, BindCraft's losses,
schedule, pLDDT gates, optimise_beta check and triage run as BindCraft runs them in both arms.

Import boundary (the ONE import-level deviation): BindCraft's `functions/__init__.py` star-imports every module
including `pyrosetta_utils` (PyRosetta: relax, interface scoring — the stages AFTER the design step). That module is not vendored and
nothing of it is executed: the driver registers a synthetic package `colabdesign_opt.bindcraft_functions` whose `__path__` is the vendored `functions/` directory and
imports `generic_utils`, `biopython_utils`, `colabdesign_utils` through it (their bytes run unmodified; their intra-package imports are
relative and resolve through that path), with `…bindcraft_functions.pyrosetta_utils` pre-seeded as a named-refusal module (any use of a
PyRosetta name raises `Refusal`). No top-level `functions` module exists anywhere, in or out of the arm.

Settings: settings.load(tree, advanced, filters) — bindcraft.py's `--advanced` / `--filters` files (default its own two, vendored;
the default advanced file checked present at the path stock/PINS.json declares), no value typed here. Inputs from the case: target PDB, chains,
binder length, hotspot (BindCraft's `target_hotspot_residues` string verbatim; absent = none), seed, the AF2 params root. Refusals BEFORE any
model is built (named, `Refusal`): a settings file off its pin; a `design_algorithm` BindCraft's loop does not know; `optimise_beta` with no
executable DSSP; a design model whose params file is absent; a settings key BindCraft reads that the file lacks (`KeyError` → named).

Outputs under <out>/ beside BindCraft's own tree (generate_directories: Trajectory/, …, failure_csv.csv): design.pdb (save_pdb of the best,
as BindCraft saves it), design.fasta (get_seqs), trajectory.jsonl (af_model._tmp["log"], one object per logged step). The run record
(`run_design`'s return value: timing from the observer's rows — units.py, one row per graded step / greedy forward / greedy round —, BindCraft's
terminate verdict + the gate that fired read back from failure_csv.csv, the outputs' sha256) stays in memory for the caller.
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import time
import types
from typing import Callable, List, Optional

from . import names, settings as _settings, units

PACKAGE = "colabdesign_opt.bindcraft_functions"                 # the synthetic package over <tree>/stock/src/bindcraft/functions
MODULES = ("generic_utils", "biopython_utils", "colabdesign_utils")
NOT_IMPORTED = "pyrosetta_utils"                                 # PyRosetta stage: a named-refusal module stands in
DESIGN_ALGORITHMS = ("2stage", "3stage", "greedy", "mcmc", "4stage")   # colabdesign_utils.py binder_hallucination: its if/elif chain; anything else prints ERROR and exit()s
FAILURE_CSV = "failure_csv.csv"                                  # bindcraft.py: os.path.join(design_path, 'failure_csv.csv')
GATE_PREFIX = "Trajectory_"                                      # the failure census columns the design step increments


class Refusal(RuntimeError):
    """A named refusal of the design step before (or instead of) running it; `.kind` is the short reason word."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class _RefusalModule(types.ModuleType):
    """Stands in for functions/pyrosetta_utils.py: importing names from it succeeds; using one raises Refusal by name."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def _refuse(*_a, **_k):
            raise Refusal("pyrosetta", f"BindCraft functions/pyrosetta_utils.py `{name}` needs PyRosetta — the relax / interface-scoring stage after the design step; this driver runs the design step only")
        _refuse.__name__ = name
        return _refuse


def tree_from_pins(pins_path: Optional[str]) -> Optional[str]:
    return os.path.dirname(os.path.dirname(os.path.abspath(pins_path))) if pins_path else None


def bindcraft_dir(tree: str) -> str:
    return os.path.join(tree, names.BINDCRAFT_DIR)


_F = {}                                                          # tree -> namespace of the imported modules


def functions(tree: str) -> types.SimpleNamespace:
    """BindCraft's generic_utils / biopython_utils / colabdesign_utils imported from the vendored tree (cached per process)."""
    if tree in _F:
        return _F[tree]
    fdir = os.path.join(bindcraft_dir(tree), "functions")
    if not os.path.isfile(os.path.join(fdir, "colabdesign_utils.py")):
        raise Refusal("vendored", f"BindCraft's functions/ not found under {fdir}")
    if PACKAGE in sys.modules and list(getattr(sys.modules[PACKAGE], "__path__", [])) != [fdir]:
        raise Refusal("vendored", f"{PACKAGE} already bound to {getattr(sys.modules[PACKAGE], '__path__', None)}, not {fdir}")
    pkg = types.ModuleType(PACKAGE, "BindCraft functions/ (vendored, stock/src/bindcraft/functions): imported file by file; its __init__ is never executed")
    pkg.__path__ = [fdir]; pkg.__package__ = PACKAGE; pkg.__file__ = None
    sys.modules[PACKAGE] = pkg
    ref = _RefusalModule(f"{PACKAGE}.{NOT_IMPORTED}", "named refusal for BindCraft functions/pyrosetta_utils.py (PyRosetta stage; never imported)")
    ref.__file__ = os.path.join(fdir, NOT_IMPORTED + ".py"); ref.__package__ = PACKAGE
    sys.modules[f"{PACKAGE}.{NOT_IMPORTED}"] = ref; setattr(pkg, NOT_IMPORTED, ref)
    parent = sys.modules.get("colabdesign_opt")
    if parent is not None:
        setattr(parent, "bindcraft_functions", pkg)
    mods = {}
    for m in MODULES:
        try:
            mod = importlib.import_module(f"{PACKAGE}.{m}")
        except ImportError as e:                                 # a package BindCraft's design step imports (numpy, pandas, scipy, matplotlib, Bio, jax, colabdesign) is absent
            raise Refusal("stack", f"BindCraft functions/{m}.py cannot be imported on this interpreter ({sys.executable}): {type(e).__name__}: {e} — the design step needs the pinned stack (stock/PINS.json pins)") from None
        if os.path.dirname(os.path.abspath(mod.__file__)) != os.path.abspath(fdir):
            raise Refusal("vendored", f"{PACKAGE}.{m} resolved to {mod.__file__}, not the vendored {fdir}")
        mods[m] = mod
    ns = types.SimpleNamespace(dir=fdir, package=pkg, **mods)
    _F[tree] = ns
    return ns


def derive(F, S: dict, *, tree: str, out_dir: str, params_dir: str, binder_len: int, seed: int, tag: str) -> dict:
    """bindcraft.py's own derivation of binder_hallucination's arguments from the settings `S` (settings.load's record: the advanced dict and
    the filters file; generic_utils), with the two installation paths of this machine: af_params_dir = --params-dir (bindcraft.py defaults it
    to its checkout folder), dssp_path = BindCraft's default (the vendored functions/dssp)."""
    G = F.generic_utils
    advanced = S["advanced"]
    try:
        design_models, prediction_models, multimer_validation = G.load_af2_models(advanced["use_multimer_design"])
        adv = G.perform_advanced_settings_check(dict(advanced), bindcraft_dir(tree))
        adv["af_params_dir"] = params_dir
        helicity_value = G.load_helicity(adv)
    except KeyError as e:
        raise Refusal("settings_key", f"settings key {e} absent from {S['file']} (read by BindCraft's derivation)") from None
    design_paths = G.generate_directories(out_dir)
    failure_csv = os.path.join(out_dir, FAILURE_CSV)
    G.generate_filter_pass_csv(failure_csv, S["filters"])
    return {"advanced": adv, "design_models": list(design_models), "prediction_models": list(prediction_models), "multimer_validation": multimer_validation,
            "helicity_value": helicity_value, "design_paths": design_paths, "failure_csv": failure_csv, "design_name": f"{tag}_l{int(binder_len)}_s{int(seed)}"}


def params_file(params_dir: str, model_index: int, use_multimer: bool) -> str:
    """The params file ColabDesign loads for design model `model_index` (mk_af_model: model_{k}_multimer_v3 | model_{k}_ptm under <data_dir>/params)."""
    name = f"model_{model_index + 1}_multimer_v3" if use_multimer else f"model_{model_index + 1}_ptm"
    return os.path.join(params_dir, "params", f"params_{name}.npz")


def executable(path: str) -> bool:
    """`path` is an executable file. A file that is there WITHOUT its executable bit (a tree fetched over HTTP loses it) gets the bit set back
    first (mode | 0o111); False only when the file is absent or the bit cannot be set here."""
    if os.path.isfile(path) and not os.access(path, os.X_OK):
        try:
            os.chmod(path, os.stat(path).st_mode | 0o111)
        except OSError:
            pass
    return os.path.isfile(path) and os.access(path, os.X_OK)


def refusals(adv: dict, derived: dict, params_dir: str) -> List[Refusal]:
    """Every reason the design step cannot run as the settings say, named — evaluated before any model is built."""
    out = []
    algo = adv.get("design_algorithm")
    if algo not in DESIGN_ALGORITHMS:
        out.append(Refusal("design_algorithm", f"settings design_algorithm={algo!r}: BindCraft's design step knows {'|'.join(DESIGN_ALGORITHMS)} (it would print ERROR and exit)"))
    if adv.get("optimise_beta"):
        dssp = adv.get("dssp_path") or ""
        if not executable(dssp):
            out.append(Refusal("dssp", f"settings optimise_beta=true runs DSSP on the stage-1 structure (biopython_utils.calc_ss_percentage); {dssp!r} is not an executable file (fix: chmod +x {dssp})"))
    for n in derived["design_models"]:
        p = params_file(params_dir, n, bool(adv.get("use_multimer_design")))
        if not os.path.isfile(p):
            out.append(Refusal("params", f"design model {n} (use_multimer_design={adv.get('use_multimer_design')}): params file absent: {p}"))
    if adv.get("save_design_trajectory_plots") or adv.get("save_design_animations"):
        import importlib.util as _u
        if _u.find_spec("matplotlib") is None:
            out.append(Refusal("matplotlib", "settings save_design_trajectory_plots / save_design_animations need matplotlib, which is not importable"))
    return out


def gate_fired(failure_csv: str) -> Optional[str]:
    """The failure-census column BindCraft's design step incremented for this trajectory (Trajectory_*), or None."""
    if not os.path.isfile(failure_csv):
        return None
    with open(failure_csv, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return None
    fired = [c for c, v in zip(rows[0], rows[-1]) if c.startswith(GATE_PREFIX) and _num(v) > 0]
    return ",".join(fired) if fired else None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _json_default(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    return str(x)


def _int_list(x):
    x = x.tolist() if hasattr(x, "tolist") else x
    return [int(v) for v in (x if isinstance(x, (list, tuple)) else [x])]


def prepare(tree: str, *, params_dir: str, out_dir: str, binder_len: int, seed: int, tag: str, advanced: Optional[str] = None,
            filters: Optional[str] = None) -> dict:
    """Load the settings (bindcraft.py's --advanced / --filters files; None = its defaults), import BindCraft's modules, derive the arguments,
    refuse by name if anything cannot be honoured, install the observer. Returns {"F", "settings", "derived"}; raises Refusal / settings.SettingsError."""
    S = _settings.load(tree, advanced, filters)
    F = functions(tree)
    D = derive(F, S, tree=tree, out_dir=out_dir, params_dir=params_dir, binder_len=binder_len, seed=seed, tag=tag)
    rs = refusals(D["advanced"], D, params_dir)
    if rs:
        raise Refusal(rs[0].kind, "; ".join(str(r) for r in rs))
    units.install()
    return {"F": F, "settings": S, "derived": D}


def run_design(prep: dict, *, target: str, chain: str, binder_len: int, hotspot: Optional[str], seed: int, out_dir: str, t_start: float,
               report=None, extra: Optional[dict] = None, step_callback: Optional[Callable[[int, dict], None]] = None) -> dict:
    """ONE trajectory: binder_hallucination as bindcraft.py calls it, then the outputs under out_dir. Returns the run record (in memory).
    `step_callback(step, state)`, when given, is called once per observed unit for this design (units.set_step_callback; a callback an
    in-process caller set beforehand stays in force when None); nothing is written for it."""
    F, S, D = prep["F"], prep["settings"], prep["derived"]
    adv = D["advanced"]
    os.makedirs(out_dir, exist_ok=True)
    units.reset()
    if step_callback is not None:
        units.set_step_callback(step_callback)
    ready_s = time.perf_counter() - t_start
    t0 = time.perf_counter()
    try:
        af = F.colabdesign_utils.binder_hallucination(D["design_name"], target, chain, hotspot or "", int(binder_len), int(seed), D["helicity_value"],
                                                     D["design_models"], adv, D["design_paths"], D["failure_csv"])
    except KeyError as e:
        raise Refusal("settings_key", f"settings key {e} absent from {S['file']} (read by binder_hallucination)") from None
    finally:
        if step_callback is not None:
            units.set_step_callback(None)
    total_s = time.perf_counter() - t0
    rows = units.rows()
    verdict = ""
    try:
        verdict = str((af.aux.get("log") or {}).get("terminate", ""))
    except Exception:                                             # noqa: BLE001 — a model without aux (never after a design) records ""
        verdict = ""
    terminate = {"verdict": verdict, "gate": gate_fired(D["failure_csv"])}
    end = units.end_row(terminate, rows)

    # ---- the outputs (host side; BindCraft's own files stay where it wrote them)
    pdb_path = os.path.join(out_dir, names.DESIGN_PDB)
    af.save_pdb(pdb_path)                                         # get_best=True: the best of the last stage that ran, as BindCraft saves it
    seqs = af.get_seqs()
    with open(os.path.join(out_dir, names.DESIGN_FASTA), "w", encoding="utf-8") as fh:
        for i, sq in enumerate(seqs):
            fh.write(f">binder_seed{seed}_{i}\n{sq}\n")
    log_rows = list(af._tmp["log"])
    for row in log_rows:
        if "models" in row:
            row["models"] = _int_list(row["models"])
    with open(os.path.join(out_dir, names.TRAJECTORY), "w", encoding="utf-8") as fh:
        for k, row in enumerate(log_rows):
            fh.write(json.dumps({"step": k, **row}, default=_json_default) + "\n")

    grads = [r for r in rows if r.get("kind") == "grad"]
    steady = [r["wall_s"] for r in grads if not r.get("first_call")]
    by_phase = {ph: _median([r["wall_s"] for r in grads if r["phase"] == ph and not r.get("first_call")]) for ph in ("soft", "temp", "hard")}
    by_phase["greedy_forward"] = _median([r["wall_s"] for r in rows if r.get("kind") == "forward" and not r.get("first_call")])
    first_calls = [r["wall_s"] for r in rows if r.get("first_call")]
    segments, recycles = report.work_segments(grads)                # equal-work cells: steady s/step per (stage, num_recycles) + where num_recycles changed
    import jax
    try:
        mem = {k: int(v) for k, v in (jax.devices()[0].memory_stats() or {}).items() if k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")}
    except Exception:                                             # noqa: BLE001
        mem = {}
    best = af._tmp.get("best") or {}
    lengths = [int(x) for x in af._lengths]
    run = {
        "tag": D["design_name"], "settings": _settings.record(S), "seed": int(seed), "hotspot": hotspot or None, "tokens": int(sum(lengths)), "lengths": lengths,
        "design_models": D["design_models"], "helicity_value": D["helicity_value"], "terminate": terminate, "n_steps": end["n_steps"], "phases_run": end["phases_run"],
        "steps": len(grads), "rows": len(rows),
        "timing": {"ready_s": ready_s, "first_calls_s": first_calls, "steady_s": _median(steady), "phase_steady_s": by_phase, "total_s": total_s,
                   "segments": segments, "recycles": recycles,
                   "process_total_s": time.perf_counter() - t_start},
        "final": {k: v for k, v in (log_rows[-1] if log_rows else {}).items() if k in ("loss", "plddt", "ptm", "i_ptm", "pae", "i_pae", "con", "i_con", "models", "recycles")},
        "best_metric": float(best["metric"]) if "metric" in best else None, "best_metric_name": af._args.get("best_metric"), "seq_best": seqs,
        "opt_num_recycles": af.opt.get("num_recycles"), "weights": {k: float(v) for k, v in af.opt["weights"].items()}, "con": dict(af.opt["con"]), "i_con": dict(af.opt["i_con"]),
        "memory": mem, "device": {"kind": str(jax.devices()[0].device_kind), "backend": jax.default_backend()},
        "files": {n: {"sha256": names.sha256_file(os.path.join(out_dir, n)), "bytes": os.path.getsize(os.path.join(out_dir, n))} for n in names.OUTPUTS},
        "bindcraft": {"commit_dir": names.BINDCRAFT_DIR, "design_paths": {k: os.path.relpath(v, out_dir) for k, v in D["design_paths"].items()}, "failure_csv": FAILURE_CSV},
        "colabdesign_file": sys.modules["colabdesign"].__file__,
        **(extra or {}),
    }
    if report is not None:
        report.log(report.run_summary_line(D["design_name"], run))
    return run


def _median(xs):
    import statistics
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None

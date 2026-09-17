"""python -m openfold3_opt {pred,check,warm} [--mode exact|fast|big|off] [--n_gpu 1|2|4|8] ...

A thin command layer over the kit code as shipped. It never re-implements a lever, a schema or a test:

* ``pred``   — ``openfold3_opt.enable(mode)``, then the stock ``run_openfold predict`` IN-PROCESS (the console script's entry point,
               ``openfold3.run_openfold:cli``) with ``--runner-yaml <the mode's configuration>``, ``--inference-ckpt-path $OPENFOLD3_CKPT``
               and upstream's own knobs exactly as given; afterwards the exit tally is printed (no kit file is written beside the outputs).
               One ``pred`` = one process = one ``run_openfold predict`` call on one query JSON — except ``--mode big`` on a query set that straddles the
               offload port's item gate: the port engages per item, so that set runs as one such pass per side of the gate (cli.run_item_groups). ``--mode off``: the stock CLI in a FRESH subprocess (``stock_pred.py``) with every
               name under stock/PINS.json's must-be-absent prefixes stripped and no kit directory on PYTHONPATH; the subprocess proves its
               environment before importing openfold3 and hands the proof to this process (a temporary file, read and removed).
               the runner yaml follows the mode (cli.row_yaml): ``off`` / ``exact`` run the stock configuration — the predict preset with the
               cuEquivariance triangle kernels on (modes.STOCK_YAML; under ``--det 1|2`` modes.STOCK_DET_YAML, the DS4Sci attention off) —,
               ``fast`` / ``big`` their kernels-off base; a caller's ``--runner-yaml`` (repeatable, argv order) is an OVERLAY on every
               route — laid on the configuration in force, the mode's own keys pinned on top and named — never a replacement; naming one
               of the tree's own configuration files under ``off`` selects it as the base (the `default` tier: shipped_predict.yml).
               under ``--mode off`` it is the stock anchor's form (needs deepspeed's evoformer_attn op installed).
               Upstream's own knobs — ``--num-model-seeds``, ``--num-diffusion-samples``, ``--use-msa-server``, ``--use-templates``,
               ``--runner-yaml`` — keep upstream's names and defaults (run_openfold.py:105-140): given, they are passed through verbatim on
               every mode exactly as under ``--mode off``; not given, they are not passed and upstream's defaults apply (1 model seed from
               start seed 42, 5 diffusion samples, 3 recycles, 200 steps, MSA server on, templates on). Inputs that carry their own MSAs and
               no templates run with ``--use-msa-server false --use-templates false``.
               ``--det 1``: the kit's deterministic reference on every arm (det.py).
* ``check``  — dry run: resolves and gates the mode on this box (pins, conflicting switches, GPU) and applies nothing.
* ``warm``   — one public-input prediction under the mode (warm.py): imports, JIT builds, first captures.

The weights: every route that takes a checkpoint hashes it once against the pinned weights (stock/PINS.json "weights"; ``weights_check``) and
prints one WEIGHTS line — ``pinned`` (the pinned checkpoint) or ``WARNING: WEIGHTS unknown …; proceeding`` (other bytes: every mode runs,
labelled on the WEIGHTS line); the digest is memoised on disk (``stack.cache_root()``/weights_digests.json, keyed by
realpath/size/mtime/inode — the key selects the entry, never decides identity; ``check`` hashes afresh).

Mode: ``--mode`` when given, else ``OPENFOLD3_OPT``, else the package default (modes.DEFAULT_MODE); a ``--mode`` that disagrees with a set
``OPENFOLD3_OPT`` is refused (rc 2). Exit codes: 0 ok, 1 the command failed, 2 usage, 3 not active / not stock.
The exit rule of the verbs that produce structures (``pred`` on both routes, ``warm``): the runner's rc as it is; then outputs short of the expected count (``expected_structures``: queries x seeds x
samples) -> ``incomplete``, rc 1; then a partial line (a requested lever without the kit's own record, ``levers_unavailable`` — the kit's
named fallback — or ``arm_complete`` false) -> rc 3, by name (a mode is all of its levers: no opt-out); the gate is printed as one line
(``exit_rule``). The env route (a plain
``run_openfold`` process under the autoload) has no verb to gate and only records.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import subprocess
import sys
from typing import Tuple, List, Optional

from . import __version__
from . import modes
from .report import PREFIX, exit_code

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3
EXIT_TEMPLATES_DROPPED = 5                                          # templ_census.EXIT_TEMPLATES_DROPPED: declared templates dropped before the model (exit_rule); --allow-template-drop is the recorded opt-out
UPSTREAM_FIX_FLAG = "--upstream-fix"                 # == opt_core.upstream_fix.FLAG (tests lock the two); spelled here so building the parser imports nothing else
UPSTREAM_KNOBS = ("--num-diffusion-samples", "--num-model-seeds", "--runner-yaml", "--use-msa-server", "--use-templates")   # run_openfold predict's own options, passed through
#   verbatim when given and OMITTED otherwise, so upstream's defaults apply as shipped (stock/src/openfold3/run_openfold.py:105-140: samples None -> 5 and seeds None -> the
#   runner yaml's list / [42], model_config.py:180 + entry_points/validator.py:264; use_msa_server True; use_templates True); --runner-yaml unnamed = the MODE's configuration (row_yaml)
UPSTREAM_USE_TEMPLATES = "true"                     # run_openfold.py:135-140 (default=True): the word a run carries when --use-templates is not given
UPSTREAM_USE_MSA_SERVER = "true"                    # run_openfold.py:128-133 (default=True)
WARM_KNOBS = {"use_msa_server": "false", "use_templates": "false", "num_model_seeds": 1, "num_diffusion_samples": 1}   # warm's one public-input prediction (warm_argv): no network, one structure


def _err(msg: str) -> None:
    print(f"{PREFIX} {msg}", file=sys.stderr, flush=True)


def resolve_mode(arg: Optional[str], environ=None) -> str:
    environ = os.environ if environ is None else environ
    from opt_core.modes import mode_argument
    envmode = (environ.get("OPENFOLD3_OPT") or "").strip().lower()
    if arg and envmode and arg.lower() != envmode:
        raise SystemExit(_usage(f"--mode {arg} disagrees with OPENFOLD3_OPT={envmode}; one invocation runs one mode — drop one of them"))
    return mode_argument(arg, envmode, modes.table())                    # command line > OPENFOLD3_OPT > the package default


def _usage(msg: str) -> int:
    _err(msg)
    return EXIT_USAGE


def _home() -> str:
    from . import env as _env
    return _env.tree_home()


def _ckpt(arg: Optional[str]) -> Optional[str]:
    return arg or os.environ.get("OPENFOLD3_CKPT")


def effective_use_templates(a) -> str:
    """The run's ``--use-templates`` word as run_openfold will see it: the flag when given, else upstream's default (UPSTREAM_USE_TEMPLATES)."""
    return str(getattr(a, "use_templates", None) or UPSTREAM_USE_TEMPLATES).strip().lower()


def resolve_upstream_fix(a, home: str, verb: str) -> List[str]:
    """``--upstream-fix``: the requested IDs, each registered (upstream_fix_registry: one file per issue under upstream_issues/) and its run
    precondition met (opt_core.upstream_fix Registry.resolve with the run's words); ValueError names an unknown ID or the unmet precondition
    (``--upstream-fix OF3-001 requires --use-templates true``). Resolved before any launch; with the flag absent nothing is imported or loaded
    (the process that runs the model applies the fixes)."""
    value = getattr(a, "upstream_fix", None)
    if not value:
        return []
    from opt_core import upstream_fix as _upstream_fix
    ids = _upstream_fix.parse(value)
    if not ids:
        return []
    from . import upstream_fix_registry
    word = WARM_KNOBS["use_templates"] if verb == "warm" else effective_use_templates(a)   # warm runs its own knobs (warm_argv); pred the caller's, else upstream's default
    return upstream_fix_registry(home).resolve(ids, run={"verb": verb, "use_templates": word})


def apply_upstream_fix(ids: List[str], home: str, what: str) -> Optional[List[dict]]:
    """Install the requested fixes in THIS process (the registry's apply: loads each issue file, prints UPSTREAM-FIX <ID> applied); ``[]`` for
    none. A fix that cannot be installed refuses the run by name (NOT ACTIVE: UPSTREAM-FIX NOT APPLIED …) → None."""
    if not ids:
        return []
    from opt_core import upstream_fix as _upstream_fix
    from . import upstream_fix_registry
    try:
        records = upstream_fix_registry(home).apply(ids, log=_err)      # _err: this package's stderr line (prefix + words) — the census line and the fix's own words
    except Exception as e:  # noqa: BLE001 — any failure of a requested fix is a refusal, never a run without it
        _err(f"NOT ACTIVE: {_upstream_fix.NOT_APPLIED} ({what} {UPSTREAM_FIX_FLAG} {','.join(ids)}): {type(e).__name__}: {e}")
        return None
    for rec in records:
        if rec.get("consumes_templates"):                                # a fix that consumes templates at inference: this process's template guard census says so
            from . import templ_guard
            templ_guard.consumed_by(rec["id"])
    return records


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="openfold3-opt", description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=f"openfold3_opt {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, ckpt=True):
        p.add_argument("--mode", default=None, help="exact|fast|big|off (default: OPENFOLD3_OPT, else the package default)")
        p.add_argument("--n_gpu", default=None, help="the memory mode's multi-GPU axis (opt_core.mem.ngpu): 1 (default; OPENFOLD3_OPT_N_GPU) = one GPU under any mode; "
                                                     "2|4|8 under --mode big = the tp line, the pair representation row-sharded over that many GPUs of this box, one rank "
                                                     "process each; refused by name under exact/fast/off and when fewer GPUs are visible")
        p.add_argument("--graphs-max-tokens", default=None, help="the CUDA-graph size gate's cap (OPENFOLD3_OPT_GRAPHS_MAX_TOKENS; opt_core.mem.graph_gate words): unset|always = no cap "
                                                                  "(the default: the lines as shipped), N = `pred` captures graphs only for a query of at most N polymer tokens, 0|off = never — available, not the default")
        p.add_argument("--no-compile", dest="no_compile", action="store_true",
                       help=f"the model-opt kits' opt-out of a kit-added torch.compile lever, an alias of {modes.ENV_LEVERS_OFF}=compile: accepted on every mode and a "
                            "named no-op in this kit — no mode compiles anything and stock does not either (the ACTIVE / DRY-RUN lines say compile=none)")
        if ckpt:
            p.add_argument("--ckpt", "--inference-ckpt-path", dest="ckpt", default=None, help="the weights file, upstream's --inference-ckpt-path (default: $OPENFOLD3_CKPT)")

    def upstream_fix_flag(p):
        p.add_argument(UPSTREAM_FIX_FLAG, dest="upstream_fix", default=None, metavar="ID[,ID...]",
                       help="apply the named fixes of confirmed upstream issues (upstream_issues/<ID>_*.py; README 'Known upstream issues') on this run — "
                            "every arm, the stock caller included; the process that runs the model prints UPSTREAM-FIX <ID> applied. "
                            "Default: none (upstream's behaviour as shipped); an unknown ID is refused by name")

    p = sub.add_parser("pred", help="one run_openfold predict call under the mode (or the stock caller, --mode off)")
    common(p)
    p.add_argument("--query-json", "--query_json", dest="query_json", required=True, help="upstream's query JSON (one call)")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    p.add_argument("--num-model-seeds", type=int, default=None, help="upstream's flag, passed through (default: upstream's — the runner yaml's seed list, else [42])")
    p.add_argument("--num-diffusion-samples", type=int, default=None, help="upstream's flag, passed through (default: upstream's 5)")
    p.add_argument("--use-templates", default=None, help="true|false: upstream's flag, passed through (default: upstream's true)")
    p.add_argument("--use-msa-server", default=None, help="true|false: upstream's flag, passed through (default: upstream's true; inputs that carry their MSAs: false)")
    p.add_argument("--runner-yaml", action="append", default=None,
                   help="a runner yaml laid ON the configuration in force — an overlay on every route, never a replacement (default base: the mode's — "
                        "off/exact: the stock configuration, stock's cuEquivariance triangle kernels, the DS4Sci attention off under --det; fast/big: their "
                        "kernels-off base): the mode's own keys are pinned on top and every override named (runner_yaml.compose); repeatable — several are "
                        "laid in argv order, a later yaml's keys on top (runner_yaml.merge_callers); under off, naming one of the tree's own configuration "
                        "files (opt/openfold3_opt/shipped_predict.yml — the shipped `default` tier —, the offload configs) selects that file as the base, as given")
    p.add_argument("--det", default="0", help="0|1: the deterministic recipe on this arm (det.py)")
    p.add_argument("--allow-template-drop", action="store_true", help="accept with exit 0 a run whose DECLARED templates reached the model as an empty stack (TEMPLATES DROPPED, "
                                                                     "else exit 5); named on the exit-rule line, the census word fallbacks=templates_dropped:… stays")
    upstream_fix_flag(p)

    p = sub.add_parser("check", help="dry run: resolve and gate the mode on this box; apply nothing (a named checkpoint is hashed afresh: its digest memo entry rewritten)")
    common(p)
    p.add_argument("--json", default=None, help="write the report here")
    p.add_argument("--det", default="0", help="the pred's recipe level, for the route refusals")
    p.add_argument("--runner-yaml", action="append", default=None, help="the pred's runner yaml(s), laid on the configuration in force as `pred` lays them, gated on this stack (default: the line's / the mode's configuration)")

    p = sub.add_parser("warm", help="one public-input prediction under the mode: imports, JIT, first captures")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--tiling", default=None, help="the public input's size: 1to1 (199 tokens, the default) .. 6to6 (1194 tokens) — n barnase + n barstar chains — or a "
                                                  "comma-separated list folded in turn by the one warm process (e.g. 2to2,4to4,6to6: the 400 / 800 / 1200-token size "
                                                  "classes' kernel variants and captures are met here instead of on the first real input of each size)")
    upstream_fix_flag(p)
    return ap


# ----------------------------------------------------------------------------------------------------------- exit rule ----
def expected_structures(query_json: Optional[str], runner_yaml: Optional[str], num_model_seeds: Optional[int], num_diffusion_samples: Optional[int]) -> Optional[int]:
    """queries x seeds x samples as upstream's writer produces them (`<query>/seed_<s>/<query>_seed_<s>_sample_<k>_model.cif`): the query JSON's
    entries; the seed count from --num-model-seeds, else the runner yaml's `experiment_settings.seeds` (an int is one seed, a list its length —
    openfold3/entry_points/validator.py:264), else upstream's default list `[42]`; the sample count from --num-diffusion-samples, else
    upstream's default 5 (projects/of3_all_atom/config/model_config.py:180). None when the query JSON or the yaml cannot be read: the exit
    rule then gates on rc and the line only, and says so."""
    try:
        with open(query_json, encoding="utf-8") as fh:
            n_queries = len(json.load(fh)["queries"])
    except (OSError, TypeError, ValueError, KeyError):
        return None
    seeds = num_model_seeds
    if seeds is None:
        seeds = 1
        if runner_yaml:
            try:
                import yaml
                with open(runner_yaml, encoding="utf-8") as fh:
                    s = ((yaml.safe_load(fh) or {}).get("experiment_settings") or {}).get("seeds")
                seeds = len(s) if isinstance(s, list) else 1
            except (OSError, ImportError, ValueError, AttributeError):
                return None
    samples = num_diffusion_samples if num_diffusion_samples is not None else 5
    return n_queries * seeds * samples


def exit_rule(what: str, rc: int, n_cif: Optional[int], expected: Optional[int], partial: bool, arm_complete,
              dropped: int = 0, allow_template_drop: bool = False, conf_not_written: int = 0) -> (int, dict):
    """The exit rule of the CLI verbs (the env route — a plain `run_openfold` process under the autoload — has no verb to gate and only
    records): 1. the runner's rc as it is; 2. structure-only model files whose confidence was not written (structure first) -> `incomplete: confidence not written for k/S`, EXIT_FAIL; outputs short of the expected count -> `incomplete`, EXIT_FAIL; 3. a partial line — a
    requested lever the kit's own record does not show installed (`levers_unavailable`, the kit's named fallback) or `arm_complete`
    false — -> EXIT_NOT_ACTIVE, by name (a mode is all of its levers on the card: no opt-out); 4. `dropped` queries whose declared templates reached the
    model as an empty stack (templ_census.judge) -> EXIT_TEMPLATES_DROPPED, unless --allow-template-drop, the recorded opt-out (the census
    word stays either way). Returns (exit code, the gate record of the run record's "exit_rule"; printed as one line)."""
    incomplete = bool(expected is not None and n_cif is not None and n_cif < expected)
    is_partial = bool(partial) or arm_complete is False
    if rc != 0:
        code, reason = rc, f"runner rc {rc}"
    elif conf_not_written:                                               # structure first: structure-only model files (pLDDT column 0.00) are on disk, their confidences are not -> never `ok`
        code, reason = EXIT_FAIL, (f"incomplete: confidence not written for {int(conf_not_written)}/{expected if expected is not None else '?'} structures "
                                   f"(structure-only model files on disk, not counted: structures {n_cif}/{expected if expected is not None else '?'})")
    elif incomplete:
        code, reason = EXIT_FAIL, f"incomplete: {n_cif}/{expected} structures"
    elif dropped and not allow_template_drop:
        code, reason = EXIT_TEMPLATES_DROPPED, f"TEMPLATES DROPPED: {dropped} templated quer{'y' if dropped == 1 else 'ies'} reached the model with a declared chain's templates dropped (no real template, or a declared chain untemplated; fallbacks=templates_dropped); --allow-template-drop is the recorded opt-out"
    elif dropped:
        code, reason = EXIT_OK, f"template drop accepted by --allow-template-drop (recorded; {dropped} quer{'y' if dropped == 1 else 'ies'}, the census word fallbacks=templates_dropped stays)"
    elif is_partial:
        code, reason = EXIT_NOT_ACTIVE, "NOT ACTIVE: the run completed with levers pending (partial: levers_pending in the exit tally); a mode is all of its levers"
    else:
        code, reason = EXIT_OK, "ok" if expected is not None else "ok (expected count unknown: rc and the line gated only)"
    gate = {"rule": "rc; incomplete -> 1; templates dropped -> 5 unless allow_template_drop; partial -> 3", "rc": rc, "n_cif": n_cif, "expected": expected,
            "incomplete": incomplete or bool(conf_not_written), "confidence_not_written": int(conf_not_written), "templates_dropped": int(dropped), "allow_template_drop": bool(allow_template_drop),
            "partial": is_partial, "arm_complete": arm_complete, "exit_code": code, "reason": reason}
    _err(f"{what}: exit rule -> {code} ({reason}; structures {n_cif}/{expected if expected is not None else '?'}; templates_dropped={int(dropped)}; partial={is_partial}; "
         f"allow_template_drop={bool(allow_template_drop)})")
    return code, gate


def gated_manifest(out_dir: str, rep: dict, what: str, rc: int, expected: Optional[int], allow_template_drop: bool = False, **kw) -> int:
    """Build the run record, judge the template guard (templates_guard), apply the exit rule to it (n_cif / templates dropped / partial / arm_complete),
    return the exit code. Nothing is written beside the outputs; a rank of the row-sharded line hands the record (exit code, gate, guard) to its
    launcher (manifest.RECORDS_ENV: the launcher's per-launch records directory, tp.launch)."""
    from . import manifest as _manifest
    man = _manifest.build(rep, out_dir=out_dir, **kw)
    man["templates"] = templates_guard(rc, kw.get("stock_proof"))
    code, gate = exit_rule(what, rc, man.get("n_cif"), expected, man.get("partial"), man.get("arm_complete"),
                           dropped=len(man["templates"]["dropped"]), allow_template_drop=allow_template_drop,
                           conf_not_written=len((man.get("structure_first") or {}).get("confidence_not_written") or []))
    man["exit_code"], man["exit_rule"] = code, gate
    records = os.environ.get(_manifest.RECORDS_ENV)                      # a rank of the row-sharded line: its launcher reads rank 0's record back (tp.launch) and removes the directory
    if records:
        _manifest.dump(os.path.join(records, f"rank{tp_rank() or 0}.json"), man)
    return code


def templates_guard(rc: int, stock_proof: Optional[dict] = None) -> dict:
    """The template guard of one call, judged in the primary process (templ_census): what the query DECLARED (templ_census.declared, taken before
    the run on every route) against what the model process FEATURISED (this process's record on the in-process routes; the stock child's, handed
    over in its proof, on the stock route). Judged only when the runner returned 0 (a failed run is the exit rule's first clause). Prints one
    TEMPLATES FEATURISED line per declared-templated query that kept real slots and one TEMPLATES DROPPED line per dropped query; returns
    {"declared", "featurised", "dropped", "judged"} for the run record."""
    from . import templ_census
    declared = templ_census.declared()
    feats = dict((stock_proof or {}).get("templates_featurised") or {}) if stock_proof is not None else templ_census.featurised()
    dropped = templ_census.judge(declared, feats) if rc == 0 else []
    for line in (templ_census.featurised_lines(declared, feats) if rc == 0 else []) + templ_census.drop_lines(dropped):
        _err(line.split("] ", 1)[1] if line.startswith("[") else line)      # _err adds the tag: one FEATURISED line per templated query that kept its templates, one DROPPED line per query that lost them
    return {"declared": declared, "featurised": feats, "dropped": dropped, "judged": rc == 0}


# --------------------------------------------------------------------------------------------------------------- weights ----
WEIGHTS_REFRESH = {"check": True, "pred": False, "warm": False}   # which verb hashes the checkpoint afresh (rewriting its digest memo entry) and which reads the memo (stack.weights_gate refresh)


def weights_check(ckpt: str, home: str, *, what: str, refresh: bool = False):
    """The checkpoint against the pinned weights (stack.weights_gate), once per process: the pin is the pin note, never a
    condition. Returns (record, exit code): EXIT_USAGE when the file is missing; else EXIT_OK with one WEIGHTS line — `pinned` (the checkpoint
    as shipped), `unknown` + WARNING (other bytes: named and the run proceeds on that checkpoint), either word
    followed by `(cached digest <utc>)` when the digest came from the memo (digest_memo.word; `refresh=True` = hash afresh, the `check` verb's)."""
    from . import digest_memo, stack
    info = stack.weights_gate(ckpt, home, refresh=refresh)
    if not info["exists"]:
        return info, _usage(f"no such checkpoint: {ckpt}")
    if info["is_pinned"]:
        _err(f"WEIGHTS {digest_memo.word('pinned', info['digest_cached_utc'])}: {ckpt} sha256={info['sha256'][:8]} == {info['pinned_file']} (stock/PINS.json): the pinned checkpoint")
        return info, EXIT_OK
    _err(f"WARNING: WEIGHTS {digest_memo.word('unknown', info['digest_cached_utc'])} sha256={info['sha256'][:8]} ({ckpt} is not the pinned checkpoint {info['pinned_file']} {info['pinned_sha256'][:8]}, stock/PINS.json); "
         f"proceeding — {what} runs on a checkpoint that is not the pinned one")
    return info, EXIT_OK


# ------------------------------------------------------------------------------------------------------------------ pred ----
def predict_argv(ckpt: Optional[str], query_json: str, output_dir: str, runner_yaml: str, *, use_msa_server=None, use_templates=None,
                 num_model_seeds=None, num_diffusion_samples=None) -> List[str]:
    """The `run_openfold predict` argument list every route builds: the query, the output directory, the runner yaml in force (row_yaml: the
    caller's, else the mode's configuration), the checkpoint, then upstream's own knobs exactly as given — a knob not given is not passed, so
    upstream's default applies as shipped (UPSTREAM_KNOBS)."""
    argv = ["--query-json", query_json, "--output-dir", output_dir, "--runner-yaml", runner_yaml]
    if ckpt:
        argv += ["--inference-ckpt-path", ckpt]
    for flag, val in (("--use-msa-server", use_msa_server), ("--use-templates", use_templates)):
        if val is not None:
            argv += [flag, str(val).strip().lower()]
    for flag, val in (("--num-model-seeds", num_model_seeds), ("--num-diffusion-samples", num_diffusion_samples)):
        if val is not None:
            argv += [flag, str(int(val))]
    return argv


def row_counts(a) -> tuple:
    """(model seeds, diffusion samples) a `pred` names: the flags when given, None = upstream's default (expected_structures resolves it)."""
    return (a.num_model_seeds, a.num_diffusion_samples)


def protocol_dims(runner_yaml: Optional[str], argv=()) -> dict:
    """The resolved protocol dimensions of a run: recycles / diffusion steps from the runner yaml in force (upstream's defaults where the yaml
    is silent: 3 recycles, 200 steps — projects/of3_all_atom/config/model_config.py), diffusion samples from the argv (else upstream's 5)."""
    dims = {"recycles": 3, "diffusion_steps": 200, "diffusion_samples": 5}
    if runner_yaml:
        import yaml
        path = runner_yaml if os.path.isabs(runner_yaml) else os.path.join(_home(), runner_yaml)
        try:
            with open(path, encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
        except (OSError, ValueError, yaml.YAMLError):                  # an unreadable yaml: the run itself fails by name on it; this line says what it could not read
            doc = None
        if doc is None:
            dims["recycles"] = dims["diffusion_steps"] = "?"
        else:
            shared = (((doc.get("model_update") or {}).get("custom") or {}).get("architecture") or {}).get("shared") or {}
            if "num_recycles" in shared:
                dims["recycles"] = int(shared["num_recycles"])
            if "no_full_rollout_steps" in (shared.get("diffusion") or {}):
                dims["diffusion_steps"] = int(shared["diffusion"]["no_full_rollout_steps"])
    argv = list(argv)
    if "--num-diffusion-samples" in argv:
        dims["diffusion_samples"] = int(argv[argv.index("--num-diffusion-samples") + 1])
    dims["trunk_passes"] = dims["recycles"] + 1 if isinstance(dims["recycles"], int) else "?"
    dims["confidence_passes"] = dims["diffusion_samples"]
    return dims


def protocol_line(runner_yaml: Optional[str], argv=()) -> str:
    """``[openfold3-opt] recycles=… trunk_passes=… diffusion_steps=… diffusion_samples=… confidence_passes=…`` — what this run computes, read
    from its runner yaml and argv (protocol_dims)."""
    d = protocol_dims(runner_yaml, argv)
    return "[openfold3-opt] " + " ".join(f"{k}={d[k]}" for k in ("recycles", "trunk_passes", "diffusion_steps", "diffusion_samples", "confidence_passes"))


STOCK_KERNEL_MODES = ("off", "exact")              # the modes that run stock's triangle kernels (modes.STOCK_YAML / STOCK_DET_YAML): the stock caller and the exact line; fast and big run their own pinned kernels-off base


def kernel_policy(mode: Optional[str], det_level: int = 0) -> Optional[dict]:
    """The eval kernel flags a call runs BY MODE: `off` and `exact` run stock's — the cuEquivariance triangle kernels on, the DS4Sci evoformer attention as
    shipped (on) at det 0 and off under the det recipe (levels 1|2: that kernel is not run-to-run deterministic) —; None for `fast` / `big` (their
    rows' yamls as written: the kernels-off base their cells and offload wrappers are pinned on) and for an unnamed mode."""
    if mode not in STOCK_KERNEL_MODES:
        return None
    return {"use_cueq_triangle_kernels": True, "use_deepspeed_evo_attention": int(det_level or 0) < 1}


def member_yaml(home: str, line: Optional[modes.Line] = None, precision: Optional[str] = None, mode: Optional[str] = None, det: int = 0) -> str:
    """The mode's own runner yaml on the tree (the yaml a call runs when the caller passes none): the line's yaml where it names one
    (modes.Line.runner_yaml: the big lines); else modes.STOCK_YAML under `off` / `exact` (modes.STOCK_DET_YAML under the det recipe);
    modes.FAST_BF16_YAML under the fast line's bf16; modes.KERNELS_OFF_YAML otherwise."""
    if line is not None and line.runner_yaml:
        return line.runner_yaml if os.path.isabs(line.runner_yaml) else os.path.join(home, line.runner_yaml)
    policy = kernel_policy(mode, det)
    if policy:
        return os.path.join(home, modes.STOCK_DET_YAML if not policy["use_deepspeed_evo_attention"] else modes.STOCK_YAML)
    return os.path.join(home, modes.FAST_BF16_YAML if precision == modes.FAST_PRECISION else modes.KERNELS_OFF_YAML)


def row_yaml(home: str, line: Optional[modes.Line] = None, precision: Optional[str] = None, out_dir: Optional[str] = None,
             runner_yaml=None, mode: Optional[str] = None, det: int = 0, as_given: bool = False) -> str:
    """The runner yaml of a call on the tree. No `--runner-yaml`: the mode's own (member_yaml). A caller's `--runner-yaml` (one path, or the
    flag repeated: a list in argv order — a later yaml's keys on top, runner_yaml.merge_callers) is an OVERLAY on EVERY route, never a
    replacement: it is COMPOSED UNDER THE CONFIGURATION IN FORCE (runner_yaml.compose) — under `off` the stock configuration (member_yaml:
    modes.STOCK_YAML, modes.STOCK_DET_YAML under the det recipe), under a kit mode the line's —: every execution key the member writes
    (runner_yaml.line_pins; the graphed line adds the alternative kernel flags off) is laid on top and every override is named; the caller's
    other keys (a seed list, num_recycles, template settings …) run as written. Naming either stock configuration file under `off` / `exact`
    names the stock FAMILY, whose member follows the det level exactly as without the flag; naming one of the tree's other configuration
    files under `off` (KIT_BASE_YAMLS: shipped_predict.yml = the `default` tier, the offload configs) selects that file as the BASE the stock
    caller runs as given — any further caller yaml is laid on it. `as_given`: a row-sharded RANK — its one yaml is the launcher's composition
    already (tp.pinned_yaml)."""
    callers = caller_yamls(runner_yaml)
    if as_given and callers:                                                         # a row-sharded rank: the launcher composed and named it
        return callers[-1]
    policy = kernel_policy(mode, det)
    if policy:                                                                       # a caller naming either stock yaml under off / exact names the stock FAMILY: the member follows the det level (kernel_policy), as without the flag
        callers = [c for c in callers if not stock_family(home, c)]
    member = member_yaml(home, line, precision, mode, det)
    from . import runner_yaml as _ry
    if mode in (None, "off"):
        bases = [c for c in callers if kit_base(home, c)]
        if bases:                                                                    # one of the tree's own configuration files under off: the configuration the stock caller runs, as given (said once) — the base of any further overlay
            member = bases[-1]
            callers = [c for c in callers if c not in bases]
            _ry.as_given_note(home, member)
    if not callers:
        return member
    out_dir = out_dir or tempfile.mkdtemp(prefix="openfold3_opt_yaml_")
    own = _ry.merge_callers(home, callers, out_dir)                                  # one caller: its own path; several: laid in argv order into one document, named
    path, record = _ry.compose(home, own, member, _ry.line_pins(line, member), out_dir, "stock" if mode == "off" else (mode or "base"))   # the note's word: `composed under the stock configuration (opt/openfold3_opt/stock_cueq_on_predict.yml)` under off
    if len(callers) > 1:
        record = dict(record, callers=list(callers))
    COMPOSED[composed_key(runner_yaml)] = record                                    # the run record's `runner_yaml_composed` (gated_manifest extra)
    return path


def caller_yamls(runner_yaml) -> List[str]:
    """A call's `--runner-yaml` value(s) in argv order: None -> [], one path -> [path], the flag repeated (argparse append) -> every path."""
    if not runner_yaml:
        return []
    return [runner_yaml] if isinstance(runner_yaml, str) else [y for y in runner_yaml if y]


def composed_key(runner_yaml):
    """The COMPOSED key of a call's caller yaml(s): the one path (as before), else the tuple of paths in argv order."""
    c = caller_yamls(runner_yaml)
    return c[0] if len(c) == 1 else tuple(c)


COMPOSED: dict = {}                            # caller yaml -> runner_yaml.compose record of this process (the last composition of each caller file)

def stock_family(home: str, path: str) -> bool:
    """Whether a runner yaml IS one of the tree's two stock configuration files (modes.STOCK_YAML, modes.STOCK_DET_YAML) — by resolved path, else by
    content digest (a staged copy): such a `--runner-yaml` under `off` / `exact` names the stock family and the det level picks the member (row_yaml)."""
    from . import manifest
    fam = [os.path.join(home, modes.STOCK_YAML), os.path.join(home, modes.STOCK_DET_YAML)]
    p = path if os.path.isabs(path) else os.path.join(home, path)
    if any(os.path.realpath(p) == os.path.realpath(f) for f in fam):
        return True
    try:
        return os.path.isfile(p) and manifest.sha256_file(p) in {manifest.sha256_file(f) for f in fam if os.path.isfile(f)}
    except OSError:
        return False


KIT_BASE_YAMLS = ("SHIPPED_YAML", "KERNELS_OFF_YAML", "FAST_BF16_YAML", "BIG_BF16_C16_YAML", "OFFLOAD_C16_YAML", "OFFLOAD_SHIPPED_C16_YAML")   # modes' names of the tree's own configuration files besides the stock pair (stock_family): named under `off`, one of them IS the configuration the stock caller runs — the `default` tier (SHIPPED_YAML), the big equality runs' stock arm (OFFLOAD_C16_YAML) — a base as given, never an overlay (row_yaml)


def kit_base(home: str, path: str) -> bool:
    """Whether a runner yaml IS one of the tree's own configuration files named in KIT_BASE_YAMLS — by resolved path, else by content digest (a
    staged copy). A copy with keys added (a seed list …) is not: it is an overlay like any caller yaml."""
    from . import manifest
    fam = [os.path.join(home, getattr(modes, n)) for n in KIT_BASE_YAMLS if getattr(modes, n, None)]
    p = path if os.path.isabs(path) else os.path.join(home, path)
    if any(os.path.realpath(p) == os.path.realpath(f) for f in fam):
        return True
    try:
        return os.path.isfile(p) and manifest.sha256_file(p) in {manifest.sha256_file(f) for f in fam if os.path.isfile(f)}
    except OSError:
        return False


KERNEL_FLAGS_PATH = ("model_update", "custom", "settings", "memory", "eval")           # the eval kernel flags' key path in a runner yaml (stock_predict.yml)
KERNEL_FLAG_PATH = KERNEL_FLAGS_PATH + ("use_deepspeed_evo_attention",)                 # the DS4Sci flag's full key path
# Upstream's third-party attention kernel flags and their shipped CUDA defaults (openfold3/projects/of3_all_atom/config/model_config.py:115-118):
# a key the runner yaml does not set keeps its default — the DS4Sci evoformer attention ON, the cuEquivariance triangle kernels OFF.
KERNEL_FLAG_DEFAULTS = {"use_deepspeed_evo_attention": True, "use_cueq_triangle_kernels": False}


def yaml_kernel_flags(path: str) -> dict:
    """The third-party attention kernel flags a runner yaml resolves to: each of KERNEL_FLAG_DEFAULTS read at KERNEL_FLAGS_PATH (the yaml
    parsed by the stack's PyYAML: any spelling YAML reads as a boolean, block or flow style), absent = upstream's shipped default. A yaml that
    does not parse, or a flag that is not a YAML boolean, is refused (raised) — never read as a default."""
    import yaml
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    node = doc
    for key in KERNEL_FLAGS_PATH:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            node = {}
            break
    flags = {}
    for flag, default in KERNEL_FLAG_DEFAULTS.items():
        value = node.get(flag, default) if isinstance(node, dict) else default
        if not isinstance(value, bool):
            raise ValueError(f"{path}: {flag} is {value!r}, not a YAML boolean")
        flags[flag] = value
    return flags


def yaml_ds4sci_off(path: str) -> bool:
    """Whether a runner yaml turns the DS4Sci evoformer attention off (`use_deepspeed_evo_attention: false` in any YAML spelling); absent =
    upstream's shipped default, on. A yaml that does not parse is refused (raised), never read as on."""
    return not yaml_kernel_flags(path)["use_deepspeed_evo_attention"]



STACK_REFUSED = "STACK REFUSED"                # the line of a call this stack cannot run as configured (rc 3): the DS4Sci kernel on without deepspeed's pre-built op
DS4SCI_STACK: dict = {}                        # the probe's record when an arm runs the DS4Sci kernel (the run record's `extra.evoformer_attn_op`)


def ds4sci_stack_check(home: str, yml: str, what: str, probe=None, code=None, cc=None) -> Optional[str]:
    """An arm whose runner yaml leaves upstream's DS4Sci evoformer attention ON (the shipped configuration: `--runner-yaml
    opt/openfold3_opt/stock_cueq_on_predict.yml`: the stock arm and `exact` at det 0; the big rows on `shipped_predict_c16.yml`) needs
    deepspeed's evoformer_attn op INSTALLED on the stack (stack.evoformer_attn_op) AND built with device code this GPU can execute
    (stack.evoformer_attn_op_code against the device's compute capability, stack.device_code_serves: an op compiled for sm_90 alone carries
    no kernel an A100, compute capability 8.0, can launch): without the op upstream catches the per-item `Unable to JIT load the
    evoformer_attn op`, with an op lacking this GPU's code the per-item `no kernel image is available for execution on the device` — either
    way it writes no structure and exits 0 — so this gate refuses the call before it starts, by name (the exit rule's `incomplete` would catch
    it only after the run). None when the yaml turns the kernel off, or the op is installed and its device code serves this GPU or cannot be
    judged (no GPU or no code list readable: no verdict, the run proceeds); else the STACK REFUSED message. `probe` / `code` / `cc` inject the
    two readers and the device's compute capability (default: stack.gpu_probe's, nvidia-smi — no CUDA context)."""
    if yaml_ds4sci_off(yml):
        return None
    from . import stack
    ok, detail = (probe or stack.evoformer_attn_op)()
    DS4SCI_STACK["evoformer_attn_op"] = [ok, detail]                          # the record of the probe (the arms that run the kernel)
    if ok is True:
        cubin, ptx, code_detail = (code or stack.evoformer_attn_op_code)()
        dev_cc = cc if cc is not None else stack.gpu_probe().get("cc")
        serves = stack.device_code_serves(dev_cc, cubin, ptx)
        DS4SCI_STACK["evoformer_attn_op_code"] = [serves, dev_cc, code_detail]   # the device-code verdict beside it (None = no verdict: no GPU / unreadable module)
        if serves is not False:
            return None
        return (f"{STACK_REFUSED}: {what}: the runner yaml {os.path.relpath(yml, home)} leaves use_deepspeed_evo_attention on (upstream's shipped "
                f"configuration) but deepspeed's {stack.DS4SCI_OP} op on this stack carries no device code for this GPU (compute capability {dev_cc}; "
                f"{code_detail}) — the op's kernels would fail with `no kernel image is available for execution on the device` and upstream would write no "
                f"structure; run on an image whose op is built for this GPU (environment/Dockerfile --build-arg DS4SCI_ARCHS with this capability in the list, "
                f"e.g. \"8.0;9.0\" for A100 and H100) or run under the det recipe (--det 1: {modes.STOCK_DET_YAML}, the DS4Sci attention off) or pass a kernels-off runner yaml ({modes.KERNELS_OFF_YAML})")
    return (f"{STACK_REFUSED}: {what}: the runner yaml {os.path.relpath(yml, home)} leaves use_deepspeed_evo_attention on (upstream's shipped "
            f"configuration) but deepspeed's {stack.DS4SCI_OP} op is not installed on this stack ({detail}); run on the stack that carries the "
            f"pre-built op or run under the det recipe (--det 1: {modes.STOCK_DET_YAML}, the DS4Sci attention off) or pass a kernels-off runner yaml ({modes.KERNELS_OFF_YAML})")


def resolve_runner_yaml(arg: Optional[str], home: str) -> Optional[str]:
    """A caller's --runner-yaml as an absolute path: as given when absolute or found from the working directory, else relative to the openfold3/
    tree (the spelling the tree's docs use: `--runner-yaml opt/openfold3_opt/shipped_predict.yml`); a path found in neither place is refused by
    name (ValueError → usage exit 2) — never handed to the runner to fail later. None stays None (the line's / the mode's yaml). A list (the flag
    repeated): every path resolved, in argv order (an empty list is None)."""
    if isinstance(arg, (list, tuple)):
        got = [resolve_runner_yaml(x, home) for x in arg if x]
        return got or None
    if not arg:
        return arg
    for cand in ((arg,) if os.path.isabs(arg) else (os.path.abspath(arg), os.path.join(home, arg))):
        if os.path.isfile(cand):
            return cand
    raise ValueError(f"--runner-yaml {arg}: no such file (looked in the working directory and in the tree {home})")


def stock_argv(a, home: str, ckpt: Optional[str], query_json: str, line: Optional[modes.Line] = None, yml: Optional[str] = None) -> List[str]:
    from . import det as _det
    if yml is None:                                                      # the call's runner yaml (row_yaml: the mode's own, or the caller's composed under the line — a composed yaml lives in a temporary directory: the output directory holds upstream's outputs only)
        mode = resolve_mode(a.mode)
        yml = row_yaml(home, line, call_precision(a, mode), runner_yaml=a.runner_yaml, mode=mode, det=_det.level(getattr(a, 'det', 0)))
    return predict_argv(ckpt, query_json, a.output_dir, yml, use_msa_server=a.use_msa_server, use_templates=a.use_templates,
                        num_model_seeds=a.num_model_seeds, num_diffusion_samples=a.num_diffusion_samples)


def run_stock_subprocess(a, home: str, ckpt: Optional[str], query_json: str, mode_env: str = "off", ckpt_info=None, det: int = 0,
                        upstream_fix: Optional[List[str]] = None) -> int:
    from . import env as _env
    prefixes = _env.must_be_absent(home)
    env = _env.strip(prefixes)
    kit_dirs = _env.kit_dirs(home)
    fd, proof_json = tempfile.mkstemp(prefix="openfold3_opt_stock_proof_", suffix=".json"); os.close(fd)   # the stock caller's proof, handed to this process and removed once read (no file beside the outputs)
    argv = stock_argv(a, home, ckpt, query_json)
    refused = ds4sci_stack_check(home, argv[argv.index("--runner-yaml") + 1], f"pred --mode {mode_env}")
    if refused:
        _err(refused)
        return EXIT_NOT_ACTIVE
    cmd = [sys.executable, "-I", "-m", "openfold3_opt.stock_pred", "--proof-json", proof_json, "--env-absent", ",".join(prefixes),
           "--kit-dirs", os.pathsep.join(kit_dirs), "--home", home, "--det", str(det)]
    if upstream_fix:                                                     # --upstream-fix: the stock caller installs the fixes in ITS process, after its proof and before the stock call
        cmd += [UPSTREAM_FIX_FLAG, ",".join(upstream_fix)]
    cmd += ["--"] + argv
    if ckpt:
        env["OPENFOLD3_CKPT"] = ckpt                                     # a declared read of the stock caller (stock/PINS.json "reads")
    os.makedirs(a.output_dir, exist_ok=True)
    _err(f"stock subprocess: {' '.join(cmd[:4])} ... (env stripped of {','.join(prefixes)}; kit dirs off PYTHONPATH; det={det})")
    rc = subprocess.call(cmd, env=env)
    proof = None
    try:
        if os.path.getsize(proof_json):
            with open(proof_json, encoding="utf-8") as fh:
                proof = json.load(fh)
    finally:
        os.unlink(proof_json)
    rep = {"active": False, "mode": "off", "reason": "stock: the stock caller in a clean subprocess", "package_version": __version__}
    yml = argv[argv.index("--runner-yaml") + 1]
    expected = expected_structures(query_json, yml, *row_counts(a))
    code = gated_manifest(a.output_dir, rep, "pred --mode off", rc, expected, allow_template_drop=getattr(a, "allow_template_drop", False), command="pred", argv=argv, checkpoint=ckpt_info or ckpt, runner_yaml=yml,
                          det=det, stock_proof=proof, extra={"stock_subprocess": cmd, **DS4SCI_STACK},
                          upstream_fix=[r["id"] for r in ((proof or {}).get("upstream_fix") or [])])   # what the stock caller's process applied (its proof's record), not merely what was asked
    if proof and not proof.get("ok"):
        return EXIT_NOT_ACTIVE
    return code


def call_precision(a, mode: str) -> Optional[str]:
    """The fast mode's precision word (modes.FAST_PRECISION: its runner yaml carries it, row_yaml), None under any other mode."""
    return modes.FAST_PRECISION if mode == "fast" else None


def query_tokens(query_json: Optional[str]) -> Optional[int]:
    """The polymer token count of the call's query file (inputs.polymer_tokens: the largest query's protein/RNA/DNA residues × copies), or
    None when the file cannot be read as a query set — the size gate then leaves the line as written and the report says n_tokens=None."""
    from . import inputs as _inputs
    try:
        return _inputs.polymer_tokens(_inputs.load(query_json))[0] if query_json else None
    except (OSError, ValueError):                                       # inputs.InputError is a ValueError; a missing/unparseable file fails upstream's own read next, by name
        return None


def apply_graphs_cap_flag(a) -> None:
    """`--graphs-max-tokens N` is the CLI spelling of OPENFOLD3_OPT_GRAPHS_MAX_TOKENS (modes.graphs_cap reads the variable): set it, refusing a
    disagreeing preset value by name (ValueError → usage)."""
    val = getattr(a, "graphs_max_tokens", None)
    if val is None:
        return
    cur = os.environ.get(modes.ENV_GRAPHS_MAX_TOKENS)
    if cur not in (None, "", str(val)):
        raise ValueError(f"--graphs-max-tokens {val} disagrees with {modes.ENV_GRAPHS_MAX_TOKENS}={cur}")
    os.environ[modes.ENV_GRAPHS_MAX_TOKENS] = str(val)
    modes.graphs_cap()                                                  # refuses a bad value by name


def apply_no_compile_flag(a) -> None:
    """`--no-compile` is the CLI spelling of MODEL_OPT_LEVERS_OFF=compile — merged into the variable so the one ablation door names it
    (modes.LEVERS_OFF_NOOP: this kit has no compile lever and stock compiles nothing, so the word is a named no-op on every mode; the ACTIVE
    line's compile=none stands with or without it). Idempotent; other names in the variable are kept in order."""
    if not getattr(a, "no_compile", False):
        return
    names = modes.levers_off_names()
    if "compile" not in names:
        os.environ[modes.ENV_LEVERS_OFF] = ",".join(names + ("compile",))


def route_options(a, mode: str):
    """The route a `pred`/`check` invocation names and its refusals (the same rules on both commands): (recipe level, the mode's line).
    Every route runs on one GPU (`--n_gpu 1`, the default) except big/tp, which runs on --n_gpu P of them (modes.check_n_gpu_route /
    check_tp_world: P in 2|4|8; P > 1 under another mode or line refused by name with opt_core.mem.ngpu's sentence); a line's parameter given
    under another line of the mode is refused by name (modes.foreign_params) — never a one-card run under a multi-GPU request."""
    from . import det as _det
    lv = _det.level(a.det)
    el = modes.line_arg(mode, n_gpu=getattr(a, "n_gpu", None))
    a.n_gpu = modes.check_n_gpu_route(mode, el, getattr(a, "n_gpu", None))     # P (1|2|4|8): P > 1 only under big/tp — opt_core.mem.ngpu's refusal sentence otherwise, by name
    a.line = el                                                          # the call's line from here on: `--n_gpu P>1` under big is tp, P = 1 resident; exact's one line; None otherwise
    foreign = modes.foreign_params(mode, el, args={"n_gpu": a.n_gpu})                # a line's parameter under a line that does not own it: usage refusal, by name
    if foreign:
        raise ValueError("; ".join(foreign))
    if (mode == "big" and el in modes.TP_LINES):
        modes.check_tp_world(a.n_gpu)
    return lv, el


# ------------------------------------------------------------------------------------------- the offload port's item gate, per item ----
# `pred --mode big` (the one-GPU line): the port engages PER ITEM by the item's own polymer token count (modes.of3o_item_groups). The port's units
# are process-wide rebinds installed when the model module is imported (of3_offload.apply_core under OF3O_LAYER, the LN-SAFE LayerNorm binding, the
# pinned host tables) and its line runs on its own runner yaml — one model configuration per `run_openfold predict` call —, while the below-gate line
# is another hook chain on the fast line's yaml: the two cannot share a process, so a query set whose items fall on both sides of the gate runs as one
# `pred` pass of this CLI per side, sequentially, in the same output directory (run_item_groups); a set on one side runs as one pass exactly as written.
ITEM_GROUP_ORDER = ("below", "engaged")                                        # the passes' order: below-gate items first, the port's last — upstream's run-level dumps at the top of the output
#                                                                                directory (experiment_config.json, model_config.json) then hold the port's configuration, as a one-pass run reaching the gate leaves them
ITEM_GROUP_LINE = {"below": modes.OF3O_ASIDE_WORD, "engaged": "resident"}     # the line word of each side on the pass lines
RC_SEVERITY = (EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE, EXIT_TEMPLATES_DROPPED)  # the named exit codes, worst first; an unnamed non-zero code (a signal, a timeout) ranks before all of them
QUERY_JSON_FLAGS = ("--query-json", "--query_json")


def worst_rc(rcs) -> int:
    """0 when every pass returned 0; else the most severe non-zero code: an unnamed code first, then runner failure (1) > usage (2) > not active (3) >
    templates dropped (5)."""
    nz = [int(r) for r in rcs if r]
    if not nz:
        return EXIT_OK
    return min(nz, key=lambda r: RC_SEVERITY.index(r) if r in RC_SEVERITY else -1)


def swap_query_json(argv: List[str], path: str) -> Optional[List[str]]:
    """`argv` with the value of the query-json flag replaced by `path` — both spellings, the `--flag=value` form and argparse's unique-prefix
    abbreviations (`--query-j …`); None unless exactly one such flag is present."""
    out, hits, i = [], 0, 0
    while i < len(argv):
        tok = str(argv[i])
        opt, eq, _ = tok.partition("=")
        if opt.startswith("--") and len(opt) > len("--query") and any(f.startswith(opt) for f in QUERY_JSON_FLAGS):
            hits += 1
            if eq:
                out.append(f"{opt}={path}")
                i += 1
            else:
                out += [opt, path]
                i += 2
            continue
        out.append(tok)
        i += 1
    return out if hits == 1 else None


def item_gate_groups(a, mode: str, el: Optional[str]) -> Optional[dict]:
    """The port's item gate decided per item of this call's query set (modes.of3o_item_groups on inputs.polymer_tokens_each) when that SPLITS the call:
    `pred --mode big` on the one-GPU line, a readable set of two or more items with members on both sides of the gate, issued through main() (the
    passes re-issue this call's own argv). None otherwise — the call then runs as one pass exactly as written: a set on one side of the gate (the knob
    `OF3O_MIN_TOKENS=none` puts every item on the port's side), exact / fast / off, the row-sharded line (its launcher's), an unreadable query or a
    malformed knob (refused by name on the one-pass route, as before)."""
    argv = getattr(a, "_argv", None)
    if mode != "big" or el in modes.TP_LINES or argv is None or swap_query_json(argv, a.query_json) is None:
        return None
    from . import inputs as _inputs
    try:
        qs = _inputs.load(a.query_json)
        each = _inputs.polymer_tokens_each(qs)
        groups = modes.of3o_item_groups(modes.LINES[(mode, el)], {k: v[0] for k, v in each.items()}, os.environ) if len(each) >= 2 else None
    except (OSError, ValueError, AttributeError, KeyError):
        return None
    if groups is None or not groups["below"] or not groups["engaged"]:
        return None
    return {**groups, "query_set": qs, "line": el}


RUN_SUMMARY = "summary.txt"                                                   # upstream's run-level rollups at the top of the output directory, one per `run_openfold predict` call:
RUN_QUERY_SET = "inference_query_set.json"                                    #  core/runners/writer.py _write_summary (the counts) and core/utils/callbacks.py LogInferenceQuerySet (the
#                                                                               query set's dump at predict start) — restored to the WHOLE call's content after the passes (merge_run_rollups)
RUN_SUMMARY_RE = re.compile(r"PREDICTION SUMMARY \(COMPLETE\).*?Total Queries Processed: (\d+)\s+- Successful Queries:\s+(\d+)\s+- Failed Queries:\s+(\d+)"
                            r"(?:\s*\n\s*Failed Queries: ([^\n]*))?", re.S)


def read_run_rollups(out_dir: str) -> dict:
    """This pass's copies of upstream's two run-level rollups, parsed: {"summary": (total, ok, failed, [failed names]) | None, "query_set": dict | None}
    (None where absent or unreadable)."""
    rec = {"summary": None, "query_set": None}
    try:
        with open(os.path.join(out_dir, RUN_SUMMARY), encoding="utf-8") as fh:
            m = RUN_SUMMARY_RE.search(fh.read())
        if m:
            rec["summary"] = (int(m[1]), int(m[2]), int(m[3]), [s.strip() for s in (m[4] or "").split(",") if s.strip()])
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(out_dir, RUN_QUERY_SET), encoding="utf-8") as fh:
            qs = json.load(fh)
        if isinstance(qs, dict) and isinstance(qs.get("queries"), dict):
            rec["query_set"] = qs
    except (OSError, ValueError):
        pass
    return rec


def merge_run_rollups(out_dir: str, records: List[dict], order: List[str]) -> List[str]:
    """After the passes, upstream's two run-level rollups describe the WHOLE call again, as one pass over the set writes them: `summary.txt` = the passes'
    counts summed (and their failed queries joined) in upstream's own format; `inference_query_set.json` = the last pass's dump with `queries` the union of
    the passes' dumps in the query file's order. Each only when every pass left a readable copy — else the last pass's stays as written. Returns the names
    restored. (`experiment_config.json` / `model_config.json` describe one model configuration per call: the last pass's — the port's — remain.)"""
    done: List[str] = []
    sums = [r.get("summary") for r in records]
    if sums and all(s is not None for s in sums):
        total, ok, failed = (sum(s[i] for s in sums) for i in range(3))
        names = sorted(set(n for s in sums for n in s[3]))
        text = ["\n" + "=" * 50, "    PREDICTION SUMMARY (COMPLETE)    ", "=" * 50, f"Total Queries Processed: {total}", f"  - Successful Queries:  {ok}",
                f"  - Failed Queries:      {failed}"]
        if names:
            text.append(f"\nFailed Queries: {', '.join(names)}")
        text.append("=" * 50 + "\n")
        with open(os.path.join(out_dir, RUN_SUMMARY), "w", encoding="utf-8") as fh:
            fh.write("\n".join(text))
        done.append(RUN_SUMMARY)
    dumps = [r.get("query_set") for r in records]
    if dumps and all(d is not None for d in dumps):
        union = {}
        for d in dumps:
            union.update(d["queries"])
        rank = {str(n): i for i, n in enumerate(order)}
        merged = {**dumps[-1], "queries": dict(sorted(union.items(), key=lambda kv: rank.get(str(kv[0]), len(rank))))}
        with open(os.path.join(out_dir, RUN_QUERY_SET), "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=4, ensure_ascii=False)
        done.append(RUN_QUERY_SET)
    return done


def pass_expected(query_json: str, a) -> Optional[int]:
    """The structures a pass over `query_json` is expected to write (expected_structures with this call's --num-model-seeds /
    --num-diffusion-samples and caller yaml): one --runner-yaml or none as `pred` counts them; the flag repeated (a list): the largest count any
    of them yields (a seed list in one of them), None when none can be read."""
    ry = getattr(a, "runner_yaml", None)
    cands = [y for y in ry if y] if isinstance(ry, (list, tuple)) else [ry]
    vals = [v for v in (expected_structures(query_json, y, *row_counts(a)) for y in (cands or [None])) if v is not None]
    return max(vals) if vals else None


def group_structures(out_dir: str, names) -> int:
    """The counted structures (manifest.count_structures) under the per-query directories `<out_dir>/<name>/` of `names`."""
    from . import manifest as _manifest
    return sum((_manifest.count_structures(os.path.join(out_dir, str(n))) or 0) for n in names)


def run_item_groups(a, groups: dict) -> int:
    """`pred --mode big` on a query set that straddles the offload port's item gate: one `pred` pass of this CLI per side of the gate, sequentially,
    in the same output directory with the same arguments — the below-gate items (a temporary subset json) on the line a below-gate set resolves to
    (modes.of3o_aside), then the items at or above the gate on the port. Each pass is a complete `pred` of its subset: its WEIGHTS / TEMPLATES /
    ACTIVE / LEVER / exit-rule lines are those of a run of that subset alone, and its outputs land in upstream's per-query directories exactly as one
    pass writes them (upstream's run-level dumps at the top of the output directory — experiment_config.json, model_config.json, query_msa.json on the
    MSA-server route — are per pass: `summary.txt` and `inference_query_set.json` are restored to the whole call's content afterwards
    (merge_run_rollups), the model-configuration dumps stay the last pass's, the port's). This process prints the ITEM GATE summary line, one line per pass (items,
    line; then rc, wall seconds, structures made/expected) and a closing line, and returns the worst pass rc (worst_rc); a pass short of its expected
    structures counts as EXIT_FAIL here even when its own exit rule — which counts the whole output directory — said ok. Cost over a one-pass run: one
    more process start (imports, weights). The subset jsons live in a temporary directory, removed afterwards."""
    import shutil
    import time
    from . import inputs as _inputs
    passes = [(side, tuple(groups[side])) for side in ITEM_GROUP_ORDER if groups.get(side)]
    _err(modes.item_gate_line(groups, engaged_line=groups.get("line") or "resident"))
    tmp = tempfile.mkdtemp(prefix=f"{__package__}_items_")
    stem = os.path.splitext(os.path.basename(str(a.query_json)))[0] or "queries"
    environ = getattr(a, "_environ", None)
    rcs: List[int] = []
    rollups: List[dict] = []
    try:
        for k, (side, names) in enumerate(passes, 1):
            qj = os.path.join(tmp, f"{stem}.{side}.json")
            _inputs.dump(_inputs.subset(groups["query_set"], names), qj)
            cmd = [sys.executable, "-m", __package__] + swap_query_json(a._argv, qj)
            expected = pass_expected(qj, a)
            _err(f"{modes.ITEM_GATE_NAME} pass {k}/{len(passes)} {side}: items={','.join(names)} line={ITEM_GROUP_LINE[side]} query_json={qj}")
            t0 = time.time()
            rc = int(subprocess.call(cmd, env=None if environ is None else dict(environ)) or 0)
            rc = rc if rc >= 0 else 128 - rc                                 # a pass killed by a signal: the shell's 128+N, as a one-pass run killed the same way exits
            made = group_structures(a.output_dir, names)
            rollups.append(read_run_rollups(a.output_dir))
            eff = EXIT_FAIL if (rc == 0 and expected is not None and made < expected) else rc
            rcs.append(eff)
            _err(f"{modes.ITEM_GATE_NAME} pass {k}/{len(passes)} {side}: rc={rc}{f' -> {eff} (incomplete)' if eff != rc else ''} wall_s={time.time() - t0:.1f} "
                 f"structures={made}/{expected if expected is not None else '?'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    restored = merge_run_rollups(a.output_dir, rollups, list((groups["query_set"].get("queries") or {}).keys())) if len(rollups) == len(passes) else []
    rc = worst_rc(rcs)
    _err(f"{modes.ITEM_GATE_NAME}: passes={len(rcs)} rc={rc} ({' '.join(f'{side}={r}' for (side, _), r in zip(passes, rcs))}) "
         f"rollups={','.join(restored) if restored else 'last-pass'}")
    return rc


def tp_rank() -> Optional[int]:
    """This process's rank when it is one of the big/tp line's rank processes (tp.py spawned it with OF3TP_RANK), else None."""
    r = (os.environ.get("OF3TP_RANK") or "").strip()
    return int(r) if r.isdigit() else None


def run_predict_inprocess(argv: List[str]) -> int:
    """The stock console script's entry point with `argv` (the levers are already active in this process)."""
    import importlib
    entry = getattr(importlib.import_module("openfold3.run_openfold"), "cli")
    try:
        entry.main(args=["predict"] + list(argv), prog_name="run_openfold", standalone_mode=True)
    except SystemExit as e:
        return exit_code(e)
    return 0


def cmd_pred(a) -> int:
    import openfold3_opt
    from . import det as _det
    home = _home()
    mode = resolve_mode(a.mode)
    ckpt = _ckpt(a.ckpt)
    if not ckpt:
        return _usage("no checkpoint: pass --ckpt or set OPENFOLD3_CKPT (configs/h100.env; stock/PINS.json \"weights\")")
    try:
        a.runner_yaml = resolve_runner_yaml(a.runner_yaml, home)          # a relative --runner-yaml: the working directory, then the tree; missing = refused by name
        apply_no_compile_flag(a)                                            # --no-compile == MODEL_OPT_LEVERS_OFF=compile (a named no-op here: nothing compiles)
        apply_graphs_cap_flag(a)
        lv, el = route_options(a, mode)
        fix_ids = resolve_upstream_fix(a, home, "pred")                     # --upstream-fix: an unknown ID / unmet precondition refused by name before any launch; applied by the process that runs the model
    except ValueError as e:
        return _usage(str(e))
    rank = tp_rank() if (mode == "big" and el in modes.TP_LINES) else None
    groups = item_gate_groups(a, mode, el) if rank is None else None              # big on one GPU: the offload port engages PER ITEM — a query set straddling its item gate runs as one `pred` pass per side (run_item_groups)
    if groups is not None:
        return run_item_groups(a, groups)
    info, rc = weights_check(ckpt, home, what=f"pred --mode {mode} --det {lv}", refresh=WEIGHTS_REFRESH["pred"])
    if rc:
        return rc
    from . import templ_census
    templ_census.record_file(a.query_json, enabled=effective_use_templates(a) == "true")   # the TEMPLATES DECLARED line (or TEMPLATES DISABLED under --use-templates false: template paths are then no workload and nothing is judged): what the query declares about templates, once per call, every route; a rank takes the census silently for its own guard (templ_census.record: report.process_role)
    if mode == "off":
        return run_stock_subprocess(a, home, ckpt, a.query_json, ckpt_info=info, det=lv, upstream_fix=fix_ids)
    if (mode == "big" and el in modes.TP_LINES) and rank is None:                   # the launcher: the ranks are `pred` processes of this CLI (tp.py)
        from . import tp as _tp
        return _tp.launch(a, home, ckpt, info, lv, modes.check_tp_world(a.n_gpu), (EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE))
    ln = modes.effective_line(mode, el, os.environ, query_tokens(a.query_json))         # the line as this call's token count runs it (modes.of3o_gate: the port's units aside below OF3O_MIN_TOKENS -> the fast line's runner yaml)
    what = f"pred --mode {mode}" + (f" (tp rank {rank})" if rank is not None else "")
    yml = row_yaml(home, ln, call_precision(a, mode), runner_yaml=a.runner_yaml, mode=mode, det=lv, as_given=rank is not None)   # a caller's yaml composed under the line, named (a rank: the launcher's composition, as given)
    refused = ds4sci_stack_check(home, yml, what)
    if refused:
        _err(refused)
        return EXIT_NOT_ACTIVE
    n_tokens = query_tokens(a.query_json)                               # the query's polymer token count: the CUDA-graph size gate (modes.graphs_gate); None = unreadable here (upstream then fails on the file itself, loud)
    if lv:
        _det.apply_env(lv, graphed=modes.graphed_call(mode, el, n_tokens))    # before the hook file runs: the kit's own OF3_DETERMINISTIC block; OF3_GRAPHS_STRICT on graphed calls only
    try:
        rep = openfold3_opt.enable(mode, strict=True, n_tokens=n_tokens, n_gpu=a.n_gpu)
    except openfold3_opt.ActivationError:
        return EXIT_NOT_ACTIVE
    argv = stock_argv(a, home, ckpt, a.query_json, ln, yml=yml)
    _err(f"pred{'' if rank is None else f' (tp rank {rank})'}: run_openfold predict {' '.join(argv)}")
    _err(protocol_line(argv[argv.index("--runner-yaml") + 1], argv))
    if not yaml_ds4sci_off(yml):                                               # the yaml turns the DS4Sci evoformer attention on (exact at det 0): the op must LOAD now — never a per-item load error or an eager path in a timed arm
        from .stack import DS4SCI_NOT_LOADED, ds4sci_load
        ok, detail = ds4sci_load()
        if not ok:
            _err(f"NOT ACTIVE: {DS4SCI_NOT_LOADED}: {detail} (runner yaml {os.path.relpath(yml, home)}; run on the stack with the pre-built evoformer_attn op or under --det 1 where that kernel is off)")
            return EXIT_NOT_ACTIVE
        _err(f"DS4SCI evoformer_attn op loaded: {detail}")
    if rank is not None and el in modes.TP_LINES:
        _err(f"[openfold3-opt tp rank {rank}] {modes.replicated_sync_policy(el, lv)}")     # the replicated-tensor sync policy by det level (bcast at det 0, guard at det 1), printed per rank
    if apply_upstream_fix(fix_ids, home, what) is None:                        # --upstream-fix: installed in this process after activation, before the model runs (UPSTREAM-FIX <ID> applied)
        return EXIT_NOT_ACTIVE
    rc = run_predict_inprocess(argv)
    if rank is not None:
        from . import tp as _tp
        _err(f"[openfold3-opt tp rank {rank}] {_tp.allocator_peak_line()}")    # the process's own allocator high-water, beside the launcher's nvidia-smi sampler
    yml = argv[argv.index("--runner-yaml") + 1]
    expected = expected_structures(a.query_json, yml, *row_counts(a)) if not rank else None   # ranks > 0 write no prediction (the add-on: rank 0 writes)
    extra = dict(DS4SCI_STACK)
    if a.runner_yaml and composed_key(a.runner_yaml) in COMPOSED:        # the caller's yaml(s) composed under the line: what was pinned / kept (runner_yaml.compose)
        extra["runner_yaml_composed"] = COMPOSED[composed_key(a.runner_yaml)]
    if rank is not None and el == "tp":                                  # the rank's schedule evidence: the core's recorded schedule words + the adapter's census + the sample_loop fields
        from .tp_rowpair import schedule_record
        extra["tp_schedule"] = schedule_record()
    return gated_manifest(a.output_dir, rep, f"pred --mode {mode}" + (f" (tp rank {rank})" if rank is not None else ""), rc, expected, allow_template_drop=getattr(a, "allow_template_drop", False), command="pred",
                          argv=argv, checkpoint=info, runner_yaml=yml, det=lv, extra=extra or None,
                          upstream_fix=fix_ids)


# ----------------------------------------------------------------------------------------------------------------- check ----
def cmd_check(a) -> int:
    from . import stack
    mode = resolve_mode(a.mode)
    try:
        apply_no_compile_flag(a)                                        # --no-compile == MODEL_OPT_LEVERS_OFF=compile (named on the DRY-RUN line's notes; compile=none either way)
        apply_graphs_cap_flag(a)
        route_options(a, mode)                                          # the pred's own route refusals (a check of the arm names the pred's options)
    except ValueError as e:
        return _usage(str(e))
    environ = None
    if modes.line_arg(mode, n_gpu=a.n_gpu) in modes.TP_LINES:                              # the dry run resolves the line as a rank of the requested world would
        environ = {**os.environ, "OF3TP_RANK": "0", "OF3TP_WORLD": str(modes.check_tp_world(a.n_gpu))}
    rep = stack.activate(mode, dry_run=True, environ=environ, n_gpu=a.n_gpu)
    from . import det as _det
    home = _home()
    try:
        a.runner_yaml = resolve_runner_yaml(a.runner_yaml, home)
    except ValueError as e:
        return _usage(str(e))
    ln = None if mode == "off" else modes.LINES.get((mode, modes.line_arg(mode, environ, n_gpu=a.n_gpu)))
    yml = row_yaml(home, ln, call_precision(a, mode), runner_yaml=a.runner_yaml, mode=mode, det=_det.level(a.det))   # a caller's yaml: composed under the line, the composition named (as `pred`)
    refused = ds4sci_stack_check(home, yml, f"check --mode {mode}")
    if refused:
        rep = dict(rep, stack_refused=refused)
        _err(refused)
    ckpt = _ckpt(a.ckpt)
    if ckpt:                                                            # a named checkpoint (--ckpt / OPENFOLD3_CKPT): hashed afresh, its digest memo entry rewritten, the WEIGHTS line printed; identity by digest, never a condition
        info, wrc = weights_check(ckpt, home, what=f"check --mode {mode}", refresh=WEIGHTS_REFRESH["check"])
        if wrc != EXIT_OK:
            return wrc
        rep = dict(rep, checkpoint=info)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, default=str)
            fh.write("\n")
    if refused:
        return EXIT_NOT_ACTIVE
    if mode == "off":
        return EXIT_OK
    from .report import DRY_RUN_OK
    return EXIT_OK if rep.get("reason") == DRY_RUN_OK else EXIT_NOT_ACTIVE


# ------------------------------------------------------------------------------------------------------------------ warm ----
def warm_argv(home: str, ckpt: str, query: str, out_dir: str, line: Optional[modes.Line] = None, precision: Optional[str] = None, mode: Optional[str] = None) -> List[str]:
    """warm's one prediction: WARM_KNOBS — MSA server off, templates off (the public query carries neither), one model seed, one diffusion sample —,
    built like every pred's (the line's yaml where it names one; the fast line's bf16 yaml — `precision` None reads as pred's does, modes.FAST_PRECISION,
    when `line` is the fast line or unnamed)."""
    if precision is None and (line is None or line is modes.LINES[("fast", None)]):
        precision = modes.FAST_PRECISION
    if mode is None and line is not None:
        mode = next((mk for (mk, lk), ln in modes.LINES.items() if ln is line), None)      # the line's mode: exact runs the stock configuration, fast / big their kernels-off base (row_yaml)
    return predict_argv(ckpt, query, out_dir, row_yaml(home, line, precision, out_dir=tempfile.mkdtemp(prefix="openfold3_opt_yaml_"), mode=mode), **WARM_KNOBS)


def cmd_warm(a) -> int:
    import openfold3_opt
    from . import warm as _warm
    home = _home()
    mode = resolve_mode(a.mode)
    if mode == "off":
        return _usage("warm has no stock form (the stock arm has nothing to warm)")
    apply_no_compile_flag(a)                                            # --no-compile == MODEL_OPT_LEVERS_OFF=compile (a named no-op here)
    ckpt = _ckpt(a.ckpt)
    if not ckpt:
        return _usage("no checkpoint: pass --ckpt or set OPENFOLD3_CKPT")
    try:
        fix_ids = resolve_upstream_fix(a, home, "warm")                     # --upstream-fix: refused by name before anything starts (warm's public query runs --use-templates false)
    except ValueError as e:
        return _usage(str(e))
    info, rc = weights_check(ckpt, home, what=f"warm --mode {mode}", refresh=WEIGHTS_REFRESH["warm"])
    if rc:
        return rc
    out = os.path.abspath(a.out)
    ln = modes.LINES[(mode, modes.line_arg(mode))]
    refused = ds4sci_stack_check(home, row_yaml(home, ln, call_precision(a, mode)), f"warm --mode {mode}")
    if refused:
        _err(refused)
        return EXIT_NOT_ACTIVE
    try:
        tilings = _warm.tilings_of(a.tiling)                             # one tiling or a comma-separated list: refused by name before anything is built
    except ValueError as e:
        return _usage(str(e))
    q = _warm.public_query(home, os.path.join(out, "query"), ",".join(tilings))
    _err(f"WARM input=1BRS tilings={','.join(tilings)} n_tokens={','.join(str(_warm.tiling_tokens(t)) for t in tilings)} queries={len(tilings)} (one process folds them in turn)")
    try:
        rep = openfold3_opt.enable(mode, strict=True)
    except openfold3_opt.ActivationError:
        return EXIT_NOT_ACTIVE
    pred_out = os.path.join(out, "pred")
    argv = warm_argv(home, ckpt, q, pred_out, ln, call_precision(a, mode))
    if apply_upstream_fix(fix_ids, home, f"warm --mode {mode}") is None:
        return EXIT_NOT_ACTIVE
    rc = run_predict_inprocess(argv)
    yml = argv[argv.index("--runner-yaml") + 1]
    expected = expected_structures(q, yml, 1, 1)                       # warm_argv: one model seed, one diffusion sample
    return gated_manifest(pred_out, rep, f"warm --mode {mode}", rc, expected, command="warm", argv=argv, checkpoint=info, runner_yaml=yml,
                          det=0, upstream_fix=fix_ids)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    a = ap.parse_args(argv)
    a._argv, a._environ = list(argv), dict(os.environ)                   # the call as issued: `pred --mode big` re-issues it per side of the port's item gate when the query set straddles it (run_item_groups)
    try:
        return {"pred": cmd_pred, "check": cmd_check, "warm": cmd_warm}[a.cmd](a)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT_USAGE
    except ValueError as e:
        return _usage(str(e))

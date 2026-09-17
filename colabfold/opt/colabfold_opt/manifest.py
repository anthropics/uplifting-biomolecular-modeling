"""The launch record: the activation report, the kit's exit counters, the verdict — the channel between `pred` and the model process it
launches. The model process writes `opt_manifest.json` into the launch's WORK directory (`COLABFOLD_OPT_WORK_DIR`, exported by `pred`: a
temporary directory removed when `pred` returns; absent — the env route without `pred` — the process writes none, its printed lines are
its record); `pred` reads it back for the verdict and completes it in memory (command, argv, exit code, settings, the stock proof, the
output listing; cli.run returns it). Nothing of the package is written under the results directory. The exit-rule record: `partial` — the levers of a kit mode's
activation with no evidence of application at exit (`partial_levers`: the report active, the lever's `calls` < 1), written by the model
process at exit (`record_exit`, the env route's record) and read by `pred`'s verdict (`kernel_not_engaged`, rc 3) — a lever that stepped
aside by a named size rule (`stepped_aside`) or whose whole call class a lever bound over it served (`superseded`) engaged by design and is
not partial; `lever_fallbacks` —
a table-backed lever that cannot run on this GPU, by name (`lever_fallback`, rc 3: a mode is all of its levers); `fallback_census` /
`fallback_excess` — the kernel's per-call fallbacks at exit against the documented class
(registry.FALLBACK_CLASSES, FALLBACK_MAX_SHARE; beyond it is `fallback_excess`: recorded and named on one line, the run's own exit code);
`verdict.missing` — the jobs without a completion artefact (`missing_outputs`, rc 1: an output shortfall, never `partial`); the completion
artefact is colabfold's own finished-job test (`completion`: `<job>.result.zip` under `--zip`, else `<job>.done.txt`, batch.py:1405-1412);
`no_model_run` — the run built no model, by the tool's own facts and by name (`num_models_0`: `--num-models 0` / `--msa-only`;
`all_jobs_done`: every job complete before the run, colabfold skips them; `af3_json`: `--af3-json` returns before `run()`): no completion
marker is owed, the applied levers had no model call to serve (`idle`, never `partial`), the run exits as stock does — the IDLE line names it.
"""
from __future__ import annotations

import json
import os
import platform
from typing import Optional

from opt_core import manifest as _core_manifest
from opt_core.gates import sha256_file                      # the shared file digest (the core's one implementation)

from . import modes as _modes, registry as _registry, report as _report

ENV_LAUNCH_ID = _modes.ENV + "_LAUNCH_ID"           # exported by `pred` into the model process (stock_pred.fast_env): every manifest that process
                                                    # writes carries it (build), and the verdict requires the value of its own launch
ENV_WORK_DIR = _modes.ENV + "_WORK_DIR"             # exported by `pred` into the model process (stock_pred.fast_env): the launch's work directory, where
                                                    # that process writes its manifest (work_dir(); absent: no manifest is written)

FILENAME = "opt_manifest.json"


def path(out_dir: str) -> str:
    return os.path.join(out_dir, FILENAME)


def work_dir(environ: Optional[dict] = None) -> Optional[str]:
    """The launch's work directory named in the environment (ENV_WORK_DIR), else None: where this process writes its manifest."""
    v = (os.environ if environ is None else environ).get(ENV_WORK_DIR)
    return v or None


def read(out_dir_or_path: str) -> dict:
    p = out_dir_or_path if out_dir_or_path.endswith(".json") else path(out_dir_or_path)
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _dump(p: str, man: dict) -> str:
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, p)
    return p


def build(report: Optional[dict], **extra) -> dict:
    rep = report or {}
    man = {"model": "colabfold", "package": "colabfold_opt", "package_version": rep.get("package_version"), "mode": rep.get("mode"),
           "active": bool(rep.get("active")), "reason": rep.get("reason"), "levers_applied": rep.get("levers_applied"),
           "levers_unavailable": rep.get("levers_unavailable"), "partial": list(rep.get("partial") or []),
           "tokens_min": rep.get("tokens_min"), "tokens_max": rep.get("tokens_max"),
           "queries": rep.get("queries"), "route": rep.get("route"), "key": rep.get("key"), "stack_key": rep.get("stack_key"),
           "gpu": rep.get("gpu"), "colabfold_version": rep.get("colabfold_version"), "alphafold_colabfold_version": rep.get("alphafold_colabfold_version"),
           "jax_version": rep.get("jax_version"), "python": platform.python_version(), "kit_env": rep.get("kit_env"),
           "kit_state_at_activation": rep.get("kit_state"), "activation_report": rep, "launch_id": os.environ.get(ENV_LAUNCH_ID),
           "no_model_run": rep.get("no_model_run"), "core": _core_manifest.core_block()}
    if rep.get("lever_fallbacks"):                                        # a table-backed lever that cannot engage on this GPU, by name (absent on a GPU whose tables serve every lever)
        man["lever_fallbacks"] = dict(rep["lever_fallbacks"])
    man.update(extra)
    return man


def write(result_dir: str, report: Optional[dict], **extra) -> str:
    """Write (or rewrite, keeping fields already recorded) the manifest under result_dir."""
    p = path(result_dir)
    old = {}
    if os.path.isfile(p):
        try:
            old = read(p)
        except (OSError, ValueError):
            old = {}
    man = build(report, **extra)
    for k in ("kit_state_exit", "lever_states_exit", "command", "argv", "exit_code", "settings", "stock_proof", "outputs", "inputs", "verdict", "idle"):
        if k in old and k not in man:
            man[k] = old[k]
    if man.get("launch_id") is None and old.get("launch_id"):
        man["launch_id"] = old["launch_id"]
    return _dump(p, man)


def lever_state_exit(man: dict, lever: str) -> Optional[dict]:
    """`lever`'s exit counters in the manifest: `lever_states_exit[lever]`, the carried kit's `kit_state_exit` standing for AF_PALLAS_ATTN
    when the per-lever record is absent."""
    states = man.get("lever_states_exit") or {}
    st = states.get(lever)
    if st is None and lever == _modes.LEVER:
        st = man.get("kit_state_exit")
    return st if isinstance(st, dict) else None


def partial_levers(man: dict) -> list:
    """The levers of the manifest's activation report with no evidence of application at exit — the report active and the lever's own
    exit counters without `calls >= 1` (AF_PALLAS_ATTN: the kernel ran no attention call; DEVICE_RESIDENT: no model call went through the
    resident apply; no exit state counts the same) — else []; a lever whose every call stepped aside by a named size rule
    (`stepped_aside_rule`: calls=0, fallbacks>=1 all attributed to registry.STEP_ASIDE_RULES), or whose every call a lever bound over it served
    (`superseded_by`: calls=0, the superseding lever's calls>=1; its own fallbacks, if any, are judged by the share rule), engaged by design and is not partial.
    A run on which colabfold_batch built no model at all (`no_model_case`: `--num-models 0` / `--msa-only`, every job already complete) gave no
    lever a call to serve: its unserved levers are `idle_levers`, by that name, and none is partial. The PARTIAL record, one reading for the model
    process (`record_exit`) and the parent's `verdict`."""
    if no_model_case(man) is not None:
        return []
    return _unserved(man)


def idle_levers(man: dict) -> list:
    """The applied levers with no call at exit on a run that built no model (`no_model_case`) — the levers `partial_levers` would name had a
    model run: idle by the tool's own facts, recorded (`idle`) and named on the IDLE line (`levers=not_applicable:<names>`) and on their LEVER
    lines (`no_model_run=<case>`); [] when a model was expected (those levers are `partial`) or every applied lever served a call."""
    if no_model_case(man) is None:
        return []
    return _unserved(man)


def _unserved(man: dict) -> list:
    """The applied levers of an ACTIVE report whose exit counters show no served call and no by-design reading (stepped aside / superseded)."""
    rep = man.get("activation_report") or {}
    if not rep.get("active"):
        return []
    out = []
    for name in list(rep.get("levers_applied") or rep.get("levers") or []):
        st = lever_state_exit(man, name)
        calls = st.get("calls") if isinstance(st, dict) else None
        if not (isinstance(calls, int) and calls >= 1) and stepped_aside_rule(st) is None and superseded_by(man, name) is None:
            out.append(name)
    return out


def superseded_by(man: dict, name: str) -> Optional[str]:
    """`<lever>:<calls>[,…]` when the calls of `name`'s served class were taken by an applied lever bound over it (modes.SUPERSEDES: TRIATTN_XLA
    over AF_PALLAS_ATTN's pair-biased sites) — `name`'s exit counters read `calls=0` (no call of its class reached its kernel: one the
    superseding lever does not serve proceeds to `name` and is counted there) while the superseding lever's read `calls>=1`. Its documented
    per-call fallbacks (registry.FALLBACK_CLASSES: the bias-free calls it hands to stock) may still be counted — they are judged by the share
    rule over the class including the superseded calls (`fallback_census`), as before the lever over it existed. The single-chain route without
    templates from TRIATTN_XLA's size floor up (no template pair stack in the model, every pair-biased head 32 channels) is that run: the flash
    kernel engaged by name and had no call left to take — by design, not a partial activation (`partial_levers` skips it; its LEVER line says
    `superseded_by=…`). None otherwise (a call of its class reached `name`, no superseding lever applied, the superseding lever served nothing)."""
    st = lever_state_exit(man, name)
    if not isinstance(st, dict):
        return None
    calls = st.get("calls")
    if not (isinstance(calls, int) and calls == 0):
        return None
    rep = man.get("activation_report") or {}
    applied = list(rep.get("levers_applied") or rep.get("levers") or [])
    out = []
    for over, under in _modes.SUPERSEDES.items():
        if name in under and over in applied:
            so = lever_state_exit(man, over)
            n = so.get("calls") if isinstance(so, dict) else None
            if isinstance(n, int) and n >= 1:
                out.append(f"{over}:{n}")
    return ",".join(out) if out else None


def superseded_calls(man: dict, name: str) -> int:
    """The calls of `name`'s class an applied lever bound over it served (modes.SUPERSEDES; 0 when none): they count in `name`'s class when
    its fallback share is judged (`fallback_census`) — the share the class showed before the lever over it took those calls."""
    rep = man.get("activation_report") or {}
    applied = list(rep.get("levers_applied") or rep.get("levers") or [])
    total = 0
    for over, under in _modes.SUPERSEDES.items():
        if name in under and over in applied:
            so = lever_state_exit(man, over)
            n = so.get("calls") if isinstance(so, dict) else None
            if isinstance(n, int) and n >= 1:
                total += n
    return total


def superseded(man: dict) -> dict:
    """lever -> `<superseding lever>:<calls>` (`superseded_by`) for the applied levers of an ACTIVE report; {} when none was. Recorded in the
    manifest at exit and in `pred`'s verdict (`superseded`); the superseded lever's LEVER line carries the word."""
    rep = man.get("activation_report") or {}
    if not rep.get("active"):
        return {}
    out = {}
    for name in list(rep.get("levers_applied") or rep.get("levers") or []):
        word = superseded_by(man, name)
        if word is not None:
            out[name] = word
    return out


def stepped_aside_rule(st: Optional[dict]) -> Optional[str]:
    """The size rule(s) by which EVERY call of a lever stepped aside — its exit state reads `calls=0`, `fallbacks>=1` and a `fallback_by`
    census (`<rule>:<count>,…`) attributing all of them to registry.STEP_ASIDE_RULES (e.g.
    `calls=0 fallbacks=16 fallback_by=below_keys_rule:16`) — as `<rule>[,<rule>]`; None otherwise (a call served, no fallback counted, a fallback
    without a named rule, a rule outside the table, counts that do not add up). Such a lever engaged and ran the stock operation BY NAME at that
    size, which its LEVER line already says: by design, not a partial activation (`partial_levers` skips it; `calls=0 fallbacks=0` stays partial)."""
    if not isinstance(st, dict):
        return None
    calls, fallbacks, by = st.get("calls"), st.get("fallbacks"), st.get("fallback_by")
    if not (isinstance(calls, int) and calls == 0 and isinstance(fallbacks, int) and fallbacks >= 1 and isinstance(by, str) and by):
        return None
    rules, total = [], 0
    for tok in by.split(","):
        rule, _sep, count = tok.partition(":")
        if rule not in _registry.STEP_ASIDE_RULES:
            return None
        try:
            total += int(count)
        except ValueError:
            return None
        rules.append(rule)
    return ",".join(rules) if rules and total == fallbacks else None


def stepped_aside(man: dict) -> dict:
    """lever -> the rule(s) it stepped aside by (`stepped_aside_rule`) for the applied levers of an ACTIVE report; {} when none did. Recorded in
    the manifest at exit and in `pred`'s verdict (`stepped_aside`), printed nowhere new: the LEVER line carries the census."""
    rep = man.get("activation_report") or {}
    if not rep.get("active"):
        return {}
    out = {}
    for name in list(rep.get("levers_applied") or rep.get("levers") or []):
        rule = stepped_aside_rule(lever_state_exit(man, name))
        if rule is not None:
            out[name] = rule
    return out


def fallback_census(state: Optional[dict], superseded_calls: int = 0) -> Optional[dict]:
    """The kernel's per-call fallbacks at exit against the documented class (registry.FALLBACK_CLASSES, FALLBACK_MAX_SHARE): `calls`,
    `fallbacks`, `superseded_calls` (the calls of its class a lever bound over it served — `superseded_calls()`; they stay in the class),
    `share` = fallbacks / (calls + superseded_calls + fallbacks), `max_share`, `within` (share <= max_share; None when nothing was counted),
    `classes`. None without an exit state. The one reading for `record_exit` and `excess_fallbacks`."""
    if not isinstance(state, dict):
        return None
    calls, fallbacks = state.get("calls"), state.get("fallbacks")
    if not (isinstance(calls, int) and isinstance(fallbacks, int)):
        return None
    n = calls + int(superseded_calls or 0) + fallbacks
    share = (fallbacks / n) if n > 0 else None
    return {"calls": calls, "fallbacks": fallbacks, "superseded_calls": int(superseded_calls or 0), "share": share, "max_share": _registry.FALLBACK_MAX_SHARE,
            "within": (share <= _registry.FALLBACK_MAX_SHARE) if share is not None else None, "classes": list(_registry.FALLBACK_CLASSES)}


def excess_fallbacks(man: dict) -> list:
    """The levers of an ACTIVE activation whose kernel fell back beyond the documented class at exit (`fallback_census` `within` false:
    more than FALLBACK_MAX_SHARE of the attention calls took the stock path) — else []. A partial state by name (`fallback_excess`), read by
    the model process (`stack.post_run_partial`) and by `pred`'s `verdict` from the same manifest fields as `partial_levers`."""
    rep = man.get("activation_report") or {}
    census = fallback_census(lever_state_exit(man, _modes.LEVER), superseded_calls(man, _modes.LEVER))
    applied = list(rep.get("levers_applied") or rep.get("levers") or [])
    if rep.get("active") and _modes.LEVER in applied and census is not None and census["within"] is False:
        return [_modes.LEVER]
    return []


def fallback_detail(levers: list, census: Optional[dict]) -> str:
    """`detail` of `fallback_excess`: the levers, then the counters and the class (the family line's <detail>)."""
    c = census or {}
    share = c.get("share")
    return (f"{','.join(levers)}: EXIT calls={c.get('calls')} fallbacks={c.get('fallbacks')} share={share:.3f} above {c.get('max_share')} "
            f"(the documented fallback classes — {'; '.join(c.get('classes') or [])} — bound the share; more attention calls than those took the stock path)"
            if share is not None else f"{','.join(levers)}: no attention call counted")


def record_exit(result_dir: str, state: Optional[dict], lever_states: Optional[dict] = None) -> Optional[str]:
    """At interpreter exit: the carried kit's final `_STATE` (the EXIT line's fields) and every lever's final counters (the LEVER lines'
    fields) into the manifest, the fallback census and the `partial` record they decide."""
    p = path(result_dir)
    if not os.path.isfile(p):
        return None
    man = read(p)
    man["kit_state_exit"] = dict(state) if state is not None else None
    man["lever_states_exit"] = {n: (dict(s) if isinstance(s, dict) else None) for n, s in (lever_states or {}).items()}
    man["fallback_census"] = fallback_census(state, superseded_calls(man, _modes.LEVER))
    man["partial"] = partial_levers(man)
    man["idle"] = idle_levers(man)
    man["stepped_aside"] = stepped_aside(man)
    man["superseded"] = superseded(man)
    man["fallback_excess"] = excess_fallbacks(man)
    return _dump(p, man)


def listing(result_dir: str, exclude=()) -> list:
    """Every file under result_dir (relative path, bytes, sha256); sorted by path."""
    out = []
    for dp, _dirs, fns in os.walk(result_dir):
        for f in fns:
            rel = os.path.relpath(os.path.join(dp, f), result_dir)
            if rel in exclude:
                continue
            out.append({"file": rel, "bytes": os.path.getsize(os.path.join(dp, f)), "sha256": sha256_file(os.path.join(dp, f))})
    return sorted(out, key=lambda e: e["file"])


DONE_SUFFIX = ".done.txt"                                # colabfold's per-query completion marker, <result_dir>/<jobname>.done.txt (batch.py:1410; touched at :1653-1654 when a model ran and --zip is off)
ZIP_SUFFIX = ".result.zip"                               # colabfold's per-query archive under --zip, <result_dir>/<jobname>.result.zip (batch.py:1405; written at :1643-1651 in place of the marker, the job's files moved into it)
COMPLETION_SUFFIXES = (ZIP_SUFFIX, DONE_SUFFIX)          # colabfold's own finished-job test, in its order (batch.py:1405-1412: a job whose archive or marker exists is done — skipped on a rerun)

NUM_MODELS_0 = "num_models_0"                            # `no_model_run` cases, by name: the command line asked for no model (--num-models 0, or --msa-only: main() sets num_models = 0, batch.py:2160-2161) — MSAs and features only, no marker
ALL_JOBS_DONE = "all_jobs_done"                          #   every job of the input was complete in <results> before the run and colabfold keeps existing results (no --overwrite-existing-results): each is skipped (batch.py:1406-1412), no model is built (:1537 is reached by a job that runs)
AF3_JSON = "af3_json"                                    #   --af3-json: main() writes the AlphaFold 3 input JSON and returns before run() (batch.py:2184-2200): the hook on run() is never entered
NO_MODEL_RUN_CASES = (NUM_MODELS_0, ALL_JOBS_DONE, AF3_JSON)


def done_marker(result_dir: str, item_id: str) -> str:
    return os.path.join(result_dir, item_id + DONE_SUFFIX)


def completion(result_dir: Optional[str], item_id: str) -> Optional[str]:
    """The completion artefact colabfold left for job `item_id` under `result_dir`, by its own finished-job test in its order
    (COMPLETION_SUFFIXES): "result.zip" (the --zip form) | "done.txt" (the marker) | None (the job is not complete: it runs on the next launch)."""
    if not result_dir:
        return None
    for suffix in COMPLETION_SUFFIXES:
        if os.path.isfile(os.path.join(result_dir, item_id + suffix)):
            return suffix[1:]
    return None


def no_model_run(num_models, item_ids, result_dir: Optional[str], keep_existing: bool = True, af3_json: bool = False) -> Optional[str]:
    """The case, by name (NO_MODEL_RUN_CASES), in which colabfold_batch builds no model on this run — decided from its own inputs before it
    runs, never from a lever's counters or a clock: `af3_json` (--af3-json: main() returns before run()), `num_models_0` (--num-models 0 /
    --msa-only), `all_jobs_done` (keep_existing and every job of `item_ids` has its completion artefact under result_dir: colabfold skips them
    all, batch.py:1406-1412); None when a model is expected to run. One decision for `pred` (before the launch, from the command line and the
    results directory) and the run hook (stack.hook_run, from run()'s own arguments)."""
    if af3_json:
        return AF3_JSON
    try:
        n = int(num_models)
    except (TypeError, ValueError):
        n = None
    if n == 0:
        return NUM_MODELS_0
    ids = list(item_ids or [])
    if keep_existing and ids and result_dir and all(completion(result_dir, i) is not None for i in ids):
        return ALL_JOBS_DONE
    return None


def no_model_case(man: Optional[dict]) -> Optional[str]:
    """The `no_model_run` word a launch record / manifest carries (top level, else its activation report's), when it is one of
    NO_MODEL_RUN_CASES; None otherwise (a model was expected: the partial rule applies in full)."""
    man = man or {}
    word = man.get("no_model_run") or (man.get("activation_report") or {}).get("no_model_run")
    return word if word in NO_MODEL_RUN_CASES else None


def no_model_detail(case: str, how: Optional[str] = None, done: Optional[dict] = None, idle: Optional[list] = None) -> str:
    """The IDLE line's <detail> for `case`: what colabfold_batch did instead of building a model (`how`: the flag as given, `--msa-only` /
    `--num-models 0`, or run()'s `num_models=0`; `done`: job -> its completion artefact before the run), then — when levers were applied and
    left without a call (`idle`) — that this is not a partial activation."""
    if case == NUM_MODELS_0:
        text = f"colabfold_batch built no model ({how or '--num-models 0'}): MSAs and input features only, no {DONE_SUFFIX[1:]} written"
    elif case == ALL_JOBS_DONE:
        done = dict(done or {})
        shown = ", ".join(f"{i}.{a}" for i, a in list(done.items())[:3]) + (f", … (+{len(done) - 3})" if len(done) > 3 else "")
        text = (f"every job was complete in the results directory before the run ({len(done)}/{len(done)}: {shown}) and colabfold_batch keeps existing results "
                f"(no --overwrite-existing-results): it skipped them all and built no model")
    elif case == AF3_JSON:
        text = "colabfold_batch wrote the AlphaFold 3 input JSON (--af3-json) and returned before its run(): the mode's hook was not entered, nothing was applied"
    else:
        text = str(case)
    if idle:
        text += " — the mode's levers had no model call to serve: not a partial activation"
    return text


def verdict(manifest_dir: str, mode: str, item_ids, child_rc: int, launch_id: Optional[str] = None,
            n_gpu: int = 1, result_dir: Optional[str] = None, no_model_run: Optional[str] = None,
            no_model_how: Optional[str] = None, done_before: Optional[dict] = None) -> dict:
    """The `pred` verdict after the model process exits, from the manifest that process wrote (under `manifest_dir`, the launch's work
    directory) and colabfold's own completion markers under `result_dir` (one per id of `item_ids`) —
    colabfold's `run` swallows per-query failures and returns normally (batch.py:1460-1462, :1476-1478, :1593-1596), so the child's
    return code alone is no verdict. One exit code per condition:
      child rc 3 / other non-zero                   -> rc 3 (its NOT ACTIVE line: `kernel_not_engaged` / `lever_fallback` when the
                                                        manifest shows that state — the strict hook's own exit —, else `not_active`;
                                                        `exit_by: model_process`) / rc 1 (`model_process_failed`)
      a kit mode with no activation report of this   -> rc 3, `hook_never_fired` (colabfold.batch.run was not entered through the hook in
        launch (the manifest's launch_id differs)        this launch; a manifest another process wrote does not count)
      a kit mode whose model process reports another  -> rc 3, `n_gpu_mismatch` (`requested=P active=Q`): the GPU count the CLI accepted
        GPU count than `--n_gpu` asked (its ACTIVE       must be the count the model process ran at — a run that dropped the axis is refused,
        report's n_gpu, or the ROWPAIR lever's at exit)  never a pass
      a kit mode active with kit_state_exit.calls<1  -> rc 3, `kernel_not_engaged`: PARTIAL (the kernel ran no attention call — the
                                                        lever with no evidence of application, `partial` names it)
      a kit mode whose lever cannot run on this GPU  -> rc 3, `lever_fallback` (the lever named with its word: a mode is all of its levers)
        (no tile table)
      a kit mode active whose kernel fell back       -> the run's own rc, `fallback_excess` recorded and named on ONE `PARTIAL fallback_excess:` line
        beyond the documented class at exit             (more than registry.FALLBACK_MAX_SHARE of the attention calls took the stock path —
                                                        `fallback_census`; those calls ran the stock operation): never an exit code of its own
      any mode, a job without its completion       -> rc 1, `missing_outputs` (the ids named: an output shortfall, never partial); the artefact
        artefact (<id>.result.zip under --zip,          is colabfold's own finished-job test (`completion`), recorded per id (`completion`)
        else <id>.done.txt)
      any mode, colabfold_batch built no model       -> rc 0, `no_model_run` names the case (`no_model_run()`: the launcher's reading before the
        (--num-models 0 / --msa-only; every job         launch, `no_model_run=`; a kit mode's model process records its own from run()'s arguments —
        already complete; --af3-json)                   the same decision): no marker is owed (`num_models_0`, `af3_json`), the applied levers with no
                                                        call are `idle`, never `partial`, and a kit mode whose hook was not entered under --af3-json
                                                        is not `hook_never_fired`; `detail` is the IDLE line's (report.idle_line)
      else                                           -> rc 0 (`detail` names the `fallback_excess` counters when present)
    The dict is recorded in the launch record (`verdict`); `detail` is the family line's <detail> (report.partial_exit_line)."""
    man = {}
    p = path(manifest_dir)
    if os.path.isfile(p):
        try:
            man = read(p)
        except (OSError, ValueError):
            man = {}
    if mode != "off" and launch_id is not None and man.get("launch_id") != launch_id:
        man = {}                                                          # not this launch's manifest: no activation report of this run
    rep = man.get("activation_report") or {}
    exit_state = man.get("kit_state_exit") or {}
    calls = exit_state.get("calls") if isinstance(exit_state, dict) else None
    case = (no_model_case(man) if mode != "off" else None) or (no_model_run if no_model_run in NO_MODEL_RUN_CASES else None)   # the model process's own reading (run()'s arguments), else the launcher's (the command line + the results directory before the launch): one decision (no_model_run())
    if case is not None and man and no_model_case(man) is None:
        man = dict(man, no_model_run=case)                                # the partial / idle readings below follow the case
    item_ids = list(item_ids or [])
    done = {i: completion(result_dir, i) for i in item_ids}               # colabfold's own finished-job test per id: result.zip (--zip) | done.txt | None
    missing = [] if case in (NUM_MODELS_0, AF3_JSON) else [i for i in item_ids if done[i] is None]   # no model asked: no marker is owed (batch.py:1653; --af3-json returns before run())
    partial = partial_levers(man) if mode != "off" else []
    idle = idle_levers(man) if mode != "off" else []
    aside = stepped_aside(man) if mode != "off" else {}
    sup = superseded(man) if mode != "off" else {}
    excess = excess_fallbacks(man) if mode != "off" else []
    census = fallback_census(exit_state, superseded_calls(man, _modes.LEVER)) if exit_state else None
    fallbacks = dict(rep.get("lever_fallbacks") or {}) if (mode != "off" and rep) else {}   # table-backed levers held on this part by name (fallback:no_tiles_cc<NN>(<kind>)): a partial state
    active_p = ngpu_active(man) if rep else None                          # the GPU count the model process ran at (None: no count to trust)
    v = {"ok": False, "rc": 1, "reason": None, "detail": "", "missing": missing, "active": rep.get("active"), "calls": calls, "child_rc": child_rc,
         "launch_id": launch_id, "partial": partial, "idle": idle, "no_model_run": case, "completion": done, "stepped_aside": aside, "superseded": sup,
         "fallback_excess": excess, "lever_fallbacks": fallbacks,
         "fallback_census": census, "exit_by": "verdict", "n_gpu_requested": int(n_gpu), "n_gpu_active": active_p}
    if child_rc == _report.EXIT_NOT_ACTIVE:                              # the model process exited by name: its own line is in the log
        v["exit_by"] = "model_process"
        if fallbacks:
            v.update(rc=_report.EXIT_NOT_ACTIVE, reason="lever_fallback", detail=_report.lever_fallback_detail(rep))
        elif partial:
            v.update(rc=_report.EXIT_NOT_ACTIVE, reason="kernel_not_engaged", detail=engaged_detail(partial, man))
        else:
            v.update(rc=_report.EXIT_NOT_ACTIVE, reason="not_active", detail="the model process refused the activation (its NOT ACTIVE line)")
    elif child_rc != 0:
        v.update(reason="model_process_failed", detail=f"rc={child_rc}")
    elif mode != "off" and not rep and case != AF3_JSON:                 # (--af3-json: colabfold_batch returns before its run() — the hook is not entered by design, named on the IDLE line)
        v.update(rc=_report.EXIT_NOT_ACTIVE, reason="hook_never_fired", detail="no activation report of this launch in the model process: colabfold.batch.run was not entered through the hook")
    elif mode != "off" and rep.get("active") and active_p != int(n_gpu):      # fail-closed: the axis the CLI accepted is the axis the model ran at, or the run is refused (never partial)
        v.update(rc=_report.EXIT_NOT_ACTIVE, reason="n_gpu_mismatch", detail=f"n_gpu_mismatch requested={int(n_gpu)} active={active_p}")
    elif fallbacks:                                                       # a lever held by name on this part (the strict hook exits 3 at the activation itself; the explicit route's record reads the same)
        v.update(rc=_report.EXIT_NOT_ACTIVE, reason="lever_fallback", detail=_report.lever_fallback_detail(rep))
    elif partial:
        v.update(rc=_report.EXIT_NOT_ACTIVE, reason="kernel_not_engaged", detail=engaged_detail(partial, man))
    elif missing:
        v.update(reason="missing_outputs", detail="no " + " / ".join(x[1:] for x in COMPLETION_SUFFIXES) + " for " + ",".join(missing))
    else:
        v.update(ok=True, rc=0)
        if case is not None:                                              # colabfold_batch built no model: named on the IDLE line (report.idle_line), the run's own exit code
            v["detail"] = no_model_detail(case, how=no_model_how or (rep.get("no_model_run_how") if rep else None),
                                          done=done_before if done_before is not None else {i: a for i, a in done.items() if a}, idle=idle)
        elif excess:
            v["detail"] = fallback_detail(excess, census)
    return v


def ngpu_active(man: dict) -> Optional[int]:
    """The GPU count the model process ran at, from the manifest it wrote: the activation report's `n_gpu` (1 when the report predates the
    axis); when P > 1 the ROWPAIR lever must be installed at exit and agree with it — else None (no count to trust: never equal to a request)."""
    rep = man.get("activation_report") or {}
    try:
        p = int(rep.get("n_gpu", 1))
    except (TypeError, ValueError):
        return None
    tp = (man.get("lever_states_exit") or {}).get(_modes.TP_LEVER)
    if isinstance(tp, dict) and tp.get("enabled"):
        try:
            return p if int(tp.get("n_gpu")) == p else None
        except (TypeError, ValueError):
            return None
    return p if p == 1 else None                                           # P > 1 reported without the lever installed at exit: no sharded run happened


def engaged_detail(partial: list, man: Optional[dict] = None) -> str:
    """`detail` of `kernel_not_engaged`: the levers with no evidence of application, each with its exit `calls`, then the reason (the family
    line's <detail>)."""
    man = man or {}
    parts = []
    for name in partial:
        st = lever_state_exit(man, name)
        calls = st.get("calls") if isinstance(st, dict) else None
        what = "the kernel ran no attention call" if name == _modes.LEVER else "no model call went through the lever"
        parts.append(f"{name}: EXIT calls={calls} ({what})")
    return "; ".join(parts)


def final(manifest_dir: str, report: Optional[dict], **fields) -> dict:
    """The launch record after the model process exits: the activation report that process wrote under `manifest_dir` (the fast
    route; `report` when the caller has it — the stock route's) with its exit records, completed with `fields` (command, argv, exit
    code, settings, the stock proof, the listing, the verdict). In memory: nothing is written."""
    p = path(manifest_dir)
    if os.path.isfile(p):
        old = read(p)
        report = report if report is not None else old.get("activation_report")
        man = build(report, **{k: v for k, v in old.items() if k in ("kit_state_exit", "lever_states_exit", "fallback_census", "stepped_aside", "superseded", "idle")})
    else:
        man = build(report)
    man.update(fields)
    return man

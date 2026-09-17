"""python -m af3_jax_opt {pred,check,warm} --variant p2 [--mode off|exact|fast|big] [--n_gpu 1|2] ... — a thin command layer over the fork's command line and the
kit code: it never re-implements a lever or a mode. ``pred`` runs one model process at the resolved mode/variant; ``check`` is the
dry run (resolves and gates, applies nothing); ``warm`` builds the mode's ``$CACHE`` for this box's key. Mode/variant resolve as ``--mode``/``--variant`` >
``AF3_JAX_OPT``/``AF3_JAX_VARIANT`` from the environment > the package default; a flag that disagrees with the set variable is refused."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from typing import List, Optional

from . import __version__, ablation as _ablation, det, inputs as _inputs, carry as _carry, modes as _modes, outputs as _outputs, report as _report, settings as _settings
from . import kernels as _kernels, peakmem as _peakmem, stack, stock_pred, variants as _variants
from opt_core.report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE   # the kit's exit table (the core's): 0 ok · 1 failed · 2 usage · 3 not active, or the mode's levers came back short (a mode is all of its levers: a run under its name with a subset exits 3 by name)
EXIT_KERNELS = _kernels.EXIT_KERNELS                                           # 5: the pass's kernel implementation is not the route's (the KERNELS line's REQUIRE guard, kernels.py)

INFER_RX = re.compile(r"Running model inference with seed (\d+) took ([0-9.]+) seconds")
JOB_RX = re.compile(r"Running fold job (.+?)\.\.\.\s*$")                      # the fork's per-input line (run_alphafold.py:1003; the kit script :1281): the PHASE line's job=
LOG_NAME = "af3_jax_opt.log"


def _resolve_switch(cli_value: Optional[str], env_name: str, what: str, default: Optional[str]) -> str:
    env_value = os.environ.get(env_name) or None
    if cli_value and env_value and cli_value != env_value:
        raise SystemExit(f"{_report.PREFIX} --{what} {cli_value} disagrees with {env_name}={env_value} in the environment; drop one (rc 2)")
    v = cli_value or env_value or default
    if v is None:
        raise SystemExit(f"{_report.PREFIX} --{what} is required ({env_name} unset) (rc 2)")
    return v


def _n_gpu(a) -> int:
    """--n_gpu wins; else AF3_JAX_N_GPU; else 1 (stack validates the value and applies the mode / visible-device rules by name)."""
    if a.n_gpu is not None:
        return a.n_gpu
    raw = os.environ.get(stack.ENV_N_GPU, "").strip()
    try:
        return int(raw) if raw else 1
    except ValueError:
        return raw                                                        # not an integer: refused by name downstream (opt_core.mem.ngpu.check_n_gpu)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--variant", choices=_variants.VARIANTS, default=None, help=f"{' | '.join(_variants.VARIANTS)} (or AF3_JAX_VARIANT)")
    p.add_argument("--mode", default=None, help=f"{' | '.join(_modes.MODES)} (or AF3_JAX_OPT; default {_modes.DEFAULT_MODE})")
    p.add_argument("--n_gpu", dest="n_gpu", type=int, default=None, help="devices for the memory mode's row-sharded pair stack: explicit, default 1 (absent == AF3_JAX_N_GPU, else 1); "
                                                                    "n_gpu > 1 is accepted under --mode big only and needs that many visible GPUs — refused by name otherwise (rc 3)")
    p.add_argument("--model_dir", dest="model_dir", default=None, help=f"a weights directory of the caller's own (holds {_variants.PARAMS_FILE}): always accepted, digested — "
                                                                     "the pinned digest prints `weights=<variant> sha256=<12> (pinned)`, any other `weights sha256=<12> NOT PINNED …` and runs; "
                                                                     "default <AF3_JAX_PARAMS_ROOT>/<variant>")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="af3-jax-opt", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=f"af3_jax_opt {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pred", help="one model process (off: the stock script; exact/fast/big: the kit script through the mode's launcher; big --n_gpu 2: one process over two GPUs)")
    _common(p)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--json_path", default=None); p.add_argument("--input_dir", default=None)
    _region_option(p)
    p = sub.add_parser("check", help="dry run: resolve + gate on this box, apply nothing"); _common(p)
    p.add_argument("--json", action="store_true")
    _region_option(p)
    p = sub.add_parser("warm", help="build the mode's $CACHE for this box's key"); _common(p)
    p.add_argument("--json_path", default=None, help="one fold input to warm exactly as `pred --json_path` will run it (its bucket, its seeds); exclusive with --input_dir")
    p.add_argument("--input_dir", default=None, help="one input per bucket to warm (default: the kit's tests/inputs); under --mode big also the input the line is sized from "
                                                    "(the class warmed is the class pred reads for it) — else pass --n_est: a big warm with neither is refused (rc 2)")
    _region_option(p)
    return ap


def _region_option(p: argparse.ArgumentParser) -> None:
    """The memory mode's sizing option, the same on pred, check and warm (modes.effective_line reads it: ONE resolver for the three verbs)."""
    p.add_argument(_modes.N_EST_FLAG, dest="n_est", type=int, default=None,
                   help="--mode big only: the caller's own token estimate of the input, in place of the one read from the fold input (inputs.token_estimate) — "
                        "the size the region rule decides on (check and warm without an input: the line of an input that size)")


def _big_region(a: argparse.Namespace, mode: str, n_gpu: int, input_path: Optional[str]) -> Optional[dict]:
    """The memory mode's region record for pred / check (None for every other mode), by THE resolver (modes.effective_line): the input's token
    estimate (pred), the caller's --n_est over it; check with nothing to size reports the reach region (the
    size-unknown rule, named). warm resolves through the same function with require_size (warm.py)."""
    return _modes.effective_line(mode, input_path=input_path, n_est=getattr(a, "n_est", None), n_gpu=n_gpu)["record"]


def ensure_fast_script() -> dict:
    """The kit's step: patches/patched_files/run_alphafold.py copied beside the stock script as run_alphafold_fast.py (sha256 of the copy
    checked against the tree's file; re-copied when it differs)."""
    src = os.path.join(stack.kit_home("fast_inference"), _modes.PATCHED_SCRIPT_RELPATH)
    dst = os.path.join(stack.repo_dir(), stack.FAST_SCRIPT)
    want = stack.sha256_file(src)
    state = "present"
    if not os.path.isfile(dst) or stack.sha256_file(dst) != want:
        shutil.copyfile(src, dst)
        state = "copied"
    return {"path": dst, "sha256": stack.sha256_file(dst), "state": state}


def cmd_pred(a: argparse.Namespace, extra: List[str]) -> int:
    mode = _resolve_switch(a.mode, stack.ENV_MODE, "mode", _modes.DEFAULT_MODE)
    variant = _resolve_switch(a.variant, stack.ENV_VARIANT, "variant", None)
    if not (a.json_path or a.input_dir) or (a.json_path and a.input_dir):
        _report.emit(f"{_report.PREFIX} pred needs exactly one of --json_path / --input_dir (rc 2)"); return EXIT_USAGE
    md_extra, extra = _variants.split_model_dir(extra)                   # the caller's weights directory, either spelling (always accepted; digested at activation, pinned or NOT PINNED by one line)
    model_dir = md_extra or a.model_dir
    caller_cache = stock_pred.caller_cache_dir(extra)                     # a --cache_dir of the caller's own: on off it replaces the EMPTY class, on a kit mode the mode's class (the levers read and write THAT directory; a cold one is a partial activation by the same rule)
    try:
        region = _big_region(a, mode, _n_gpu(a), os.path.abspath(a.json_path) if a.json_path else os.path.abspath(a.input_dir))
        rep = stack.activate(mode, variant, n_gpu=_n_gpu(a), model_dir=model_dir, region=region,
                             stated=_settings.stated(extra), caller_cache_dir=caller_cache if mode != "off" else None)   # a kit mode gates the caller's directory as its class; off keeps its EMPTY-class rule below
    except ValueError as e:
        _report.emit(f"{_report.PREFIX} {e} (rc 2)"); return EXIT_USAGE
    if mode == "off" and caller_cache is not None:
        rep["cache_dir"] = caller_cache                                      # off: the caller's own --cache_dir, verbatim (a kit mode took it as its class at activation)
    rep["autotune"] = ("caller_cache_dir" if caller_cache is not None else "skipped(cache_dir empty)") if mode == "off" else "class_cache"   # off: the EMPTY class skips tokamax autotune, a caller's --cache_dir holds the fork's own; kit modes: the class holds it
    _report.emit(_report.activation_line(rep))
    for note in rep.get("gpu_notes") or []:                              # a device below the kit's reference hardware line: named, never refused — the mode engages its whole lever set; a lever that cannot run there refuses the run by name
        _report.emit(f"{_report.PREFIX} GPU NOTE {note}")
    if rep.get("region"):                                                # the memory mode: ONE line naming the lever set the region decided inactive, and the rule's words (decided up front, never a fallback)
        _report.emit(f"{_report.PREFIX} {_modes.region_note(rep['region'])}")
    if (rep.get("xla_pool") or {}).get("applies"):                      # --n_gpu P at the row-sharded stack's pool precondition: ONE line naming the fraction the model process runs with and who set it
        _report.emit(_report.xla_pool_note(rep["xla_pool"]))
    eff = rep.get("mode_effective") or mode                              # big in region fast runs the fast line's program (modes.effective_line, resolved at activation): its resolution, evidence rules and executables index
    wl = _variants.weights_line(rep.get("params"))
    if wl:
        _report.emit(wl)                                                  # `WEIGHTS weights=p2 sha256=<12> (pinned)` | `WEIGHTS weights sha256=<12> NOT PINNED — …` (the run proceeds either way)
    if not rep["active"]:
        return EXIT_NOT_ACTIVE
    out_dir = os.path.abspath(a.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    preexisting = _outputs.prediction_files(out_dir)                        # models already under --output_dir (an earlier pass's job: upstream writes this pass's into a timestamped sibling) — not this pass's to count
    empty_cache = mode == "off" and caller_cache is None                     # off with no --cache_dir of the caller's: `--cache_dir=` — the fork skips its tokamax autotune cache and keeps its JAX cache at ./jax under the cwd,
    proc_cwd = out_dir if empty_cache else stack.repo_dir()                  #  so that process runs IN the pass's output directory (a fresh JAX cache per pass); every other route runs in the repo dir
    if empty_cache:
        rep["cache_dir"] = os.path.join(out_dir, "jax")
    cache_dir = rep["cache_dir"]
    res = _modes.with_levers_off(_modes.with_n_gpu(_modes.resolve(eff, cache_dir, size=rep.get("region")), rep["n_gpu"]), rep.get("levers_ablated") or [], rep["n_gpu"])   # activation's composition (MODEL_OPT_LEVERS_OFF applied there, validated by name)
    if not empty_cache:
        os.makedirs(cache_dir, exist_ok=True)                               # the levers' class, or the caller's --cache_dir on off
    fast = ensure_fast_script() if mode != "off" else None
    if fast:
        _report.emit(_report.line("SCRIPT", path=fast["path"], sha256=fast["sha256"][:16], state=fast["state"]))
    user = [f"--output_dir={out_dir}"] + ([f"--json_path={os.path.abspath(a.json_path)}"] if a.json_path else [f"--input_dir={os.path.abspath(a.input_dir)}"]) + extra   # the caller's stock flags (extra) ride last: absl's last-wins order
    launcher = res["launcher"]                                              # the mode's launcher prefix (modes.launcher): nothing of the caller's rides on it
    argv = stock_pred.compose(variant, cache_dir, user, script=res["script"], mode_flags=res["flags"], launcher=launcher, model_dir=model_dir)   # the caller's stock flags last, verbatim
    _report.emit(stock_pred.buckets_line(stock_pred.effective_buckets(argv)))       # the padding buckets the model process runs with, parsed AFTER composition (a caller's later --buckets wins by absl order)
    env = stack.model_process_env(mode_env=res["env"], core_path=bool(res["launcher"]), n_gpu=rep["n_gpu"])
    peak_hook = _peakmem.arm(env)                                            # the allocator-peak probe (opt_core.mem.peak) rides every model process of the pass, the stock route's too: hook dir LAST on PYTHONPATH
    kexp = _kernels.expected(mode, res["levers"], stock_pred.effective_flash_impl(argv)["impl"])   # the route's expected kernel reading, from the mode table and the composed request alone
    _kernels.arm(env, peak_hook["hook"], kexp)                              # the KERNELS probe beside the peak instrument (one sitecustomize boots both); the expectation named so the process refuses before its first timed item
    pr = stock_pred.proof(env, argv) if mode == "off" else None
    if pr:
        _report.emit(stock_pred.proof_line(pr))
        if not pr["ok"]:
            return EXIT_NOT_ACTIVE
    _report.emit(_report.line("COMMAND", argv=" ".join(argv), lever_env=" ".join(f"{k}={v}" for k, v in res["env"].items()) or "none"))
    for ln in _report.phase_notes():                                       # the per-item PHASE line's NA fields, named once per pass (report.ONE_PROGRAM): the same words on every route
        _report.emit(ln)
    _report.emit(det.cache_line(cache_dir, "before"))
    tk_dir = None if empty_cache else cache_dir                              # `--cache_dir=`: the fork keeps no tokamax table path at all (run_alphafold.py _autotune_cache_path is None)
    tk_before = det.tokamax_table(tk_dir)
    log_path = os.path.join(out_dir, LOG_NAME)
    tk_lines = []                                                            # the transcript's tokamax autotune lines (det.TOKAMAX_RX): loaded | saved | unavailable
    ck_lines = []                                                            # the transcript's CACHEKEY line (det.CACHEKEY_RX): the model process's persistent-cache key rebinding
    infer, lever_lines, klines, job, jobs = [], [], [], [None], []   # job: the fold input the transcript last named (the PHASE line's item); jobs: every input named, in order; klines: the KERNELS reader's transcript lines
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        def on_line(ln: str) -> None:
            log.write(ln); log.flush()
            m = JOB_RX.search(ln)
            if m:
                job[0] = m.group(1); jobs.append(m.group(1))
            m = INFER_RX.search(ln)
            if m:
                infer.append({"seed": int(m.group(1)), "took_s": float(m.group(2))})
                _report.emit(_report.phase_line(int(m.group(1)), float(m.group(2)), job=job[0]))   # ONE PHASE line per item at the fork's own per-seed timer, every route alike
            if _modes.is_evidence_line(ln):
                lever_lines.append(ln.rstrip("\n"))
            if _kernels.is_kernels_line(ln):
                klines.append(ln.rstrip("\n"))
            if det.is_tokamax_line(ln):
                tk_lines.append(ln.rstrip("\n"))
            if det.is_cachekey_line(ln):
                ck_lines.append(ln.rstrip("\n"))
            sys.stderr.write(ln); sys.stderr.flush()
        stack.launched(mode, variant, rep["n_gpu"], region=(rep.get("region") or {}).get("region"))
        rc = stock_pred.run_logged(argv, env, proc_cwd, on_line)
    wall = time.time() - t0
    if not empty_cache and not det.xla_caches_state(env)["carried"]:              # the recipe's carrier switched off in this environment (JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES without the autotune dir): named once, never refused — exact == off is then not expected
        _report.emit(det.detclass_line(cache_dir, env))
    _report.emit(det.cache_line(cache_dir, "after", key=det.cache_key_word(ck_lines)))   # cache_key=device_kind under the tree's launchers, =topology when the process ran jax's own key (mode off)
    tokamax = det.tokamax_account(tk_before, det.tokamax_table(tk_dir), tk_lines, argv)
    _report.emit(det.tokamax_line(tokamax))                                  # ONE TOKAMAX line per pass, every route: the class's tokamax autotuning table (absent | written | loaded), the attempt's fate, the cache-miss policy
    peak = _peakmem.collect(peak_hook)                                       # the pass's allocator peak (one PEAK line; a measurement, never a gate)
    _report.emit(_peakmem.line(peak))
    for ln in _report.peak_item_lines(peak, jobs):                           # the cross-engine per-item JAX grammar (`PEAK item=<job> inuse_gib=<f>`) when the pass had one input; else ONE named PEAK-NOTE
        _report.emit(ln)
    kacc = _kernels.account(mode=mode, n_gpu=rep["n_gpu"], levers=res["levers"], argv=argv, env=env, lines=klines, rc=rc, caller_environ=os.environ)
    _report.emit(kacc["line"])                                               # ONE KERNELS line per pass, every route: the implementation each accelerator site actually ran (kernels.py) — a gate: refused = FAILED, exit 5
    shutil.rmtree(peak_hook["hook"], ignore_errors=True)                      # the start-up hook directory (instrument, probe, per-process peak records): read above, nothing of it is an output
    n_pred = _outputs.count_predictions(out_dir, before=preexisting)    # the models THIS pass wrote
    input_path = os.path.abspath(a.json_path) if a.json_path else os.path.abspath(a.input_dir)
    expected = _outputs.expected_predictions(argv, _inputs.seeds_in(input_path, _outputs.num_seeds(argv)))   # --num_seeds N: upstream expands every input to N seeds
    templates = _modes.templates_census(lever_lines, _inputs.templates_declared(input_path), eff)   # the template census (inprocess/templates.py): monitoring, recorded on the DONE line; never a failure or partial condition
    evidence = _modes.lever_evidence(eff, lever_lines, res["levers"], n_gpu=int(res.get("n_gpu") or 1))
    for ln in _report.lever_lines(eff, {**evidence["per_lever"], **_modes.ablated_per_lever(res.get("ablated") or [])}):   # one LEVER line per lever of the composition (the per-lever census), then one per ablated lever (state=off reason=ablated)
        _report.emit(ln)
    failures, partial, named = [], [], list(rep["partial_conditions"])   # one code per condition: a failed run (1) / the mode's levers short (3, by name: a mode is all of its levers). named = a cold cache class: on the ACTIVE and DONE lines, never an exit code
    if _settings.flag_false(argv, "run_inference") and rc == 0:           # --norun_inference (upstream's own flag: no model inference this run; with the kit's --norun_data_pipeline upstream loads the inputs and exits 0): nothing ran that the census could judge and no prediction is due — mirrored, exit 0, named
        _report.emit(f"{_report.PREFIX} NOTE no inference requested (--norun_inference): nothing for the census to judge; predictions={n_pred} expected=0")
        _report.emit(_report.line("DONE", status="ok", rc=rc, predictions=f"{n_pred}/0", templates=templates["token"], wall=f"{wall:.1f}s", reason=None,
                                  partial="no_inference_requested") + " " + _report.ngpu_fields(rep["n_gpu"]) + _ablation.token(rep.get("levers_ablated") or []))
        return EXIT_OK
    refused = _modes.refusal_reason(lever_lines) if rc == EXIT_NOT_ACTIVE else None   # the model process refused by name before launch: a lever of the mode cannot engage on this GPU (fpf_launch.refuse_held)
    krefused = not kacc["ok"] and (rc in (0, EXIT_KERNELS) or kacc["scan"]["refused_inprocess"])   # the REQUIRE guard: an accelerator absent, fallen back, or a call site owned by other than the route's lever set — refused by name (exit 5), whatever rc a process the probe stopped ended with;
    if krefused and not refused:                                           # a process that died of something else (its own rc) is that failure, its missing reading a consequence (named on the DONE line, exit 1)
        failures.append(_kernels.done_reason(kacc))
    if rc != 0 and not krefused and not refused:
        failures.append(f"model process exited {rc}" + ("" if kacc["ok"] else f" (kernels reading: {kacc['words']['verdict']})"))
    if n_pred != expected and not refused:
        failures.append(f"predictions={n_pred} expected={expected}")
    soft = []                                                              # levers_short words that are NOT an exit code: kernels that engaged and ran the stock class at some call sites (modes: softened)
    if not evidence["ok"]:
        (soft if evidence.get("softened") else partial).append(f"levers_short: {evidence['reason']}")
    status = "FAILED" if failures else ("not_active" if refused else ("partial" if partial else "ok"))
    for cond in soft:                                                       # ONE named line per softened condition: those call sites ran the stock class; the run keeps its own exit code
        _report.emit(f"{_report.PREFIX} PARTIAL {cond}; those call sites ran the stock class — recorded, exit 0")
    _report.emit(_report.line("DONE", status=status, rc=rc, predictions=f"{n_pred}/{expected}", templates=templates["token"], wall=f"{wall:.1f}s",
                              reason="; ".join(failures) if failures else (refused if refused else None), partial="; ".join(named + partial + soft) if (named or partial or soft) else None)
                 + " " + _report.ngpu_fields(rep["n_gpu"]) + _ablation.token(rep.get("levers_ablated") or []))   # `` ablated=<names>`` last, only under MODEL_OPT_LEVERS_OFF (the ACTIVE and DONE lines both name the ablation)
    if krefused and not refused:
        return EXIT_KERNELS                                                   # one code per condition: a kernels refusal is exit 5
    return EXIT_OK if status == "ok" else (EXIT_FAIL if status == "FAILED" else EXIT_NOT_ACTIVE)


def cmd_check(a: argparse.Namespace, extra: Optional[List[str]] = None) -> int:
    mode = _resolve_switch(a.mode, stack.ENV_MODE, "mode", _modes.DEFAULT_MODE)
    variant = _resolve_switch(a.variant, stack.ENV_VARIANT, "variant", "") or None
    extra = list(extra or [])
    try:
        model_dir = a.model_dir
        rep = stack.check(mode, variant, n_gpu=_n_gpu(a), model_dir=model_dir, refresh_digest=True, stated=_settings.stated(extra),
                          region=_big_region(a, mode, _n_gpu(a), None))   # the check verb hashes the weights afresh and rewrites the memo entry
    except ValueError as e:
        _report.emit(f"{_report.PREFIX} {e} (rc 2)"); return EXIT_USAGE
    caller_cache = stock_pred.caller_cache_dir(extra)
    if mode == "off" and caller_cache is not None:
        rep["cache_dir"] = caller_cache                                       # the caller's own --cache_dir, verbatim (mode off only), as pred takes it
    carry = _carry.carry_check()                                   # every add-on directory: reported; the mode's own add-ons gate (stack.activate)
    rep["carry_all"] = {k: carry[k] for k in ("kits", "files", "ok", "missing")}
    if rep.get("active"):
        eff = rep.get("mode_effective") or mode                          # modes.effective_line's, resolved at activation
        res = _modes.with_levers_off(_modes.with_n_gpu(_modes.resolve(eff, rep["cache_dir"] or "$CACHE", size=rep.get("region")), rep["n_gpu"]), rep.get("levers_ablated") or [], rep["n_gpu"])
        rep["command"] = " ".join(stock_pred.compose(variant, rep["cache_dir"] or "$CACHE",
                                                      ["--output_dir=<out>", "--json_path=<json>", *extra], script=res["script"], mode_flags=res["flags"],
                                                      launcher=res["launcher"], model_dir=model_dir))
    else:
        rep["command"] = None
    rep["autotune"] = ("caller_cache_dir" if caller_cache is not None else "skipped(cache_dir empty)") if mode == "off" else "class_cache"   # the ACTIVE line's autotune= word, as pred says it
    _report.emit(_report.activation_line(rep))
    if rep.get("region"):
        _report.emit(f"{_report.PREFIX} {_modes.region_note(rep['region'])}")
    if rep.get("command"):
        _report.emit(stock_pred.buckets_line(stock_pred.effective_buckets(rep["command"].split())))   # the composed command's effective --buckets
    if (rep.get("xla_pool") or {}).get("applies"):
        _report.emit(_report.xla_pool_note(rep["xla_pool"]))
    wl = _variants.weights_line(rep.get("params"))
    if wl:
        _report.emit(wl)
    _report.emit(_report.line("CARRY", status="ok" if carry["ok"] else "FAILED", files=carry["files"], missing=carry["missing"] or None))
    if a.json:
        print(_report.dump(rep))
    return EXIT_OK if rep["active"] else EXIT_NOT_ACTIVE


def cmd_warm(a: argparse.Namespace, extra: Optional[List[str]] = None) -> int:
    from . import warm as _warm
    mode = _resolve_switch(a.mode, stack.ENV_MODE, "mode", _modes.DEFAULT_MODE)
    variant = _resolve_switch(a.variant, stack.ENV_VARIANT, "variant", None)
    extra = list(extra or [])
    if a.json_path and a.input_dir:
        _report.emit(f"{_report.PREFIX} warm takes at most one of --json_path / --input_dir (rc 2)"); return EXIT_USAGE
    try:
        res = _warm.warm(variant, mode=mode, input_dir=a.input_dir, json_path=a.json_path, n_gpu=_n_gpu(a), model_dir=a.model_dir,
                         n_est=a.n_est, user_args=extra)                 # the memory mode's line: sized from --input_dir as pred sizes it, or --n_est (warm.py, modes.effective_line)
    except ValueError as e:                                               # modes.BigUnsized among them: `big: the program depends on the input size — pass --input_dir or --n_est N`
        _report.emit(f"{_report.PREFIX} {e} (rc 2)"); return EXIT_USAGE
    _report.emit(_warm.summary_line(res))
    return EXIT_OK if res["status"] in ("PASS", "SKIP") else (EXIT_FAIL if res["status"] == "FAIL" else EXIT_NOT_ACTIVE)   # PARTIAL (the mode's levers came back short): exit 3 by name; SKIP: mode off with no --cache_dir keeps a fresh JAX cache per pass — nothing to warm, said by name, not a failure


def main(argv: Optional[List[str]] = None) -> int:
    from ._autoload import TAG
    from ._core_gate import gate
    gate(__file__, tag=TAG)                                               # THE pin gate first (_core_gate.py: a core absent or older than the pin → NOT ACTIVE by name, exit 3); the producer table below is the module-granular second word
    ap = build_parser()
    a, extra = ap.parse_known_args(argv)
    if a.cmd not in ("pred", "check", "warm") and extra:                   # pred, check and warm take the stock command line's own flags verbatim after the package's
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    bad = stack.undeclared_env()
    if bad:
        _report.emit(f"{_report.PREFIX} NOT ACTIVE: undeclared variable(s) {', '.join(bad)} (declared: {', '.join(stack.DECLARED_ENV)}); exit 3"); return EXIT_NOT_ACTIVE
    missing = _modes.missing_producers()                                  # every verb, before anything resolves: an older core is named, never a traceback (rc 3)
    if missing:
        _report.emit(f"{_report.PREFIX} NOT ACTIVE mode={getattr(a, 'mode', None) or os.environ.get(stack.ENV_MODE) or '?'} reason={_modes.producer_missing_reason(missing)}; exit 3"); return EXIT_NOT_ACTIVE
    try:
        if a.cmd == "pred":
            return cmd_pred(a, extra)
        if a.cmd == "check":
            return cmd_check(a, extra)
        if a.cmd == "warm":
            return cmd_warm(a, extra)
    except SystemExit as e:
        if isinstance(e.code, str):
            _report.emit(e.code); return EXIT_USAGE
        raise
    except stack.ActivationError as e:
        _report.emit(f"{_report.PREFIX} {e}"); return EXIT_NOT_ACTIVE
    except FileNotFoundError as e:
        _report.emit(f"{_report.PREFIX} {e}"); return EXIT_NOT_ACTIVE
    return EXIT_USAGE

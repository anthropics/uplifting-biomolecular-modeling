"""`warm`: build this box's ``$CACHE`` (XLA autotune + compilation cache, and the tokamax autotuning table ``$CACHE/tokamax_autotune.json``
when the model process's tokamax.autotune() succeeds — det.TOKAMAX_TABLE; the TOKAMAX line says which) — the first step of the deterministic recipe
(det.py); default inputs are the kit's own ``tests/inputs/`` (WARM_INPUTS). A compiled program's cache key includes device kind and the model config, so warming
with other model-shape flags than the warmed ones leaves its first real process compiling its own (the stock flags stated pass through to the
warm-up process verbatim). Each mode of the kit warms its own class (``exact`` <key>/, ``fast`` its own suffixed dir); ``--mode off`` has nothing to warm
(each pass keeps a fresh JAX cache) unless the caller names a ``--cache_dir``; ``--mode big`` additionally needs a size signal
(``--input_dir`` / ``--json_path`` / ``--n_est``) and refuses (rc 2) without one — a class warmed blind is one pred does not read. ``--json_path X`` warms
exactly the program `pred --json_path X` runs (the file's bucket and seeds)."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from typing import Optional, Sequence

from . import settings as _settings
from . import det, inputs as _inputs, modes as _modes, outputs as _outputs, report as _report, stack, stock_pred
from . import variants as _variants

WARM_INPUTS = os.path.join("tests", "inputs")                            # the default warm-up inputs, under the base kit (stack.kit_home)


def warm(variant: str, mode: str = _modes.DEFAULT_MODE, input_dir: Optional[str] = None, out_dir: Optional[str] = None,
         n_gpu: int = 1, model_dir: Optional[str] = None, n_est: Optional[int] = None,
         user_args: Sequence[str] = (), json_path: Optional[str] = None) -> dict:
    """Build the mode's class in the composition it will run (the mode's row); status PASS | FAIL (the process, its time, the count) |
    PARTIAL (the levers reported short: the mode ran with a subset of its lever set — the class is built all the same; the cli exits 3 by name, a mode
    is all of its levers). The cold class at activation is warm's purpose: it is recorded in the activation (``partial_conditions``), never a refusal. ``n_est``: the caller's ``--n_est`` (the
    memory mode's line, modes.effective_line — else sized from ``input_dir``; a big warm with nothing to size raises modes.BigUnsized). ``user_args``: the caller's stock flags, passed to the warm-up process verbatim."""
    if json_path and input_dir:
        raise ValueError("warm takes at most one of --json_path / --input_dir")
    input_path = os.path.abspath(json_path) if json_path else (os.path.abspath(input_dir) if input_dir else None)   # ``--json_path``: ONE fold input warmed exactly as `pred --json_path` runs it (the file's bucket and seeds)
    line = _modes.effective_line(mode, input_path=input_path, n_est=n_est, n_gpu=n_gpu,
                                 require_size=True)                      # THE resolver, FIRST: the line this input runs under pred (its region, program and cache class) — a big warm with
    user_args = list(user_args or [])
    rep = stack.activate(mode, variant, n_gpu=n_gpu, model_dir=model_dir, stated=_settings.stated(user_args),          # nothing to size is refused by name before
                         region=line["record"], caller_cache_dir=stock_pred.caller_cache_dir(user_args))                          # anything activates; the class activated (and built below) is the one pred reads for the same input
    caller_cache = stock_pred.caller_cache_dir(user_args)
    if mode == "off" and caller_cache is not None:
        rep["cache_dir"] = caller_cache                                      # the caller's own --cache_dir, verbatim (mode off; refused under the kit modes by the cli)
    rep["autotune"] = ("caller_cache_dir" if caller_cache is not None else "skipped(cache_dir empty)") if mode == "off" else "class_cache"   # the ACTIVE line's word, as pred says it
    _report.emit(_report.activation_line(rep))
    if rep.get("region"):                                                # the memory mode: ONE line naming the lever set the region decided inactive and the rule's words (pred's NOTE, the same words)
        _report.emit(f"{_report.PREFIX} {_modes.region_note(rep['region'])}")
    wl = _variants.weights_line(rep.get("params"))
    if wl:
        _report.emit(wl)                                                  # the weights verdict: pinned | NOT PINNED (the warm proceeds either way)
    if not rep["active"]:
        return {"status": "NOT_ACTIVE", "mode": mode, "variant": variant, "reason": rep["reason"], "activation": rep, "region": rep.get("region")}
    eff = rep.get("mode_effective") or mode                              # the mode whose program runs (modes.effective_line: big in region fast IS the fast line) — resolved, indexed and judged as pred does
    cache_dir = rep["cache_dir"]
    if mode == "off" and caller_cache is None:                               # off runs each pass with `--cache_dir=` (a fresh ./jax under the pass's output dir): nothing to warm
        out = {"status": "SKIP", "mode": mode, "variant": variant, "key": rep["key"], "cache_dir": cache_dir, "executables": [],
               "reason": "mode off keeps a fresh JAX cache per pass (--cache_dir= : ./jax under the pass's output directory) — nothing to warm; name a --cache_dir to warm a cache of your own"}
        _report.emit(summary_line(out) + f" reason={out['reason'].split(' — ')[0].replace(' ', '_')}")
        return out
    before = det.cache_digest(cache_dir)
    _report.emit(det.cache_line(cache_dir, "before"))
    tk_before = det.tokamax_table(cache_dir)
    tk_lines = []                                                           # the transcript's tokamax autotune lines (det.TOKAMAX_RX)
    ck_lines = []                                                           # the transcript's CACHEKEY line (det.CACHEKEY_RX)
    input_dir = os.path.abspath(json_path) if json_path else os.path.abspath(input_dir or os.path.join(stack.kit_home(), WARM_INPUTS))   # a file (--json_path) or a directory
    input_flag = f"--json_path={input_dir}" if json_path else f"--input_dir={input_dir}"
    tmp = out_dir or tempfile.mkdtemp(prefix="af3_jax_opt_warm_")
    res = _modes.with_levers_off(_modes.with_n_gpu(_modes.resolve(eff, cache_dir, size=rep.get("region")), rep["n_gpu"] or n_gpu), rep.get("levers_ablated") or [], rep["n_gpu"] or n_gpu)   # pred's composition, call for call (MODEL_OPT_LEVERS_OFF included)
    os.makedirs(cache_dir, exist_ok=True)                                   # the levers' class, or the caller's --cache_dir on off
    if mode != "off":
        from .cli import ensure_fast_script
        ensure_fast_script()
    argv = stock_pred.compose(variant, cache_dir, [f"--output_dir={tmp}", input_flag, *user_args],
                              script=res["script"], mode_flags=res["flags"], launcher=res["launcher"], model_dir=model_dir)
    env = stack.model_process_env(mode_env=res["env"], core_path=bool(res["launcher"]), n_gpu=int(rep.get("n_gpu") or 1))   # a launcher = a lever mode: the core's kernels by path, as pred composes it (cli.py); off: nothing of the tree
    _report.emit(_report.line("COMMAND", argv=" ".join(argv), lever_env=" ".join(f"{k}={v}" for k, v in res["env"].items()) or "none"))
    log_path = os.path.join(tmp, "warm.log")
    lever_lines = []
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        def on_line(ln: str) -> None:
            log.write(ln); log.flush()
            if _modes.is_evidence_line(ln):
                lever_lines.append(ln.rstrip("\n"))
            if det.is_tokamax_line(ln):
                tk_lines.append(ln.rstrip("\n"))
            if det.is_cachekey_line(ln):
                ck_lines.append(ln.rstrip("\n"))
            sys.stderr.write(ln); sys.stderr.flush()
        stack.launched(mode, variant, rep["n_gpu"] or n_gpu, region=(rep.get("region") or {}).get("region"))
        rc = stock_pred.run_logged(argv, env, stack.repo_dir(), on_line)
    wall = time.time() - t0
    after = det.cache_digest(cache_dir)
    evidence = _modes.lever_evidence(eff, lever_lines, res["levers"], n_gpu=int(res.get("n_gpu") or 1))
    execs = det.executables_report(cache_dir)
    _report.emit(det.cache_line(cache_dir, "after", key=det.cache_key_word(ck_lines)))
    _report.emit(det.detclass_line(cache_dir, env))                         # the recipe's carrier inside the class (XLA's per-fusion autotune results) and whether this environment lets it ride: named on every warm
    tokamax = det.tokamax_account(tk_before, det.tokamax_table(cache_dir), tk_lines, argv)
    _report.emit(det.tokamax_line(tokamax))                                 # the class's tokamax autotuning table after the warm-up process: written | loaded | absent (the attempt's fate named)
    n_pred = _outputs.count_predictions(tmp)
    expected = _outputs.expected_predictions(argv, _inputs.seeds_in(input_dir, _outputs.num_seeds(argv)))
    for ln in _report.lever_lines(eff, {**evidence["per_lever"], **_modes.ablated_per_lever(res.get("ablated") or [])}):
        _report.emit(ln)
    failures, partial = [], []                                          # one code per condition (cli): FAIL -> 1, PARTIAL (levers short) -> 3
    if rc != 0:
        failures.append(f"model process exited {rc}")
    if n_pred != expected:
        failures.append(f"predictions={n_pred} expected={expected}")
    if not evidence["ok"]:
        partial.append(f"levers_short: {evidence['reason']}")
    status = "FAIL" if failures else ("PARTIAL" if partial else "PASS")
    out = {"status": status, "mode": mode, "variant": variant, "key": rep["key"], "cache_dir": cache_dir,
           "cache": {"before": before, "after": after}, "cache_key": det.cache_key_word(ck_lines), "tokamax": tokamax, "executables": execs, "predictions": n_pred,
           "predictions_expected": expected, "lever_evidence": evidence, "exit_code": rc, "wall_s": round(wall, 1),
           "log": log_path, "command": argv, "input_dir": input_dir, "activation": rep, "reasons": failures, "partial": bool(partial),
           "partial_conditions": partial, "mode_effective": eff, "region": rep.get("region")}
    if failures or partial:
        out["reason"] = "; ".join(failures + partial)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def summary_line(res: dict) -> str:
    c = res.get("cache") or {}
    after = (c.get("after") or {}).get("sha256", "")[:16] if c else ""
    return (f"{_report.PREFIX} WARM {res.get('status')} mode={res.get('mode')} variant={res.get('variant')} key={res.get('key')} "
            f"cache={res.get('cache_dir')} cache_sha256={after or 'absent'} executables={len(res.get('executables') or [])} "
            + (f"partial={'; '.join(res['partial_conditions'])} " if res.get("partial_conditions") else "")
            + f"predictions={res.get('predictions')} rc={res.get('exit_code')} wall={res.get('wall_s')}s log={res.get('log')}"
            + (f" reason={res['reason']}" if res.get("reason") else ""))

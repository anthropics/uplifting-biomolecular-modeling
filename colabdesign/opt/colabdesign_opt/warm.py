"""`warm` — one design through the mode's own line: BindCraft's bundled example — the case its own `settings_target/PDL1.json` names
(`example/PDL1.pdb`, its chains, its hotspot, the first of its binder `lengths`; read from the vendored tree by `example_case`, nothing typed
here) — or the case named by `--starting-pdb/--chains/--binder-len`, in the mode's subprocess route (`design --mode M`), its outputs under
`<out>/warm/`. Its product is the measured one-time costs — the arm's start-up up to the design
call (`ready`: imports, the settings, BindCraft's modules) and the first call of each stage (`first_calls`: the parameters' load and the
compiles) beside the steady per-step wall; the numbers are the design's `[run]` line read back
(cli.design's `runs`) and `ready` is that line's SETTINGS predecessor's arrival (report.run_lines).

Stock ColabDesign enables no persistent compilation cache (`colabdesign/__init__.py:3-4` sets `XLA_FLAGS` only), so under `off` a warm
process shortens no later process; under the kit-route modes lever `compilecache` (compilecache_jax.py) persists the design step's
executables, so a later process on this machine at the same token count, flags and stack loads them instead of compiling. Result (`run`'s
return value, in memory): {status, mode, levers, tokens, steps, timing:
{ready_s, first_calls_s, phase_steady_s, steady_s, total_s}, files: {name: sha256}, evidence, input: {target, chains, binder_len, hotspot, seed, case},
exit_code, wall_s, out_dir}; the line `WARM PASS|FAIL|NOT ACTIVE mode=... levers=... tokens=... ready=... first_calls=... steady=... total=... rc=... out=...`.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

from opt_core.modes import levers_label

from . import names, outputs, report


class WarmError(RuntimeError):
    pass


def example_case(tree: str) -> dict:
    """BindCraft's bundled example as its own target settings file names it (names.EXAMPLE_SETTINGS under the tree `tree`): {name,
    target, chains, hotspot, binder_len, settings} — `target` = the file's `starting_pdb` by basename under the vendored example/ directory,
    `binder_len` = the first of its `lengths`, `hotspot` = its `target_hotspot_residues` (None when empty)."""
    path = os.path.join(tree, names.EXAMPLE_SETTINGS)
    try:
        with open(path, encoding="utf-8") as fh:
            s = json.load(fh)
        case = {"name": str(s["binder_name"]), "target": os.path.join(tree, names.EXAMPLE_DIR, os.path.basename(str(s["starting_pdb"]))),
                "chains": str(s["chains"]), "hotspot": str(s["target_hotspot_residues"]).strip() or None, "binder_len": int(s["lengths"][0]),
                "settings": names.EXAMPLE_SETTINGS.replace(os.sep, "/")}
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        raise WarmError(f"the bundled example settings {path}: {type(e).__name__}: {e}") from None
    if not os.path.isfile(case["target"]):
        raise WarmError(f"the bundled example target {case['target']} is absent ({case['settings']} starting_pdb)")
    return case


def run(mode: Optional[str], out_dir: str, params_dir: Optional[str], *, target: Optional[str] = None, chain: Optional[str] = None,
        binder_len: Optional[int] = None, seed: int = 0,
        log: Callable[[str], None] = report.log) -> dict:
    """One design of the bundled example (or of `target`/`chain`/`binder_len`) through `design --mode M` in this process; the result dict.
    The exit code is the design's; `partial` / the timing (the design's `[run]` line read back) / the outputs' digests come from the
    design's in-memory record (cli.design_argv) — nothing but the design's outputs and its run.log is written."""
    from . import cli, stack                                        # the design command is the kit's one design line; imported here (cli imports warm)
    out_dir = os.path.abspath(out_dir)
    design_dir = os.path.join(out_dir, "warm")
    os.makedirs(out_dir, exist_ok=True)
    res = {"status": "FAIL", "mode": mode, "levers": None, "out_dir": design_dir, "seed": seed, "settings": None}
    t0 = time.perf_counter()
    if target is None:
        if chain is not None or binder_len is not None:
            raise WarmError("--chains / --binder-len go with --starting-pdb; the bundled example needs none of them")
        case = example_case(stack.tree_home())
        target, chain, binder_len, hotspot = case["target"], case["chains"], case["binder_len"], case["hotspot"]
        log(f"warm case: BindCraft's example ({case['settings']}): {os.path.basename(target)} chains {chain} hotspot {hotspot or 'none'} binder length {binder_len}")
    else:
        if chain is None or binder_len is None:
            raise WarmError("--starting-pdb needs --chains and --binder-len")
        case, hotspot, target = None, None, os.path.abspath(target)
    res["input"] = {"target": target, "chains": chain, "binder_len": int(binder_len), "hotspot": hotspot, "seed": seed, "case": case,
                    "target_sha256": outputs.sha256_file(target) if os.path.isfile(target) else None}
    argv = ["--starting-pdb", target, "--chains", chain, "--binder-len", str(binder_len), "--seed", str(seed), "--out", design_dir, "--binder-name", "warm"]
    if hotspot:
        argv += ["--target-hotspot-residues", hotspot]
    if mode is not None:
        argv += ["--mode", mode]
    if params_dir is not None:
        argv += ["--params-dir", params_dir]
    rec = cli.design_argv(argv)                                    # the design in this process: its record stays in memory, nothing but the outputs and run.log on disk
    rc = rec["exit_code"]
    res["exit_code"] = rc
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    act = rec.get("activation") or {}
    if act:
        verdict = rec.get("verdict") or {}
        res.update({"mode": act.get("mode", mode), "levers": act.get("levers"), "evidence": (rec.get("evidence") or {}).get("state"), "activation_line": report.active_line(act),
                    "gpu": act.get("gpu"), "case": rec.get("case"), "partial": verdict.get("partial") or {}, "gated": verdict.get("gated") or {}})
    runs = rec.get("runs") or []                                   # one design per warm: its [run] line, read back (report.run_lines)
    if runs:
        run_rec = runs[-1]
        t = run_rec.get("timing") or {}
        res["tokens"] = run_rec.get("tokens")
        res["timing"] = {k: t.get(k) for k in ("ready_s", "first_calls_s", "phase_steady_s", "steady_s", "total_s")}
        res["steps"] = run_rec.get("steps")
        res["files"] = {k: v.get("sha256") for k, v in (rec.get("files") or {}).items()}   # the outputs' digests as the design command read them from --out (outputs.files)
    res["status"] = "PASS" if rc == 0 else ("NOT ACTIVE" if rc == cli.EXIT_NOT_ACTIVE else "FAIL")
    if rc == 0 and not res.get("timing"):
        res["status"], res["reason"] = "FAIL", "the design printed no run line"
    log(summary_line(res))
    return res


def _fmt(x, nd=1) -> str:
    if x is None:
        return "none"
    if isinstance(x, (list, tuple)):
        return ",".join(_fmt(v, nd) for v in x)
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def summary_line(res: dict) -> str:
    t = res.get("timing") or {}
    return (f"WARM {res.get('status')} mode={res.get('mode')} levers={levers_label(tuple(res.get('levers') or ()))} tokens={res.get('tokens')} "
            f"ready={_fmt(t.get('ready_s'))} first_calls={_fmt(t.get('first_calls_s'))} steady={_fmt(t.get('steady_s'), 3)} total={_fmt(t.get('total_s'))} "
            f"rc={res.get('exit_code')} out={res.get('out_dir')}" + (f" reason={res['reason']}" if res.get("reason") else "")
            + (f" partial={','.join(res['partial'])}" if res.get("partial") else ""))

"""warm — make a shape ready for `exact`: the weights staged, the features frozen, the P1 files populated — the kit's own
warm-up, once per shape:

  0. `python tools/fetch_public_inputs.py --out <shape dir>/inputs_public --weights` — the kit's own weights fetch + sha256 check
     (`$MOSAIC_CACHE_DIR/boltz/boltz2_conf.ckpt`; a no-op download when present);
  1. the stock arm A (modes.ROWS `A_stock1`): the driver with every lever off and `--features-out <root>/<shape>/features_L<L>_c<N>.npz` —
     one stock design (seed `--seed`, default 0) that freezes the shape's features (P3's input, the recipe's "identical inputs");
  2. the populate row B (`B_p1populate_p2`): P1 dump + P2 + P3 into `<root>/<shape>/xla_cache/` (`xla_autotune_results.pb` + the compilation cache).

Both designs are the driver's own runs, written under `<out>/<A tag>/`, `<out>/<B tag>/` (default `<root>/<shape>/warm/`). Result:
``{"status": PASS|FAIL|NOT ACTIVE, "exit", "steps": [...], "features": {...}, "p1": {...}, "levers_shown_by_driver", "partial", "wall_s", ...}``;
a shape is warm when its features file and its autotune file exist — what `check` reports and `design --mode exact` requires. The
populate row's own manifest is read back like `design`'s: a lever of the row it does not show applied (P1's cache directory, P2's fast
load, P3's frozen features) is a PARTIAL activation — `NOT ACTIVE`, exit 3, unless ``allow_partial`` (recorded either way). The
populate row always starts from an empty P1 directory and a fresh features file: a shape that already holds either is refused
(``AlreadyWarm``).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Callable, Optional

from . import det, driver, inputs, outputs, report, settings, stack
from .modes import resolve

FETCH_RELPATH = os.path.join("tools", "fetch_public_inputs.py")
TAG_A, TAG_B = "warm_A_stock", "warm_B_populate"
TAG_T = "warm_T_{mode}"                                                     # a transparent row's warm-up design (fast: warm_T_fast, big: warm_T_big)
WARM_T_STEPS = (3, 3)                                                      # its stage lengths: the executables (trunk step, both stages, the refold) do not depend on the step counts, so the shortest design that visits every one fills the cache


class AlreadyWarm(ValueError):
    """The shape has a features file or P1 files already: re-warming would re-freeze the features (a new `ref_pos` draw) and populate onto a
    primed cache — a procedure the kit never runs (the populate row starts from an empty directory); removing the shape's directory warms it again."""


def shape_state(root: str, shape: inputs.Shape) -> dict:
    """What the shape already holds under the cache root: its features file and its P1 directory's entries."""
    feats, p1 = inputs.features_path(root, shape), inputs.p1_dir(root, shape)
    entries = sorted(os.listdir(p1)) if os.path.isdir(p1) else []
    return {"features": feats, "features_present": os.path.isfile(feats), "p1_dir": p1, "p1_entries": len(entries), "autotune_present": stack.p1_state(p1)["autotune_present"]}


def stage_weights(shape_dir: str, env: dict, log: Callable[[str], None], timeout: Optional[float]) -> dict:
    """Step 0: the kit's fetch_public_inputs.py --weights (its own sha256 check of the checkpoint)."""
    kit = stack.kit_home()
    cmd = [sys.executable, os.path.join(kit, FETCH_RELPATH), "--out", os.path.join(shape_dir, "inputs_public"), "--weights"]
    lines = []
    from .driver import run_logged
    t0 = time.perf_counter()
    rc, killed = run_logged(cmd, lambda s: (lines.append(s), log(s)), timeout=timeout, cwd=os.path.join(kit, "tools"), env=env)
    prov = next((l for l in lines if l.startswith("provenance:")), None)
    return {"step": "weights", "command": cmd, "exit_code": rc, "timed_out": killed, "wall_s": round(time.perf_counter() - t0, 1), "provenance": prov,
            "sha256_ok": any("sha256_ok" in l and "true" in l.lower() for l in lines)}


def warm(mode: str, shape: inputs.Shape, *, seed: int = 0, out_dir: Optional[str] = None, timeout: Optional[float] = None,
         det_level: int = det.DEFAULT_LEVEL, log: Optional[Callable[[str], None]] = None, allow_partial: bool = False) -> dict:
    """Result: ``status`` PASS | FAIL | NOT ACTIVE and ``exit`` (report.exit_for on the populate row: 0, 1, or 3 when the row's levers are
    not all shown applied by the driver's manifest — ``partial`` — and ``allow_partial`` is not given), the driver's lever evidence
    (``levers_shown_by_driver``), the shape's files."""
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    if mode in stack.MODES and mode != "off" and stack.mode_needs(mode).get("p1_form") == "transparent":
        return warm_transparent(mode, shape, seed=seed, out_dir=out_dir, timeout=timeout, det_level=det_level, log=log)   # fast / big: their compilation cache filled ahead of time by one short design of the row
    if mode not in stack.MODES or not stack.mode_needs(mode)["populate"]:                # a pinned P1 (a populate row: exact) has files to warm
        raise ValueError(f"warm applies to a kit mode whose row carries P1 (exact; fast / big for their compilation cache); got {mode!r}")
    root = stack.cache_root()
    if not root:
        raise stack.ActivationError(f"{stack.ENV_CACHE_ROOT} unset (README \"Variables\"): the P1 files live under it")
    kit = stack.kit_home()
    sdir = inputs.shape_dir(root, shape)
    feats, p1 = inputs.features_path(root, shape), inputs.p1_dir(root, shape)
    st0 = shape_state(root, shape)
    if st0["features_present"] or st0["p1_entries"]:
        raise AlreadyWarm(f"shape {shape.key} under {root} already holds {'its features file' if st0['features_present'] else ''}"
                          f"{' and ' if st0['features_present'] and st0['p1_entries'] else ''}"
                          f"{str(st0['p1_entries']) + ' P1 entries' if st0['p1_entries'] else ''}: the populate row runs from an empty directory "
                          f"— remove {sdir} to warm the shape again")
    os.makedirs(sdir, exist_ok=True); os.makedirs(p1, exist_ok=True)
    out_dir = os.path.abspath(out_dir or os.path.join(sdir, "warm"))
    os.makedirs(out_dir, exist_ok=True)
    eff = settings.effective(kit)                                            # the driver's own step counts: warm runs the rows at the driver's defaults
    t0 = time.perf_counter()
    steps = []
    # step 0: weights (the environment of the stock arm, which reads MOSAIC_CACHE_DIR)
    res_a = resolve("off")
    argv_a, env_a, notes_a = driver.compose(res_a, seed=seed, out_dir=out_dir, tag=TAG_A, shape_flags=shape.flags(), settings_flags=settings.flags(eff),
                                            det_level=det_level, features_out=feats)
    w = stage_weights(sdir, env_a, log, timeout)
    steps.append(w)
    if w["exit_code"] != 0:
        return {"status": "FAIL", "reason": f"weights staging failed (rc={w['exit_code']})", "steps": steps, "wall_s": round(time.perf_counter() - t0, 1), "out_dir": out_dir}
    # step 1: the stock arm A with --features-out
    log(report.activation_line(dict(active=True, mode="off", route="driver", row=res_a.row, row_line=res_a.row, levers_applied=[], levers_unavailable=[], p1={}, upstream=stack.upstream_versions(), gpu=stack.gpu_identity())))
    rc, killed = driver.launch(argv_a, env_a, cwd=notes_a["cwd"], timeout=timeout, on_line=lambda s: log(report.relay(s)))
    steps.append({"step": "stock featurize (arm A)", "tag": TAG_A, "command": argv_a, "exit_code": rc, "timed_out": killed})
    fs = inputs.features_state(feats)
    if rc != 0 or killed or not fs["present"]:
        return {"status": "FAIL", "reason": f"stock arm A failed (rc={rc}, timed_out={killed}, features present={fs['present']})", "steps": steps,
                "features": fs, "wall_s": round(time.perf_counter() - t0, 1), "out_dir": out_dir}
    # step 2: the populate row B
    res_b = resolve(mode, cache_dir=p1, features=feats, features_sha=fs["sha256"], phase="populate")
    argv_b, env_b, notes_b = driver.compose(res_b, seed=seed, out_dir=out_dir, tag=TAG_B, shape_flags=shape.flags(), settings_flags=settings.flags(eff), det_level=det_level)
    log(report.activation_line(dict(active=True, mode=mode, route="driver", row=res_b.row, row_line=res_b.row, levers_applied=list(res_b.levers),
                                    levers_unavailable=[], p1={"cache_dir": p1, "autotune": "dump"}, upstream=stack.upstream_versions(), gpu=stack.gpu_identity())))
    rc, killed = driver.launch(argv_b, env_b, cwd=notes_b["cwd"], timeout=timeout, on_line=lambda s: log(report.relay(s)))
    steps.append({"step": "populate (row B)", "tag": TAG_B, "command": argv_b, "env_exported": notes_b["row_env"], "exit_code": rc, "timed_out": killed})
    st = stack.p1_state(p1)
    results_b = outputs.read_results(out_dir, TAG_B)
    shown = outputs.classify(results_b)                                      # the populate row's own evidence: every lever of the row (P1 dump, P2, P3)
    ran = rc == 0 and not killed and st["autotune_present"] and results_b is not None
    missing = [k for k in res_b.levers if not shown.get(k)] if ran else []
    act = {"active": True, "mode": mode, "route": "driver", "row": res_b.row, "levers_applied": list(res_b.levers), "levers_fallback": missing, "partial": bool(missing)}
    code = report.exit_for(rc, not ran, act, allow_partial, what=f"populate row {TAG_B}, see {outputs.results_path(out_dir, TAG_B)}")
    res = {"status": {report.EXIT_OK: "PASS", report.EXIT_NOT_ACTIVE: "NOT ACTIVE"}.get(code, "FAIL"), "exit": code, "mode": mode,
           "shape": shape.record(),
           "steps": steps, "features": fs, "p1": st, "levers_shown_by_driver": shown, "partial": missing, "allow_partial": bool(allow_partial),
           "wall_s": round(time.perf_counter() - t0, 1), "out_dir": out_dir, "shape_dir": sdir}
    if not ran:
        res["reason"] = f"populate row failed (rc={rc}, timed_out={killed}, autotune present={st['autotune_present']}, results present={results_b is not None})"
    return res


def warm_transparent(mode: str, shape: inputs.Shape, *, seed: int = 0, out_dir: Optional[str] = None, timeout: Optional[float] = None,
                     det_level: int = det.DEFAULT_LEVEL, log: Optional[Callable[[str], None]] = None) -> dict:
    """`warm --mode fast|big`: the row's compilation cache for the shape (`<root>/<shape>/xla_cache_<mode>/`, P1's transparent form) filled
    AHEAD of time by ONE short design of the row itself (`--steps1 3 --steps2 3`: every executable the design compiles — the trunk step of
    both stages and the refold — is shape-keyed, not step-count-keyed), so the first real design of the shape starts warm. Nothing is
    frozen or pinned (no features file, no autotune results: those are `exact`'s); an already-filled directory is not an error — the run
    reads what is there and adds what is missing. Result as `warm()`: ``status`` PASS | FAIL, ``exit``, ``p1`` (entries before / after)."""
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    root = stack.cache_root()
    if not root:
        raise stack.ActivationError(f"{stack.ENV_CACHE_ROOT} unset (README \"Variables\"): the {mode} row's compilation cache lives under it — nothing to warm without it")
    kit = stack.kit_home()
    sdir = inputs.shape_dir(root, shape)
    p1 = inputs.p1_dir(root, shape, mode)
    why = stack.transparent_dir_unusable(p1)
    if why:
        raise stack.ActivationError(why)
    out_dir = os.path.abspath(out_dir or os.path.join(sdir, f"warm_{mode}"))
    os.makedirs(out_dir, exist_ok=True)
    tag = TAG_T.format(mode=mode)
    eff = settings.effective(kit, steps1=WARM_T_STEPS[0], steps2=WARM_T_STEPS[1])
    t0 = time.perf_counter()
    before = stack.p1_state(p1, "transparent")
    res_t = resolve(mode, cache_dir=p1)
    argv, env, notes = driver.compose(res_t, seed=seed, out_dir=out_dir, tag=tag, shape_flags=shape.flags(), settings_flags=settings.flags(eff), det_level=det_level)
    w = stage_weights(sdir, env, log, timeout)                              # step 0 as `exact`'s: the checkpoint staged and sha256-checked once
    steps = [w]
    if w["exit_code"] != 0:
        return {"status": "FAIL", "exit": report.EXIT_FAIL, "mode": mode, "shape": shape.record(), "reason": f"weights staging failed (rc={w['exit_code']})", "steps": steps,
                "p1": before, "wall_s": round(time.perf_counter() - t0, 1), "out_dir": out_dir, "shape_dir": sdir}
    log(report.activation_line(dict(active=True, mode=mode, route="driver", row=res_t.row, row_line=res_t.row + f"[{','.join(res_t.levers)}]", levers_applied=list(res_t.levers),
                                    levers_unavailable=[], p1=before, upstream=stack.upstream_versions(), gpu=stack.gpu_identity(),
                                    notes=[f"warm-up design of the {mode} row at --steps1 {WARM_T_STEPS[0]} --steps2 {WARM_T_STEPS[1]}: fills {p1}"])))
    rc, killed = driver.launch(argv, env, cwd=notes["cwd"], timeout=timeout, on_line=lambda s: log(report.relay(s)))
    steps.append({"step": f"warm-up design (row {res_t.row})", "tag": tag, "command": argv, "env_exported": notes["row_env"], "exit_code": rc, "timed_out": killed})
    after = stack.p1_state(p1, "transparent")
    results_t = outputs.read_results(out_dir, tag)
    ran = rc == 0 and not killed and results_t is not None and after["n_cache_entries"] > 0
    res = {"status": "PASS" if ran else "FAIL", "exit": report.EXIT_OK if ran else report.EXIT_FAIL, "mode": mode, "shape": shape.record(), "steps": steps,
           "features": None, "p1": dict(after, entries_before=before["n_cache_entries"]), "levers_shown_by_driver": outputs.classify(results_t) if results_t else {},
           "partial": [], "allow_partial": False, "wall_s": round(time.perf_counter() - t0, 1), "out_dir": out_dir, "shape_dir": sdir}
    if not ran:
        res["reason"] = f"warm-up design failed (rc={rc}, timed_out={killed}, results present={results_t is not None}, cache entries={after['n_cache_entries']})"
    return res


def summary_line(res: dict) -> str:
    st, fs = res.get("p1") or {}, res.get("features") or {}
    feats = ("present" if fs.get("present") else "absent") if res.get("features") is not None or st.get("form") != "transparent" else "n/a"
    auto = "off" if st.get("form") == "transparent" else ("present" if st.get("autotune_present") else "absent")
    return (f"{report.PREFIX} WARM {res.get('status')} mode={res.get('mode')} shape={(res.get('shape') or {}).get('key')} features={feats} "
            f"autotune={auto} cache_entries={st.get('n_cache_entries')}" + (f" entries_before={st['entries_before']}" if "entries_before" in st else "") + f" wall={res.get('wall_s')}s"
            + report.partial_field({"partial": bool(res.get("partial")), "levers_fallback": res.get("partial"), "allow_partial": res.get("allow_partial")})
            + f" out={res.get('out_dir')}" + (f" reason={res['reason']}" if res.get("reason") else ""))

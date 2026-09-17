"""``pred``: one ``rf3 fold`` per (input file, seed), on the interpreter the mode names, with the mode's environment row.

Kit modes run the upstream CLI on the PATCHED interpreter (``stack.opt_python()``; state ``patched`` proved by sha
first) with the row exported at the process boundary — the add-on's own command line, e.g. ``RF3_CUDAGRAPH=1 RF3_HOIST=1 rf3 fold ...``
— plus ``ROSETTAFOLD3_OPT=<mode>`` so the child's own hook (_autoload.py) proves the tree, prints the ACTIVE / APPLIED / EXIT lines
and writes the exit tally ``pred`` reads for its verdict. ``off`` is the stock caller (stock_fold.py) on the pristine interpreter. The
file set per seed is the CLI's own.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Sequence

from . import _core
from . import leversoff as _leversoff
from . import big as _big
from . import weights as _weights
from . import report as _report
from . import settings as _settings
from . import stack as _stack
from . import stock_fold as _stock
from . import tree as _tree
from .modes import ENV, reach_env, resolve


class NotActive(RuntimeError):
    """A refusal BEFORE any fold process started (the ``--n_gpu`` rules): the CLI maps it to exit 3, the NOT ACTIVE line is printed once."""


def kit_env(res, environ=None, tally_file: Optional[str] = None,
            n_gpu: int = 1, allow_partial: bool = False) -> Dict[str, str]:
    """The patched child's environment: the row's switches, ROSETTAFOLD3_OPT=<row>, the tally file, the
    memory mode's ``--n_gpu`` (``ROSETTAFOLD3_OPT_N_GPU``: the hook in the fold process gates and reports the same P) and its partial-unit
    opt-out (``big.ALLOW_PARTIAL_ENV``, set from ``--allow-partial`` only: a value in the caller's environment is not passed on)."""
    env = dict(os.environ if environ is None else environ)
    env.pop(_big.ALLOW_PARTIAL_ENV, None)
    if allow_partial:
        env[_big.ALLOW_PARTIAL_ENV] = "1"
    conflicts = _stack.env_conflicts(res, env)
    if conflicts:
        raise RuntimeError("environment contradicts row %s: %s" % (res.row, ", ".join(f"{k}={v!r}" for k, v in conflicts.items())))
    env.update(res.switches)
    env.update(reach_env(n_gpu))                      # the row-sharded line's ROWPAIR_* levers at --n_gpu > 1 (modes.TP_ENV)
    env[ENV] = res.row
    env["PYTHONDONTWRITEBYTECODE"] = env.get("PYTHONDONTWRITEBYTECODE", "1")
    if tally_file:
        env[_stack.ENV_TALLY_FILE] = tally_file
    env[_stack.ENV_N_GPU] = str(int(n_gpu))
    return env


def launch(cmd: List[str], env: Dict[str, str], n_gpu: int, *, stdout=None) -> int:
    """Start ONE fold: ``n_gpu == 1`` — one child process (one interpreter, one GPU; nothing of the family launcher is imported);
    ``n_gpu == P > 1`` — the P rank processes of the same command through the row-sharding adapter's launcher (``rowpair.launch_argv`` →
    ``opt_core.mem.rowpair.launch.run_rank_processes``: rank r bound to visible device r, a bounded rendezvous, any rank's failure tears the
    group down and is ONE named line; ranks > 0 write under ``<out_dir>.ranks/rank<r>``, rank 0's directory is the run's output and its
    transcript streams into ``stdout``). One seed on every rank: the command's ``seed=<int>``, or one drawn here and printed when the command has none. Returns 0 when every rank returned 0, else the failing rank's code. Users never type torchrun."""
    if int(n_gpu) == 1:
        r = subprocess.run(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT if stdout else None)
        return r.returncode
    if cmd_seed(cmd) is None:                                   # the ranks must fold under ONE seed (per-rank MSA subsampling / noise draws would fold a chimera):
        seed = random.SystemRandom().randrange(2 ** 31)          # unseeded (stock's default), one seed is drawn here for every rank and named on the line
        cmd = list(cmd) + [f"seed={seed}"]
        print(f"{_report.PREFIX} n_gpu={int(n_gpu)} seed={seed} drawn=1 (one seed for every rank; pass seed=<s> to choose it)", file=sys.stderr, flush=True)
    from . import rowpair as _rowpair
    runner, kwargs = _rowpair.launch_argv(cmd, env, int(n_gpu))
    return int(runner(stdout=stdout, **kwargs))


def seed_dir(out_dir: str, seed: Optional[int]) -> str:
    """Where one fold's outputs go: ``<out_dir>/seed-<s>/`` for a seed of ``--seeds``, ``<out_dir>/`` itself for the one unseeded fold ``pred`` runs
    without ``--seeds`` (``rf3 fold`` without ``seed=``: upstream's default, ``seed: null``)."""
    return _stock._seed_dir(out_dir, seed)


def seed_label(seed: Optional[int]) -> str:
    """``seed <s>`` / ``unseeded`` — how the verdict's failure clauses name a fold."""
    return f"seed {seed}" if seed is not None else "unseeded"


def cmd_seed(cmd: List[str]) -> Optional[int]:
    """The ``seed=<int>`` hydra token of a fold command line (``settings.overrides`` writes one per seed of ``pred``); None when absent or not an int."""
    for t in cmd:
        if isinstance(t, str) and t.startswith("seed="):
            try:
                return int(t[len("seed="):])
            except ValueError:
                return None
    return None


def run_kit(mode: str, *, inputs: str, out_dir: str, ckpt: str, overrides: Sequence[str] = (), seeds: Sequence[Optional[int]] = (),
            log_path: Optional[str] = None, n_gpu: int = 1, allow_partial: bool = False) -> dict:
    res = resolve(mode, _stack.kit_home(), fpf_home=_stack.fpf_home())
    rep_gate = _stack.n_gpu_gate(res.mode, n_gpu)               # --n_gpu P: big only, P GPUs visible, the adapter present (opt_core.mem.ngpu words; ActivationError)
    n_gpu = rep_gate["n_gpu"]
    python = _stack.opt_python()
    lists = _stack.tree_digests()
    ts = _tree.state_of(python, lists)
    _tree.expect(ts, res.tree_state)
    print(f"{_report.PREFIX} TREE {ts.line()}", file=sys.stderr, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    runs = []
    tallies = {}
    rank_blocks: Dict[str, Dict[int, dict]] = {}           # n_gpu > 1: seed -> {rank r > 0: that rank's exit tally} (rank 0's is tallies[seed])
    rank_census: Dict[str, list] = {}
    for s in seeds:
        od = seed_dir(out_dir, s)
        tally_file = os.path.join(out_dir, f".tally_seed-{s}.json" if s is not None else ".tally_unseeded.json")
        env = kit_env(res, tally_file=tally_file, n_gpu=n_gpu, allow_partial=allow_partial)
        cmd = _stock.command(python, inputs, od, ckpt, overrides, s)          # the same argv the stock route runs, on the patched interpreter
        t0 = time.time()
        with (open(log_path, "a", encoding="utf-8") if log_path else open(os.devnull, "w")) as lf:
            lf.write("$ " + " ".join(f"{k}={v}" for k, v in res.switches.items()) + f" {ENV}={res.row} {_stack.ENV_N_GPU}={n_gpu} " + " ".join(cmd) + "\n")
            lf.flush()
            rc = launch(cmd, env, n_gpu, stdout=lf if log_path else None)
        runs.append({"seed": s, "cmd": cmd, "rc": rc, "wall_s": round(time.time() - t0, 1), "out_dir": od, "n_gpu": n_gpu})
        if os.path.exists(tally_file):
            try:
                tallies[str(s)] = json.load(open(tally_file, "r", encoding="utf-8"))
            finally:
                os.remove(tally_file)
        rank_tallies = sorted(p for p in os.listdir(out_dir) if p.startswith(os.path.basename(tally_file) + ".rank"))
        for p in rank_tallies:                              # ranks > 0 write their exit tallies beside rank 0's (report.register_exit_tally); read (the all-rank lever verdict), counted, then removed
            try:
                rank_blocks.setdefault(str(s), {})[_rank_of(p)] = json.load(open(os.path.join(out_dir, p), "r", encoding="utf-8"))
            except (OSError, ValueError):
                pass                                        # an unreadable rank tally is a missing one: the verdict names the rank
            os.remove(os.path.join(out_dir, p))
        runs[-1]["ranks_reported"] = (1 if str(s) in tallies else 0) + len(rank_tallies)
        if rc != 0:
            break
    ok = all(x["rc"] == 0 for x in runs) and len(runs) == len(seeds)
    failures = [f"{seed_label(x['seed'])}: rc={x['rc']}" for x in runs if x["rc"] != 0]
    mismatch = n_gpu_failures(n_gpu, runs, tallies)             # fail-closed: the P the fold processes REPORT (exit tally n_gpu, one tally per rank) equals the P requested
    if mismatch:
        for m in mismatch:
            print(f"{_report.PREFIX} NOT ACTIVE: reason={m}", file=sys.stderr, flush=True)
        failures += mismatch
        ok = False
    if res.fpf_arm:
        failures += fpf_failures(seeds, runs, tallies)     # a fast run is a run only when its kernels acted (report.fpf_tally ok; on the big row a kernel silent above its size gate is a named disengagement, not a failure)
    for name in ("xtr", "dtk", "mkdit", "confhoist", "confln"):                      # the package levers with a verdict of their own (pf / dtk / mkdit describe ok): a core refusal at launch, a dead seam, a fallback, a recorded error or a displaced alias fails the run by name
        if name not in res.levers:
            continue
        if int(n_gpu) == 1:
            failures += lever_failures(name, runs, tallies)
        else:                                               # n_gpu > 1: EVERY rank's tally is read (the model runs in the rank processes; the lever is judged where it ran or was declined by name)
            bad, records = lever_rank_failures(name, runs, tallies, rank_blocks, int(n_gpu))
            failures += bad
            rank_census[name] = records
            for rec in records:
                print(f"{_report.PREFIX} RANKS " + " ".join(f"{k}={v}" for k, v in rec.items()), file=sys.stderr, flush=True)
    if res.row == _stack.BIG_MODE:
        failures += big_failures(seeds, runs, tallies)   # a big run is a run only when its census closed (big.state exit verdict 0)
    ok = ok and not failures                                # every row: a lever of the mode that did not engage (no kernel row for this GPU, a dead seam, a silent lever)
    return {"status": "PASS" if ok else "FAIL", "runs": runs, "tree": ts.line(), "tree_state": ts.state, "python": python, "row": res.row,
            "switches": dict(res.switches), "tallies": tallies, "fpf_arm": res.fpf_arm, "failures": failures,
            "n_gpu": n_gpu, "n_gpu_mismatch": bool(mismatch), **({"rank_levers": rank_census} if rank_census else {})}


def n_gpu_failures(n_gpu: int, runs: List[dict], tallies: Dict[str, dict]) -> List[str]:
    """Fail-closed accounting of the resource axis: for every seed that ran, the P its fold process reported at exit (the tally's
    ``n_gpu``, written by the hook in that process) and the number of rank processes that reported (rank 0's tally + one ``.rank<r>``
    tally per rank > 0) must equal the P ``pred`` was asked for — ``n_gpu_mismatch requested=P active=Q ranks_reported=R`` otherwise
    (a fold on fewer GPUs than requested is never a pass)."""
    out = []
    for x in runs:
        if x["rc"] != 0:
            continue
        t = tallies.get(str(x["seed"])) or {}
        active = int(t.get("n_gpu") or 0)
        reported = int(x.get("ranks_reported") or 0)
        if active != int(n_gpu) or reported != int(n_gpu):
            out.append(f"n_gpu_mismatch requested={int(n_gpu)} active={active} ranks_reported={reported} (seed {x['seed']})")
    return out


def lever_failures(name: str, runs: List[dict], tallies: Dict[str, dict]) -> List[str]:
    """Total accounting of a package lever with a verdict of its own (``dtk``, ``mkdit``): every seed that ran must carry the lever's tally
    block with ``ok``; a missing block or a false verdict is a named failure."""
    out = []
    for x in runs:
        if x["rc"] != 0:
            continue
        t = (tallies.get(str(x["seed"])) or {}).get(name)
        if not t:
            out.append(f"{seed_label(x['seed'])}: no {name} tally (the lever's counters were not written at exit)")
        elif not t.get("ok"):
            out.append(f"{seed_label(x['seed'])}: {name} {t.get('reason') or 'not ok'}")
    return out


def _rank_of(tally_name: str) -> int:
    """``.tally_seed-<s>.json.rank<r>`` -> r."""
    return int(tally_name.rsplit(".rank", 1)[1])


LEVER_CENSUS_KEYS = ("calls", "served", "gated", "fallback")   # the gate counters summed over ranks on the RANKS line (a lever without a census contributes 0)


def lever_rank_failures(name: str, runs: List[dict], tallies: Dict[str, dict], rank_blocks: Dict[str, Dict[int, dict]],
                        n_gpu: int) -> Tuple[List[str], List[dict]]:
    """``n_gpu > 1``: the all-rank verdict of a package lever. For every seed that ran, EVERY rank's exit tally (rank 0's = ``tallies[seed]``,
    rank r's = ``rank_blocks[seed][r]``) must carry the lever's block, and each block must be either declined by name (``conflict``: the
    row-sharding adapter owns the lever's site under ``n_gpu > 1`` — rowpair.CONFLICTS) or on with a clean verdict of its own (``ok``: for dtk
    that is calls >= 1 and no fallback, dtk.problems); every rank in the SAME state. Anything else is a named failure (``seed s rank r: ...``).
    Returns ``(failures, records)`` — one record per seed for the RANKS line: lever, seed, n_gpu, ranks_reported, state, the summed counters
    (``calls served gated fallback``) and the per-rank calls.
    """
    failures: List[str] = []
    records: List[dict] = []
    for x in runs:
        if x["rc"] != 0:
            continue
        s = str(x["seed"])
        blocks = {0: (tallies.get(s) or {}).get(name)}
        for r, t in sorted((rank_blocks.get(s) or {}).items()):
            blocks[int(r)] = (t or {}).get(name)
        states = {}
        sums = {k: 0 for k in LEVER_CENSUS_KEYS}
        per_rank = []
        for r in range(int(n_gpu)):
            b = blocks.get(r)
            if not b:
                failures.append(f"{seed_label(x['seed'])} rank {r}: no {name} tally (the lever's block was not written at that rank's exit)")
                states[r] = "missing"
                per_rank.append("-")
                continue
            c = b.get("census") or {}
            for k in LEVER_CENSUS_KEYS:
                sums[k] += int(c.get(k) or 0)
            per_rank.append(str(int(c.get("calls") or 0)))
            if b.get("conflict"):
                states[r] = f"off:conflict:{b['conflict']}"
            elif b.get("on") and b.get("ok"):
                states[r] = "on"
            else:
                states[r] = "fail"
                failures.append(f"{seed_label(x['seed'])} rank {r}: {name} {b.get('reason') or 'not ok'}")
        kinds = sorted(set(v for v in states.values() if v not in ("missing", "fail")))
        if len(kinds) > 1:
            failures.append(f"{seed_label(x['seed'])}: {name} ranks disagree ({', '.join(f'rank {r}={v}' for r, v in sorted(states.items()))})")
        records.append({"lever": name, "seed": s, "n_gpu": int(n_gpu), "ranks_reported": sum(1 for r in range(int(n_gpu)) if blocks.get(r)),
                        "state": (kinds[0] if len(kinds) == 1 and len(states) == int(n_gpu) and all(v == kinds[0] for v in states.values()) else "mixed"),
                        **sums, "per_rank_calls": ",".join(per_rank)})
    return failures, records


def fpf_failures(seeds: List[int], runs: List[dict], tallies: Dict[str, dict]) -> List[str]:
    """Total accounting of an FPF arm's seeds: every seed that ran must carry an adapter tally with ``ok`` (report.fpf_tally); a
    missing tally or a false verdict is a named failure."""
    out = []
    for x in runs:
        if x["rc"] != 0:
            continue
        t = (tallies.get(str(x["seed"])) or {}).get("fpf")
        if not t:
            out.append(f"{seed_label(x['seed'])}: no FPF tally (the adapter's counters were not written at exit)")
        elif not t.get("ok"):
            out.append(f"{seed_label(x['seed'])}: FPF {t.get('reason') or 'not ok'}")
    return out


def _no_rollout(t: dict) -> Optional[str]:
    """The fold process's upstream census (report.tally 'upstream'): ``zero_items`` | ``early_stop`` when nothing rolled out, else None."""
    from . import upstream as _upstream
    up = (t or {}).get("upstream")
    return _upstream.no_rollout_reason(up) if isinstance(up, dict) else None


def big_failures(seeds: List[int], runs: List[dict], tallies: Dict[str, dict]) -> List[str]:
    """Total accounting of the big mode's seeds: every seed that ran must carry the big block of the tally (big.state) with the
    exit verdict 0 — every expected lever applied, every unit's census complete (a refused lever, a fallback, a unit without the
    lever's mark, or a process where the trigger never fired is a named failure; the --allow-partial opt-out is recorded)."""
    out = []
    for x in runs:
        if x["rc"] != 0:
            continue
        b = (tallies.get(str(x["seed"])) or {}).get("big")
        if not b:
            out.append(f"{seed_label(x['seed'])}: no big tally (the mode's record was not written at exit)")
            continue
        v = b.get("exit") or {}
        why = _no_rollout(tallies.get(str(x["seed"])) or {})
        if v.get("exit_code", 1) != 0 and why and not int(b.get("units") or 0) > 0 and not (b.get("census") or {}).get("n_partial"):
            continue                                        # upstream predicted nothing / early-stopped every input: the mode's seams were never reached (named on the EXIT line), not a failure
        if v.get("exit_code", 1) != 0:
            reasons = v.get("reasons") or {}
            out.append(f"{seed_label(x['seed'])}: big exit {v.get('exit_code')} partial={','.join(v.get('partial') or [])} "
                       + "; ".join(f"{k}: {r}" for k, r in reasons.items()))
    return out


def run(mode: str, *, inputs: str, out_dir: str, ckpt: str, overrides: Sequence[str] = (), seeds: Sequence[Optional[int]] = (),
        log_path: Optional[str] = None, n_gpu: int = 1, allow_partial: bool = False) -> dict:
    """Dispatch on the mode; return the run record (``status`` PASS|FAIL). ``n_gpu``: the memory mode's resource
    axis (``big`` only; refused by name elsewhere, in the words of ``opt_core.mem.ngpu``)."""
    pins = _stack.pins()
    ckpt = ckpt or _stack.checkpoint_path()
    if not ckpt:
        raise ValueError(f"no checkpoint: pass --ckpt or set {_stack.ENV_CKPT} (stock/PINS.json weights)")
    if not os.path.isfile(ckpt):
        raise ValueError(f"checkpoint not found: {ckpt}")
    _weights.announce(ckpt, pins, verb="pred")                  # WEIGHTS checkpoint=pinned|unknown: an unknown checkpoint is a named warning; the run proceeds in every mode
    overrides = _settings.check(overrides)                  # rf3 fold's own key=value tokens, verbatim; the kit's four keys refused by name
    try:
        res = resolve(mode, _stack.kit_home(), fpf_home=_stack.fpf_home())
    except _leversoff.WithholdError as e:                        # MODEL_OPT_LEVERS_OFF refused by name before any interpreter starts: the NOT ACTIVE line, rc 3
        print(f"{_report.PREFIX} NOT ACTIVE: {e}", file=sys.stderr, flush=True)
        raise NotActive(str(e)) from e
    try:
        _stack.n_gpu_gate(res.mode, n_gpu)                    # refused by name before any interpreter starts (off / exact / fast with P > 1; P > visible)
    except _stack.ActivationError as e:
        print(f"{_report.PREFIX} NOT ACTIVE: {e}", file=sys.stderr, flush=True)
        raise NotActive(str(e)) from e
    if res.row == "off":
        lists = _stack.tree_digests()
        python = _stack.stock_python()
        rec = _stock.run(python, pins, lists, _stack.opt_root(), inputs=inputs, out_dir=out_dir, ckpt=ckpt, overrides=overrides, seeds=seeds,
                         log_path=log_path)
    else:
        rec = run_kit(mode, inputs=inputs, out_dir=out_dir, ckpt=ckpt, overrides=overrides, seeds=seeds,
                      log_path=log_path, n_gpu=n_gpu, allow_partial=allow_partial)
    rec["settings"] = _settings.describe(overrides)
    return rec


FAIL_ESCAPE = "a lever of the mode did not engage on this GPU / stack: the run is refused, exit 1; `--mode off` runs stock"   # the one escape a FAIL verdict names


def summary_line(rec: dict) -> str:
    walls = ",".join(str(r["wall_s"]) for r in rec["runs"])
    fails = ("; ".join(rec["failures"])) if rec.get("failures") else ""
    return (f"{_report.PREFIX} pred {rec['status']} row={rec.get('row', 'off')} runs={len(rec['runs'])} wall_s={walls} tree={rec['tree_state']}"
            + (f" fpf={rec['fpf_arm']}" if rec.get("fpf_arm") else "")
            + (f" failures={fails}" if fails else "")
            + (f" ({FAIL_ESCAPE})" if fails and rec.get("status") == "FAIL" and rec.get("row", "off") != "off" else ""))

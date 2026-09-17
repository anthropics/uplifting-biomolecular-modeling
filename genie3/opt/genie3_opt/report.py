"""Observability for genie3_opt: the activation line, the run line, the evidence line, the ready line and the exit
tally (all on stderr, prefixed ``[genie3-opt]``).

* activation — ``ACTIVE mode=<m> attach=driver tier=<1|2> genie3=<v> stack=<torch/lightning/triton> gpu=<name(smNN,key)> levers=<ids: the effective set at the request's batch size and the pass's --trimul>
  line=<the kit line>`` (or ``DRY-RUN ...`` / ``NOT ACTIVE: <reason>``), formatted from the activation report `genie3_opt.enable()` /
  `status()` return;
* refused — ``NOT ACTIVE: mode=<m> cannot serve <features> — refused by name (exit 3), nothing ran: <feature: mechanism; …>; run --mode off for
  the stock path`` (design.refused_line) when a request names a computation the kit line does not run (design.kit_refusals: upstream's beam
  search, the sidechain stage, more than one device); printed before anything runs;
* lever skipped — ``LEVER lever=<id> state=skipped reason=<why> mode=<m> served_by=module_forward`` when ONE planned lever stood down for the
  request by its declared gate (registry.Lever.declined: L7 under the kernel's token floor) — the pass ran, the lever's calls took the module's
  own forward (lever_skipped_line);
* run — ``RUN mode=<m> problems=<n> designs=<n> seed=<s|unseeded> batch=<B|request> precision=<fp32|tf32> out=<dir> [shard=<K>/<M>]`` once per
  request pass (``shard=`` under --num-shards M --shard-id K: ``designs`` is then the shard's share), then
  ``EVIDENCE levers=<applied> missing=<ids> forbidden=<n>`` from the driver's log and timings JSON (registry evidence), then
  ``NUMERICS mode=<m> line=<fp32|tf32> matmul_tf32=<bool> matmul=<word> cudnn_tf32=<bool> batch=<ran>/<asked> verdict=<ok|FAIL: reason>`` —
  the driver's own numerics readback judged against the line (design.numerics_verdict: an fp32 line reads TF32 off, the tf32 line on);
* capacity — ``CAPACITY mode=<m> batch=<B> n_token=<N> need_gb=<x> verdict=refused suggest_batch=<b>`` after a driver pass the driver ended by
  refusing a batch before running it (its memory model against the device; the pass is `failed`, the driver's exit 5);
* kernels — ``KERNELS route=stock tf32_matmul=<bool> cudnn_tf32=default float32_matmul_precision=default torch=<version> overrides=<none|NAME,…>
  source=environment`` before a stock pass: the child's matmul numerics as its stripped environment leaves them (design.kernels_census; no probe
  process runs);
* ready — ``ready mode=<m> t=<s> pass=<tag> first=<problem>/pdbs/<name>.pdb`` once per child process that wrote a design: the seconds
* stock — ``STOCK proof=ok|FAIL pinned=… files_checked=… git_head=… modified=… forbidden_env=… kit_modules=… kit_dirs=… autoload=… torch=… genie3=…`` once per
  stock pass, the caller's proof read back (stock_cli.py env_proof + the pins report);
  from that process's start to its first written PDB (design.first_design) — on the stock line the cold start (start-up, model load,
  the first design); on the driver line start-up, model set-up, featurisation and the first batch (fast) / the first design (exact) —
  the driver writes each batch's PDBs as the batch completes (g3batch.py write_batch); the driver's own ``model_setup_s`` /
  first ``graph_capture_s`` are the cold-start figures there (``opt_manifest.json`` ``timings``);
* exit — the driver's own final figures of the pass, read from its timings JSON (``n_designs``, ``sampling_loop_s``,
  ``s_per_design_sampling``, ``graph_capture_s`` count, ``D59_global_state_checks_passed``, ``rng_final_state_matches_stock_sampler``),
  printed at interpreter exit of the `design` process; when no driver ran (stock, or nothing ran) the tally says so.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Dict, List, Optional

from .codes import TAG

PREFIX = f"[{TAG}]"
TALLY_KEYS = ("n_designs", "model_setup_s", "featurize_s", "sampling_loop_s", "s_per_design_sampling", "total_s", "D59_global_state_checks_passed",
              "rng_final_state_matches_stock_sampler", "shard", "matmul_precision", "seed")

_TALLY: Dict[str, object] = {"sources": [], "registered": False, "printed": False}


def emit(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


REFUSED_LEVERS = PREFIX + " NOT ACTIVE: mode={mode} refused — lever(s) {levers} could not run on this box (their LEVER / KERNELS / driver lines above say why); a mode is all of its levers, never a subset under its name: exit 3 (the files this pass wrote are not mode {mode}'s; --mode off runs stock)"
FORBIDDEN_HEAD, FORBIDDEN_TAIL = f"{PREFIX} FAILED: ", " forbidden driver-log line(s) (a traceback, an assertion, an RNG-bookkeeping or graph-replay divergence: registry FORBIDDEN_LINES) — the outputs are not trusted; exit 1"


NOTE_HEAD = f"{PREFIX} NOTE "


def note_line(text: str) -> str:
    """A named fact the activation NOTES and proceeds on (never a refusal): one line, `[genie3-opt] NOTE <fact>` (stack.hardware_notes)."""
    return NOTE_HEAD + str(text)


def levers_refused_line(mode: str, levers: str) -> str:
    """The mode's refusal after a pass whose records lack a planned lever's evidence: ``[genie3-opt] NOT ACTIVE: mode=<m> refused — lever(s) <ids>
    could not run on this box (…); a mode is all of its levers, never a subset under its name: exit 3 (…)`` — `levers` = the ids in registry order
    (``L7``), as the EVIDENCE line lists them; printed once by design.run. A lever that cannot run (a kernel that will not compile or launch, an
    unsupported shape) refuses the mode by name; uncertainty about the environment alone never does (the levers engage and the lines name it)."""
    return REFUSED_LEVERS.format(mode=mode, levers=levers)


BROKEN_HEAD, BROKEN_TAIL = f"{PREFIX} FAILED: lever(s) ", " ran but their own record contradicts the lever's contract (a judged predicate: registry.Lever.evidence) — the outputs are not trusted; exit 1"


def broken_line(detail: str) -> str:
    """``[genie3-opt] FAILED: lever(s) <ids> ran but their own record contradicts the lever's contract (…) — the outputs are not trusted; exit 1``:
    a planned lever whose `judged` evidence predicate fails (L8's hoist anomalies, L11's batch table) ran and is broken — the pass failed (1),
    distinct from a lever that could not run (the mode refuses by name, 3)."""
    return f"{BROKEN_HEAD}{detail}{BROKEN_TAIL}"


def forbidden_line(n: int) -> str:
    """``[genie3-opt] FAILED: <n> forbidden driver-log line(s) (…) — the outputs are not trusted; exit 1``: a registry.FORBIDDEN_LINES match voids the pass."""
    return f"{FORBIDDEN_HEAD}{n}{FORBIDDEN_TAIL}"


def _join(items) -> str:
    return ",".join(str(x) for x in items) if items else "none"


def gpu_label(gpu) -> str:
    if not isinstance(gpu, dict) or not gpu.get("name"):
        return "none"
    bits = [b for b in (gpu.get("sm"), gpu.get("key")) if b]
    return f"{gpu['name']}({','.join(bits)})" if bits else str(gpu["name"])


def _stack(rep: dict) -> str:
    st = rep.get("stack") or {}
    return f"torch={st.get('torch')} lightning={st.get('lightning')} triton={st.get('triton')} pinned={st.get('pinned')}"


def activation_line(rep: Optional[dict]) -> str:
    rep = rep or {}
    head = f"mode={rep.get('mode')}"
    if rep.get("active") or (rep.get("dry_run") and not rep.get("reason")):
        body = (f"{head} attach={rep.get('attach')} tier={rep.get('tier')} genie3={rep.get('genie3_version')} {_stack(rep)} "
                f"gpu={gpu_label(rep.get('gpu'))} levers={_join(rep.get('levers_planned'))} line={rep.get('line')}")   # levers / line = the EFFECTIVE set at the run's --batch / --precision (modes.effective)
        if rep.get("target_gpu_mismatch"):
            body += f" note={rep['target_gpu_mismatch']}"
        return f"{PREFIX} {'DRY-RUN' if rep.get('dry_run') else 'ACTIVE'} {body}"
    return f"{PREFIX} NOT ACTIVE: {rep.get('reason')} ({head})"


def run_line(mode: str, n_problems: int, n_designs, seed, out_dir: str, batch="request", precision: str = "fp32", shard: Optional[str] = None) -> str:
    """``RUN mode=<m> problems=<n> designs=<n> seed=<s|unseeded> batch=<B|request> precision=<fp32|tf32> out=<dir> [shard=<K>/<M>]`` — the pass and its
    run parameters (the batch: the kit line's --batch-size or the stock request's generation.dataset.batch_size, `request` when the file leaves it
    to upstream's default; the matmul precision the line runs at; ``shard`` only under --num-shards M --shard-id K, ``designs`` then the shard's
    share of the request)."""
    return f"{PREFIX} RUN mode={mode} problems={n_problems} designs={n_designs} seed={seed} batch={batch} precision={precision} out={out_dir}" + (f" shard={shard}" if shard else "")


def numerics_line(mode: str, word: str, live: Optional[dict], why: Optional[str], batch=None, planned=None) -> str:
    """After a kit pass, the line's numerics judged from the driver's own readback (design.numerics_verdict): ``NUMERICS mode=<m>
    line=<fp32|tf32> matmul_tf32=<bool> matmul=<word> cudnn_tf32=<bool> batch=<ran>/<asked> verdict=<ok|FAIL: reason>``."""
    live = live or {}
    return (f"{PREFIX} NUMERICS mode={mode} line={word} matmul_tf32={live.get('matmul_tf32')} matmul={live.get('matmul')} cudnn_tf32={live.get('cudnn_tf32')} "
            f"batch={batch}/{planned} verdict={'ok' if why is None else 'FAIL: ' + str(why)}")


def evidence_line(applied: List[str], missing: List[str], forbidden: List[str]) -> str:
    return f"{PREFIX} EVIDENCE levers={_join(applied)} missing={_join(missing)} forbidden={len(forbidden)}"


def refused_line(mode: str, refusals: List[str]) -> str:
    """``NOT ACTIVE: mode=<m> cannot serve <feature,…> — refused by name (exit 3), nothing ran: <feature: mechanism; …>; run --mode off for the stock
    path`` — a request naming a computation the kit line does not run (design.kit_refusals, each entry ``<feature>: <mechanism>``)."""
    from .codes import EXIT_NOT_ACTIVE
    names = ",".join(r.split(":", 1)[0] for r in refusals)
    return (f"{PREFIX} NOT ACTIVE: mode={mode} cannot serve {names} — refused by name (exit {EXIT_NOT_ACTIVE}), nothing ran: "
            f"{'; '.join(refusals)}; run --mode off for the stock path")


def lever_skipped_line(mode: str, lever: str, reason: str) -> str:
    """``LEVER lever=<id> state=skipped reason=<why> mode=<m> served_by=module_forward`` — a planned lever that stood down for THIS request by its
    declared gate (registry.Lever.declined: L7 under the kernel's token floor) and whose calls ran the module's own forward; the core's grammar
    (``state=skipped reason=<gate>``, as the driver's own ``LEVER name=… state=skipped`` line in design.log), on the console. The pass is neither
    partial nor refused, and the lever is not among the EVIDENCE line's applied levers."""
    return f"{PREFIX} LEVER lever={lever} state=skipped reason={reason} mode={mode} served_by=module_forward"


def ready_line(mode: str, t_first_s: float, pass_tag: str, first: str) -> str:
    """The cold-start line of one child process: seconds from its start to its first written PDB."""
    return f"{PREFIX} ready mode={mode} t={float(t_first_s):.2f} pass={pass_tag} first={first}"


def kernels_line(census: dict) -> str:
    """The stock route's KERNELS census, printed before the stock child starts: ``KERNELS route=stock tf32_matmul=<bool> cudnn_tf32=default
    float32_matmul_precision=default torch=<version> overrides=<none|NAME,…> source=environment`` (design.kernels_census: the child's matmul numerics
    as its stripped environment leaves them; torch's own initialisation, no probe process)."""
    ov = census.get("overrides") or {}
    return (f"{PREFIX} KERNELS route=stock tf32_matmul={census.get('tf32_matmul')} cudnn_tf32={census.get('cudnn_tf32')} "
            f"float32_matmul_precision={census.get('float32_matmul_precision')} torch={census.get('torch')} overrides={','.join(sorted(ov)) if ov else 'none'} source={census.get('source')}")


def stock_line(proof: dict) -> str:
    """The stock pass's proof, read back from <out>/stock_env_proof.json after the child exits: ``STOCK proof=ok|FAIL pinned=<bool>
    files_checked=<n> git_head=<sha> modified=<n> forbidden_env=<n> kit_modules=<n> kit_dirs=<n> autoload=<bool> torch=<bool> genie3=<bool>``
    (the caller's own record, stock_cli.py env_proof + the pins report; ``error=<fact>`` when the caller refused)."""
    co = ((proof.get("pins_check") or {}).get("detail") or {}).get("checkout") or {}   # design.write_pins_check: {"bad", "detail": {"checkout", "stack"}}
    line = (f"{PREFIX} STOCK proof={'ok' if proof.get('ok') else 'FAIL'} pinned={co.get('pinned')} files_checked={co.get('files_checked')} git_head={co.get('git_head')} "
            f"modified={len(co.get('git_modified') or [])} forbidden_env={len(proof.get('forbidden_env_present') or [])} kit_modules={len(proof.get('kit_modules_loaded') or [])} "
            f"kit_dirs={len(proof.get('kit_dirs_on_sys_path') or [])} autoload={proof.get('autoload_armed')} torch={proof.get('torch_imported')} genie3={proof.get('genie3_imported')}")
    if proof.get("error"):
        line += f" error={proof['error']}"
    return line


# ----------------------------------------------------------------------------------------------------------------- exit tally
def tally_from_timings(path: str) -> dict:
    """The driver's final figures from its timings JSON (only the keys it wrote)."""
    try:
        r = json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out = {k: r[k] for k in TALLY_KEYS if k in r}
    if isinstance(r.get("graph_capture_s"), list):
        out["graph_captures"] = len(r["graph_capture_s"])
    if isinstance(r.get("per_batch"), list) and r["per_batch"]:
        out["batches"] = len(r["per_batch"])
        out["max_batch"] = max(len(b.get("names") or []) for b in r["per_batch"])
    return out


def add_tally_source(tag: str, timings_json: Optional[str]) -> None:
    _TALLY["sources"].append({"tag": tag, "timings": timings_json})


def exit_tally_line(pid: Optional[int] = None) -> str:
    pid = os.getpid() if pid is None else pid
    parts = []
    for src in _TALLY["sources"]:
        stats = tally_from_timings(src["timings"]) if src.get("timings") and os.path.isfile(src["timings"]) else {}
        if stats:
            parts.append(f"{src['tag']}:" + "{" + ",".join(f"{k}={v}" for k, v in stats.items()) + "}")
        else:
            parts.append(f"{src['tag']}: no driver figures (the driver wrote no timings JSON; stock, or the pass did not finish)")
    if not parts:
        return f"{PREFIX} EXIT pid={pid} no driver figures: no driver pass ran in this process"
    return f"{PREFIX} EXIT pid={pid} source=timings-json " + " | ".join(parts)


def _print_exit_tally() -> None:
    if _TALLY["printed"]:
        return
    _TALLY["printed"] = True
    try:
        emit(exit_tally_line())
    except Exception as e:  # noqa: BLE001
        emit(f"{PREFIX} EXIT tally failed: {e!r}")


def register_exit_tally() -> None:
    """Print the exit tally at interpreter exit (once per process)."""
    if _TALLY["registered"]:
        return
    _TALLY["registered"] = True
    atexit.register(_print_exit_tally)


def reset_for_tests() -> None:
    _TALLY["sources"] = []
    _TALLY["printed"] = False

"""Observability for rfdiffusion1_opt: the activation line, the run line and the exit tally (all on stderr, prefixed
``[rfdiffusion1-opt]``).

* activation — ``ACTIVE mode=<m> attach=driver tier=<1|2> rfdiffusion=<v> stack=<torch/dgl/triton> gpu=<name(smNN,key)>
  levers=<ids> numerics=<policy> numerics_source=declared matmul=… matmul_tf32=… cudnn_tf32=… autocast=… env=<NAME=v,...> line=<the kit line>``
  — the documented line, printed once the first pass's application evidence (the levers' applied-lines, or the stock process's environment
  proof) has been read; at activation, before anything runs, the same fields are printed under the word ``PLAN`` (or ``DRY-RUN ...`` / ``NOT ACTIVE: <reason>``), formatted from the activation
  report `rfdiffusion1_opt.enable()` / `status()` return;
* run — ``RUN mode=<m> cases=<n> designs=<n> process=<per-case|resident|served K=k> out=<dir>`` once per driver / stock pass, then ``EVIDENCE levers=<applied>
  missing=<ids> forbidden=<n>`` from the driver log (registry evidence lines);
* exit — the kits' own final counters of the pass, read from the driver's ``<tag>_timings.json`` (``fullgraph_final``, ``prep_final``,
  ``einsum_route_final``, ``triton_ln_final``), printed at interpreter exit of the `design` process; when no
  driver ran (stock, or nothing ran) the tally says so;
* the exit rule (``verdict``: the one decision after a pass, the design route and the served route alike; the codes EXIT_OK /
  EXIT_FAIL / EXIT_USAGE / EXIT_NOT_ACTIVE live here and nowhere else in the package, equal to the core's family codes, opt_core.report) — a pass whose levers left no evidence line, or printed a
  forbidden line, is ``partial`` (a lever on the stock path): ``NOT ACTIVE: partial activation — <levers>: <reason>; exit 3``
  (``not_active_partial_line``), EXIT_NOT_ACTIVE with the outputs kept and ``status: partial`` in the manifest. Outputs short of the request are ``incomplete`` (EXIT_FAIL) whatever else
  happened; a failed process or the launcher's own verdict is ``failed`` (EXIT_FAIL); a stock process whose environment proof is not
  ok is not the stock line (EXIT_NOT_ACTIVE). Never silent: every degraded pass is named in the manifest and on its line.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

TAG = "rfdiffusion1-opt"                                                  # the kit's tag in the core's line grammar (opt_core.report.prefix(TAG) == PREFIX; test_cli_outputs_stock)
PREFIX = "[" + TAG + "]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3            # the one home of the exit codes (cli / design / serve import them); == opt_core.report's family codes (test_cli_outputs_stock)
NOT_ACTIVE_FMT = "{prefix} NOT ACTIVE: {reason}"                         # the one refusal line: nothing ran, or the pass is not the mode's line
PARTIAL_DETAIL_FMT = "{levers}: {reason}"                                # <detail> of the family grammar: the lever names first, then the engine's own reason
NOT_ACTIVE_PARTIAL_FMT = "partial activation — {detail}; exit {code}"
NOTE_FMT = "{prefix} NOTE: {text}"                                        # a named fact that gates nothing: printed, recorded in opt_manifest.json, the run proceeds (a card above the stack's capability ceiling; the memory estimate of the served line)
STATUS_WORDS = ("ok", "partial", "failed", "incomplete", "refused", "dry-run")   # manifest `status`; the verdict's precedence is verdict()'s
TALLY_MAX_FIELDS = 40
STATS_KEYS = ("fullgraph_final", "prep_final", "einsum_route_final", "triton_ln_final")
_P = re.escape(PREFIX)
NOT_ACTIVE_RE = re.compile(_P + r" NOT ACTIVE: (?P<reason>.*)$", re.M)
NOT_ACTIVE_PARTIAL_RE = re.compile(_P + r" NOT ACTIVE: partial activation — (?P<detail>.*); exit (?P<code>\d+)$", re.M)

_TALLY: Dict[str, object] = {"sources": [], "registered": False, "printed": False}


def emit(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


# ------------------------------------------------------------------------------------------------------------------- exit rule
def not_active_line(reason: str) -> str:
    """``[rfdiffusion1-opt] NOT ACTIVE: <reason>`` — the one refusal line of the package (a refusal before anything ran, the launcher's
    refusal, a stock process not proven clean, a partial pass)."""
    return NOT_ACTIVE_FMT.format(prefix=PREFIX, reason=reason)


def note_line(text: str) -> str:
    """``NOTE: <text>`` — a named fact, never a gate: the run proceeds (stack._finish prints the report's notes — a card above the capability
    ceiling; serve.py prints the launcher's memory estimate)."""
    return NOTE_FMT.format(prefix=PREFIX, text=text)


def partial_detail(levers: Sequence[str], reason: Optional[str]) -> str:
    """``<levers>: <reason>`` — the lever names first (``none`` when only forbidden lines were printed), then the engine's own reason."""
    return PARTIAL_DETAIL_FMT.format(levers=_join(list(levers)), reason=reason or "no reason recorded")


def not_active_partial_line(levers: Sequence[str], reason: Optional[str], code: int = EXIT_NOT_ACTIVE) -> str:
    """The partial-exit line: ``NOT ACTIVE: partial activation — <levers>: <reason>; exit 3``."""
    return not_active_line(NOT_ACTIVE_PARTIAL_FMT.format(detail=partial_detail(levers, reason), code=code))



def verdict(*, failed: Optional[str] = None, not_active: Optional[str] = None, partial: Sequence[str] = (), partial_reason: Optional[str] = None,
            n_pdb: Optional[int] = None, expected: Optional[int] = None) -> dict:
    """The exit rule's one decision after a pass, in this precedence — one status word and one code per condition:
    ``not_active`` (a stock process whose environment proof is not ok: not the stock line; a driver process that exited non-zero before any
    lever applied: activation failed in the driver) -> ``failed``, EXIT_NOT_ACTIVE;
    ``failed`` (a driver process or worker rc != 0, the launcher's own verdict) -> ``failed``, EXIT_FAIL;
    ``incomplete`` (``n_pdb`` != ``expected``, counted whatever else happened) -> ``incomplete``, EXIT_FAIL;
    ``partial`` (levers without their evidence line / a forbidden line) -> ``partial``, EXIT_NOT_ACTIVE; otherwise ``ok``, EXIT_OK.
    Returns {status, exit_code, reason, partial, partial_reason, incomplete}: the partial and incomplete records are kept whichever code wins."""
    partial = list(partial)
    incomplete = None if (n_pdb is None or expected is None or n_pdb == expected) else f"{n_pdb}/{expected}"
    if not_active:
        status, code, reason = "failed", EXIT_NOT_ACTIVE, not_active
    elif failed:
        status, code, reason = "failed", EXIT_FAIL, failed
    elif incomplete:
        status, code, reason = "incomplete", EXIT_FAIL, f"outputs short of the request: {incomplete} des_*.pdb"
    elif partial or partial_reason:
        status, code, reason = "partial", EXIT_NOT_ACTIVE, partial_detail(partial, partial_reason)
    else:
        status, code, reason = "ok", EXIT_OK, None
    return {"status": status, "exit_code": code, "reason": reason, "partial": partial, "partial_reason": partial_reason, "incomplete": incomplete}


def emit_verdict(v: dict) -> None:
    """The exit rule's line for a partial pass: the NOT ACTIVE form (exit 3); nothing for the other statuses
    (their facts are on the RUN / EVIDENCE lines and in the manifest)."""
    if v["status"] == "partial":
        emit(not_active_partial_line(v["partial"], v["partial_reason"], v["exit_code"]))
    elif v["partial"] or v["partial_reason"]:                                # partial and something worse: the partial fact still named beside the winner
        emit(f"{PREFIX} PARTIAL recorded: {partial_detail(v['partial'], v['partial_reason'])} (status {v['status']}, exit {v['exit_code']})")


def _join(items) -> str:
    return ",".join(str(x) for x in items) if items else "none"


def gpu_label(gpu) -> str:
    if not isinstance(gpu, dict) or not gpu.get("name"):
        return "none"
    bits = [b for b in (gpu.get("sm"), gpu.get("key")) if b]
    return f"{gpu['name']}({','.join(bits)})" if bits else str(gpu["name"])


def _stack(rep: dict) -> str:
    st = rep.get("stack") or {}
    return f"torch={st.get('torch')} dgl={st.get('dgl')} triton={st.get('triton')} pinned={st.get('pinned')}"


def activation_line(rep: Optional[dict], word: Optional[str] = None) -> str:
    """The one activation line: ``ACTIVE`` (the documented line: printed only AFTER the first pass's application evidence is read), ``PLAN`` (the
    same fields, printed at activation before any driver or stock process runs — a plan, not a proof), ``DRY-RUN``, or ``NOT ACTIVE: <reason>``."""
    rep = rep or {}
    head = f"mode={rep.get('mode')}" + (" route=served" if rep.get("served") else "")
    if rep.get("active") or (rep.get("dry_run") and not rep.get("reason")):
        env = _join(f"{k}={v}" for k, v in (rep.get("env") or {}).items())
        body = (f"{head} attach={rep.get('attach')} tier={rep.get('tier')} rfdiffusion={rep.get('rfdiffusion_version')} {_stack(rep)} "
                f"gpu={gpu_label(rep.get('gpu'))} levers={_join(rep.get('levers_planned'))}"
                + f" {_kv(numerics_words(rep.get('numerics')))} "
                f"env={env} overrides={','.join(rep.get('overrides') or []) or 'none'}"
                + (f" replaced={_join(rep.get('compose'))}" if rep.get("compose") else "") + f" line={rep.get('line')}")
        if rep.get("target_gpu_mismatch"):
            body += f" note={rep['target_gpu_mismatch']}"
        return f"{PREFIX} {word or ('DRY-RUN' if rep.get('dry_run') else 'ACTIVE')} {body}"
    return not_active_line(f"{rep.get('reason')} ({head})")




def run_line(mode: str, n_cases: int, n_designs: int, out_dir: str, process: str) -> str:
    return f"{PREFIX} RUN mode={mode} cases={n_cases} designs={n_designs} process={process} out={out_dir}"


def evidence_line(applied: List[str], missing: List[str], forbidden: List[str], log_error: Optional[str] = None) -> str:
    return f"{PREFIX} EVIDENCE levers={_join(applied)} missing={_join(missing)} forbidden={len(forbidden)}" + (f" log_unreadable={log_error}" if log_error else "")


# ----------------------------------------------------------------------------------------------------------------- exit tally
def _fmt(d: dict, budget: int) -> str:
    items = [(k, v) for k, v in d.items() if not isinstance(v, (dict, list))][:budget]
    return "{" + ",".join(f"{k}={v}" for k, v in items) + "}"


def tally_from_timings(path: str) -> Dict[str, dict]:
    """The driver's final counters from its <tag>_timings.json (only the keys it wrote)."""
    try:
        r = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"error": repr(e)}                                                # the read failure is named in the tally, not folded into an empty one
    return {k: r[k] for k in STATS_KEYS if isinstance(r.get(k), dict)}


def numerics_from_timings(path: str) -> Dict[str, object]:
    """The torch numerics the driver recorded per case (``precision``: param_dtype, allow_tf32_matmul, cudnn_tf32, autocast_enabled) — one dict when
    every case agrees; ``{"error": ...}`` names an unreadable file or disagreeing cases (never folded into an empty record)."""
    try:
        r = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"error": repr(e)}
    precs = [c.get("precision") for c in (r.get("cases") if isinstance(r, dict) else r) or [] if isinstance(c, dict) and c.get("precision")]
    if not precs:
        return {"error": "no per-case precision record in the driver's timings"}
    if any(p != precs[0] for p in precs):
        return {"error": f"cases disagree: {precs}"}
    return dict(precs[0])


def _onoff(v) -> str:
    return "none" if v is None else ("on" if v is True else ("off" if v is False else str(v)))


def _kv(parts: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in parts.items())


def numerics_words(num: Optional[dict]) -> dict:
    """``numerics=<policy> numerics_source=declared|torch`` and, for a declared policy, ``matmul= matmul_tf32= cudnn_tf32= autocast=`` (+
    ``numerics_mismatch=<key,...>`` when the values the driver recorded differ from the table) — ``modes.NUMERICS`` / ``modes.numerics``."""
    if not num:
        return {}
    out = {"numerics": num.get("policy"), "numerics_source": num.get("source", "declared")}
    if num.get("policy") != "untouched":
        out.update(matmul=num.get("matmul"), matmul_tf32=_onoff(num.get("matmul_tf32")), cudnn_tf32=_onoff(num.get("cudnn_tf32")),
                   autocast=_onoff(num.get("autocast")))
        if num.get("mismatch"):
            out["numerics_mismatch"] = ",".join(num["mismatch"])                    # recorded torch values that differ from modes.NUMERICS: named, not refused
    return out


def numerics_line(pass_tag: str, record: Dict[str, object], num: Optional[dict]) -> str:
    """``NUMERICS pass=<tag> numerics=<policy> numerics_source=torch matmul=… matmul_tf32=… cudnn_tf32=… autocast=… [numerics_mismatch=<keys>]`` — one
    pass's driver record against the mode's declaration (named, never refused); an unreadable record is named instead."""
    if "error" in record:
        return f"{PREFIX} NUMERICS pass={pass_tag} numerics_source=unreadable ({record['error']})"
    return f"{PREFIX} NUMERICS pass={pass_tag} {_kv(numerics_words(num))}"


def io_line(pass_tag: str, counters: Optional[dict]) -> str:
    """``IO pass=<tag> io=numpy confirmed=<n> mismatches=<m> calls=<c> seconds=<s>`` — lever IO1's counters for one pass (design.io_counters); ``io=unreadable``
    when the driver did not print them (design.io_defects names that as a defect of the pass)."""
    if not counters:
        return f"{PREFIX} IO pass={pass_tag} io=unreadable"
    return (f"{PREFIX} IO pass={pass_tag} io=numpy confirmed={counters.get('n_verified')} mismatches={counters.get('n_mismatch')} "
            f"calls={counters.get('n_calls')} seconds={float(counters.get('seconds') or 0.0):.2f}")


def add_tally_source(tag: str, timings_json: Optional[str], log: Optional[str]) -> None:
    _TALLY["sources"].append({"tag": tag, "timings": timings_json, "log": log})


def exit_tally_line(pid: Optional[int] = None) -> str:
    pid = os.getpid() if pid is None else pid
    parts = []
    for src in _TALLY["sources"]:
        stats = {}
        if src.get("timings") and os.path.isfile(src["timings"]):
            stats.update(tally_from_timings(src["timings"]))
        if stats:
            parts.append(f"{src['tag']}:" + " ".join(f"{k}={_fmt(v, TALLY_MAX_FIELDS // max(1, len(stats)))}" for k, v in stats.items()))
        else:
            parts.append(f"{src['tag']}: no lever counters (the driver wrote no final stats; stock, or the pass did not finish)")
    if not parts:
        return f"{PREFIX} EXIT pid={pid} no lever counters: no driver pass ran in this process"
    return f"{PREFIX} EXIT pid={pid} source=driver-files " + " | ".join(parts)


def _exit_tally_at_exit() -> str:
    """The whole line the core's hook prints at interpreter exit (once per process: opt_core.report.register_exit_tally is idempotent per tag)."""
    _TALLY["printed"] = True
    return exit_tally_line()


def register_exit_tally() -> None:
    """Print the exit tally at interpreter exit (once per process) — opt_core.report.register_exit_tally with this package's whole line."""
    if _TALLY["registered"]:
        return
    _TALLY["registered"] = True
    from opt_core.report import register_exit_tally as _register
    _register(TAG, _exit_tally_at_exit)


def reset_for_tests() -> None:
    _TALLY["sources"] = []
    _TALLY["printed"] = False

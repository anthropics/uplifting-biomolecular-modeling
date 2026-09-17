"""Observability for colabdesign_opt: every line the package prints, on stderr, prefixed ``[colabdesign-opt]``.

* ``ACTIVE mode=<off|exact|fast|word> [base=<mode> ablated=<ids> [restored=<ids>]] route=<subprocess|env|api|check> tier=<none|1|2> class=<stock|exact|precision|approx|ablation> colabdesign=<version>@<commit8>
  jax=<v> haiku=<v> gpu=<card> levers=<a+b|none> skipped=<ids|none> arm=<stock|kit> settings=<the settings file's name> line=<the mode's guarantee>`` — formatted from the activation
  report (`stack.activate`), with ``stack_drift=<dist>:<installed>!=<pinned>[,…]`` appended when a stack distribution is off its pin (named, never
  refused; absent on the pinned stack) and ``base= ablated=`` present only for the mode word's subtractive form `<mode>-no-<lever>` (modes.resolve:
  tier=none class=ablation, the EVIDENCE and EXIT lines then carry ``ablated=<ids>`` too); ``NOT ACTIVE reason=<...>`` (exit 3) when the mode cannot be activated: colabdesign absent or off its
  pinned commit (what "stock" means), the kit's core absent, an unknown mode, a late activation — configuration, never a lever: a LEVER that
  cannot engage here steps aside BY NAME (its `LEVER … state=skipped reason=cannot_run|no_attention_kernel` line in the arm) and the mode runs
  with the rest of its set; `levers=` = what is on, `skipped=` = the levers that stepped aside (on the dry-run routes: foretold from `no GPU
  visible`, else none — the arm's LEVER lines are the record);
* ``RUN mode=<m> target=<file> chains=<c> binder_len=<L> tokens=<T> hotspot=<spec|none> seed=<s> steps<=<n> greedy_rounds=<n>`` — the case,
  before the arm starts;
* ``LEVER name=<id> state=<on|skipped|off> ...`` — one per lever, printed IN the kit arm by the lever (nosub.py at model build, the kernel levers at exit);
* ``EVIDENCE levers=<applied> [ablated=<ids> [restored=<ids>]] gated=<ids> fallback=<ids> missing=<ids> skipped=<ids> grad_subbatch=<v> fn_subbatch=<v>
  compile_cache=<hits>/<requests> compile_threads=<n> prev_on_device=<n> <lever>_served=<n> <lever>_fallback_by=<reason:n|none> (per census lever)
  pallas_served=<n> pallas_fallback=<n> pallas_fallback_by=<reason:n|none>`` — the kit arm's lever state, derived only from those lines (evidence.py);
* ``SETTINGS <name> file=<path> sha256=<16> design_models=<list> helicity=<x> hotspot=<spec|none> trajectory=<tag>`` — printed by the design
  script in the arm as each design call starts (`settings_line`), then, when that design ends, ``[run] <tag>: tokens <T> steps <n> (soft <n>
  temp <n> hard <n> greedy_rounds <n> greedy_forward <n>) terminate=<complete|terminated:<gate>> steady <s> s/step soft <s> temp <s> hard <s>
  greedy_forward <s> first_calls <s>,<s>,… total <s> s loss[-1]=<x> plddt=<x> i_ptm=<x> design.pdb sha256=<16> recycles=<r>:<k0>-<k1>[,…]
  segments=<stage>/r<r>:<n>@<s>[,…]`` — the last two words are the EQUAL-WORK cells: `recycles=` the runs of graded steps over which BindCraft's
  num_recycles was constant (its optimise_beta branch can raise it mid-trajectory), `segments=` the steady s/step (first calls excluded) of each
  (stage, num_recycles) cell with its step count (`run_summary_line`,
  bindcraft.py) — the design's counts and timing; the calling process reads them back from that line (`parse_run_line` / `run_lines`, the
  renderer's inverse — cli.py's channel from its arm);
* ``EXIT rc=<n> mode=<m> levers=<a+b|none> evidence=<...> out=<dir>`` — once per `design`.
"""
from __future__ import annotations

import re
import sys
from typing import Callable, List, Optional, Sequence

from opt_core import report as _core_report

from .names import PREFIX, DESIGN_PDB

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = _core_report.EXIT_OK, _core_report.EXIT_FAIL, _core_report.EXIT_USAGE, _core_report.EXIT_NOT_ACTIVE   # the tree's exit codes: 0 done, 1 ran and failed, 2 usage, 3 mode not active / partial refused


def log(line: str, *, stream=None) -> None:
    (stream or sys.stderr).write(f"{PREFIX} {line}\n")
    (stream or sys.stderr).flush()


def _after_prefix(template: str, detail: str) -> str:
    """A core line template rendered for this package's `log` (which writes the prefix itself)."""
    return template.format(prefix=PREFIX, detail=detail)[len(PREFIX) + 1:]


PARTIAL_HEAD = _after_prefix(_core_report.PARTIAL_REFUSED, "\0").split("\0")[0]                                  # `NOT ACTIVE: partial activation — ` — the tree's form (opt_core.report)
PARTIAL_TAIL = "; exit 3 (an installed lever that did not engage is a defect, named - a lever that cannot run here steps aside at install instead, state=skipped, exit 0)"


def partial_line(detail: str) -> str:
    """The partial-exit line, one fixed grammar: ``NOT ACTIVE: partial activation — <detail>; exit 3 (an installed lever that did not engage
    is a defect, named - …)`` — `detail` = the lever names first, then the kit's reason (``nosub: fallback: grad_fn subbatch=4 (...)``); every
    entry point that exits on a partial activation prints it through here (cli.partial_exit). A lever that INSTALLED and then served nothing,
    or fell back for an undeclared reason, means the outputs are not what the LEVER lines claim: a defect, exit 3. A lever that cannot run here
    is not this case: it steps aside at install (`state=skipped reason=cannot_run|no_attention_kernel`, levers.py), named, exit 0."""
    return f"{PARTIAL_HEAD}{detail}{PARTIAL_TAIL}"


def _join(items) -> str:
    items = [str(x) for x in (items or []) if x is not None and str(x) != ""]
    return ",".join(items) if items else "none"


def gpu_label(gpu: Optional[dict]) -> str:
    """`<name>(<smNN>,<MiB>MiB)`: the card, its compute capability and memory as nvidia-smi reports them; `none` without a GPU."""
    if not gpu or not gpu.get("name"):
        return "none"
    name = str(gpu["name"]).replace(" ", "-")
    bits = []
    if gpu.get("cc"):
        bits.append(f"sm{str(gpu['cc']).replace('.', '')}")
    if gpu.get("memory_mib"):
        bits.append(f"{int(gpu['memory_mib'])}MiB")
    return f"{name}({','.join(bits)})" if bits else name


def active_line(rep: dict) -> str:
    up = rep.get("upstream") or {}
    cd = up.get("colabdesign") or {}
    cd_s = f"{cd.get('version') or '?'}@{(cd.get('commit') or '?')[:8]}"
    if not rep.get("active"):
        return f"NOT ACTIVE reason={rep.get('reason') or 'unknown'}"
    abl = f" base={rep.get('base')} ablated={_join(rep.get('ablated'))}" if rep.get("ablated") else ""   # the subtractive word only (modes.resolve): never on a table mode's line
    abl += f" restored={_join(rep.get('restored'))}" if rep.get("restored") else ""         # ... and only when a dropped lever's replaced lever is back in the set (registry Supersession)
    return (f"ACTIVE mode={rep.get('mode')}{abl} route={rep.get('route')} tier={rep.get('tier') if rep.get('tier') is not None else 'none'} "
            f"class={rep.get('numerics_class') if rep.get('numerics_class') is not None else 'stock'} "
            f"colabdesign={cd_s} jax={up.get('jax') or '?'} haiku={up.get('dm-haiku') or '?'} gpu={gpu_label(rep.get('gpu'))} "
            f"levers={_levers(rep.get('levers'))} skipped={_join(rep.get('skipped'))} arm={rep.get('arm') or '?'} settings={rep.get('settings') or '?'} line={rep.get('line')}"
            + (f" stack_drift={','.join(f'{k}:{v}' for k, v in rep['stack_drift'].items())}" if rep.get("stack_drift") else ""))   # a stack distribution off its pin: named, never refused; absent on the pinned stack


def _levers(levers) -> str:
    from opt_core.modes import levers_label
    return levers_label(tuple(levers or ()))


def run_line(case: dict) -> str:
    return (f"RUN mode={case.get('mode')} target={case.get('target')} chains={case.get('chains')} binder_len={case.get('binder_len')} "
            f"tokens={case.get('tokens')} hotspot={case.get('hotspot') or 'none'} seed={case.get('seed')} steps<={case.get('steps')} greedy_rounds={case.get('greedy_rounds')}")


def _na(v):
    """A field of a lever outside the run's set (it printed no line): `n/a`, never Python's None."""
    return "n/a" if v is None else v


def _fb(d) -> str:
    """`reason:n,...` of a kernel's fallback census, `none` when empty, `n/a` when the lever printed no line."""
    if d is None:
        return "n/a"
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items())) or "none"


def _hits_of(ev: dict) -> str:
    """`<hits>/<requests>` of the compile cache, or `n/a` when the lever printed no line."""
    r, h = ev.get("compile_requests"), ev.get("compile_hits")
    return "n/a" if r is None else f"{h if h is not None else 0}/{r}"


def evidence_line(ev: dict) -> str:
    from .evidence import CENSUS_LEVERS                                            # the census levers' field pairs, registry order
    fb = ev.get("pallas_fallback_by") or {}
    abl = f" ablated={_join(ev.get('ablated'))}" if ev.get("ablated") else ""             # the subtractive word only
    abl += f" restored={_join(ev.get('restored'))}" if ev.get("restored") else ""
    return (f"EVIDENCE levers={_join(ev.get('applied'))}{abl} gated={_join(ev.get('gated'))} fallback={_join(ev.get('fallback'))} "
            f"missing={_join(ev.get('missing'))} skipped={_join(ev.get('skipped'))} grad_subbatch={_na(ev.get('grad_subbatch'))} fn_subbatch={_na(ev.get('fn_subbatch'))} compile_cache={_hits_of(ev)} "
            f"compile_threads={_na(ev.get('parcompile_threads'))} prev_on_device={_na(ev.get('prev_device_steps'))} "
            + "".join(f"{l}_served={_na(ev.get(l + '_served'))} {l}_fallback_by={_fb(ev.get(l + '_fallback_by'))} " for l in CENSUS_LEVERS) +
            f"pallas_served={_na(ev.get('pallas_served'))} pallas_fallback={_na(ev.get('pallas_fallback'))} "
            f"pallas_fallback_by={','.join(f'{k}:{v}' for k, v in sorted(fb.items())) or 'none'}")


SETTINGS_HEAD = "SETTINGS "                                                   # the design script's line as each design call starts (settings_line)
RUN_HEAD = "[run] "                                                            # the design script's line as each design ends (run_summary_line)
N_STEPS_WORDS = ("soft", "temp", "hard", "greedy_rounds", "greedy_forward")    # the count words in the `[run]` line's parenthesis (units.end_row n_steps), in printed order
PHASES_TIMED = ("soft", "temp", "hard", "greedy_forward")                      # the phase words of its steady columns (bindcraft.run_design phase_steady_s), in printed order


def settings_line(S: dict, derived: dict, hotspot: Optional[str]) -> str:
    """``SETTINGS <name> file=<path> sha256=<16> design_models=<list> helicity=<x> hotspot=<spec|none> trajectory=<tag>`` — the pinned settings
    the design runs at (settings.load's record) and bindcraft.derive's derivation, printed by the design script right before the design call:
    its arrival at the calling process marks that call's start (run_lines `ready_s`)."""
    return (f"{SETTINGS_HEAD}{S['name']} file={S['file']} sha256={S['sha256'][:16]} design_models={derived['design_models']} "
            f"helicity={derived['helicity_value']} hotspot={hotspot or 'none'} trajectory={derived['design_name']}")


def _f(x, nd=3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "nan"


def _recycles_int(v):
    """A row's num_recycles as an int: Python numbers and array scalars (the model's aux carries a device/numpy scalar) alike; None otherwise."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == f else None


def work_segments(grads: list) -> tuple:
    """The equal-work cells of one design's graded steps (units rows, kind grad, in step order): (segments, recycles) —
    segments = [{"phase", "recycles", "n", "steady_s"}] per (stage, num_recycles) cell in order of first appearance (steady_s = the median wall of
    the cell's steps that are not first calls; None when the cell has none), recycles = [{"recycles", "k0", "k1"}] = the maximal runs of
    consecutive steps with one num_recycles value (k = the row's step ordinal). A row without a recycles value counts as recycles None."""
    cells, order, runs = {}, [], []
    for i, r in enumerate(grads):
        rc = _recycles_int(r.get("recycles"))
        key = (r.get("phase"), rc)
        if key not in cells:
            cells[key] = []; order.append(key)
        if not r.get("first_call"):
            cells[key].append(float(r["wall_s"]))
        k = r.get("k", i); k = int(k) if isinstance(k, (int, float)) else i
        if runs and runs[-1]["recycles"] == rc:
            runs[-1]["k1"] = k
        else:
            runs.append({"recycles": rc, "k0": k, "k1": k})
    counts = {}
    for r in grads:
        rc = _recycles_int(r.get("recycles"))
        counts[(r.get("phase"), rc)] = counts.get((r.get("phase"), rc), 0) + 1
    segs = [{"phase": ph, "recycles": rc, "n": counts[(ph, rc)], "steady_s": _median_or_none(cells[(ph, rc)])} for ph, rc in order]
    return segs, runs


def _median_or_none(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _rc(v) -> str:
    return "na" if v is None else str(int(v))


def segments_words(t: dict) -> str:
    """`recycles=<r>:<k0>-<k1>,… segments=<stage>/r<r>:<n>@<s>,…` from a timing dict (`none` for an absent or empty list)."""
    rec = t.get("recycles") or []; seg = t.get("segments") or []
    w1 = ",".join(f"{_rc(x.get('recycles'))}:{x.get('k0')}-{x.get('k1')}" for x in rec) or "none"
    w2 = ",".join(f"{x.get('phase')}/r{_rc(x.get('recycles'))}:{x.get('n')}@{_f(x.get('steady_s'))}" for x in seg) or "none"
    return f"recycles={w1} segments={w2}"


def parse_segments_words(recycles: str, segments: str) -> tuple:
    """The inverse of `segments_words` at printed precision: ([{recycles,k0,k1}], [{phase,recycles,n,steady_s}])."""
    rec, seg = [], []
    if recycles and recycles != "none":
        for w in recycles.split(","):
            r_, span = w.split(":", 1); k0, k1 = span.split("-", 1)
            rec.append({"recycles": None if r_ == "na" else int(r_), "k0": int(k0), "k1": int(k1)})
    if segments and segments != "none":
        for w in segments.split(","):
            head, val = w.split("@", 1); cell, n = head.rsplit(":", 1); ph, r_ = cell.rsplit("/r", 1)
            seg.append({"phase": ph, "recycles": None if r_ == "na" else int(r_), "n": int(n), "steady_s": _float_word(val)})
    return rec, seg


def run_summary_line(tag: str, run: dict) -> str:
    t = run.get("timing") or {}
    fc = t.get("first_calls_s") or []
    ph = t.get("phase_steady_s") or {}
    fin = run.get("final") or {}
    ns = run.get("n_steps") or {}
    term = run.get("terminate") or {}
    verdict = term.get("gate") or term.get("verdict") or ""
    return (f"{RUN_HEAD}{tag}: tokens {run.get('tokens')} steps {run.get('steps')} ({' '.join(f'{k} {ns.get(k)}' for k in N_STEPS_WORDS)}) "
            f"terminate={('terminated:' + verdict) if verdict else 'complete'} steady {_f(t.get('steady_s'))} s/step "
            f"{' '.join(f'{k} {_f(ph.get(k))}' for k in PHASES_TIMED)} "
            f"first_calls {','.join(_f(x, 1) for x in fc) or 'none'} total {_f(t.get('total_s'), 1)} s "
            f"loss[-1]={_f(fin.get('loss'), 4)} plddt={_f(fin.get('plddt'), 4)} i_ptm={_f(fin.get('i_ptm'), 4)} "
            f"design.pdb sha256={(run.get('files') or {}).get(DESIGN_PDB, {}).get('sha256', '?')[:16]} "
            f"{segments_words(t)}")


# run_summary_line read back: one pattern over the words the renderer above writes, in its order, from the same heads and word tuples
# (tests/test_inputs_outputs round-trips the two).
RE_RUN = re.compile(re.escape(RUN_HEAD) + r"(?P<tag>\S+): tokens (?P<tokens>\S+) steps (?P<steps>\S+) "
                    r"\(" + " ".join(rf"{k} (?P<{k}>\S+)" for k in N_STEPS_WORDS) + r"\) "
                    r"terminate=(?P<terminate>\S+) steady (?P<steady>\S+) s/step "
                    + " ".join(rf"{k} (?P<p_{k}>\S+)" for k in PHASES_TIMED) + " "
                    r"first_calls (?P<first_calls>\S+) total (?P<total>\S+) s "
                    r"loss\[-1\]=(?P<loss>\S+) plddt=(?P<plddt>\S+) i_ptm=(?P<i_ptm>\S+) design\.pdb sha256=(?P<pdb16>\S+)"
                    r"(?: recycles=(?P<recycles>\S+) segments=(?P<segments>\S+))?")


def _int_word(s: str) -> Optional[int]:
    try:
        return int(s)
    except (TypeError, ValueError):                                          # `None` as the renderer prints an absent count
        return None


def _float_word(s: str) -> Optional[float]:
    """A number word as `_f` printed it; `nan` (an absent value) reads back as None."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def parse_run_line(line: str) -> Optional[dict]:
    """The `[run]` line read back — the inverse of `run_summary_line` over what the line carries, at its printed precision: {tag, tokens, steps,
    n_steps: {soft, temp, hard, greedy_rounds, greedy_forward}, terminate: `complete` | `terminated:<gate>`, timing: {steady_s, phase_steady_s:
    {soft, temp, hard, greedy_forward}, first_calls_s: [..], total_s}, final: {loss, plddt, i_ptm}, design_pdb_sha256_16 (as printed: 16 hex,
    `?` when no design.pdb was digested)}; None when `line` holds no `[run]` line. An absent number (`nan` / `None` / `none` on the line) reads
    back as None / []."""
    m = RE_RUN.search(str(line).rstrip("\n"))
    if not m:
        return None
    g = m.groupdict()
    fc = [] if g["first_calls"] == "none" else [_float_word(x) for x in g["first_calls"].split(",")]
    return {"tag": g["tag"], "tokens": _int_word(g["tokens"]), "steps": _int_word(g["steps"]),
            "n_steps": {k: _int_word(g[k]) for k in N_STEPS_WORDS}, "terminate": g["terminate"],
            "timing": {"steady_s": _float_word(g["steady"]), "phase_steady_s": {k: _float_word(g["p_" + k]) for k in PHASES_TIMED},
                       "first_calls_s": fc, "total_s": _float_word(g["total"]),
                       **dict(zip(("recycles", "segments"), parse_segments_words(g.get("recycles") or "", g.get("segments") or "")))},
            "final": {k: _float_word(g[k]) for k in ("loss", "plddt", "i_ptm")}, "design_pdb_sha256_16": g["pdb16"]}


def is_settings_line(line: str) -> bool:
    """True for the design script's SETTINGS line, with the package prefix as the arm prints it (`[colabdesign-opt] SETTINGS …`) or bare."""
    line = str(line)
    return line.startswith(SETTINGS_HEAD) or f"{PREFIX} {SETTINGS_HEAD}" in line


def run_lines(lines: Sequence[str], stamps: Optional[Sequence[float]] = None) -> List[dict]:
    """Every `[run]` line of an arm's transcript, read back (`parse_run_line`), in the order printed — one per design that ended. With `stamps`
    (each line's arrival in seconds since the arm was launched, one per line: driver.launch), each run's timing gains `ready_s` = the stamp of
    the SETTINGS line last printed before it — the arm's start-up up to that design call (imports, the settings, BindCraft's modules); None
    when no SETTINGS line preceded it. Without `stamps` no run carries `ready_s`."""
    if stamps is not None and len(stamps) != len(lines):
        raise ValueError(f"run_lines: {len(lines)} lines but {len(stamps)} stamps — one stamp per line (driver.launch)")
    out, started = [], None
    for i, line in enumerate(lines):
        if is_settings_line(line):
            started = stamps[i] if stamps is not None else None
            continue
        run = parse_run_line(line)
        if run is None:
            continue
        if stamps is not None:
            run["timing"]["ready_s"] = started
        out.append(run)
        started = None
    return out


def exit_line(rc: int, res, ev: Optional[dict], out_dir: str) -> str:
    evs = "n/a" if ev is None else f"applied={_join(ev.get('applied'))};fallback={_join(ev.get('fallback'))};missing={_join(ev.get('missing'))}"
    abl = f" ablated={_join(res.ablated)}" if getattr(res, "ablated", ()) else ""          # the subtractive word only
    abl += f" restored={_join(res.restored)}" if getattr(res, "restored", ()) else ""
    return f"EXIT rc={rc} mode={res.mode}{abl} levers={_levers(res.levers)} evidence={evs} out={out_dir}"


Logger = Callable[[str], None]

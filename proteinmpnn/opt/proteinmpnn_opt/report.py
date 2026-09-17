"""Observability for proteinmpnn_opt: the activation line, the stock line, the kit's own lines and the exit tally (all prefixed
``[proteinmpnn-opt]``; the activation and exit lines go to stderr, the kit's own lines are echoed as the kit prints them).

* activation — ``ACTIVE mode=<m> variant=<v> proteinmpnn=<commit8> gpu=<name(smNN)> [target=<name>[/<MiB>MiB] match=yes|no] levers=<expanded
  flags> fallbacks=<x,...> [opted_out=<levers>] [partial=<levers>] [allow_partial=yes] [stock_args=<the stock options given>]`` (or ``DRY-RUN ...`` from ``check`` /
  ``NOT ACTIVE: <reason>``; ``opted_out`` = probe-gated levers the command line left out by name, ``--hybrid_gemm 0``;
  ``partial`` = the requested levers without evidence of application, by name),
  formatted from the activation report of ``enable()`` / ``status()``; ``target``/``match`` appear when the configuration names a target GPU
  (MODEL_OPT_TARGET_GPU, with MODEL_OPT_TARGET_GPU_MEM_MIB the memory total the match asserts); ``stock_args`` when a pass carries stock options;
* weights — the line after the activation line, for the one file the pass loads: ``weights=<name> sha256=<12 hex> (pinned)`` when its
  digest is a ``stock/PINS.json`` entry, ``weights sha256=<12 hex> NOT PINNED — proceeding (stock/PINS.json names the tested weights)``
  for any other bytes (the pass runs either way); the text is ``stock/check_pins.py``'s (``weights_line`` there, carried in the report's
  ``weights`` record), this module adds the prefix only;
* the kit's own evidence lines are the worker's — ``DEVICE CELL: ...`` and ``hybrid_gemm: PROBE PASS|FAIL ...`` (opt/forward/mpnn_exact_worker/
  addon/mpnn_worker2.py); the package records them in opt_manifest.json, never rewrites them;
  the per-item ``ITEM`` and ``PEAK`` lines, the one ``STACK`` line and the one ``OUTPUTS_WRITTEN`` line the kit executables print are
  stage.py's (``ITEM_LINE_FMT`` / ``ITEM_LINE_RE``, ``PEAK_LINE_FMT`` / ``PEAK_LINE_RE``, ``STACK_LINE_FMT`` / ``STACK_LINE_RE``, ``OUTPUTS_WRITTEN_FMT`` /
  ``OUTPUTS_WRITTEN_RE``), echoed as printed;
* cmd — ``CMD route=<stock|kit> variant=<v> argv=<the child's argv, shell-quoted>`` (``CMD_LINE_FMT`` / ``CMD_LINE_RE``), on stderr
  right before every ``design`` launches its design process (stock_run / kit_run), flushed so it stands even when the child dies: the one place a
  pass's whole command line — its seed, its shape, its levers — is stated as run;
* note — ``NOTE <text>`` (``NOTE_LINE_FMT``, ``note_line``), on stderr before the design process, for a consequence of the pass's own options that
  holds on both routes and that neither executable prints: ``NOTE designs per target requested=<n> produced=<(n//b)*b> (upstream's rule: whole
  batches of <b>)`` when ``--num_seq_per_target`` is not a whole number of ``--batch_size`` batches (``DESIGNS_NOTE_FMT``, settings.designs_per_target;
  opt_manifest ``designs_per_target``), and ``NOTE --chain_id_jsonl <f> not read for a single-PDB input (…)`` (``CHAIN_ID_UNREAD_NOTE_FMT``);
* exit — ``EXIT pid=<pid> mode=<m> variant=<v> route=<stock|worker> rc=<child rc> <output counts> wall_s=<s> [probe=PASS|FAIL|unobserved]
  [partial=<levers>] [allow_partial=yes] [incomplete=<n/m>] exit=<code>`` once per command that launched a process (``probe=``: the worker's own
  on-device probe verdict for its probe-gated lever, present whenever the line requested it — kit_run.probe_word); when nothing was launched the
  tally says so (never silent) — a pass mode exact cannot serve (settings.worker_refuses) is one NOT ACTIVE line naming each option with its
  mechanism and that tally, exit 3.
"""
from __future__ import annotations

import os
import re
import shlex
import sys
from typing import Sequence

from opt_core import report as core_report
from opt_core.report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE, gpu_label   # one code per condition; gpu: ``name(smNN)`` · ``name`` · ``none``

from .modes import OPT_OUTS
from . import TAG                                                                    # "proteinmpnn-opt": the package root holds the one spelling

PREFIX = core_report.prefix(TAG)                                                     # "[proteinmpnn-opt]"
_join = core_report.join                                                             # ``a,b,c`` — or ``none`` for an empty list
CMD_LINE_FMT = PREFIX + " CMD route=%s variant=%s argv=%s"                           # argv: shlex.join of the child's argv (cmd_line)
NOTE_LINE_FMT = PREFIX + " NOTE %s"                                                    # a named consequence of the pass's own options that neither route prints itself (note_line)
DESIGNS_NOTE_FMT = "designs per target requested=%d produced=%d (upstream's rule: whole batches of %d)"   # --num_seq_per_target not a whole number of --batch_size batches (settings.designs_per_target)
CHAIN_ID_UNREAD_NOTE_FMT = "--chain_id_jsonl %s not read for a single-PDB input (protein_mpnn_run.py assigns a --pdb_path input's chains from --pdb_path_chains, else designs every chain)"
CMD_LINE_RE = re.escape(PREFIX) + r" CMD route=(?P<route>\S+) variant=(?P<variant>\S+) argv=(?P<argv>.+)"


def _versions(rep: dict) -> str:
    return f"proteinmpnn={rep.get('proteinmpnn_version')}"


def _levers(rep: dict) -> str:
    items = list(rep.get("levers_applied") or [])
    if rep.get("bb_batch") is not None:
        items.append(f"bb_batch={rep['bb_batch']}")
    return _join(items)


def _target(rep: dict) -> str:
    """`` target=<name>[/<mem>MiB] match=yes|no|none`` when the configuration names a target GPU (MODEL_OPT_TARGET_GPU); ``none`` = no GPU
    identified; empty when no target is named."""
    tg = rep.get("target_gpu")
    if not tg:
        return ""
    mem = rep.get("target_gpu_mem_mib")
    m = rep.get("gpu_matches_target")
    return f" target={tg}" + (f"/{mem}MiB" if mem else "") + f" match={'yes' if m else ('none' if m is None else 'no')}"


def _extra(rep: dict) -> str:
    """`` stock_args=<the stock options given>`` when a pass carries any; empty otherwise."""
    ex = rep.get("stock_args") or []
    return f" stock_args={' '.join(str(t) for t in ex)}" if ex else ""


def _partial(rep: dict) -> str:
    """`` opted_out=<levers>`` (probe-gated levers the command line left out by name, --hybrid_gemm 0), `` partial=<levers>`` (the requested levers
    without evidence of application, by name) and `` allow_partial=yes`` when recorded."""
    out = f" opted_out={_join(rep['opted_out'])}" if rep.get("opted_out") else ""
    out += f" partial={_join(rep['partial'])}" if rep.get("partial") else ""
    return out + (" allow_partial=yes" if rep.get("allow_partial") else "")


def activation_line(rep: dict | None) -> str:
    rep = rep or {}
    mode, variant = rep.get("mode"), rep.get("variant")
    head = f"mode={mode} variant={variant}"
    if rep.get("active"):
        line = (f"{PREFIX} ACTIVE {head} {_versions(rep)} gpu={gpu_label(rep.get('gpu'))}{_target(rep)} levers={_levers(rep)} "
                f"fallbacks={_join(rep.get('levers_fallback'))}" + _partial(rep))
        return line + _extra(rep)
    if rep.get("dry_run"):
        return (f"{PREFIX} DRY-RUN {head} route={rep.get('route')} {_versions(rep)} gpu={gpu_label(rep.get('gpu'))}{_target(rep)} "
                f"levers={_levers(rep)} probe={rep.get('probe', {}).get('verdict')}" + _partial(rep)
                + (f" reason={rep.get('reason')}" if rep.get("reason") else ""))
    reason = rep.get("reason") or "not activated"
    return f"{PREFIX} NOT ACTIVE: {reason}" + (f" ({head})" if mode else "") + _extra(rep)


def weights_line(weights: dict | None) -> str | None:
    """``[proteinmpnn-opt] <the weights line>`` from the report's ``weights`` record (stock/check_pins.py ``check_weights``), or None when no
    weights file was inspected (the checkout or the checkpoint directory itself was refused)."""
    return f"{PREFIX} {weights['line']}" if weights and weights.get("line") else None


def log_weights(weights: dict | None, stream=None) -> str | None:
    line = weights_line(weights)
    if line:
        print(line, file=stream or sys.stderr, flush=True)
    return line


def log_activation(rep: dict | None, stream=None) -> str:
    """The activation line, then the weights line of the pass when the report carries one."""
    line = activation_line(rep)
    print(line, file=stream or sys.stderr, flush=True)
    log_weights((rep or {}).get("weights"), stream)
    return line


# the family lines of a partial exit are the core's two templates (opt_core.report PARTIAL_REFUSED / PARTIAL_ALLOWED): PREFIX + " " + head + detail + tail
PARTIAL_HEAD, PARTIAL_TAIL = core_report.PARTIAL_REFUSED.replace("{prefix} ", "", 1).split("{detail}")
PARTIAL_ALLOWED_HEAD, PARTIAL_ALLOWED_TAIL = core_report.PARTIAL_ALLOWED.replace("{prefix} ", "", 1).split("{detail}")


def partial_detail(rep: dict) -> str:
    """``<detail>`` of the family line: the requested levers without evidence of application, by name, then this package's reason —
    ``levers_unavailable`` (the worker's own CPU rule: no CUDA device at activation) before ``gated`` (the worker's on-device probe
    refused the lever on this box: its verdict) before ``levers_fallback`` (the worker's end-of-run record shows the lever off)."""
    partial = list(rep.get("partial") or [])
    cpu = [l for l in partial if l in (rep.get("levers_unavailable") or [])]
    gated = {l: v for l, v in (rep.get("gated") or {}).items() if l in partial and l not in cpu}
    ran_without = [l for l in partial if l not in cpu and l not in gated]
    parts = []
    if cpu:
        parts.append(f"{_join(cpu)}: switched off by the worker's own CPU rule (no CUDA device)")
    if gated:
        parts.append("; ".join(f"{l}: the worker's on-device probe refused it on this box ({v})" + (f" — {OPT_OUTS[l]} 0 runs the mode without it" if l in OPT_OUTS else "")
                               for l, v in gated.items()))
    if ran_without:
        parts.append(f"{_join(ran_without)}: the worker's end-of-run record shows the lever off")
    return "; ".join(parts) + f" (mode={rep.get('mode')} variant={rep.get('variant')})"


def partial_escape(rep: dict) -> str:
    """How a partial pass proceeds, for the reason text: the opt-out of every partial lever that has one (``--hybrid_gemm 0``), and ``--allow-partial``."""
    outs = [f"{OPT_OUTS[l]} 0" for l in (rep.get("partial") or []) if l in OPT_OUTS]
    return (f"{' '.join(outs)} runs the mode without {'it' if len(outs) == 1 else 'them'}; " if outs else "") + "--allow-partial records and proceeds"


def partial_line(rep: dict) -> str:
    """The one line every partial exit prints (``design``, ``warm`` through it, ``check`` on a partial plan): ``NOT ACTIVE: partial
    activation — <detail>; exit 3 (--allow-partial records and proceeds)``, or with ``allow_partial`` recorded ``PARTIAL allowed:
    <detail> (--allow-partial, recorded)`` and the exit is the pass's own."""
    if rep.get("allow_partial"):
        return core_report.PARTIAL_ALLOWED.format(prefix=PREFIX, detail=partial_detail(rep))
    return core_report.PARTIAL_REFUSED.format(prefix=PREFIX, detail=partial_detail(rep))


def log_partial(rep: dict, stream=None) -> str:
    line = partial_line(rep)
    print(line, file=stream or sys.stderr, flush=True)
    return line


def cmd_line(route: str, variant: str, argv: Sequence[str]) -> str:
    """``[proteinmpnn-opt] CMD route=<stock|kit> variant=<v> argv=<shlex.join(argv)>`` — the design process's whole command line."""
    return CMD_LINE_FMT % (route, variant, shlex.join(str(a) for a in argv))


def note_line(text: str) -> str:
    """``NOTE <text>``: one named line for a consequence of the pass's own options that holds on both routes and that neither executable prints —
    ``DESIGNS_NOTE_FMT`` (fewer designs than ``--num_seq_per_target`` asks: upstream designs whole batches), ``CHAIN_ID_UNREAD_NOTE_FMT``."""
    return NOTE_LINE_FMT % text


def log_note(text: str, stream=None) -> str:
    line = note_line(text)
    print(line, file=stream or sys.stderr, flush=True)
    return line


def designs_note(designs: dict) -> str | None:
    """The DESIGNS note text for ``settings.designs_per_target``'s record, None when the request is a whole number of batches."""
    if designs["batch_size"] > 0 and designs["produced"] and designs["produced"] != designs["requested"]:
        return DESIGNS_NOTE_FMT % (designs["requested"], designs["produced"], designs["batch_size"])
    return None


def log_cmd(route: str, variant: str, argv: Sequence[str], stream=None) -> str:
    """Print the CMD line (stderr, flushed) right before the child is launched; returns it."""
    line = cmd_line(route, variant, argv)
    print(line, file=stream or sys.stderr, flush=True)
    return line


def env_clean_line(proof: dict) -> str:
    """The stock subprocess's environment proof, one line: ``ENV-CLEAN ok`` or ``ENV-CLEAN present=<names>`` (listed, never judged)."""
    present = proof.get("present") or []
    kit_on_path = proof.get("kit_dirs_on_path") or []
    if not present and not kit_on_path:
        return f"{PREFIX} ENV-CLEAN ok (absent: {', '.join(proof.get('checked_prefixes') or [])}; PYTHONPATH and sys.path free of the kit directories)"
    return f"{PREFIX} ENV-CLEAN present={_join(present)} kit_dirs_on_path={_join(kit_on_path)}"


def exit_tally_line(stats: dict | None, pid: int | None = None) -> str:
    pid = os.getpid() if pid is None else pid
    if not stats:
        return f"{PREFIX} EXIT pid={pid} nothing launched: no stock or kit process ran in this command"
    fields = [f"{k}={v}" for k, v in stats.items() if v is not None]
    return f"{PREFIX} EXIT pid={pid} " + " ".join(fields)


def log_exit_tally(stats: dict | None, stream=None) -> str:
    line = exit_tally_line(stats)
    print(line, file=stream or sys.stderr, flush=True)
    return line

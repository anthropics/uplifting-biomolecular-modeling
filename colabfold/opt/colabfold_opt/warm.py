"""`warm` — one public-input prediction through `pred`: the route end to end on this machine.

The input is the example complex shipped with the kit, `tests/inputs/1BRS_AD.a3m` (barnase-barstar, PDB 1BRS chains A and D, 199 residues, the complex
form with no homologs), predicted into `--out <dir>` with the colabfold_batch options the caller adds, if any. Without `--out` the results
directory is a fresh private one (tempfile.mkdtemp, mode 0700, named on a `WARM out=` line before the launch): colabfold_batch unpickles
result files it finds in a results directory, so `warm` never reads or writes a fixed name in the shared temporary directory. A given
`--out` that exists is refused by name (rc 2, nothing launched) when another account could have written into it: writable by group or
other, or owned by neither this user nor root (uid 0 accepts any owner: a container's root over a bind-mounted host directory). There is
no cache the package manages; the JAX compilation cache is the environment's own (stock/PINS.json image.env JAX_COMPILATION_CACHE_DIR).

PASS = `pred`'s verdict (cli.pred → manifest.verdict: rc 0 — for a kit mode the activation report present and, when active,
`calls >= 1` at exit; every job's completion marker). The line reports the launch record's facts: the ranked PDBs found against the
expectation (`pdbs=<found>/<expected>`, settings.models_per_seed × num_seeds of the colabfold_batch options passed — 5 × 1 at upstream's
defaults), `active`, `calls` (the record's `kit_state_exit`), `partial` (the levers of a PARTIAL state, else none). Printed as
`[colabfold-opt] WARM PASS|FAIL|NOT ACTIVE ...`; the exit code is `pred`'s own (0 / 1 / 3), never collapsed.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import List, Optional, Tuple

from . import cli as _cli, manifest as _manifest, modes as _modes, report, settings as _settings, stack

WARM_INPUT = os.path.join("tests", "inputs", "1BRS_AD.a3m")
OUT_PREFIX = "colabfold_opt_warm-"                                                       # the private results directory's name prefix under the temporary directory (tempfile.mkdtemp)


def input_path() -> str:
    return os.path.join(stack.tree_home(), WARM_INPUT)


def out_refusal(path: str) -> Optional[str]:
    """Why an existing ``--out`` directory is refused — None when it is absent (colabfold_batch makes it) or when nothing another account
    could have planted would be read from it: refused when it is writable by group or other, or when its owner is neither this uid nor root
    (uid 0 accepts any owner). colabfold_batch unpickles the result files it finds there before it predicts, so the owner / mode is the check."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    uid = os.geteuid()
    if st.st_mode & 0o022:
        return f"{path} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {path}, or name a directory of your own"
    if uid != 0 and st.st_uid not in (uid, 0):
        return f"{path} belongs to uid {st.st_uid}, not to this process (uid {uid}) or root; fix: name a directory of your own"
    return None


def results_dir(out: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(results directory, None), or (None, the refusal): ``--out`` as given (absolute) when acceptable (out_refusal); without ``--out`` a
    fresh directory private to this user (mkdtemp: mode 0700, a name no other process predicts)."""
    if out:
        path = os.path.abspath(out)
        why = out_refusal(path)
        return (None, why) if why else (path, None)
    return os.path.abspath(tempfile.mkdtemp(prefix=OUT_PREFIX)), None                    # absolute: mkdtemp under a relative TMPDIR may name it relatively


def main(a, mode: str, options: List[str]) -> Tuple[int, Optional[dict]]:
    results, why = results_dir(a.out)
    if why:                                                                               # an --out another account could have written: refused by name, nothing launched
        report.emit(report.line("WARM REFUSED", mode=mode, rc=_cli.EXIT_USAGE, reason="out_dir", out=os.path.abspath(a.out)) + f": {why}")
        return _cli.EXIT_USAGE, None
    if not a.out:
        report.emit(report.line("WARM", mode=mode, out=results) + " (a private temporary directory made for this run; --out DIR names your own)")
    rc, rec = _cli.pred(a, mode, [input_path(), results] + list(options))
    pdbs = sorted(f for f in os.listdir(results) if re.match(r".*_unrelaxed_rank_\d{3}_.*\.pdb$", f)) if os.path.isdir(results) else []
    expected = _settings.models_per_seed(options) * _settings.num_seeds(options)         # ranked model files per job: the passed --num-models × --num-seeds, else upstream's 5 × 1
    man = rec or {}                                                                       # pred's launch record (None: refused before the launch)
    v = man.get("verdict") or {}
    exit_state = _manifest.lever_state_exit(man, _modes.LEVER if mode == "fast" else _modes.HOST_LEVER) or {}   # the mode's headline lever: the kernel's attention calls (fast), the resident apply calls (exact)
    calls = exit_state.get("calls") if isinstance(exit_state, dict) else None
    active = bool(man.get("active"))
    ok = rc == 0
    tag = "WARM PASS" if ok else ("WARM NOT ACTIVE" if rc == _cli.EXIT_NOT_ACTIVE else "WARM FAIL")
    report.emit(report.line(tag, mode=mode, rc=rc, reason=v.get("reason") or "none", pdbs=f"{len(pdbs)}/{expected}", active=int(active), calls=calls,
                            partial=",".join(list(v.get("lever_fallbacks") or {}) + list(v.get("partial") or [])) or "none", out=results))
    if a.json:
        print(report.dump({"ok": ok, "rc": rc, "verdict": v, "pdbs": pdbs, "expected_pdbs": expected, "active": active, "calls": calls, "out": results}))
    return rc, man or None

"""python -m rfdiffusion1_opt {design,check,warm} [--mode off|exact|fast] [KEY=VALUE ...] [--pack K] [--det 0|1] [--dry-run] ...

A thin command layer over the kits' code and the upstream command line. It never re-implements a lever, a mode or a test:

* ``design``  — designs (design.py). The input is upstream's own and nothing else: hydra ``KEY=VALUE`` overrides exactly as
               ``scripts/run_inference.py`` takes them (``inference.input_pdb=… 'contigmap.contigs=[…]' 'ppi.hotspot_res=[…]'
               inference.output_prefix=… inference.num_designs=… inference.design_startnum=…``: one target per invocation, outputs at the typed
               prefix, the run record beside them). ``--mode off`` = the stock command line in a proven-clean subprocess; a kit mode = the mode's line
               (the resident driver) in one resident process, every typed override composed on it, its evidence lines read back; a request the kit
               line cannot serve (symmetry, cyclic peptides, fold conditioning, …: upstream_args.refused) is refused by name, exit 3, before anything
               runs (``--mode off`` runs it); ``--pack K`` = K resident workers of the exact line on one GPU under CUDA MPS (serve.py, the packed line);
               ``opt_manifest.json`` beside the outputs.
* ``check``   — dry run: resolve and gate the mode on this box (GPU / stack / checkout / weights); reports the upstream pin; touches nothing;
               exit 3 when this box would refuse; ``--json`` for the full report and the mode table.
* ``warm``    — the cold start of every shape of the kit's bundled example targets through ``design`` (first design per target: JIT
               specialisation, CUDA-graph capture, allocator pools), off the results.

No mode given = `fast`, on the design route and the packed line alike (modes.DEFAULT_MODE_RULE).

Exit codes (report.py, the one home: EXIT_OK / EXIT_FAIL / EXIT_USAGE / EXIT_NOT_ACTIVE): 0 ok · 1 the run failed, or outputs short of the
request · 2 usage · 3 not active: refused before anything ran (``check``: would refuse), or a lever on the stock path after the pass (no
evidence line / a forbidden line — ``partial``: ``[rfdiffusion1-opt] NOT ACTIVE: partial activation — <levers>: <reason>; exit 3``, the outputs
kept); outputs short of the request are exit 1 (report.verdict). The stock command line's own exit code is passed through on ``--mode off``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE

VERBS = ("design", "check", "warm")
VALUE_FLAGS = ("--mode", "--pack", "--det", "--out_dir")   # the kit flags that take a value token (--out_dir is warm's: where its scratch designs go)
PACK_WORKER_GB_ENV, PACK_WORKER_GB_DEFAULT = "MODEL_OPT_PACK_WORKER_GB", 10.0   # the per-worker device footprint (GB) the packing launcher's memory estimate uses for `--pack K` (a NOTE, never a gate; configs/<card>.env may export another value)


def _worker_gb() -> float:
    v = os.environ.get(PACK_WORKER_GB_ENV)
    try:
        return float(v) if v and v.strip() else PACK_WORKER_GB_DEFAULT
    except ValueError:
        return PACK_WORKER_GB_DEFAULT


def _common(p: argparse.ArgumentParser) -> None:
    from .modes import MODE_NAMES, default_mode   # the modes and the default, from the one table; any other name is refused by name at resolve (exit 3), never by argparse
    p.add_argument("--mode", default=None, help=f"{' | '.join(MODE_NAMES)}; or RFDIFFUSION1_OPT; default: {default_mode()}; any other name is refused by name (exit 3)")
    p.add_argument("overrides", nargs="*", metavar="KEY=VALUE", help="upstream's hydra overrides, verbatim — the target itself (inference.input_pdb, contigmap.contigs, ppi.hotspot_res, inference.output_prefix, inference.num_designs, inference.design_startnum, …) and any other key; none = upstream's defaults. --mode off passes them to run_inference.py unchanged; a kit mode composes them on its resident driver's configuration and refuses by name, exit 3, what its line cannot serve (symmetry, cyclic peptides, fold conditioning, …; --mode off runs those)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rfdiffusion1-opt", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"rfdiffusion1_opt {__version__}")
    sub = ap.add_subparsers(dest="verb")
    d = sub.add_parser("design", help="designs: upstream's hydra overrides for one target (scripts/run_inference.py's own KEY=VALUE tokens); outputs at inference.output_prefix, the run record (opt_manifest.json, run.log, run_timings.json) beside them"); _common(d)
    d.add_argument("--pack", type=int, default=None, metavar="K", help="the packed line: K resident workers of the kit line (exact or fast) on this GPU under CUDA MPS (common/mps_packing/mps_workers.sh), each a disjoint slice of the target's designs (inference.num_designs a multiple of K); K = 1 runs one worker through the same launcher; refused by name on --mode off, as a request the kit line cannot serve is under a kit mode")
    d.add_argument("--det", type=int, default=0, choices=[0, 1], help="1: the deterministic recipe on both arms — upstream's per-design seed (inference.deterministic=True) on the stock command line and on the kit line alike, so exact's outputs are byte-comparable with off's at the same design index; 0 (default): upstream's unseeded default on every arm")
    d.add_argument("--dry-run", action="store_true", help="resolve, gate and print the commands; run nothing")
    c = sub.add_parser("check", help="dry run: resolve and gate the mode on this box; report the upstream pin"); _common(c)
    c.add_argument("--pack", type=int, default=None, metavar="K", help="check the packed line's gates too (the launcher, nvidia-cuda-mps-control)")
    c.add_argument("--json", action="store_true", help="print the full activation report, the pin report and the mode table")
    w = sub.add_parser("warm", help="run the first design of each bundled example target (the cold start, off the results)"); _common(w)
    w.add_argument("--out_dir", required=True, help="the scratch directory the warm-up designs and their record go to (not a result)")
    return ap


def cmd_design(a) -> int:
    from . import design
    if a.pack is not None:
        from . import serve
        rc, man = serve.run(a.mode, a.overrides, k=a.pack, worker_gb=_worker_gb(), dry_run=a.dry_run, det_flag=bool(a.det))
    else:
        rc, man = design.run(a.mode, a.overrides, dry_run=a.dry_run, det_flag=bool(a.det))
    if man.get("status") in ("refused", "usage"):
        return EXIT_USAGE if man.get("status") == "usage" else EXIT_NOT_ACTIVE
    if a.dry_run:
        print(json.dumps({k: man.get(k) for k in ("status", "reason", "mode", "line", "levers_planned", "env_row", "driver_cmd", "stock_cmds", "serve", "cases")}, indent=1, default=str))
        return EXIT_OK if man.get("status") == "dry-run" else EXIT_NOT_ACTIVE
    print(json.dumps({"status": man.get("status"), "reason": man.get("reason"), "out_dir": man.get("out_dir"), "n_pdb": (man.get("outputs") or {}).get("n_pdb"),
                      "levers_evidenced": man.get("levers_evidenced"), "levers_missing": man.get("levers_missing"),
                      "manifest": os.path.join(man.get("out_dir") or "", "opt_manifest.json")}, indent=1))
    return rc


def cmd_check(a) -> int:
    from . import design, modes, stack
    mode, pack = a.mode, a.pack
    _, refused, overrides = design.request(mode, a.overrides)            # a request the kit line cannot serve: the same NOT ACTIVE line design prints, exit 3
    if refused:
        return EXIT_NOT_ACTIVE
    rep = stack.activate(mode, overrides, dry_run=True, served=pack is not None)
    pins = stack.pins_report()                                           # the upstream pin and the installed stack against stock/PINS.json: reported, never a gate
    for line in pins.get("lines") or []:
        sys.stderr.write(line + "\n")
    if a.json:
        print(json.dumps({"report": rep, "pins": pins, "mode_table": modes.table()}, indent=1, default=str))
    if rep.get("reason") or rep.get("would_refuse"):                    # not a mode here (unknown), or the box would refuse: the NOT ACTIVE line printed by activate
        return EXIT_NOT_ACTIVE
    return EXIT_OK


def cmd_warm(a) -> int:
    from . import warm
    res = warm.run(a.mode, a.out_dir, a.overrides)
    print(json.dumps({k: res.get(k) for k in ("status", "reason", "first_designs")}, indent=1, default=str))
    if "rc" in res:
        return res["rc"]                                                                 # the design pass's exit code, unchanged
    return EXIT_OK if res.get("status") == "ok" else (EXIT_NOT_ACTIVE if res.get("status") == "refused" else EXIT_FAIL)


def main(argv: Optional[List[str]] = None) -> int:
    from .upstream_args import is_override, split_hydra_flags
    ap = build_parser()
    hydra_flags, rest = split_hydra_flags(list(sys.argv[1:] if argv is None else argv))   # Hydra's own flags (--config-name symmetry, -cn, --cfg job, …) are upstream's arguments: they ride with the overrides, verbatim
    upstream, ours, prev = [], [], ""
    for t in rest:                                                                            # upstream's KEY=VALUE / +KEY / ~KEY tokens wherever they stand among the kit's flags (Hydra takes them anywhere); a value of a kit flag stays with it
        (upstream if is_override(t) and prev not in VALUE_FLAGS else ours).append(t)             # a hydra override token (KEY=VALUE, +KEY=…, ++KEY=…, ~KEY[=…]): upstream_args' one grammar
        prev = t
    a = ap.parse_args(ours)
    if hasattr(a, "overrides"):
        a.overrides = hydra_flags + upstream + list(a.overrides)
        if a.verb in ("design", "warm"):                                                       # the verbs that run upstream (check is dry)
            from . import schedules
            a.overrides += schedules.override(a.overrides, dry_run=bool(getattr(a, "dry_run", False)))   # upstream's schedule cache directory when the checkout's own cannot or must not be used (schedules.py); nothing when one is typed or the checkout's is this user's to write
    elif hydra_flags or upstream:
        ap.error(f"upstream's arguments {hydra_flags + upstream} belong to design / check / warm")
    if not a.verb:
        ap.print_help()
        return EXIT_USAGE
    try:
        return {"design": cmd_design, "check": cmd_check, "warm": cmd_warm}[a.verb](a)
    except KeyboardInterrupt:
        return 130

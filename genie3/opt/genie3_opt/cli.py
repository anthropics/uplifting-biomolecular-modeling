"""python -m genie3_opt {design,check,warm} [--mode off|exact|fast] ...

A thin command layer over the kit's code and the upstream command line. It never re-implements a lever or a mode:

* ``design``  — designs from a request file (design.py): ``--mode off`` = the stock command line in a proven-clean subprocess, with upstream's
               own `generate` flags passed through (``--verbose --log-dir --num-devices --shard-id --num-shards``); a kit mode = the mode's
               line (the kit's driver) in one process, cwd = the checkout — dataset shards (``--num-shards --shard-id``) and the sequence stage
               (``predict_sequence``) on every mode, a request naming a computation the kit line does not run (beam search, the sidechain
               stage, more than one device) refused by name before anything runs (exit 3; ``--mode off`` runs it); outputs as upstream writes
               them plus ``opt_manifest.json``.
* ``check``   — the activation dry run (stack.activate dry_run): resolves and gates the mode on this box, applies nothing, prints the
               ``DRY-RUN`` / ``NOT ACTIVE`` line, the stock pins report and the weights digests (``--json``: the whole report and the mode
               table). Exit 0 when the mode would activate, 3 when it would not (named fact), 2 on usage. No --mode = the default mode
               (modes.DEFAULT_MODE, fast).
* ``warm``    — one public request through the mode's line (warm.py).

Exit codes: 0 ok · 1 the run failed (a driver / stock process error, a forbidden line in the driver's log, a numerics readback that contradicts
the line), or the PDB count differs from the request (``incomplete: <found>/<expected>`` in the manifest) · 2 usage · 3 not active / refused
before anything ran (a deployment fact: no CUDA device, no checkout, a weight file missing, the shared core absent; on a kit mode a request naming a
computation the kit line does not run — beam search, the sidechain stage, more than one device: ``NOT ACTIVE: mode=<m> cannot serve …``; a shard
index outside 0 <= K < M; or a batch the driver's memory
model refuses on this card, with the batch size that fits named; or, after the pass, a planned lever that could not run on this box — a kernel that
would not compile or launch, an unsupported shape: ``NOT ACTIVE: mode=<m> refused — lever(s) <ids> could not run …``, ``levers_missing`` in the
manifest — because a mode is all of its levers, never a subset under its name). Uncertainty about the environment alone (another GPU, another
torch / triton patch level, a card without a cell row) is named on the activation lines and every lever engages: never a refusal. ``warm`` exits as
its design does. The environment route (``GENIE3_OPT=<mode> genie3 generate …``, _autoload.py) hands the command to ``design`` and exits as it does,
and refuses any other genie3 command under a kit mode (exit 3); the in-process route returns the code from ``run_design``.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .codes import EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE, TAG

VERBS = ("design", "check", "warm")


def _common(p: argparse.ArgumentParser) -> None:
    from .modes import DEFAULT_MODE, MODE_NAMES
    p.add_argument("--mode", default=None, help=f"{' | '.join(MODE_NAMES)}; or GENIE3_OPT; default: {DEFAULT_MODE} (modes.DEFAULT_MODE); any other name is refused before anything runs (exit {EXIT_NOT_ACTIVE})")


def _stock_flags(p: argparse.ArgumentParser) -> None:
    """Upstream's own `genie3 generate` flags, spelled as upstream spells them (src/genie3/cli.py:57-110): passed verbatim to the stock child; on a
    kit mode `--shard-id` / `--num-shards` slice the request as upstream does (the driver's own flags) and `--num-devices` > 1 is refused by name
    (design.kit_refusals: the kit line is one process on one GPU; shards are its multi-GPU form)."""
    g = p.add_argument_group("upstream generate flags (passed to `genie3 generate` verbatim)")
    g.add_argument("--verbose", action="store_true", help="upstream's --verbose (detailed runtime output); on a kit line, the driver's per-batch detail")
    g.add_argument("--log-dir", dest="log_dir", default=None, help="upstream's --log-dir (default on the stock route: <out_dir>/logs when --out_dir is given, else upstream's own logs/runs)")
    g.add_argument("--num-devices", dest="num_devices", type=int, default=None, metavar="N", help="upstream's --num-devices (overrides runtime.num_devices); a kit mode is one process on one GPU and refuses N > 1 by name (exit 3): shards are its multi-GPU form, or --mode off")
    g.add_argument("--shard-id", dest="shard_id", type=int, default=None, help="upstream's --shard-id K (0 <= K < M): this pass writes shard K's share of every problem's designs, on every mode")
    g.add_argument("--num-shards", dest="num_shards", type=int, default=None, help="upstream's --num-shards M: the request sliced as upstream slices it, on every mode")


def build_parser() -> argparse.ArgumentParser:
    from . import det
    ap = argparse.ArgumentParser(prog="genie3-opt", description=__doc__.split("\n\n")[0], allow_abbrev=False)   # flags are spelled in full: no prefix of one flag selects another
    ap.add_argument("--version", action="version", version=f"genie3_opt {__version__}")
    sub = ap.add_subparsers(dest="verb")
    d = sub.add_parser("design", help="designs from a request file (upstream's experiment YAML)", allow_abbrev=False); _common(d)
    d.add_argument("--input", "-c", "--config", dest="input", required=True, help="upstream's experiment YAML (the file `genie3 generate -c/--config` reads: paths, generation.dataset, generation.sampler); `-c` / `--config` are upstream's own spellings (run.sh keeps `--config <card>` for the kit's card before `--`; upstream's argv rides verbatim after `--`)")
    d.add_argument("--out_dir", default=None, help="paths.rootdir for this pass (default: the request's own paths.rootdir / generation.io.outdir, as upstream resolves it)")
    d.add_argument("--tag", default="run")
    d.add_argument("--n", type=int, default=None, help="generation.dataset.n_sample for this pass (designs per problem)")
    d.add_argument("--selections", default=None, help="generation.dataset.selections for this pass (comma-separated problem keys)")
    d.add_argument("--seed", type=int, default=None, help="experiment.seed for this pass (default: the request file's own; absent = upstream's unseeded default)")
    d.add_argument("--batch_size", dest="batch_size", type=int, default=None, metavar="B", help="generation.dataset.batch_size for this pass, every mode: designs per denoiser call (upstream reads the key on the stock route; a kit line runs --batch-size B). Default: the request file's own, else upstream's 1 on every mode; --batch_size 8 is the batched throughput configuration of the kit modes")
    d.add_argument("--det", type=int, default=det.DEFAULT_LEVEL, choices=det.LEVELS, help="0 (default): upstream's own numerics — experiment.seed as the request or --seed leaves it; 1: the deterministic recipe (det.py): the recipe's seed 0 unless the request or --seed sets one, plus the kit line's graph-vs-eager check")
    _stock_flags(d)
    c = sub.add_parser("check", help="activation dry run on this box", allow_abbrev=False); _common(c)
    c.add_argument("--json", action="store_true")
    w = sub.add_parser("warm", help="one public request through the mode's line", allow_abbrev=False); _common(w)
    w.add_argument("--out_dir", required=True)
    w.add_argument("--input", default=None, help="a request file (default: upstream's binder-design example)")
    w.add_argument("--n", type=int, default=1)
    return ap


def cmd_design(a) -> int:
    from . import design, report
    report.register_exit_tally()
    rc, man = design.run(a.input, a.out_dir, a.mode, tag=a.tag, seed=a.seed, n_sample=a.n, selections=a.selections, det_level=a.det,
                         batch=a.batch_size, verbose=bool(a.verbose), log_dir=a.log_dir, num_devices=a.num_devices, shard_id=a.shard_id, num_shards=a.num_shards)
    if rc == EXIT_NOT_ACTIVE:
        print(f"[{TAG}] refused: {man.get('reason') or man.get('capacity_note') or man.get('status')}", file=sys.stderr)
    return rc


def cmd_check(a) -> int:
    from . import modes, stack
    rep = stack.activate(a.mode, dry_run=True)
    ok = not rep.get("reason") and not rep.get("would_refuse")
    if a.json:
        print(json.dumps({"activation": rep, "mode_table": modes.table(), "default_mode": modes.DEFAULT_MODE, "would_activate": ok}, indent=1, default=str))
    else:
        from .report import note_line
        w = rep.get("would_refuse") or ([rep["reason"]] if rep.get("reason") else [])
        for line in w:
            print(f"[{TAG}] would refuse: {line}")
        for note in rep.get("notes") or []:                                        # noted, never refused (stack.hardware_notes, the stock pins report)
            print(note_line(note))
        co = ((rep.get("pins") or {}).get("detail") or {}).get("checkout") or {}
        print(f"[{TAG}] pins checkout={co.get('root')} pinned={co.get('pinned')} files_checked={co.get('files_checked')} git_head={co.get('git_head')} "
              f"modified={len(co.get('git_modified') or [])} differences={len((rep.get('pins') or {}).get('bad') or [])} weights_pinned={rep.get('weights_pinned')}")
        print(f"[{TAG}] check mode={rep.get('mode')} would_activate={ok} gpu={(rep.get('gpu') or {}).get('name')} class={(rep.get('gpu') or {}).get('class')} "
              f"key={(rep.get('gpu') or {}).get('key')} pinned_stack={(rep.get('stack') or {}).get('pinned')} genie3_root={rep.get('genie3_root')} weights={rep.get('weights')} "
              f"line={rep.get('line')}")
    return EXIT_OK if ok else EXIT_NOT_ACTIVE


def cmd_warm(a) -> int:
    from . import warm
    res = warm.run(a.mode, a.out_dir, request_path=a.input, n=a.n)
    return int(res["rc"]) if "rc" in res else EXIT_NOT_ACTIVE                     # the design's own code; refused before anything ran: 3


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.verb not in VERBS:
        ap.print_usage(sys.stderr)
        return EXIT_USAGE
    return {"design": cmd_design, "check": cmd_check, "warm": cmd_warm}[a.verb](a)


if __name__ == "__main__":
    sys.exit(main())

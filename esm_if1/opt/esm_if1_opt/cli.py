"""The command line: ``esm_if1-opt design|check …`` == ``python -m esm_if1_opt design|check …``.

    design (PDBFILE | --input <dir|file>) [--out <dir> | --outpath FILE] [--mode off|fast] [--det [0|1]] [--seed N] [--batch_size B]
           [--chain C] [--temperature T] [--num-samples N] [--multichain-backbone | --singlechain-backbone] [--nogpu]
    check  [--mode M]

``design --mode fast`` (the default): the ACTIVE line (lever ``batched_sampling``, on the single-chain and the ``--multichain-backbone``
route alike), a NOT APPLIED line for ``--det 1``, the WEIGHTS line (``stack.stage_torch_home``), then ONE kit child — ``python -m
esm_if1_opt.kit_design <driver arguments>`` — the batched driver over every structure, printing its clock lines (its STARTUP line names the
device: ``cuda`` when torch sees a GPU and ``--nogpu`` is absent, else ``cpu``); this process prints the INVOCATION line, the LEVER line
composed from the child's ITEM records (``state=on batches=… rows=…``, or ``state=off reason=no_batch_ran`` when the child ended before its
first forward), accounts for every structure (``outputs.census_paths``), writes ``opt_manifest.json`` and prints EXIT.
``design --mode off`` (stock): the NOT ACTIVE banner, a NOT APPLIED line naming ``--seed`` / ``--batch_size`` (and ``--det 1``) — upstream is
unseeded and samples one sequence per call — the WEIGHTS line, then per structure ONE proven stock child — ``python -s -m
esm_if1_opt.stock_design --proof-json <out>/stock_env_proof.json --env-absent <prefixes> --kit-dirs "" -- <script> <the script's own argv>``
(``opt_core.stock_proof.stock_command``) in the stripped environment (``stack.stock_env``); each child proves itself clean (``ENV-CLEAN ok`` |
``NOT STOCK`` exit 3) and runs upstream's ``sample_sequences.py`` as shipped on that structure; one INVOCATION line per child, then the census,
the manifest and EXIT. ``check``: what the tree carries and what the box has (CHECK lines: upstream pin, fair-esm, torch, the weights file
with its size and sha256 beside the expected values, the GPU against ``MODEL_OPT_TARGET_GPU``) and one DRY-RUN line for the mode; exit 3
when the mode cannot run here. Exit codes: 0 ok · 1 failed / incomplete · 2 usage · 3 not active / not stock.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

from opt_core import gates, stock_proof
from opt_core.home import HomeError

from . import ActivationError, inputs, manifest, modes, outputs, report, settings, stack, upstream_fix

VERBS = ("design", "check")
PROOF_FILE = "stock_env_proof.json"


def _usage(msg: str) -> int:
    report.emit(f"{report.PREFIX} usage: {msg}")
    return report.EXIT_USAGE


def cmd_design(tokens: List[str]) -> int:
    t0 = time.time()
    p = settings.parser()
    mode = None
    try:
        ns = p.parse_args(tokens)
        mode = modes.resolve(ns.mode)
        kit = settings.check_values(ns)
        input_path = inputs.input_arg(ns)                               # upstream's positional pdbfile or --input, exactly one
        files, stems = inputs.resolve(input_path)
        out_dir, fasta_paths = outputs.plan(ns.out, ns.outpath, files, stems)   # --out DIR (seqs/<stem>.fasta each) or --outpath FILE (one structure)
        script = stack.stock_script()                                   # the tree (opt/, stock/, upstream_issues/) must be where the package expects it
        fixes = upstream_fix.resolve(upstream_fix.parse(ns.upstream_fix), stack.tree_home())   # --upstream-fix <ID>: refused by name when unknown (usage); nothing loads here
        rep = stack.enable(mode)                                        # off: inactive (stock) · fast: batched_sampling on; needs fair-esm + torch (exit 3)
    except SystemExit as e:                                             # argparse: usage (2) / --help (0)
        return int(e.code) if isinstance(e.code, int) else report.EXIT_USAGE
    except (modes.ModeError, settings.SettingsError, inputs.InputError, outputs.OutputError, upstream_fix.UnknownUpstreamFix) as e:
        return _usage(str(e))
    except (ActivationError, HomeError) as e:
        report.emit(report.refused(str(e), mode))
        return report.EXIT_NOT_ACTIVE
    route = modes.route_of(mode)
    fix_tokens = upstream_fix.child_tokens(fixes)                        # [] unless --upstream-fix: the leading hand-off pair of every child
    outpath = None if ns.out else fasta_paths[stems[0]]                 # the --outpath layout: one FASTA, the run record beside it
    if mode == modes.OFF:
        report.emit(report.not_active_stock())
        ignored = [("seed", kit["seed"]), ("batch_size", kit["batch"])] + ([("det", 1)] if kit["det"] else [])
        report.emit(report.not_applied(route, ignored, "mode off runs upstream's sample_sequences.py as shipped: unseeded, one sequence per model call"))
    else:
        report.emit(report.active_line(rep, batch_size=kit["batch"], seed=kit["seed"], det=kit["det"], multichain=bool(ns.multichain_backbone), nogpu=bool(ns.nogpu)))
        if kit["det"]:
            report.emit(report.not_applied(route, [("det", 1)], "no deterministic-algorithm switch is set or needed: the seed makes the sampled records repeat"))
    weights = stack.stage_torch_home()
    report.emit(report.weights_line(weights))
    for d in {os.path.dirname(p) for p in fasta_paths.values()} | {out_dir}:
        os.makedirs(d, exist_ok=True)
    proof_json = os.path.join(out_dir, PROOF_FILE)
    if mode == modes.OFF:
        env, removed = stack.stock_env(out_dir)
    else:
        env, removed = stack.kit_env(out_dir), []
    for stale in (env[stack.TIMING_ENV], proof_json):
        if os.path.exists(stale):
            os.unlink(stale)                                            # this pass's records and proof only
    children = []
    if mode == modes.OFF:                                               # upstream: one interpreter per structure, the script's own argv
        for f, stem in zip(files, stems):
            cmd = stock_proof.stock_command(sys.executable, report.STOCK_CHILD, proof_json=proof_json, env_absent=stack.MUST_BE_ABSENT_PREFIXES,
                                            kit_dirs=[], module_prefixes=[], args=fix_tokens + [script] + settings.script_argv(ns, os.path.abspath(f), fasta_paths[stem]))
            launch_ts = time.time()
            rc = subprocess.run(cmd, env=env, cwd=out_dir).returncode
            exit_ts = time.time()
            children.append({"name": stem, "argv": cmd, "launch_ts": round(launch_ts, 3), "exit_ts": round(exit_ts, 3), "wall_s": round(exit_ts - launch_ts, 3), "rc": rc})
            report.emit(report.invocation_line(route, stem, os.path.basename(script), launch_ts, exit_ts, rc, stack.cache_word()))
            if rc == stock_proof.EXIT_NOT_STOCK and not manifest.proof_ok(proof_json):
                return report.EXIT_NOT_ACTIVE                           # the child printed NOT STOCK; nothing ran
    else:                                                               # the kit child: the batched driver over every structure
        cmd = [sys.executable, "-m", report.KIT_CHILD] + fix_tokens + settings.driver_argv(ns, os.path.abspath(input_path), out_dir=out_dir, outpath=outpath)
        launch_ts = time.time()
        rc = subprocess.run(cmd, env=env, cwd=out_dir).returncode
        exit_ts = time.time()
        children.append({"name": "pass", "argv": cmd, "launch_ts": round(launch_ts, 3), "exit_ts": round(exit_ts, 3), "wall_s": round(exit_ts - launch_ts, 3), "rc": rc})
        report.emit(report.invocation_line(route, "pass", report.KIT_CHILD, launch_ts, exit_ts, rc, stack.cache_word()))
    rc = next((c["rc"] for c in children if c["rc"] != 0), 0)          # the first failing child's status stands; 0 when every child returned 0
    cache = stack.cache_word()
    census = outputs.census_paths(fasta_paths, int(ns.num_samples))
    v = report.verdict(rc, rep, census["incomplete"])
    census["pass_status"] = "ok" if v["exit_code"] == 0 else ("failed" if rc != 0 else "incomplete")
    records = manifest.read_records(os.path.join(out_dir, manifest.TIMING))     # the children's evidence records (timing.jsonl): STARTUP names the device the pass ran on, ITEM each batched forward
    if mode != modes.OFF:
        for text in report.lever_lines(rep, batch_size=kit["batch"], records=records):   # the LEVER line after the child: what its records show ran (state=on batches=… rows=…), not the plan
            report.emit(text)
    kit_block = {"route": route, "levers": list(rep.get("levers_applied") or []), "device": manifest.device_of(records),
                 "settings": {"num_samples": int(ns.num_samples), "temperature": float(ns.temperature), "chain": ns.chain,
                              "multichain_backbone": bool(ns.multichain_backbone), "nogpu": bool(ns.nogpu)},
                 "seed": {"value": kit["seed"], "applied": mode != modes.OFF}, "batch": {"value": kit["batch"], "applied": mode != modes.OFF},
                 "det": {"requested": kit["det"], "applied": False}, "upstream_fix": [fid for fid, _ in fixes],
                 "inputs": {"path": os.path.abspath(input_path), "n": len(files), "stems": list(stems)},
                 "outputs": {"dir": out_dir, "seqs": (outputs.SEQS_DIR if ns.out else ""), "files": [os.path.relpath(fasta_paths[s], out_dir) for s in stems]},
                 "weights": dict(weights, cache=cache), "children": children, "environment_removed": {"variables": removed},
                 "proof": manifest.read_proof(proof_json) if mode == modes.OFF else None, "script": script if mode == modes.OFF else None}
    path = manifest.write_run(out_dir, mode=mode, route=route, report_=rep, command=list(sys.argv), kit=kit_block, census=census, exit_=v,
                              wall_s=time.time() - t0, records=records)
    report.emit(manifest.line(path))
    report.emit(report.exit_line(mode=mode, route=route, rc=rc, inputs=len(files), n_fasta=census["n_items_complete"], designs_written=census["n_records"],
                                 designs_expected=len(files) * int(ns.num_samples), wall=time.time() - t0, incomplete=census["incomplete"], exit_code=v["exit_code"]))
    return int(v["exit_code"])


def cmd_check(tokens: List[str]) -> int:
    p = argparse.ArgumentParser(prog="esm_if1-opt check", description="Report what the tree carries and what this box has; nothing runs.")
    p.add_argument("--mode", default=None, metavar="M", help=f"{'|'.join(modes.MODES)}")
    try:
        ns = p.parse_args(tokens)
        mode = modes.resolve(ns.mode)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else report.EXIT_USAGE
    except modes.ModeError as e:
        return _usage(str(e))
    try:
        f = stack.facts()
    except HomeError as e:
        report.emit(report.refused(str(e), mode))
        return report.EXIT_NOT_ACTIVE
    g = stack.gpu()
    tm = stack.target_match(g)
    wpath = f["weights"]["env"] if f["weights"]["env_is_file"] else (f["weights"]["hub_entry"] if f["weights"]["hub_entry_present"] else None)
    size = os.path.getsize(wpath) if wpath else None
    digest = gates.sha256_file(wpath) if wpath else None
    report.emit(report.line("CHECK", upstream=f"facebookresearch/esm@{stack.UPSTREAM_COMMIT[:8]}", fair_esm=f["upstream"]["installed"] or "none",
                            carried=stack.FAIR_ESM_VERSION, archive=stack.ARCHIVE, torch=f["torch"] or "none", stack=f["stack_key"], python=sys.executable))
    report.emit(report.line("CHECK", weights=stack.WEIGHTS_FILE, path=wpath or "none", bytes=size if size is not None else "none",
                            expected_bytes=stack.WEIGHTS_BYTES, sha256=digest or "none", expected_sha256=stack.WEIGHTS_SHA256,
                            match=("none" if digest is None else str(digest == stack.WEIGHTS_SHA256)), torch_home=f["weights"]["torch_home"]))
    ok = bool(f["upstream"]["installed"] and f["torch"])
    why = None if ok else f"mode {mode} cannot run: fair-esm / torch not installed on this interpreter (run.sh install; STOCK.md)"
    report.emit(report.dry_run_line(mode=mode, route=modes.route_of(mode), levers=(",".join(modes.levers(mode)) or "none"), esm=f["upstream"]["installed"] or "none",
                                    torch=f["torch"] or "none", gpu=report.gpu_word(g), target=tm["target"] or "none",
                                    match=("none" if tm["match"] is None else str(tm["match"])),
                                    weights=("env" if f["weights"]["env_is_file"] else ("hub" if f["weights"]["hub_entry_present"] else "absent")), ok=str(ok)))
    if not ok:
        report.emit(report.refused(why, mode))
        return report.EXIT_NOT_ACTIVE
    return report.EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        sys.stderr.write(__doc__.split("\n\n")[1] + "\n")
        return report.EXIT_OK
    if not argv or argv[0] not in VERBS:
        sys.stderr.write(__doc__.split("\n\n")[1] + "\n")
        return report.EXIT_USAGE
    verb, tokens = argv[0], argv[1:]
    return cmd_design(tokens) if verb == "design" else cmd_check(tokens)


if __name__ == "__main__":
    sys.exit(main())

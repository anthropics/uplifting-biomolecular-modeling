"""proteinmpnn-opt — the command line: design | check | warm.

    proteinmpnn-opt design --mode off|exact [--variant V] [--bb_batch K] [--det 0|1] [--hybrid_gemm 0|1] [--allow-partial] <input> <output folder> [<stock options>]
                    input:  --jsonl_path <parsed.jsonl> | --pdb_path <file.pdb>  (upstream's own)  |  --input <dir of PDB files | parsed.jsonl | file.pdb>
                    output: --out_folder <dir>  (upstream's own)  |  --out <dir>
    proteinmpnn-opt check  --mode exact [--variant V] [--bb_batch K] [--det 0|1] [--hybrid_gemm 0|1] [--json]     dry run: resolves and gates the mode on this box, applies nothing (DRY-RUN line)
    proteinmpnn-opt warm   --mode exact [--variant V] [--bb_batch K] [--det 0|1] [--hybrid_gemm 0|1] [--keep] [--allow-partial] [--json]   one public-input design pass in the mode, its lines relayed

Mode = --mode, else $PROTEINMPNN_OPT; neither is a usage error (exit 2) whose one line names off|exact — the family default is fast, a tier this
engine does not ship, so it has no default mode; a --mode that disagrees with a set $PROTEINMPNN_OPT is refused. Variant (the weight set) =
--variant, else $PROTEINMPNN_VARIANT, else modes.DEFAULT_VARIANT — vanilla, upstream's own default; upstream's ``--use_soluble_model`` /
``--path_to_model_weights`` pass through and select the weights exactly as protein_mpnn_run.py reads them. The input and the output folder are
given once each, under upstream's names (``--jsonl_path`` / ``--pdb_path``, ``--out_folder``: an existing protein_mpnn_run.py command line runs
as it is behind ``design --mode M``) or the kit's (``--input`` also takes a directory of PDB files: the parse step runs first; ``--out``); two
spellings of one thing is a usage error. ``--bb_batch K`` (kit modes): backbones per worker batch, the mode line's 16 unless given — speed and
GPU memory only, the outputs do not depend on it (modes.resolve). ``--det 0|1`` is accepted and inert (det.py: both routes are deterministic by
construction at the pass's own --seed). ``design --mode off`` is the stock route (stock_run.py:
NOT ACTIVE: mode off, the ENV-CLEAN line, the stock exit code); ``check`` / ``warm`` take the kit modes only. Beyond ``--mode``, the inputs and
the outputs, ``design`` takes only the stock command line's own options, spelled and defaulted as upstream defines them, and hands them to the
route's executable verbatim (settings.py: an option upstream does not define, a kit lever or an option the package supplies itself is refused by
name, exit 2, as upstream's argparse refuses an unknown option); they are printed on the activation line (``stock_args=``) and recorded in the
manifest. Mode exact serves the design pass with every option of it (the seed — 0 draws one per process as upstream draws it —, the sequence
counts and temperatures, the omitted / biased residues and PSSM dictionaries, a PDB directory, a parsed jsonl or one PDB file with its
``--pdb_path_chains``); what its worker cannot serve — the CA-only model, the scoring-only and probability-only passes, tied decoding,
``--backbone_noise`` above 0 (settings.worker_refuses names each with the mechanism) — is refused by name before any process starts: one NOT ACTIVE
line, exit 3, ``--mode off`` runs the stock command line with those options. ``--hybrid_gemm 0`` leaves the base variants' probe-gated lever out of the
exact set by name (modes.OPT_OUTS; 1, the default, requests it). Every ``design`` prints one CMD line before each process it starts
(report.log_cmd: the route, the variant and the child's whole argv); every ``design`` / ``warm`` writes opt_manifest.json beside
the outputs and ends with the EXIT tally line. Exit codes: 0 ok, 1 the pass failed (or ``incomplete``: fewer ``seqs/*.fa`` than inputs, recorded
as ``incomplete: <n>/<m>``), 2 usage (an unknown mode or variant name included), 3 the mode refused by name (NOT ACTIVE). A mode is all of its
levers: it engages every lever of its line or refuses by name — it never runs under its name with a subset. It refuses when something a pass
cannot run without is absent (the carried kit files, the ProteinMPNN checkout, its weights file), when the checkout's HEAD is another commit than
the pin (another commit is another "stock"), when a lever of the line cannot run on this box (no CUDA device: the worker's CUDA-graph levers —
refused at activation, nothing launched, ``partial=<levers>`` names them; ``check`` shows the same plan and exits 3), and when the worker's own
on-device probe finds that a lever cannot engage bit-identically on this device (``hybrid_gemm: PROBE FAIL`` then ``hybrid_gemm: REFUSED``: the
worker stops before any output, exit 3 naming ``--hybrid_gemm 0`` — the line without the lever, by name; ``fused_draw: REFUSED`` likewise on a
stack without Triton, exit 3). What is
merely untested — another GPU model or arch, another torch or driver, weights of another digest — is named on the lines (``gpu=… match=no``, the
STACK line, ``NOT PINNED``) and runs. ``--mode fast`` is ``proteinmpnn ships no fast tier: select --mode exact``, before anything resolves.
``--allow-partial`` (or ``PROTEINMPNN_OPT_ALLOW_PARTIAL=1``) is the one recorded override: the levers that can run do, ``allow_partial: true`` in the
manifest, ``allow_partial=yes`` on the lines, the exit the pass's own; it does not apply to the probe's refusal (nothing partial ran). After a run,
a requested lever the worker's end-of-run record shows off is ``partial`` too (exit 3 unless ``--allow-partial``); a run that leaves no lever record
is ``partial_detection: none (...)``, exit by its rc. ``warm`` follows ``design``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

from . import ActivationError
from . import inputs as _inputs, kit_run, manifest, modes, outputs as _outputs, report, settings as _settings, stack, stock_run, warm

PROG = "proteinmpnn-opt"
USAGE = __doc__.split("\n\n")[1] + "\n"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", default=None, help=f"{'|'.join(modes.MODES)} (else $PROTEINMPNN_OPT; no default: this engine ships no fast tier, so a command names its mode)")
    p.add_argument("--variant", default=None, help=f"the weight set, {'|'.join(modes.VARIANTS)} (default: $PROTEINMPNN_VARIANT, else {modes.DEFAULT_VARIANT} — upstream's; "
                                                  "upstream's --use_soluble_model / --path_to_model_weights among the stock options select the weights as protein_mpnn_run.py does)")
    p.add_argument("--bb_batch", type=_whole, default=None, metavar="K",
                   help="kit modes: backbones per worker batch (default: the mode line's, 16) — larger is faster per backbone and takes more GPU memory, "
                        "1 designs one backbone at a time; the outputs do not depend on it")
    p.add_argument("--det", type=int, choices=(0, 1), default=0,
                   help="accepted and inert: both routes are deterministic by construction at the pass's own --seed (det.py: the recipe is argument values, nothing in the environment), so 1 changes nothing")


def _whole(text: str) -> int:
    """argparse type: a whole number, 1 or more."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"{n}: 1 or more")
    return n


def _opt_out_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hybrid_gemm", type=int, choices=(0, 1), default=1,
                   help="base variants, kit modes: 1 (default) requests the worker's probe-gated batched message GEMMs; 0 leaves the lever out of the set by name (modes.OPT_OUTS)")


def _opt_out(a: argparse.Namespace) -> List[str]:
    """The probe-gated levers the command line leaves out by name (``--hybrid_gemm 0``)."""
    return [name for name, flag in modes.OPT_OUTS.items() if getattr(a, flag.lstrip("-"), 1) == 0]


def _allow_partial_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--allow-partial", action="store_true",
                   help=f"a partial activation (a requested lever without evidence of application) is recorded and the pass proceeds; exit 3 otherwise "
                        f"(environment spelling: {stack.ENV_ALLOW_PARTIAL}=1)")


def _allow_partial(a: argparse.Namespace) -> bool:
    return bool(a.allow_partial) or stack.allow_partial_env()                     # the flag ORs with the environment spelling (one reader)


def _fold_evidence(rep: dict, res: modes.Resolution, ev: dict) -> None:
    """The run's lever evidence (kit_run.lever_evidence) into the activation report: ``partial`` = every requested lever the activation
    marked unavailable, the worker's record shows dropped, or the worker's probe gated off (resolution order); ``gated`` (the probe's verdict
    per lever) and ``levers_unobserved`` named beside it;
    ``partial_detection`` when the run left no record; ``evidence_missing`` when the worker route left none (the levers cannot be judged)."""
    dropped = set(rep.get("levers_unavailable") or []) | set(ev["dropped"]) | set(ev["gated"])
    rep["levers_fallback"] = [l for l in res.levers if l in ev["dropped"] or l in ev["gated"]]
    rep["levers_unobserved"] = list(ev["unobserved"])
    rep["gated"] = dict(ev["gated"])
    rep["partial"] = [l for l in res.levers if l in dropped]
    rep["levers_applied"] = [l for l in res.levers if l not in dropped and l not in ev["gated"]]
    if ev["detection"]:
        rep["partial_detection"] = ev["detection"]
        rep["evidence_missing"] = res.route == "worker"


def _mode_variant(a) -> tuple:
    try:
        return stack.resolve_mode_and_variant(a.mode, a.variant)
    except modes.UnsupportedMode as e:                                          # a standard mode this engine does not ship (fast): refused by name, 3
        raise CliError(str(e), EXIT_NOT_ACTIVE) from e
    except (ActivationError, modes.ModeError) as e:
        raise CliError(str(e)) from e


# ----------------------------------------------------------------------------------------------------------------- design
def cmd_design(argv: List[str], echo: bool = True) -> int:
    p = argparse.ArgumentParser(prog=f"{PROG} design", allow_abbrev=False, description="one design pass: the stock command line (off) or the kit's line (exact)",
                                epilog="Every other `--option [value]` is an option of the stock command line itself (protein_mpnn_run.py), "
                                       "with upstream's name and default, handed to the design process verbatim: --seed, "
                                       "--num_seq_per_target, --batch_size, --sampling_temp, --omit_AAs, --omit_AA_jsonl, --bias_AA_jsonl, --bias_by_res_jsonl, "
                                       "--pssm_jsonl ..., --model_name, --save_score, --save_probs, ... (mode exact refuses by name, exit 3, the few it cannot "
                                       "serve — --ca_only, --score_only, the probability-only passes, --tied_positions_jsonl, --backbone_noise > 0: --mode off runs them).")
    _common(p)
    p.add_argument("--jsonl_path", default=None, help="upstream's own: the parsed jsonl (helper_scripts/parse_multiple_chains.py's output)")
    p.add_argument("--pdb_path", default=None, help="upstream's own: one PDB file (with --pdb_path_chains as upstream reads it; upstream reads no --chain_id_jsonl for one)")
    p.add_argument("--input", default=None, help="a directory of PDB files (parsed first), a parsed.jsonl or one PDB file — one spelling of the input")
    p.add_argument("--out_folder", default=None, help="upstream's own: the output folder (seqs/ scores/ probs/)")
    p.add_argument("--out", default=None, help="the same output folder, the kit's spelling — one of the two")
    p.add_argument("--chain_id_jsonl", default=None, help="the stock --chain_id_jsonl ({name: [[designed chains], [fixed chains]]}); not given = upstream's default (every chain of each entry designed, none fixed)")
    p.add_argument("--fixed_positions_jsonl", default=None, help="the stock --fixed_positions_jsonl")
    _opt_out_arg(p)
    _allow_partial_arg(p)
    a, stock_args = p.parse_known_args(argv)                                    # what the design parser does not define is the stock command line's (settings.parse decides, by name)
    a.input = _given_once(a, (("--jsonl_path", "jsonl_path"), ("--pdb_path", "pdb_path"), ("--input", "input")),
                          "an input: --jsonl_path <parsed.jsonl> | --pdb_path <file.pdb> (upstream's own) or --input <directory of PDB files | parsed.jsonl | file.pdb>")
    a.out = _given_once(a, (("--out_folder", "out_folder"), ("--out", "out")), "an output folder: --out_folder <dir> (upstream's own) or --out <dir>")
    try:
        kind = _inputs.classify_base(a.input)
    except _inputs.InputError as e:                                             # an input that cannot be read: the pass failed, 1 (the stock route's code for the same input)
        raise CliError(str(e), EXIT_FAIL) from e
    if a.jsonl_path is not None and kind != "parsed":
        raise CliError(f"--jsonl_path {a.jsonl_path}: not a parsed jsonl (a directory of PDB files is --input <dir>, one PDB file is --pdb_path <file>)")
    if a.pdb_path is not None and kind != "pdb":
        raise CliError(f"--pdb_path {a.pdb_path}: not a .pdb file (a parsed jsonl is --jsonl_path <file>, a directory of PDB files is --input <dir>)")
    return run_design(a, stock_args, echo=echo)


def _given_once(a: argparse.Namespace, spellings, what: str) -> str:
    """The one value among the spellings of one design argument; none or two given is a usage error naming the spellings."""
    got = [(flag, getattr(a, attr)) for flag, attr in spellings if getattr(a, attr) is not None]
    if len(got) != 1:
        raise CliError((f"design takes {what}" if not got else
                        f"design takes {what} — one of them, given once (got {' and '.join(f for f, _ in got)})"))
    return str(got[0][1])


def run_design(a: argparse.Namespace, stock_args: List[str], echo: bool = True) -> int:
    """One design pass from the parsed design arguments (``cmd_design``; warm.run calls it with ``echo`` False: the child output is not relayed)."""
    mode, variant = _mode_variant(a)
    allow_partial = _allow_partial(a)
    opt_out = _opt_out(a)
    bb_batch = getattr(a, "bb_batch", None)
    if bb_batch is not None and mode == "off":
        try:
            modes.resolve(mode, variant, stack.kit_home(), bb_batch=bb_batch)      # the table's own words: the stock route has no batch of backbones
        except modes.ModeError as e:
            raise CliError(str(e)) from e
    t0 = time.time()
    try:
        pairs = _settings.parse(stock_args, variant)                             # stock options only: no lever, nothing the package supplies, nothing upstream does not define
        model_name = _settings.model_name(pairs, variant)                        # the weights file the pass loads (--model_name as given, else upstream's default)
        weights_file = _settings.weights_path(pairs, variant, stack.mpnn_dir())   # the one file, by the same rule that renders the argv: the pin check digests and names it
        refused = _settings.worker_refuses(pairs, variant) if mode != "off" else []   # what mode exact cannot serve in this pass, by name with the mechanism: refused below before any process (--mode off runs it)
        designs = _settings.designs_per_target(pairs, variant)                  # upstream's whole-batches rule, the same on both routes: named on one NOTE line below when it trims the request
    except _settings.SettingsError as e:                                        # a token upstream's argparse would refuse: usage, 2
        raise CliError(str(e)) from e
    except _inputs.InputError as e:                                             # an input that cannot be read: the pass failed, 1 (the stock route's code for the same input)
        raise CliError(str(e), EXIT_FAIL) from e
    rep: Optional[dict] = None
    stats: dict = {"mode": mode, "variant": variant}
    rc = EXIT_FAIL
    try:
        if refused:                                                             # mode exact cannot serve this pass: refused by name, nothing staged or launched (the EXIT tally says so), exit 3
            rep = {"active": False, "mode": mode, "variant": variant, "stock_args": list(stock_args), "refused": [token for token, _ in refused],
                   "reason": _settings.refusal_reason(mode, refused)}
            report.log_activation({**rep, "dry_run": False})                    # NOT ACTIVE: each option with its mechanism, exit 3, --mode off named
            manifest.write(a.out, rep, command="design", argv=sys.argv[1:], exit_code=EXIT_NOT_ACTIVE, stock_args=stock_args, extra={"refused": rep["refused"]})
            rc = EXIT_NOT_ACTIVE
            return rc
        notes = [n for n in (report.designs_note(designs),
                             report.CHAIN_ID_UNREAD_NOTE_FMT % a.chain_id_jsonl if a.chain_id_jsonl and _inputs.classify_base(a.input) == "pdb" else None) if n]
        for n in notes:                                                         # named consequences of the pass's own options, the same on both routes (report.note_line)
            report.log_note(n)
        extra = {"designs_per_target": designs, "notes": notes}
        if mode == "off":
            rep = stack.activate("off", variant, dry_run=True, model_name=model_name, weights_file=weights_file)   # the stock route's plan (no lever requested); the weights file the pass loads is the one named
            rep["mode"] = mode
            rep["reason"] = rep.get("reason") or "mode off (stock route)"
            rep["stock_args"] = list(stock_args)
            report.log_activation({**rep, "active": False, "dry_run": False})
            if not rep["pins"].get("pinned"):
                raise CliError("stock pin: " + "; ".join(rep["pins"].get("findings") or []), EXIT_NOT_ACTIVE)
            rec = stock_run.run(variant, a.input, a.out, stock_args, chain_id_jsonl=a.chain_id_jsonl, fixed_positions_jsonl=a.fixed_positions_jsonl, echo=echo)
            stack.mark_launched()
            kit_rec = None
        else:
            rep = stack.enable(mode, variant, model_name=model_name, opt_out=opt_out, weights_file=weights_file, bb_batch=bb_batch,
                               allow_partial=allow_partial)                     # levers that cannot run on this box refuse the mode by name here (ActivationError) unless --allow-partial
            rep["stock_args"] = list(stock_args)                               # the whole line: the row's levers and every stock option given
            rep["allow_partial"] = allow_partial
            report.log_activation(rep)
            res = modes.resolve(mode, variant, stack.kit_home(), opt_out=opt_out, bb_batch=bb_batch)
            rec = kit_run.run(res, a.input, a.out, stock_args, chain_id_jsonl=a.chain_id_jsonl, fixed_positions_jsonl=a.fixed_positions_jsonl, echo=echo)
            kit_rec = {"lines": rec.get("lines"), "worker_record": rec.get("worker_record"),
                       "commands": [{k: v for k, v in c.items() if k not in ("lines", "worker_record")} for c in rec["commands"]]}
            if rec.get("lowmem"):
                kit_rec["lowmem"] = rec["lowmem"]                               # the derived executable (stage.stage_lowmem)
            wr = rec.get("worker_record") or {}
            if isinstance(wr.get("hybrid_gemm_probe"), dict):                 # the worker's own cell: what it saw and decided (never rewritten), its word folded into probe.verdict
                rep["probe"] = dict(rep.get("probe") or {}, worker=wr["hybrid_gemm_probe"], verdict=kit_run.probe_word(wr["hybrid_gemm_probe"]))
            elif (rep.get("probe") or {}).get("requested"):
                rep["probe"] = dict(rep["probe"], verdict="unobserved")         # requested, and the record carries no probe cell: named, never guessed
            if wr.get("refused"):                                              # the worker refused the job by name before any output (its PROBE FAIL / REFUSED lines above): the refusing
                lever = str(wr["refused"]).split(":")[0].strip()                # lever ("<lever>: <why>"), gated off by its own start-up probe (the record's "<lever>_probe" cell); nothing was designed
                rep["gated"] = {lever: f"{lever} probe: {kit_run.probe_word(wr.get(lever + '_probe'))}"}
            else:                                                              # the job ran: each requested lever judged against the worker's end-of-run record
                _fold_evidence(rep, res, kit_run.lever_evidence(res, rec.get("worker_record")))
        rc = EXIT_OK if rec["rc"] == 0 else EXIT_FAIL
        listing = _outputs.listing(a.out, variant)
        counts = _outputs.counts(listing)
        n_in = rec.get("n_inputs")
        incomplete = f"{counts.get('seqs', 0)}/{n_in}" if n_in is not None and counts.get("seqs", 0) < n_in else None   # outputs short of the request
        if kit_rec is not None and (rec.get("worker_record") or {}).get("refused"):   # a mode is all of its levers: the worker's on-device probe refused one, so the job was refused by name
            rc, rep["active"], incomplete = EXIT_NOT_ACTIVE, False, None
            why = str(rec["worker_record"]["refused"])
            rep["reason"] = (f"mode {mode} refused by name on this device: {why} — the lever cannot engage bit-identically here and the line is all of its levers; nothing was designed "
                             + (f"({modes.OPT_OUTS.get('hybrid_gemm', '--hybrid_gemm')} 0 runs the line without the lever, by name)" if why.startswith("hybrid_gemm")
                                else "(--mode off runs the stock command line)"))
            report.log_activation({**rep, "dry_run": False, "weights": None})   # the NOT ACTIVE line, after the worker's own PROBE FAIL / REFUSED lines
        elif rc == EXIT_OK and rep.get("evidence_missing"):                  # the worker route without its end-of-run record: nothing proves the levers
            rc, rep["active"] = EXIT_NOT_ACTIVE, False
            rep["reason"] = "no end-of-run record from the worker: the requested levers cannot be judged (the worker's output above)"
        elif rc == EXIT_OK and rep.get("partial") and not allow_partial:
            rc, rep["active"] = EXIT_NOT_ACTIVE, False
            rep["reason"] = f"partial activation: {','.join(rep['partial'])} requested without evidence of application ({report.partial_escape(rep)})"
        elif rc == EXIT_OK and incomplete:
            rc = EXIT_FAIL
        if rep.get("partial") and rec["rc"] == 0 and not rep.get("evidence_missing"):
            report.log_partial(rep)                                              # the family line: NOT ACTIVE (3) or PARTIAL allowed (recorded)
        stats.update({"route": rec["route"], "rc": rec["rc"], "inputs": n_in, **{f"n_{k}": v for k, v in counts.items()}, "wall_s": round(rec["wall_s"], 1),
                      "probe": (rep.get("probe") or {}).get("verdict") if (rep.get("probe") or {}).get("requested") else None,   # the worker's on-device verdict: PASS | FAIL | unobserved
                      "partial": ",".join(rep["partial"]) if rep.get("partial") else None, "allow_partial": "yes" if rep.get("allow_partial") else None,
                      "incomplete": incomplete, "exit": rc})
        manifest.write(a.out, rep, command="design", argv=sys.argv[1:], exit_code=rc, stock_args=stock_args,
                       inputs={"path": os.path.abspath(a.input), "n_items": n_in, "parsed": rec.get("parsed"), "assigned": rec.get("assigned"),
                               "parser": rec.get("parser"), "order": rec.get("order")},   # order: who fixed the backbone order (the parse helper's file-system order for a directory; given for a jsonl / one file)
                       outputs={"counts": counts, "listing": listing}, kit=kit_rec,
                       stock=({"env_proof": rec.get("env_proof"), "commands": rec.get("commands")} if rec["route"] == "stock" else None), wall_s=time.time() - t0,
                       extra={"incomplete": incomplete, **extra})
    except (stock_run.StockError, kit_run.KitRunError, _inputs.InputError, _settings.SettingsError) as e:
        raise CliError(str(e), EXIT_FAIL) from e
    except ActivationError as e:
        if a.out:
            manifest.write(a.out, {"active": False, "mode": mode, "variant": variant, "reason": str(e)}, command="design", argv=sys.argv[1:], exit_code=EXIT_NOT_ACTIVE, stock_args=stock_args)
        raise CliError(str(e), EXIT_NOT_ACTIVE) from e
    finally:
        report.log_exit_tally(stats if "rc" in stats else None)
    return rc


# ----------------------------------------------------------------------------------------------------------------- check
def cmd_check(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog=f"{PROG} check", allow_abbrev=False, description="dry run: resolve and gate the mode on this box; nothing is applied")
    _common(p)
    _opt_out_arg(p)
    p.add_argument("--json", action="store_true", help="print the activation report as JSON on stdout")
    a = p.parse_args(argv)
    mode, variant = _mode_variant(a)
    if mode == "off":
        raise CliError("check --mode off: nothing to resolve for the stock route (design --mode off runs it)")
    rep = stack.activate(mode, variant, dry_run=True, opt_out=_opt_out(a), bb_batch=a.bb_batch)
    for note in (rep.get("pins") or {}).get("notes") or []:
        print(f"{report.PREFIX} check: {note}", file=sys.stderr, flush=True)   # the checkout / weights facts stock/check_pins.py reports (never a gate)
    report.log_activation(rep)
    if rep.get("partial") and not rep.get("reason"):
        report.log_partial(rep)                                                  # the family line on a partial plan (allow_partial: the environment spelling)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    partial_refused = bool(rep.get("partial")) and not rep.get("allow_partial")
    return EXIT_OK if not rep.get("reason") and not partial_refused else EXIT_NOT_ACTIVE      # the code design would give on this box


# ----------------------------------------------------------------------------------------------------------------- warm
def cmd_warm(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog=f"{PROG} warm", allow_abbrev=False, description="one public-input design pass in the mode: graph capture on this box")
    _common(p)
    _opt_out_arg(p)
    p.add_argument("--keep", action="store_true", help="keep the outputs (a temporary directory otherwise)")
    p.add_argument("--json", action="store_true", help="print the result as JSON on stdout")
    _allow_partial_arg(p)
    a = p.parse_args(argv)
    mode, variant = _mode_variant(a)
    if mode == "off":
        raise CliError("warm --mode off: the stock route has nothing to warm (design --mode off runs it)")
    try:
        res = warm.run(mode, variant, keep=a.keep, allow_partial=_allow_partial(a), opt_out=_opt_out(a), bb_batch=a.bb_batch)
    except FileNotFoundError as e:
        raise CliError(str(e), EXIT_NOT_ACTIVE) from e
    print(warm.summary_line(res), flush=True)
    if a.json:
        print(json.dumps(res, indent=1, default=str))
    return EXIT_OK if res["status"] == "PASS" else (EXIT_NOT_ACTIVE if res["rc"] == EXIT_NOT_ACTIVE else EXIT_FAIL)   # design's own code


# ----------------------------------------------------------------------------------------------------------------- entry
COMMANDS = {"design": cmd_design, "check": cmd_check, "warm": cmd_warm}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, end="")
        return EXIT_OK if argv else EXIT_USAGE
    fn = COMMANDS.get(argv[0])
    if fn is None:
        print(f"{report.PREFIX} unknown command {argv[0]!r}\n{USAGE}", end="", file=sys.stderr)
        return EXIT_USAGE
    try:
        return fn(argv[1:])
    except CliError as e:
        print(f"{report.PREFIX} ERROR: {e}", file=sys.stderr, flush=True)
        return e.code

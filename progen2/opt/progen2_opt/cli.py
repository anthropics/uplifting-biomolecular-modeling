"""python -m progen2_opt {sample,score,check} [--mode exact|off] <the stock script's own flags> ...

A thin command layer over the kits and the upstream CLIs; the flags of `sample` / `score` ARE sample.py's / likelihood.py's own (names, types,
defaults), plus the mode word. It never re-implements an optimization, a mode or a test:

* ``sample``  — sample.py. ``--mode off``: ONE clean stock `sample.py` process with exactly the flags given (stock_cli.py: the environment
                proved clean of package variables), its stdout / stderr this process's own, its exit code returned. ``--mode exact``: in-process
                (generate.py): the model loaded once as sample.py loads it with the generation kit's levers installed, one unit per call;
                stdout = the block the stock prints (the context line, per sample a blank line + index + completion, `done.`).
* ``score``   — likelihood.py. ``--mode off``: ONE clean stock `likelihood.py` process. ``--mode exact``: in-process (score.py): the stock load
                order, the scoring kit's ``apply``, its ``Scorer`` at B=1 per direction; stdout = `ll_sum=` / `ll_mean=`.
  The one flag upstream lacks, on both: ``--input FILE --out_dir DIR`` — a multi-item job through ONE loaded model (outputs.py: FILE = JSON lines
  keyed by the per-item flags' dests; DIR/items/<item_id>/block.txt per item, nothing else); under exact the job's batch sizes size the
  load: the model load paid once, the static K/V slots of the job's `num_samples` values held from the load.
* ``check``   — the activation line for (mode, size) without applying anything: kit line, what would be on, the pins (stock files hashed,
                the stack compared), kit dirs present, GPU + card class, stack key. Runs nothing.

Every flag reaches the route as the stock script types it, under both modes: ``--mode off`` hands them to the stock script untouched;
``--mode exact`` passes ``--fp16`` / ``--device`` / ``--rng-deterministic`` to the in-process load, every unit's seed call and (score) the
forwards' autocast, ``--rng-seed`` / the per-item flags to each unit, runs likelihood.py's own sanity section before the scoring when
``--sanity true`` (its lines precede the ll pair) and sample.py's sanity witness once per load (whatever the flag: the stock's own order). A
setting outside the tested defaults (fp16=false, rng_deterministic=false, another card) is NAMED on the stack line and the levers engage —
every one of them: a mode is all of its levers. ``--device cpu`` (or a box without a CUDA device) is refused by name (exit 3): the levers run
on CUDA; ``--mode off --device cpu`` runs the stock on the cpu.

Exit codes: 0 ok | 1 failed: check, the load, or an item (the outputs short of the request) | 2 usage (a flag or mode word not on the
surface: `--mode fast` is not shipped for this model) | 3 not active by name — the stock files at the stock dir off the pinned commit (that
changes what stock means: every mode refuses), the kit tree not whole, a kit refusal by name (``KitRefused``), ``score``'s partial activation
(the scoring kit's rotary-table check or counted-path assertion contradicts its composition: the outputs stay), no CUDA device /
``--device cpu``, or a lever that cannot run here (``<lever> cannot run: <reason>`` — a mode never runs under its name with a subset) |
otherwise the stock's exit code.

Every stock process the package spawns runs under ``$PROGEN2_PYTHON`` when set (STOCK.md 'Variables'), else this interpreter (stack.python),
with one wall budget (stack.PROCESS_TIMEOUT_S; a process past it is killed and reported); the card is ``CUDA_VISIBLE_DEVICES``'s, as for
the stock scripts.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import report
from .modes import DEFAULT_MODE, MODEL_NAMES, MODES, NOT_SHIPPED_MODES, STOCK_DEFAULT_MODEL, variant_of_model
from .outputs import SAMPLE_ARG_OF
from .score import KIT_REFUSED                       # the kits' fail-loud class, by name

PROG = "python -m progen2_opt"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3

# The stock scripts' own flags (sample.py main() L112-122, likelihood.py main() L122-128): dest -> flag. Every value is kept as the string given
# and handed to the stock script as given; the stock's own types (int / float / `str(x).lower() == 'true'`) are applied only where the package
# reads a value itself (the in-process load, seed and forwards: stock_settings below).
JOB_FLAGS = {"sample": {"device": "--device", "rng_deterministic": "--rng-deterministic", "fp16": "--fp16", "sanity": "--sanity"},
             "score": {"device": "--device", "rng_seed": "--rng-seed", "rng_deterministic": "--rng-deterministic", "fp16": "--fp16", "sanity": "--sanity"}}
ITEM_FLAGS = {"sample": dict(SAMPLE_ARG_OF), "score": {"context": "--context"}}
STOCK_ORDER = {"sample": ("device", "rng_seed", "rng_deterministic", "p", "t", "max_length", "num_samples", "fp16", "context", "sanity"),
               "score": ("device", "rng_seed", "rng_deterministic", "fp16", "context", "sanity")}
JOB_DEFAULTS = {"sample": {"device": "cuda:0", "rng_deterministic": "true", "fp16": "true", "sanity": "true"},                       # sample.py L113-121
                "score": {"device": "cuda:0", "rng_seed": "42", "rng_deterministic": "true", "fp16": "true", "sanity": "false"}}    # likelihood.py L123-128
FIRST_CARD = ("cuda:0", "cuda")                       # the stock's default device, the one the levers were tested on: the first visible card (CUDA_VISIBLE_DEVICES picks it)


def stock_bool(x) -> bool:
    """sample.py / likelihood.py's own boolean type: `lambda x: (str(x).lower() == 'true')`."""
    return str(x).lower() == "true"


def stock_settings(cmd: str, a) -> dict:
    """The job-level flags as the stock script types them, its own defaults where not given (JOB_DEFAULTS): device (str), fp16 /
    rng_deterministic / sanity (its boolean type) and, for score, rng_seed (int) — what the exact routes load, seed and run with."""
    given = {dest: (getattr(a, dest) if getattr(a, dest) is not None else JOB_DEFAULTS[cmd][dest]) for dest in JOB_FLAGS[cmd]}
    out = {"device": str(given["device"]), "fp16": stock_bool(given["fp16"]), "rng_deterministic": stock_bool(given["rng_deterministic"]), "sanity": stock_bool(given["sanity"])}
    if "rng_seed" in given:
        out["rng_seed"] = int(given["rng_seed"])
    return out


def outside_defaults(settings: dict) -> list:
    """The settings given outside the tested defaults, as the stack line names them (the levers engage: an uncertainty named, never a
    reason to disengage): fp16=false, rng_deterministic=false, device=<a card other than the first>."""
    out = []
    if not settings["fp16"]:
        out.append("fp16=false")
    if not settings["rng_deterministic"]:
        out.append("rng_deterministic=false")
    if settings["device"] not in FIRST_CARD and settings["device"].split(":")[0] != "cpu":
        out.append(f"device={settings['device']}")
    return out


USAGE = f"""usage: {PROG} <command> [--mode {'|'.join(MODES)}] <the stock script's own flags> ...

commands (--model = the stock's checkpoint name: {' | '.join(MODEL_NAMES)}; progen2-bfd90 is accepted for progen2-BFD90)
  sample  [--mode M] [--model progen2-large] [--device cuda:0] [--rng-seed 42] [--rng-deterministic true] [--p 0.95] [--t 0.2]
          [--max-length 256] [--num-samples 1] [--fp16 true] [--context 1] [--sanity true]
                                            sample.py's own flags and defaults. mode off: the stock script in ONE clean process, its own
                                            stdout / stderr and exit code; exact: in-process — the model loaded once as sample.py loads it,
                                            the generation kit's levers installed, one unit — stdout = the block the stock prints (the
                                            context line, per sample a blank line + index + completion, `done.`)
  score   [--mode M] [--model progen2-base] [--device cuda:0] [--rng-seed 42] [--rng-deterministic true] [--fp16 true]
          [--context <likelihood.py's own default>] [--sanity false]
                                            likelihood.py's own flags and defaults. mode off: the stock script in ONE clean process; exact:
                                            the scoring kit in-process — stdout = `ll_sum=<float>` / `ll_mean=<float>` (after likelihood.py's
                                            own sanity lines when --sanity true)
    the one flag upstream lacks, on both: --input FILE --out_dir DIR
                                            a multi-item job through ONE loaded model (the load paid once; exact holds the static K/V slots of
                                            the job's batch sizes from the load). FILE = JSON lines (or a JSON list), one object
                                            per item keyed by the per-item flags' dests (sample: context, max_length, num_samples, t, p,
                                            rng_seed; score: context; plus item_id, default item<index>); a missing key takes the stock
                                            default; those flags are then refused on the command line (--input names them per item), the
                                            others apply to every item. Output: DIR/items/<item_id>/block.txt per item, nothing else
  check   [--mode M] [--model M]            dry run: the pins + the activation resolved and gated on this box; nothing applied

modes    {' | '.join(MODES)} (default {DEFAULT_MODE}); {' / '.join(NOT_SHIPPED_MODES)} are not shipped for this model (refused by name, exit 2)
         every flag reaches the route as the stock types it under both modes (exact: --fp16 / --device / --rng-deterministic to the load,
         each unit's seed call and each forward; score --sanity true runs likelihood.py's sanity section first; sample.py's sanity witness
         runs once per load whatever the flag). A setting outside the tested defaults is named on the stack line and the levers
         engage — all of them: a mode is all of its levers or it refuses by name
exit     0 ok | 1 failed (check, the load, an item) | 2 usage | 3 not active by name: the stock files off the pinned commit,
         no CUDA device / --device cpu (--mode off runs the stock on the cpu), a lever that cannot run here (named with its reason),
         score's partial activation (reason printed)
"""


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


class _Parser(argparse.ArgumentParser):
    def error(self, message):                          # argparse's own wording, the package's exit table (2 = usage)
        raise CliError(f"{self.prog}: {message}")


def check_mode(mode) -> str:
    """--mode when given, else the package default; an unknown word is a usage error (exit 2)."""
    mode = DEFAULT_MODE if mode is None or str(mode).strip() == "" else str(mode).strip().lower()
    if mode in NOT_SHIPPED_MODES:
        raise CliError(f"--mode {mode}: not shipped for this model (modes: {' | '.join(MODES)})")
    if mode not in MODES:
        raise CliError(f"unknown mode {mode!r} (choose from {', '.join(MODES)})")
    return mode


def variant_of(model, route: str) -> str:
    """The size word of --model (the stock script's own default when absent: sample.py progen2-large, likelihood.py progen2-base)."""
    try:
        return variant_of_model(model if model is not None else STOCK_DEFAULT_MODEL[route])
    except ValueError as e:
        raise CliError(f"argument {e}") from e


def activate(mode: str, variant, route, dry_run: bool = False, device=None, outside=None) -> dict:
    from . import stack
    try:
        rep = stack.activate(mode, variant, dry_run=dry_run, route=route, device=device, outside_defaults=outside)
    except Exception as e:  # noqa: BLE001
        return {"active": False, "mode": mode, "variant": variant, "route": route, "reason": f"activation raised {e!r}"}
    rep.setdefault("mode", mode)
    return rep


def _stock_parser(cmd: str) -> _Parser:
    p = _Parser(prog=f"{PROG} {cmd}", allow_abbrev=False, add_help=False)
    p.add_argument("-h", "--help", action="store_true")
    p.add_argument("--mode", default=None)
    p.add_argument("--model", default=None)
    for dest in STOCK_ORDER[cmd]:
        p.add_argument((JOB_FLAGS[cmd].get(dest) or ITEM_FLAGS[cmd].get(dest)), dest=dest, default=None)
    p.add_argument("--input", default=None)
    p.add_argument("--out_dir", default=None)
    return p


def given_argv(cmd: str, a, dests) -> list:
    """The flags among `dests` the user gave, in the stock's own order, with the values as given."""
    out = []
    for dest in STOCK_ORDER[cmd]:
        if dest in dests and getattr(a, dest) is not None:
            out += [(JOB_FLAGS[cmd].get(dest) or ITEM_FLAGS[cmd][dest]), getattr(a, dest)]
    return out


def _prepare(cmd: str, argv: list):
    """Parse, then the usage rules shared by sample / score: returns (a, mode, variant, job_argv, items | None, settings)."""
    from . import outputs, stack
    a = _stock_parser(cmd).parse_args(argv)
    if a.help:
        print(USAGE, end="")
        raise CliError("", EXIT_OK)
    mode = check_mode(a.mode)
    variant = variant_of(a.model, cmd)
    given_items = given_argv(cmd, a, ITEM_FLAGS[cmd])
    items = None
    if a.input is not None:
        if given_items:
            raise CliError(f"--input names them per item: {' '.join(given_items[0::2])} cannot be given on the command line with --input")
        if not a.out_dir:
            raise CliError("--input FILE goes with --out_dir DIR (the job's items/<item_id>/block.txt)")
        if not os.path.isfile(a.input):
            raise CliError(f"--input {a.input}: no such file")
        try:
            items = outputs.load_items(a.input, cmd, stack.stock_dir())
        except ValueError as e:
            raise CliError(f"--input {a.input}: {e}") from e
    elif a.out_dir:
        raise CliError(f"{cmd} --out_dir goes with --input FILE (a single call prints its block on stdout and writes nothing)")
    return a, mode, variant, given_argv(cmd, a, JOB_FLAGS[cmd]), items, stock_settings(cmd, a)


def single_item(cmd: str, a) -> dict:
    """The one item of a single call: the per-item flags given (typed as the stock types them), the stock defaults otherwise; item_id item0000."""
    from . import outputs, stack
    it = {dest: getattr(a, dest) for dest in ITEM_FLAGS[cmd] if getattr(a, dest) is not None}
    return outputs.load_items_from([it], cmd, stack.stock_dir())[0]


# ----------------------------------------------------------------------------------------------------------------- the stock arm
def run_stock(cmd: str, a, variant: str, job_argv: list, items) -> int:
    """The stock route (mode off): ONE clean stock process per call (its streams this process's own), or one per item of a job (block.txt per item)."""
    from . import outputs, stack, stock_cli
    mode = "off"
    pins = stack.pins()
    wd = stack.workdir(variant)
    up = stack.upstream_name(variant)
    report.register_exit_tally(mode, cmd)                          # the EXIT line of a stock process: items + kit_modules=none (the proof no kit was on the path)
    spec = stock_cli.env_absent_spec(pins)
    allowed = stock_cli.env_allowed(pins, stack.DATA_ENV)
    env = stock_cli.clean_env(dict(os.environ), spec, allowed)
    script = stock_cli.script_of(cmd)
    if items is None:                                              # the single call: --model + the flags as given, nothing else
        print(f"{report.PREFIX} stock: one fresh `{script}` process from {wd} (env stripped of {','.join(spec)})", file=sys.stderr, flush=True)
        r = stock_cli.run_passthrough(cmd, ["--model", up] + job_argv + given_argv(cmd, a, ITEM_FLAGS[cmd]), up, wd, env, spec, allowed=allowed)
        print(report.item_line(outputs.default_item_id(0), r["wall_s"]) + ("" if r["rc"] == 0 else f" FAILED rc={r['rc']}"), file=sys.stderr, flush=True)
        report.note_items(1, mode, cmd)
        return r["rc"]
    writer = outputs.Writer(a.out_dir, cmd)
    print(f"{report.PREFIX} stock: one fresh `{script}` process per item from {wd} (env stripped of {','.join(spec)})", file=sys.stderr, flush=True)
    n_ok, rc_last, first = 0, 0, True
    for it in items:
        r = stock_cli.run_item(cmd, it, up, wd, env, spec, allowed=allowed, job_argv=job_argv)
        if first:
            first = False
            for line in (r["stderr"] or "").splitlines():
                if line.startswith(stock_cli.PREFIX):
                    print(line, file=sys.stderr, flush=True)
            if r["load_s"] is not None:
                print(report.ready_line(variant, r["load_s"]) + " (the stock's own `loading parameters took`)", file=sys.stderr, flush=True)
        if r["rc"] == 0 and r["block"]:
            writer.write_item(it, r["block"])
            n_ok += 1
        else:                                                      # a failed item is named with the child's own last words, never silent
            for line in (r["stderr"] or "").strip().splitlines()[-12:]:
                print(f"{report.PREFIX} item {it['item_id']} stderr: {line}", file=sys.stderr, flush=True)
        print(report.item_line(it["item_id"], r["wall_s"]) + ("" if r["rc"] == 0 and r["block"] else f" FAILED rc={r['rc']}"), file=sys.stderr, flush=True)
        rc_last = r["rc"]
    report.note_items(len(items), mode, cmd)
    return 0 if n_ok == len(items) else (rc_last or EXIT_FAIL)


# ----------------------------------------------------------------------------------------------------------------- sample / score
def cmd_sample(argv: list) -> int:
    from . import generate, outputs, stack
    a, mode, variant, job_argv, items, settings = _prepare("sample", argv)
    rep = activate(mode, variant, "sample", device=settings["device"], outside=outside_defaults(settings))
    if rep.get("refused"):                                         # refused by name, whatever the mode (the line is out: stack.activate): 3, nothing run
        return EXIT_NOT_ACTIVE
    if mode == "off":
        return run_stock("sample", a, variant, job_argv, items)
    if not rep.get("active"):                                      # refused by name (the line is out: stack.activate): 3, nothing generated
        return EXIT_NOT_ACTIVE
    report.register_exit_tally(mode, "sample")
    if items is None:
        units, emit = [single_item("sample", a)], (lambda it, block: (sys.stdout.write(block), sys.stdout.flush()))
    else:
        writer = outputs.Writer(a.out_dir, "sample")
        units, emit = items, writer.write_item
    try:
        summ = generate.run_items(units, variant, rep, emit, settings)
    except stack.ActivationError:                                  # a lever the load did not install: refused by name (the line is out), nothing generated
        return EXIT_NOT_ACTIVE
    except Exception as e:  # noqa: BLE001
        print(f"{report.PREFIX} sample FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        if type(e).__name__ == generate.OUT_OF_MEMORY:             # the load or the job's slots ran out of memory: named (the kit's OOM_POLICY: nothing retried or rerouted), exit 1
            return EXIT_FAIL
        raise
    return summ["rc"]


def cmd_score(argv: list) -> int:
    from . import outputs, score
    a, mode, variant, job_argv, items, settings = _prepare("score", argv)
    rep = activate(mode, variant, "score", device=settings["device"], outside=outside_defaults(settings))
    if rep.get("refused"):                                         # refused by name, whatever the mode (the line is out: stack.activate): 3, nothing run
        return EXIT_NOT_ACTIVE
    if mode == "off":
        return run_stock("score", a, variant, job_argv, items)
    if not rep.get("active"):                                      # refused by name (the line is out): 3, nothing scored
        return EXIT_NOT_ACTIVE
    report.register_exit_tally(mode, "score")
    if items is None:
        units, emit = [single_item("score", a)], (lambda it, block: (sys.stdout.write(block), sys.stdout.flush()))
    else:
        writer = outputs.Writer(a.out_dir, "score")
        units, emit = items, writer.write_item
    try:
        summ = score.run_items(units, variant, rep, emit, seed=settings["rng_seed"], fp16=settings["fp16"], device=settings["device"],
                               rng_deterministic=settings["rng_deterministic"], sanity=settings["sanity"])
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ != KIT_REFUSED:                        # the kit refused by name (apply, its pins): 3 via main(); anything else: the run failed, 1 with its name
            print(f"{report.PREFIX} score FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
    rc = summ["rc"]
    if summ.get("partial"):                                        # the rotary check or the counted path contradicts the composition: the kit did not compute what it names — the outputs stay, the exit is 3
        print(report.partial_line(summ["partial"], "the outputs stay", mode, variant, "score"), file=sys.stderr, flush=True)
        rc = EXIT_NOT_ACTIVE
    return rc


# ----------------------------------------------------------------------------------------------------------------- check
def cmd_check(argv: list) -> int:
    """stock/check_pins.py (the pin's own checker: the stock files' and the size's weights' digests refused off their pins; the interpreter and
    packages reported against the pinned stack) + the activation dry run."""
    from . import stack
    p = _Parser(prog=f"{PROG} check", allow_abbrev=False)
    p.add_argument("--mode", default=None)
    p.add_argument("--model", default=None)
    a = p.parse_args(argv)
    mode = check_mode(a.mode)
    variant = variant_of(a.model, "sample") if a.model is not None else None
    rc_pins, pins_out = 0, ""
    script = os.path.join(stack.tree_home(), "stock", "check_pins.py")
    if os.path.isfile(script):
        import subprocess
        cmd = [stack.python(), "-I", script] + stack.check_pins_argv(variant)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=stack.PROCESS_TIMEOUT_S)
        rc_pins, pins_out = r.returncode, (r.stdout + r.stderr)
        for line in pins_out.splitlines():
            print(f"{report.PREFIX} pins: {line}", file=sys.stderr, flush=True)
        print(f"{report.PREFIX} check_pins rc={rc_pins} ({' '.join(cmd[2:])})", file=sys.stderr, flush=True)
    else:
        print(f"{report.PREFIX} pins: stock/check_pins.py absent from {stack.tree_home()}", file=sys.stderr, flush=True)
        rc_pins = EXIT_FAIL
    rep = activate(mode, variant, None, dry_run=True)
    if rc_pins != 0:
        return EXIT_FAIL
    return EXIT_OK if not rep.get("would_refuse") else EXIT_NOT_ACTIVE


COMMANDS = {"sample": cmd_sample, "score": cmd_score, "check": cmd_check}


def main(argv=None) -> int:
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
        msg = str(e)
        if msg:
            print(msg if msg.startswith(report.PREFIX) else f"{report.PREFIX} ERROR: {msg}", file=sys.stderr, flush=True)
        return e.code
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ != KIT_REFUSED:                                # the kits' fail-loud class, by name: a refusal, exit 3 (anything else is the traceback, exit 1)
            raise
        print(report.refused_line(f"the kit refused: {e}"), file=sys.stderr, flush=True)
        return EXIT_NOT_ACTIVE

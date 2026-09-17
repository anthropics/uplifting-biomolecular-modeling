"""python -m boltzgen_opt {design,check,warm} [--mode exact|fast|big|off] ...

Console script ``boltzgen-opt``; ``run.sh`` wraps the same commands (plus ``install``).

* ``design``  — one design job from a spec YAML (design.py), in upstream's own words: ``design <spec.yaml> --output DIR [any `boltzgen run`
  option]`` — every option the kit does not define is upstream's and reaches its ``configure`` as given (upstream validates it; ``--cache``
  defaults to $BOLTZGEN_CACHE) —, then the mode's launch — the runner (``xa_run.py``) in a child process for a kit mode, the stock caller per
  pipeline step for ``off`` (stock_design.py). ``--seed N``: step i of the pipeline runs with ``seed + i`` and upstream's conformer generator
  is seeded from that state (the one recipe on every arm, stock_design.SEED_LINE + seed_default_rng); without it ``off`` runs upstream
  unseeded — upstream's own form — and a kit mode, whose runner is seeded by construction, is not active (exit 3, the line names ``--seed``).
  The kit modes run one GPU: ``--devices N > 1`` is refused by name there and served by ``--mode off``.
* ``check``   — dry run: resolve the mode against the lever directories' files and the GPU, apply nothing (stack.activate dry_run).
* ``warm``    — one design on the example spec that ships in the kit (warm.py) in the requested mode: JIT warm-up of the caches.

The mode comes from ``--mode`` or ``BOLTZGEN_OPT``; both given and different is a usage error. Neither given: the package default
(modes.DEFAULT_MODE = fast, the tolerance class; ``exact`` (bitwise), ``big`` and ``off`` are asked for by name). A kit mode that
cannot activate (a lever directory or the pinned upstream missing, no GPU, no ``--seed``) exits 3 with a NOT ACTIVE line naming the
escape and never runs upstream silently in its place; the card itself is never a refusal (a card other than the pinned one is named on a
NOTE line); ``--mode off`` is the explicit stock route and serves every request upstream serves. A mode is all of its levers: a PARTIAL
activation — the mode's launch ran but a lever of the mode could not run on this box (a kernel that does not import, compile or launch, an
unsupported shape or dtype) — is a refusal by name: one NOT ACTIVE stderr line naming the levers and their reasons, the activation line's
``partial=`` field, the manifest's ``partial`` list, exit 3 (the outputs are not the mode's); nothing runs under a mode's name with a subset
of its levers, and there is no opt-out. An untested card, driver or library patch level is never that: it is named and the levers engage.
What the accelerator census finds after a run (census.py; its account, kernels.py) is recorded in ``opt_manifest.json``, never an exit.
Every line the package prints is prefixed ``[boltzgen-opt]`` on stderr; ``opt_manifest.json`` is written beside the outputs by ``design``
and ``warm``.

Exit codes (codes.py — the one definition; imported below): 0 ok · 1 the run failed · 2 usage · 3 not active (a refused or partial mode).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import design, manifest, modes, report, stack, warm
from .codes import EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE   # noqa: E402 — the one definition (codes.py)

USAGE = """usage: boltzgen-opt <command> [options]
  design  <spec.yaml> --output DIR [--mode exact|fast|big|off] [--seed N] [<any `boltzgen run` option: --protocol --num_designs --diffusion_batch_size --steps --cache ...>]
  check   [--mode exact|fast|big|off]
  warm    [--mode exact|fast|big|off]
"""
IN_RUN_WHERE = "in the run: the outputs under {out} are not the mode's"      # design's <detail> tail: the partial state is judged after the run


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


def exit_for(child_rc: int, rep: dict, where: str = None) -> int:
    """The running verbs' exit from the child's rc and the report: EXIT_NOT_ACTIVE for a child that refused its own activation (it exited 3
    itself) and for a partial activation (a lever of the mode could not run on this box: the mode refuses by name — one NOT ACTIVE line,
    the report's ``partial``; a mode is all of its levers), EXIT_FAIL for a failed child, else EXIT_OK. The accelerator census
    (``rep["kernels"]``) is a record, not an exit."""
    if child_rc == EXIT_NOT_ACTIVE:
        return EXIT_NOT_ACTIVE
    if child_rc != 0:
        return EXIT_FAIL
    if rep.get("partial"):
        report.say(report.partial_exit_line(stack.partial_levers(rep), rep.get("fallback_reasons"), where=where))
        return EXIT_NOT_ACTIVE
    return EXIT_OK


def effective_mode(flag: Optional[str]) -> str:
    env_mode = modes.mode_from_env()
    if flag and env_mode and flag != env_mode:
        raise CliError(f"--mode {flag} disagrees with {modes.ENV}={env_mode} in the environment: unset one")
    try:
        return modes.check_mode(flag or env_mode)
    except ValueError as e:
        raise CliError(str(e)) from e


def devices_refusal(mode: str, n: int) -> str:
    """The NOT ACTIVE reason of a kit mode asked to run on several devices: the runner drives one GPU."""
    return f"mode {mode} runs the kits' runner on one GPU and was asked for --devices {n}; --mode off passes --devices to upstream; exit {EXIT_NOT_ACTIVE}"


def seed_refusal(mode: str) -> str:
    """The NOT ACTIVE reason of a kit mode asked for without ``--seed``: the runner is seeded by construction (xa_run.py <run_dir>
    <seed>) and no seed is invented; ``--mode off`` runs upstream unseeded."""
    return f"mode {mode} runs the kits' seeded runner and needs --seed N (step i runs with N + i); --mode off runs upstream unseeded; exit {EXIT_NOT_ACTIVE}"


# Upstream's `boltzgen run` / `configure` options (boltzgen 0.3.2, cli/boltzgen.py add_configure_arguments + add_models_download_options +
# the configure parser's --steps), by arity: 1 = one value, "+" = one or more values, 0 = a flag. `design` forwards each, as given and in the
# caller's order, to upstream's configure; `--output`, `--cache` and `--devices` are the three it also reads itself (defined on its parser).
# A test holds this table equal to the vendored wheel's parser (tests/test_cli_forms.py).
UPSTREAM_OPTIONS = {
    "--protocol": 1, "--config": "+", "--num_workers": 1, "--config_dir": 1, "--use_kernels": 1, "--moldir": 1, "--reuse": 0,
    "--num_designs": 1, "--diffusion_batch_size": 1, "--design_checkpoints": "+", "--step_scale": 1, "--noise_scale": 1,
    "--skip_inverse_folding": 0, "--inverse_fold_num_sequences": 1, "--inverse_fold_checkpoint": 1, "--inverse_fold_avoid": 1,
    "--only_inverse_fold": 0, "--folding_checkpoint": 1, "--affinity_checkpoint": 1, "--budget": 1, "--alpha": 1, "--filter_biased": 1,
    "--metrics_override": "+", "--additional_filters": "+", "--size_buckets": "+", "--refolding_rmsd_threshold": 1,
    "--force_download": 0, "--models_token": 1, "--steps": "+",
}
KIT_HANDLED_UPSTREAM_OPTIONS = ("--output", "--cache", "--devices")     # upstream's too; `design` reads them (its parser) and forwards --cache / --devices itself


def split_design_argv(argv: List[str]):
    """``(own, upstream)``: the tokens for `design`'s own parser and upstream's options in the caller's order. An upstream option takes the
    values upstream gives it (UPSTREAM_OPTIONS: one, or every token up to the next ``--option``); every token after a bare ``--`` is
    upstream's verbatim; an option neither defines is a usage error naming it (upstream's ``run --help`` lists upstream's)."""
    own, up, i = [], [], 0
    while i < len(argv):
        w = argv[i]
        name = w.split("=", 1)[0] if w.startswith("--") else None
        if w == "--":
            up.extend(argv[i + 1:]); break
        if name in UPSTREAM_OPTIONS:
            n = UPSTREAM_OPTIONS[name]
            up.append(w); i += 1
            if "=" in w or n == 0:
                continue
            if n == 1:
                if i >= len(argv):
                    raise CliError(f"design: upstream's option {name} takes a value")
                up.append(argv[i]); i += 1
            else:
                j = i
                while j < len(argv) and not argv[j].startswith("--"):
                    up.append(argv[j]); j += 1
                if j == i:
                    raise CliError(f"design: upstream's option {name} takes one or more values")
                i = j
            continue
        if name is not None and name not in KIT_HANDLED_UPSTREAM_OPTIONS + ("--mode", "--seed", "--help"):
            raise CliError(f"design: no such option {name} (design's own: --output --mode --seed --cache --devices; upstream's: UPSTREAM_OPTIONS, `boltzgen run --help`)")
        own.append(w); i += 1
    return own, up


def _gpu_for_report():
    return stack.gpu_probe()


def _not_active(mode: str, reason: str, out_dir: Optional[str], argv, gpu=None, gate=None, res=None, command: str = "design") -> int:
    rep = {"active": False, "mode": mode, "form": "process", "reason": reason, "gpu": gpu, "gpu_gate": gate}
    if res is not None:
        rep["switches"] = modes.describe_line(res)
    report.emit(rep)
    if out_dir:
        manifest.write(out_dir, rep, command=command, argv=argv, exit_code=EXIT_NOT_ACTIVE)
    return EXIT_NOT_ACTIVE


# ----------------------------------------------------------------------------------------------------------------- design
def cmd_design(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog="boltzgen-opt design", add_help=True, usage="boltzgen-opt design <spec.yaml> --output DIR [--mode exact|fast|big|off] [--seed N] [<boltzgen run options>]",
                                description="Upstream's own options (`boltzgen run --help`: --protocol --num_designs --diffusion_batch_size --steps --design_checkpoints "
                                            "--num_workers --use_kernels --config ... — cli.UPSTREAM_OPTIONS) are accepted as given and passed to its configure step.")
    p.add_argument("spec", help="the design spec YAML (upstream's positional argument)")
    p.add_argument("--output", required=True, help="output directory: upstream's run directory, plus opt_manifest.json and the kit's logs")
    p.add_argument("--mode", default=None, choices=list(modes.MODES))
    p.add_argument("--seed", type=int, default=None, help="base seed: step i of the pipeline runs with seed + i and upstream's conformer generator is seeded from it, on every arm; absent: `off` runs upstream unseeded, a kit mode is not active")
    p.add_argument("--cache", default=None, help=f"upstream's --cache (default: ${stack.ENV_CACHE})")
    p.add_argument("--devices", type=int, default=None, help="upstream's --devices; the kit modes run one GPU and refuse N > 1 by name")
    own, upstream = split_design_argv(argv)
    a = p.parse_args(own)
    a.output = os.path.abspath(a.output)                             # one run directory for this process, `configure` and the steps, which run with its parent as cwd
    a.spec = os.path.abspath(a.spec)                                 # likewise resolved here, not against that parent
    a.out = a.output
    mode = effective_mode(a.mode)
    res = modes.resolve(mode, stack.opt_home())                      # a ValueError (a lever directory's file that does not match the mode table) propagates uncaught
    cache = a.cache or os.environ.get(stack.ENV_CACHE)
    user_args = upstream + (["--devices", str(a.devices)] if a.devices is not None else [])
    gpu = _gpu_for_report()
    gate = "n/a"
    if res.active:
        if a.seed is None:                                            # the runner is seeded by construction; no seed is invented
            return _not_active(mode, seed_refusal(mode), a.out, argv, gpu, res=res)
        if a.devices is not None and a.devices > 1:                  # the runner drives one GPU
            return _not_active(mode, devices_refusal(mode, a.devices), a.out, argv, gpu, res=res)
        reason, gate = stack.gates(res, gpu)
        if reason is not None:
            return _not_active(mode, reason, a.out, argv, gpu, gate, res)
    os.makedirs(a.out, exist_ok=True)
    where = IN_RUN_WHERE.format(out=a.out)
    try:
        conf = design.configure(a.spec, a.out, user_args, cache)
        exp = design.kernel_expectation(mode, a.out, conf)
        if res.active:
            run = design.run_kit(res, a.out, a.seed, echo=True, exp=exp)
        else:
            run = design.run_stock(a.out, a.seed, echo=True)
        run["gpu_gate"] = gate
        run["designs"] = design.account_designs(a.out, run)         # the DESIGNS line: requested / produced / the batches upstream skipped — every mode
        rep = design.process_report(res, run, gpu)
        report.emit(rep)
        rc = exit_for(run["rc"], rep, where=where)
        manifest.write(a.out, rep, command="design", argv=argv, exit_code=rc, designs=run["designs"],
                       extra={"configure": conf, "seed": a.seed, "run": {k: v for k, v in run.items() if k not in ("levers_applied", "levers_fallback", "fallback_reasons", "designs")}})
        return rc
    except design.DesignError as e:
        report.say(f"{report.PREFIX} ERROR: {e}")
        manifest.write(a.out, {"active": False, "mode": mode, "form": "process", "reason": str(e), "gpu": gpu}, command="design", argv=argv, exit_code=EXIT_FAIL)
        return EXIT_FAIL


# ------------------------------------------------------------------------------------------------------------------ check
def cmd_check(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog="boltzgen-opt check", add_help=True)
    p.add_argument("--mode", default=None, choices=list(modes.MODES))
    a = p.parse_args(argv)
    mode = effective_mode(a.mode)
    try:
        rep = stack.activate(mode, dry_run=True)
    except (ValueError, stack.ActivationError) as e:
        raise CliError(str(e), EXIT_NOT_ACTIVE) from e
    report.emit(rep)
    if mode == "off":
        return EXIT_OK
    return EXIT_OK if rep.get("reason") is None else EXIT_NOT_ACTIVE


# ------------------------------------------------------------------------------------------------------------------- warm
def cmd_warm(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog="boltzgen-opt warm", add_help=True)
    p.add_argument("--mode", default=None, choices=list(modes.MODES))
    a = p.parse_args(argv)
    mode = effective_mode(a.mode)
    res = modes.resolve(mode, stack.opt_home())
    gpu = _gpu_for_report()
    if res.active:
        reason, gate = stack.gates(res, gpu)
        if reason is not None:
            return _not_active(mode, reason, None, argv, gpu, gate, res, command="warm")
    r = warm.run(mode, gpu=gpu)
    report.say(f"{report.PREFIX} {warm.summary_line(r)}")
    return r.get("exit", EXIT_FAIL)                                         # design's rule: 0 ok, 1 failed, 3 not active (a child refused its activation, or a lever could not run: partial)


# ------------------------------------------------------------------------------------------------------------------ entry
COMMANDS = {"design": cmd_design, "check": cmd_check, "warm": cmd_warm}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, end="")
        return EXIT_OK if argv else EXIT_USAGE
    cmd = argv[0]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"{report.PREFIX} unknown command {cmd!r}\n{USAGE}", end="", file=sys.stderr)
        return EXIT_USAGE
    try:
        return fn(argv[1:])
    except CliError as e:
        report.say(f"{report.PREFIX} ERROR: {e}")
        return e.code
    except ValueError as e:
        report.say(f"{report.PREFIX} ERROR: {e}")
        return EXIT_USAGE

"""python -m chai1_opt {pred,check,warm} --mode off|exact|fast|big ...

A thin command layer over the kit code and the upstream API. It never re-implements a lever, a mode or a test:

* ``pred``   — predictions from an items file (inputs.py: the package's items JSON, the kit's PACK.json or one FASTA) at the fold
               settings given — stock's own run_inference flags (settings.py) — in the one output schema (outputs.py).
               ``--mode off``: the stock caller ``stock_fold.py`` in ONE clean subprocess per invocation, looping the items (every kit and recipe variable stripped
               and proved absent, no kit directory on sys.path). Kit modes: the kit driver as shipped in a subprocess per seed list
               (driver.py: ``kit/chai_worker.py --levels <the mode's list>``; the fold settings through the driver's own RUN_KW).
               ``--det 1`` applies the kit's deterministic recipe in either case (det.py). A mode's eager line (modes.KitMode.eager
               + KitMode.dstep + KitMode.pairtrack) is installed in the driver process by the package through the kits' own builders
               (stack.apply_eager: the add-on's ``build_lever_parts`` over the eager stack's ``build_parts`` / ``make_loader``).
               A row's implied levers (modes.KitMode.implied_optin: ``tf32`` on fast / big, ``alloc``) ride with the mode; there is no
               request for a lever beside a mode.
* ``check``  — dry run: resolves and gates the mode on this box (kit present, stock pin, weights, GPU) and applies nothing; prints
               the DRY-RUN line; works without torch.
* ``warm``   — one public-input prediction through ``pred`` (warm.py): the route end to end; then, for a mode carrying the compiled step,
               its ahead-of-time step packages for this card (warm_aoti.py: built once per card and stack key; a fold then loads a crop's step instead of compiling it).

Mode = ``--mode`` when given, else ``CHAI1_OPT`` from the environment, else ``modes.DEFAULT_MODE`` (``fast``, the package default; on a box where
fast's kernels cannot build — no C compiler — it is refused by name, ``stack.mode_stack_check``; ``modes.py``); ``--mode`` and a set ``CHAI1_OPT`` must agree.
A torch / CUDA stack other than the pinned one is recorded on every route (``NOTE STACK not pinned: …``)
and refused by name only under ``CHAI1_OPT_STRICT_STACK=1`` (stack.strict_stack_refusal). Exit codes: 0 ok · 1 failed · 2 usage · 3 not active.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

from . import _core, det as _det, inputs, modes, ngpu, report, settings, stack, warm

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3   # one code per condition, every verb: 0 ok · 1 the work failed (a fold, the install, a weight file off its pin)
                                                              # · 2 usage · 3 the mode is not active — refused by name; never a fold on it (a PARTIAL activation
                                                              # included, unless the caller passed --allow-partial: then it is named on the PARTIAL line and proceeds)

USAGE = f"""usage: chai1-opt <command> --mode {'|'.join(modes.MODES)} [options]
  pred    --input <items.json|PACK.json|x.fasta> --out_dir <dir> [fold flags] [--seed S | --seeds S,S] [--msa_dir|--msa-directory <dir>]
          [--tag <name>] [--det 0|1] [--n_gpu {'|'.join(str(p) for p in ngpu.SUPPORTED)}]
  check   [fold flags] [--n_gpu P] [--json]
  warm    [--det 0|1] [--keep] [--json] [--crops all|N,N]
Fold flags = stock's own (`chai-lab fold`: typer over run_inference), stock's defaults, every mode: --num-trunk-recycles 3
  --num-diffn-timesteps 200 --num-diffn-samples 5 --num-trunk-samples 1 --recycle-msa-subsample 0 --use-esm-embeddings|--no-use-esm-embeddings
  (on) --low-memory|--no-low-memory (on) --use-msa-server|--no-use-msa-server (off) --use-templates-server|--no-use-templates-server (off);
  --msa-server-url URL --constraint-path PATH --template-hits-path PATH --device DEV: every one passes through on every mode (settings.py).
  Seeds: --seed <int> (stock's flag; --seeds 0,1 folds a list),
  else the items' own; outputs are per seed (<key>/seed_<s>/) so a kit mode without one is refused by name; off without one is stock's
  unseeded call, written under <key>/seed_none/.
Mode: --mode, else CHAI1_OPT, else the package default {modes.DEFAULT_MODE} (README.md, the Modes bullet). --n_gpu: GPUs per fold,
default 1; P ∈ {{{','.join(str(p) for p in ngpu.SUPPORTED)}}} on chai1 — any other value is refused by name (exit 3), never shrunk. Exit: 0 ok,
1 failed, 2 usage, 3 not active (refused by name; a partial activation or a partial memory-mode ITEM without --allow-partial).
"""


class CliError(Exception):
    def __init__(self, msg, code=EXIT_USAGE):
        super().__init__(msg); self.code = code


def resolve_mode(cli_mode: Optional[str]) -> str:
    env_mode = (os.environ.get(modes.ENV) or "").strip().lower() or None
    m = (cli_mode or "").strip().lower() or None
    if m and env_mode and m != env_mode:
        raise CliError(f"--mode {m} disagrees with {modes.ENV}={env_mode}; a run has one mode — drop one of them")
    m = m or env_mode or modes.DEFAULT_MODE
    if m not in modes.MODES:
        raise CliError(f"{m!r} is not a mode ({'|'.join(modes.MODES)})")
    return m


def resolve_n_gpu(cli_n_gpu, mode: str) -> tuple:
    """``(P, refusal)``: the accepted GPUs-per-fold count (``--n_gpu`` absent = 1) or the refusal sentence the NOT ACTIVE line prints
    (``ngpu.resolve``: the core's mode sentence under exact / fast, chai1's not-applicable sentence under big); a malformed value is a
    usage error."""
    try:
        return ngpu.resolve(cli_n_gpu, mode), None
    except ValueError as e:
        raise CliError(str(e)) from None
    except Exception as e:  # noqa: BLE001 — opt_core.mem.ngpu.NGpuRefused: its reason is the sentence
        reason = getattr(e, "reason", None)
        if reason is None:
            raise
        return None, str(reason)


def _argparser(cmd: str):
    import argparse
    ap = argparse.ArgumentParser(prog=f"chai1-opt {cmd}", add_help=True)
    ap.add_argument("--mode", default=None)
    if cmd in ("pred", "check"):
        ap.add_argument("--n_gpu", default=None, help=f"GPUs per fold: explicit, default 1; chai1 ships P in {ngpu.SUPPORTED} — a larger P is refused by name (opt_core.mem.ngpu), never shrunk")
    if cmd in ("pred", "check"):
        settings.add_fold_arguments(ap)                                         # stock's own fold flags, stock's defaults (settings.py); warm folds at its own constant
    if cmd in ("pred", "warm"):
        ap.add_argument("--det", type=int, default=_det.DEFAULT_LEVEL, choices=list(_det.LEVELS))
    if cmd == "pred":
        ap.add_argument("--no-compile", action="store_true", help="the sanctioned opt-out of the compiled denoiser step (fast / big): an alias of "
                        "MODEL_OPT_LEVERS_OFF=compiled — the first item of each crop size skips the per-process compile, every item runs the eager statements")
        ap.add_argument("--allow-partial", action="store_true", help="the recorded opt-out of the mode's completeness rules: a partial ACTIVATION (a lever of the mode the run's own switches turned off) is refused by name without it and proceeds, named (PARTIAL, one NOTE line), with it; the memory mode's per-item gate at exit likewise (a partial ITEM exits 3 without it)")
        ap.add_argument("--input", required=True)
        ap.add_argument("--out_dir", required=True)
        ap.add_argument("--seed", "--seeds", dest="seeds", default=None, metavar="S[,S…]", help="the seed(s) folded per input (stock's --seed; a list folds each in turn), else the items' own seeds; a kit mode needs one (outputs are per seed), --mode off without one is stock's unseeded call")
        ap.add_argument("--msa_dir", "--msa-directory", dest="msa_dir", default=None, help="chai-lab's msa_directory (its --msa-directory): the run's .aligned.pqt files")
        ap.add_argument("--tag", default=None)
    if cmd == "warm":
        ap.add_argument("--crops", default=None, help="all | N[,N…] of chai-lab's model sizes (256 384 512 768 1024 1536 2048): also fold one synthetic single-chain "
                        "input per crop in one process so the mode's kernel caches under the JIT root are filled for those crops (the compiled step's per-process trace remains)")
        ap.add_argument("--keep", action="store_true")
        ap.add_argument("--quiet", action="store_true")
        ap.add_argument("--log", default=None)
    if cmd in ("check", "warm"):
        ap.add_argument("--json", action="store_true")
    return ap


def _kit_proto():
    """The kit's own chai_proto (stdlib at import), from the carried kit directory — the pred process only."""
    ok, why = stack.kit_present()
    if not ok:
        raise CliError(why, EXIT_NOT_ACTIVE)
    kit_pkg = stack.kit_pkg_dir()                                             # the kit's own modules (kit/); the worker the tree runs lives beside the kit
    if kit_pkg not in sys.path:
        sys.path.insert(0, kit_pkg)
    import chai_proto
    return chai_proto


# ------------------------------------------------------------------------------------------------------------------ pred
def cmd_pred(argv: List[str]) -> int:
    a = _argparser("pred").parse_args(argv)
    mode = resolve_mode(a.mode)
    if getattr(a, "no_compile", False):
        modes.no_compile_to_env()                                    # --no-compile == MODEL_OPT_LEVERS_OFF=compiled (one mechanism; the driver inherits the variable)
    env_off = modes.apply_levers_off()                             # MODEL_OPT_LEVERS_OFF: the rows reduced by its names (an ablation); a bad name is refused below by name (modes.refusal)
    if env_off and mode == "off":
        sys.stderr.write(f"{report.PREFIX} NOTE {modes.ENV_LEVERS_OFF}={os.environ.get(modes.ENV_LEVERS_OFF)!r} ignored under --mode off (stock: no kit lever to switch off)\n")
    optin = modes.with_implied(mode, ())                           # the row's implied levers (tf32 on fast / big, alloc): modes.KitMode.implied_optin
    n_gpu, why_ngpu = resolve_n_gpu(a.n_gpu, mode)
    if why_ngpu:
        report.print_activation({"mode": mode, "reason": why_ngpu}); return EXIT_NOT_ACTIVE
    st = settings.from_args(a)                                    # the fold settings: stock's run_inference keywords, stock's defaults unless a flag says otherwise
    lv = _det.level(a.det)
    tag = a.tag or mode
    out_dir = os.path.abspath(a.out_dir)
    its = inputs.load(a.input)
    inputs.resolve_keys(its, _kit_proto())
    cli_seeds = settings.parse_seeds(a.seeds)
    seeds_by_item = {it.key: settings.seeds_for(cli_seeds, it.seeds) for it in its.items}
    given = a.msa_dir or its.msa_dir
    msa_dir = os.path.abspath(given) if given else None          # the run's alignments (chai-lab's msa_directory); None = chai-lab's own default
    os.makedirs(os.path.join(out_dir, tag), exist_ok=True)
    why = modes.refusal(mode) or modes.optin_refusal(mode, optin) or settings.seed_refusal(mode, seeds_by_item)   # a kit mode with no seed named: refused by name, never a made-up seed
    if why:
        report.print_activation({"mode": mode, "reason": why}); return EXIT_NOT_ACTIVE
    if mode == "off":
        return _pred_stock(its, seeds_by_item, st, lv, msa_dir, out_dir, tag)    # stock: chai-lab's form (stock_fold.msa_directory_for); no seed named = its unseeded call
    msa_dir = driver_msa_dir(msa_dir, out_dir)                    # the kit driver's own contract: one directory always
    rep = stack.activate(mode, dry_run=True, trigger="pred", det=lv, optin=optin, n_gpu=n_gpu)   # pred's pre-flight gate (digest memo, not afresh)
    if rep.get("reason"):
        return EXIT_NOT_ACTIVE
    groups = {}
    for it in its.items:
        groups.setdefault(tuple(seeds_by_item[it.key]), []).append(it)
    rcs = []
    env = dict(os.environ); env.pop(modes.ENV, None)                  # the driver's --levels list is the switch; no env route inside
    from opt_core.oom import is_oom
    try:
        env.update(stack.optin_env(optin))                                 # alloc: the allocator configuration the driver child reads at its first CUDA allocation
    except Exception as e:  # noqa: BLE001 — the lever module's refusal wording
        if is_oom(e): raise                                                # noqa: E701 — an out-of-memory error propagates, never a refusal
        report.print_activation({"mode": mode, "optin": list(optin), "reason": str(e)}); return EXIT_NOT_ACTIVE
    for seeds, items in groups.items():
        cmd = [sys.executable, "-m", "chai1_opt.driver", "--mode", mode] + st["flags"] + (["--device", st["device"]] if st.get("device") else []) + ["--uids", inputs.uids_arg(items),
               "--seeds", ",".join(str(s) for s in seeds), "--out_dir", out_dir, "--tag", tag, "--msa_dir", msa_dir, "--det", str(lv),
               "--n_gpu", str(n_gpu)] + (["--optin", ",".join(optin)] if optin else []) + (["--allow-partial"] if a.allow_partial else [])
        sys.stderr.write(f"{report.PREFIX} driver: {' '.join(cmd[2:])}\n"); sys.stderr.flush()
        rcs.append(subprocess.call(cmd, env=env, cwd=out_dir))
        if rcs[-1] < 0:                                                        # the driver process was ended by a signal inside native code (no EXIT line of its own): say which — never a bare exit 1
            sys.stderr.write(f"{report.PREFIX} ERROR: the driver process (seeds {','.join(str(s) for s in seeds)}) was ended by signal {_signal_name(-rcs[-1])} (rc {rcs[-1]}) inside native code — "
                             f"its folds have no EXIT line; the run exits {EXIT_FAIL}\n"); sys.stderr.flush()
    if any(rc == EXIT_NOT_ACTIVE for rc in rcs):
        return EXIT_NOT_ACTIVE                                                 # a driver that refused never folded
    return EXIT_OK if all(rc == 0 for rc in rcs) else EXIT_FAIL


def _signal_name(num: int) -> str:
    """``SIGILL`` / ``SIGSEGV`` / … for a signal number (``signal <n>`` when the platform has no name for it)."""
    try:
        import signal
        return signal.Signals(num).name
    except Exception:  # noqa: BLE001
        return f"signal {num}"


def driver_msa_dir(msa_dir: Optional[str], out_dir: str) -> str:
    """The kit driver's ``--msa_dir`` argument (``cp.ensure_msa_pqt`` in ``chai_worker.py`` takes a directory): the run's alignments,
    else ``<out_dir>/msa_empty``. The fold's ``msa_directory`` per item is chai-lab's own form — that directory when a chain's
    ``.aligned.pqt`` is in it, ``None`` otherwise (``stack.install_msa_form``, the off route's rule ``stock_fold.msa_directory_for``)."""
    d = msa_dir or os.path.join(out_dir, "msa_empty")
    os.makedirs(d, exist_ok=True)
    return d


def _pred_stock(its, seeds_by_item, st, lv, msa_dir, out_dir, tag) -> int:
    from . import stock_fold
    rep = stack.base_report("off", trigger="pred", dry_run=False, det=lv)
    rep.update(route="stock", levels=None)
    why = stack.gates(rep, need_gpu=True, use_torch=False)
    if why:
        rep.update(active=False, reason=why); report.print_activation(rep); return EXIT_NOT_ACTIVE
    rep.update(active=False, reason="off: stock — nothing applied, no environment set")
    report.print_activation(rep)
    env = stock_fold.clean_environment()
    plan = inputs.to_plan(its, list(its.items), seeds_by_item, st, lv, msa_dir)        # ONE clean stock process for the whole invocation: every item, in order,
    plan_path = os.path.join(tempfile.mkdtemp(prefix="chai1_opt_stock_"), "stock.plan.json")                           # through one interpreter — upstream's own per-process state (the resident
    with open(plan_path, "w", encoding="utf-8") as fh:                                   # ESM2 model, the TorchScript loads) is paid as a user's own loop pays it
        json.dump(plan, fh, indent=1)
    cmd = [sys.executable, "-s", "-m", "chai1_opt.stock_fold", "--plan", plan_path, "--out_dir", out_dir, "--tag", tag]
    sys.stderr.write(f"{report.PREFIX} stock: {' '.join(cmd[1:])} (items={len(its.items)}, one process)\n"); sys.stderr.flush()
    rc_stock = subprocess.call(cmd, env=env, cwd=out_dir)
    rc = EXIT_OK if rc_stock == 0 else EXIT_FAIL
    return rc


# ----------------------------------------------------------------------------------------------------------------- check
def cmd_check(argv: List[str]) -> int:
    a = _argparser("check").parse_args(argv)
    mode = resolve_mode(a.mode)
    n_gpu, why_ngpu = resolve_n_gpu(a.n_gpu, mode)
    if why_ngpu:
        rep = {"mode": mode, "reason": why_ngpu, "active": False}
        report.print_activation(rep)
        if a.json:
            print(json.dumps(rep, indent=1, default=str), flush=True)
        return EXIT_NOT_ACTIVE
    st = settings.from_args(a)                                                            # the fold settings the same flags give pred (stock's defaults unless given)
    rep = stack.activate(mode, dry_run=True, trigger="check", n_gpu=n_gpu)              # activate merges the row's implied levers; check hashes the weights afresh
    if a.json:
        rep = dict(rep, settings=dict(st, library_defaults=settings.library_defaults(),   # the fold keywords this invocation resolves to (pinned source, AST)
                                     library_defaults_installed=settings.installed_signature_defaults()))   # None unless chai_lab.chai1 is already imported
        print(json.dumps(rep, indent=1, default=str), flush=True)
    if mode == "off":
        return EXIT_NOT_ACTIVE if rep.get("optin") else EXIT_OK                          # off names its reason (stock, nothing applied); an opt-in request under off is a refusal
    return EXIT_OK if not rep.get("reason") else EXIT_NOT_ACTIVE


# ------------------------------------------------------------------------------------------------------------------ warm
def cmd_warm(argv: List[str]) -> int:
    a = _argparser("warm").parse_args(argv)
    mode = resolve_mode(a.mode)
    try:
        crops = warm.parse_crops(a.crops)
    except ValueError as e:
        sys.stderr.write(f"{report.PREFIX} {e}\n"); return 2
    res = warm.run(mode, det_level=a.det, log_path=a.log, echo=not a.quiet, keep=a.keep, crops=crops)
    print(warm.summary_line(res), flush=True)
    if res["status"] == "PASS":                                                # statement two: the mode's compiled step built ahead of time for this card (chai1_fastln.aoti packages under
        from . import warm_aoti                                                # $MODEL_OPT_JIT_ROOT/<key>/aoti, built once; a fold then loads a crop's step instead of compiling it) — skipped BY NAME for a
        res["aoti"] = warm_aoti.step(mode, echo=not a.quiet)                  # mode without the lever, no JIT root, no CUDA headers; a failed build fails warm (the fold itself still compiles)
        print(warm_aoti.step_line(res["aoti"]), flush=True)
    if a.json:
        print(json.dumps(res, indent=1, default=str), flush=True)
    return EXIT_OK if res["status"] == "PASS" and (res.get("aoti") or {}).get("status") != "failed" else EXIT_FAIL


# ----------------------------------------------------------------------------------------------------------------- entry
COMMANDS = {"pred": cmd_pred, "check": cmd_check, "warm": cmd_warm}


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
        _core.entry_gate()                                                     # statement one: the pin gate, then the producers check — an absent, stale or older core exits 3 by name here
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT_NOT_ACTIVE
    try:
        return fn(argv[1:])
    except CliError as e:
        print(f"{report.PREFIX} ERROR: {e}", file=sys.stderr, flush=True)
        return e.code
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"{report.PREFIX} ERROR: {e}", file=sys.stderr, flush=True)
        return EXIT_USAGE

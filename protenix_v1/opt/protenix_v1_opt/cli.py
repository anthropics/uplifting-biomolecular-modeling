"""protenix-v1-opt <command> [--mode exact|fast|big|off] [--det 0|1] [--n_gpu P] [--allow-partial] ...  (== python -m protenix_v1_opt ...; run.sh wraps it after sourcing a config)

  pred    [--mode M] [--det 0|1] [--n_gpu P] [--allow-partial] <stock `protenix pred` arguments>
          one process, the stock CLI with the mode's levers on its runner; every other argument is stock's own (`protenix pred
          --help`: --seeds, --cycle, --step, --sample, --dtype, --use_msa, …) and passes through verbatim, defaults included
          (`--mode off`: the stock CLI in a clean subprocess, stock_pred.py). `--det 1` = the deterministic recipe (det.py) on either
          route: the stock arm under the recipe is what the kit's Tier-1 claim is stated against (a comparison runs both arms
          under it; a stock run under the recipe is not stock's default numerics).
          Exit: the stock CLI's code; 3 when the mode did not activate or the activation was PARTIAL (below).
  check   [--mode M] [--det 0|1] [--n_gpu P] [--allow-partial]      dry run: the mode's arm, the gates (the --n_gpu rules included) on this machine; the kit's lever code is not imported
  warm    [--mode M] [--det 0|1] [--allow-partial] --out_dir DIR   one prediction (the kit's p995_1brs input, 1 sample): weights load, graph capture

Mode = --mode when given, else PROTENIX_V1_OPT from the environment, else modes.DEFAULT_MODE; --n_gpu = devices for one
prediction (default 1; P>1 only under --mode big, else refused by name — ngpu.py; every verb ends with the EXIT line carrying
`n_gpu=P sharding=<rowpair|none>`); a --mode that disagrees with a set
PROTENIX_V1_OPT is refused (rc 2). Exit codes (report.py): 0 ok, 1 failed, 2 usage, 3 not active — including a PARTIAL activation: a
lever of the mode the kit did not apply, applied with no served call, or fell back from at run time (the kit's own counters,
report.kit_evidence) — `[protenix-v1-opt] NOT ACTIVE: partial activation — <levers>: <reason>; exit 3 (--allow-partial records and
proceeds)`. `--allow-partial` (every command) names the allowance on the PARTIAL line and the exit is the run's own; `check` carries the
flag for a uniform line and has no partial plan (the gates refuse or admit; the partial state exists once the kit has applied and run).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

from opt_core import cli as CORE_CLI
from opt_core import stock_proof as SP

from . import __version__
from . import kit as K
from . import modes as M
from . import report as R
from .modes import DEFAULT_MODE, MODES
from .report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE

USAGE = __doc__


class CliError(CORE_CLI.CliError):
    """A usage error found while running a verb (opt_core.cli.CliError) with the exit code it carries (EXIT_USAGE unless the verb says)."""

    def __init__(self, msg, code=EXIT_USAGE):
        super().__init__(msg)
        self.code = code


def _parser(cmd: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"protenix-v1-opt {cmd}", add_help=True)
    p.add_argument("--mode", choices=MODES, default=None)
    p.add_argument("--det", choices=("0", "1"), default=None, help="the deterministic recipe (det.py); default 0, or PROTENIX_V1_OPT_DET")
    p.add_argument(R.ALLOW_PARTIAL_FLAG, dest="allow_partial", action="store_true", default=False,
                   help="name a PARTIAL activation on its line and proceed (exit = the run's own); default: exit 3")
    p.add_argument("--n_gpu", default=None, metavar="P",
                   help="devices for one prediction: 1 (the default; the single-GPU path of any mode) or, under --mode big, P>1 = the row-sharded pair stack over P devices (ngpu.py; or PROTENIX_V1_OPT_N_GPU)")
    return p


def _effective_mode(args) -> str:
    from .stack import ENV_MODE
    env = (os.environ.get(ENV_MODE) or "").strip().lower() or None
    if args.mode and env and args.mode != env:
        raise CliError(f"--mode {args.mode} disagrees with {ENV_MODE}={env} from the environment; one run has one mode — drop one of them")
    return M.check_mode(args.mode or env or DEFAULT_MODE)


def _effective_det(args) -> bool:
    from .stack import ENV_DET
    if args.det is not None:
        return args.det == "1"
    return (os.environ.get(ENV_DET) or "").strip() == "1"


def _effective_allow_partial(args) -> bool:
    """`--allow-partial` on the command line (the one opt-out; no environment spelling)."""
    return bool(getattr(args, "allow_partial", False))


def _effective_n_gpu(args) -> int:
    """`--n_gpu` when given, else PROTENIX_V1_OPT_N_GPU, else 1 (ngpu.effective); a non-integer or P<1 is a usage error."""
    from . import ngpu
    try:
        return ngpu.effective(getattr(args, "n_gpu", None))
    except ValueError as e:
        raise CliError(f"--n_gpu: {e}")


def _admit_n_gpu(n_gpu: int, mode: str):
    """The n_gpu rules for this verb (ngpu.admit): None when admitted, else the refusal sentence (already printed as the NOT ACTIVE line)."""
    from . import ngpu
    try:
        ngpu.admit(n_gpu, mode)
        return None
    except ngpu.NG.NGpuRefused as e:
        R.log(R.not_active_line(str(e.reason)))
        return str(e.reason)


def _out_dir(stock_args: List[str]) -> str:
    """The `-o/--out_dir` the caller passes to the stock CLI (its default: ./output)."""
    out = "./output"
    it = iter(range(len(stock_args)))
    for i in it:
        a = stock_args[i]
        if a in ("-o", "--out_dir") and i + 1 < len(stock_args):
            out = stock_args[i + 1]
        elif a.startswith("--out_dir="):
            out = a.split("=", 1)[1]
    return out


# ----------------------------------------------------------------------------------------------------------------------------------- pred
def _stock_env() -> tuple:
    """The environment of the stock subprocess (opt_core.stock_proof): the package switches and each name carrying one of the kit's prefixes stripped
    (PINS.json stock_environment.must_be_absent_prefixes), kit PYTHONPATH entries removed."""
    from . import stack
    env, stripped = SP.strip_env(os.environ, stack.pins()["stock_environment"]["must_be_absent_prefixes"])
    kept, removed = SP.strip_pythonpath(env.get("PYTHONPATH"), [K.kit_home()])
    if kept:
        env["PYTHONPATH"] = kept
    else:
        env.pop("PYTHONPATH", None)
    return env, stripped, removed


def _run_stock(stock_args: List[str], det: bool) -> int:
    from . import stock_pred
    env, _stripped, _removed = _stock_env()
    out_dir = _out_dir(stock_args)
    os.makedirs(out_dir, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="protenix_v1_opt_stock_")                  # the child's environment proof (opt_core.stock_proof) is its printed PROOF / STOCK lines; the core's proof file lives in this per-launch scratch directory, never beside the outputs
    cmd = stock_pred.command(sys.executable, proof_json=os.path.join(scratch, "stock_env_proof.json"), kit_dirs=[K.kit_home()], stock_args=stock_args, det=det)
    try:
        proc = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True)
        for line in proc.stderr:
            sys.stderr.write(line)
        rc = proc.wait()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    R.log(R.exit_line("off", rc, 1))
    return rc


def cmd_pred(argv: List[str]) -> int:
    p = _parser("pred")
    args, stock_args = p.parse_known_args(argv)
    mode = _effective_mode(args)
    det = _effective_det(args)
    allow = _effective_allow_partial(args)
    n_gpu = _effective_n_gpu(args)
    refused = _admit_n_gpu(n_gpu, mode)                                # the --n_gpu rules first: a refused P never loads a weight
    if refused:
        R.log(R.exit_line(mode, EXIT_NOT_ACTIVE, n_gpu))
        return EXIT_NOT_ACTIVE
    try:
        K.frozen_weights_check(argv=stock_args)                        # the weights boot gate, every route (kit.frozen_weights_check): a missing file refuses by name; a present checkpoint is digested — named pinned or NOT PINNED — and the run proceeds
    except K.FrozenWeightsError as e:
        R.log(R.not_active_line(f"frozen weights: {e}"))
        return EXIT_NOT_ACTIVE
    if mode == "off":
        from . import ablation as A                                    # MODEL_OPT_LEVERS_OFF under the stock route: refused by name, nothing runs (the route applies no lever)
        if A.requested():
            R.log(R.not_active_line(A.off_refusal(A.requested())))
            R.log(R.exit_line(mode, EXIT_NOT_ACTIVE, n_gpu))
            return EXIT_NOT_ACTIVE
        return _run_stock(stock_args, det)
    from . import rowpair
    if n_gpu > 1 and not rowpair.is_rank_process():                    # the launching process of a P>1 run: P rank processes of this same verb
        return _run_ranks(n_gpu, argv, _out_dir(stock_args), mode)
    import protenix_v1_opt
    from . import stack
    try:
        protenix_v1_opt.enable(mode, strict=True, det=det, allow_partial=allow, n_gpu=n_gpu)
    except protenix_v1_opt.ActivationError:
        R.log(R.exit_line(mode, EXIT_NOT_ACTIVE, n_gpu))
        return EXIT_NOT_ACTIVE
    from runner.batch_inference import protenix_cli
    rc = EXIT_OK
    try:
        protenix_cli.main(args=["pred"] + stock_args, prog_name="protenix", standalone_mode=False)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (EXIT_OK if e.code is None else EXIT_FAIL)
    except protenix_v1_opt.ActivationError as e:              # the kit refused the arm on the runner, or the activation was partial (stack._wrap_runner)
        if stack.status().get("reason") != str(e):             # stack._refuse printed the line already; a bare ActivationError has not
            R.log(R.not_active_line(str(e)))
        rc = EXIT_NOT_ACTIVE
    except Exception as e:                                    # click's own usage errors surface as exceptions under standalone_mode=False
        if not type(e).__module__.startswith("click"):
            raise
        print(f"protenix pred: {e}", file=sys.stderr, flush=True)
        rc = EXIT_USAGE
    rep = stack.refresh_levers()                               # the kit's account after the run: the evidence the exit rule reads
    if rc == EXIT_OK and not rep.get("active"):
        R.log(R.not_active_line("at exit: the stock runner was never built, the levers never applied"))
        rc = EXIT_NOT_ACTIVE
    res = M.resolve(mode)
    v = R.verdict(rep, res.trimul, res.levers, allow, run_ok=(rc == EXIT_OK and bool(rep.get("active"))))
    R.log_exit_lines(v, rep)
    if v["exit_code"] is not None:
        rc = v["exit_code"]
    rc = R.items_exit_code(rc, rep.get("items_census"))             # a failed or open item is exit 1 even when the stock CLI returned 0 (its per-item handler continues)
    R.log(R.exit_line(mode, rc, n_gpu))
    return rc


def _run_ranks(n_gpu: int, argv: List[str], out_dir: str, mode: str) -> int:
    """`pred --mode big --n_gpu P>1` in the launching process: P rank processes of `pred <argv>` (rowpair.run_ranks over the family
    launcher: rank environment, one visible device per rank, fail-fast), rank 0's transcript relayed to stderr; the exit code is rank 0's; a
    failed rank is the named ROWPAIR event line and a non-zero exit."""
    from . import ngpu, rowpair
    t0 = time.time()
    try:
        ngpu.large_input_regime(os.environ)                                # the ranks inherit the row-sharded pair stack's host-parking exports (ngpu.TP_EXPORTS) unless the caller set them; the P=1-only memory levers are off by each rank's in-process selection (big.rowpair_switches)
        recs = rowpair.run_ranks(n_gpu, argv, out_dir)
        rc = int(next((r.get("rc", EXIT_FAIL) for r in recs if r.get("rank") == 0), EXIT_FAIL) or 0)
        if rc == 0:                                                            # fail-closed: P transcripts, each reporting `n_gpu=P sharding=rowpair` on its ACTIVE line — never a pass on fewer devices than asked
            try:
                ngpu.assert_ranks_report(n_gpu, [rowpair.rank_log(out_dir, r) for r in range(len(recs))])
            except ngpu.NG.NGpuRefused as e:
                R.log(R.not_active_line(str(e.reason)))
                rc = EXIT_NOT_ACTIVE
    except rowpair.L.RankFailed as e:
        R.log(rowpair.rank_failed_line(e))
        code = getattr(e, "exitcode", None)
        rc = code if isinstance(code, int) and code > 0 else EXIT_FAIL
    except rowpair.D.RowpairRefused as e:
        R.log(R.not_active_line(str(getattr(e, "reason", e))))
        rc = EXIT_NOT_ACTIVE
    except ngpu.NG.NGpuRefused as e:
        R.log(R.not_active_line(str(e.reason)))
        rc = EXIT_NOT_ACTIVE
    R.log(f"{R.PREFIX} ROWPAIR event=ranks_done n_gpu={n_gpu} rc={rc} wall_s={round(time.time() - t0, 1)} logs={os.path.join(out_dir, 'rowpair')}")
    R.log(R.exit_line(mode, rc, n_gpu))
    return rc


# ---------------------------------------------------------------------------------------------------------------------------------- check
def dry_run_report(mode: str, det: bool = False, allow_partial: bool = False, n_gpu: int = 1) -> dict:
    """The mode's resolution and this box's gates, nothing armed (importable on a CPU box: torch is imported only for the GPU gate).
    `partial` is [] by construction: the plan is refused or admitted whole (stack.gates); `allow_partial` records the flag for the run;
    the `--n_gpu` rules (ngpu.admit) are a gate like the others: a refused P is the first entry of `gates`."""
    from . import ngpu, stack
    rep = {"mode": mode, "det": bool(det), "allow_partial": bool(allow_partial), "partial": [], "package_version": __version__, "kit": K.kit_home(),
           "protenix_version": stack.dist_version("protenix"), **ngpu.record(n_gpu)}
    try:
        ngpu.admit(n_gpu, mode)
        ngpu_reasons = []
    except ngpu.NG.NGpuRefused as e:
        ngpu_reasons = [str(e.reason)]
    if mode == "off":
        from . import ablation as A                                       # MODEL_OPT_LEVERS_OFF under the stock mode: a gate reason, by name
        abl_reasons = [A.off_refusal(A.requested())] if A.requested() else []
        rep.update(arm=M.STOCK_ARM, gates=ngpu_reasons + abl_reasons, ok=not (ngpu_reasons or abl_reasons), stock_env_violations=stack.stock_env_violations())
        return rep
    try:
        res = M.resolve(mode)
    except RuntimeError as e:                                             # a mode whose composition refuses on this input / environment (big: an undeclared PROTENIX_V1_BIG_* name, a malformed size statement): a gate reason, worded as the activation words it
        rep.update(gates=ngpu_reasons + [f"mode {mode}: {e}"], ok=False)
        return rep
    rep.update(arm=res.arm, line=res.line, tier=res.tier, trimul=res.trimul, levers=list(res.levers))
    if res.capped or res.graph_cap:                                       # the sampler-graph token cap (modes.graph_cap): the cap, its source and, when sized above it, the levers withheld
        rep.update(levers_capped=list(res.capped), graph_cap=res.graph_cap)
    if res.ablated:                                                       # MODEL_OPT_LEVERS_OFF: the levers withheld and the mode's own arm, so the dry run shows the ablation it would run
        rep.update(levers_ablated=list(res.ablated), mode_arm=res.mode_arm)
    reasons = ngpu_reasons + stack.gates(res, det, refresh_weights=True)   # `check` hashes the weights afresh and rewrites the digest memo entry
    rep.update(gates=reasons, ok=not reasons, gpu=stack.torch_gpu_info(), kit_files=K.check_files())
    return rep


def cmd_check(argv: List[str]) -> int:
    p = _parser("check")
    args = p.parse_args(argv)
    mode = _effective_mode(args)
    n_gpu = _effective_n_gpu(args)
    rep = dry_run_report(mode, _effective_det(args), _effective_allow_partial(args), n_gpu)
    print(json.dumps(rep, indent=1, default=str))
    if not rep["ok"]:
        R.log(R.not_active_line("; ".join(rep["gates"])))
    rc = EXIT_OK if rep["ok"] else EXIT_NOT_ACTIVE
    R.log(R.exit_line(mode, rc, n_gpu))
    return rc


# ----------------------------------------------------------------------------------------------------------------------------------- warm
WARM_INPUT = "p995_1brs"                                                                  # warm's one prediction: the kit's bundled 1BRS input, one seed, one sample (warm's own constants, not flags)
WARM_SEEDS = "101"
WARM_SAMPLE = "1"


def kit_input_json(name: str, work_dir: str) -> str:
    """The kit's bundled input <name>.json (inputs/), copied under work_dir with every relative `precomputed_msa_dir` made absolute: the bundled
    inputs name their MSA directories relative to this model's directory (kit.tree_home(): protenix_v1/, the README's working directory), so
    the copy runs from any working directory. work_dir is a scratch directory, never the prediction's output directory (the output directory
    holds the stock CLI's files only). A named directory that does not exist is a usage error, by path."""
    kit = K.kit_home()
    src = os.path.join(kit, "inputs", f"{name}.json")
    if not os.path.isfile(src):
        raise CliError(f"no kit input {name!r} under {os.path.join(kit, 'inputs')}")
    with open(src, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    for item in items:
        for entry in item.get("sequences", []):
            msa = (entry.get("proteinChain") or {}).get("msa") or {}
            d = msa.get("precomputed_msa_dir")
            if d and not os.path.isabs(d):
                d = os.path.join(K.tree_home(), d)
                if not os.path.isdir(d):
                    raise CliError(f"kit input {name!r}: MSA directory {d} does not exist")
                msa["precomputed_msa_dir"] = d
    dst = os.path.join(work_dir, "inputs", f"{name}.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=1)
    return dst


def cmd_warm(argv: List[str]) -> int:
    p = _parser("warm")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args(argv)
    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    js = kit_input_json(WARM_INPUT, tempfile.mkdtemp(prefix="protenix_v1_opt_warm_"))
    fwd = ["--input", js, "--out_dir", out, "--seeds", WARM_SEEDS, "--sample", WARM_SAMPLE, "--use_msa", "true"]
    modeargs = ((["--mode", args.mode] if args.mode else []) + (["--det", args.det] if args.det is not None else [])
                + [R.ALLOW_PARTIAL_FLAG]                                   # warm fills caches and loads weights: a lever that serves no call on the small warm input
                                                                           # (its size gate) is recorded (the PARTIAL line) and the verb still exits 0 — activation is check / pred's
                + (["--n_gpu", str(args.n_gpu)] if args.n_gpu is not None else []))
    return cmd_pred(modeargs + fwd)


COMMANDS = {"pred": cmd_pred, "check": cmd_check, "warm": cmd_warm}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"protenix-v1-opt: unknown command {cmd!r}\n{USAGE}", file=sys.stderr)
        return 2
    try:
        return int(COMMANDS[cmd](rest) or 0)
    except SystemExit as e:                                   # argparse's own usage exits (bad option / choice): rc 2, message already printed
        return e.code if isinstance(e.code, int) else 2
    except CliError as e:
        print(f"protenix-v1-opt {cmd}: {e}", file=sys.stderr)
        return e.code
    except ValueError as e:
        print(f"protenix-v1-opt {cmd}: {e}", file=sys.stderr)
        return 2

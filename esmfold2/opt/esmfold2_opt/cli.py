"""python -m esmfold2_opt {pred,check,warm} --variant fast|full_msa|full_nomsa [--mode exact|fast|big|off] ...

A thin command layer over the kit code and the upstream API. It never re-implements a lever or a mode:

* ``pred``   — predictions from an input file (inputs.py: upstream's prediction-input JSON) at the upstream API's own
               settings (settings.py: the library's knobs passed through by name), written in the one output schema (outputs.py).
               ``--mode off``: the stock caller ``stock_fold.py`` in a clean subprocess (every kit variable stripped and proved absent,
               no kit directory on sys.path — nothing from the kit on the path). Kit modes: ``esmfold2_opt.enable(mode, variant)``, upstream's model
               load for the variant's checkpoint (stock/PINS.json), the kit server's own ``configure()`` on that model (eager), then the
               same fold loop as the stock caller. ``--det 1`` applies the kit's deterministic recipe (det.py) in either case.
* ``check``  — the activation line for (mode, variant) without applying anything: server mode, resolved levers, the levers the
               variant cannot use, kernel key, weights resolution; ``--json`` prints the full report.
* ``warm``   — one prediction on the kit's public input through ``pred`` so the mode's Triton kernels are compiled (warm.py); pred's exit rule.

Exit codes (one per condition, every verb): 0 ok | 1 failed, or the pass incomplete (outputs short of the request:
the INCOMPLETE line names <n>/<m>) | 2 usage | 3 levers not active (reason printed) — a mode is all of its levers on a GPU class: when the
kit's own records show a lever of the mode's set not applied on this device (a kernel that could not compile or launch here) pred
refuses by name before its first fold (the APPLIED line's fallbacks, the LEVER lines, one ``NOT ACTIVE: partial activation …`` line,
exit 3) — it never runs under the mode's name with a subset. A GPU class or library version the kit's tables hold no measured row for
is named on those lines and engaged, never a refusal. The kit's
declared guards (the MK hoist off at num_diffusion_samples > 1) are ``gated``: recorded by name, exit 0. ``check`` names in its notes a lever whose
measured tables do not cover this box; ``warm`` follows pred. On the environment route (``ESMFOLD2_OPT=<mode>``
in front of your own script) the partial state is recorded (the APPLIED line, ``status()["partial"]``, the EXIT tally) — never the host
process's exit: ``pred`` is the gated form.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from . import report, tp, warm
from ._core_gate import gate as core_gate                                 # the kit's core pin gate (byte-identical copy of the release tree's kit_template/_core_gate.py; stdlib only)
from ._producers import kit_anchor
from .modes import DEFAULT_MODE, MODES, VARIANTS, kit_mode
from .report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE      # the one exit-code table (report.py): every verb

PROG = "python -m esmfold2_opt"

USAGE = f"""usage: {PROG} <command> --variant fast|full_msa|full_nomsa [--mode exact|fast|big|off] ...

commands
  pred    --variant V [--mode M] --input FILE --out_dir DIR [--seeds 0,1] [--num_loops N] [--num_sampling_steps N] [--num_diffusion_samples N] [--msa_max_depth N]
          [--lm_dropout F] [--msa_column_mask_rate F] [--remove_insertions true|false] [--max_sequences N] [--det 0|1|2] [--backend fused|shipped]
          [--noise_scale F] [--step_scale F] [--max_inference_sigma F] [--lm_mask_pct F] [--early_exit true|false]   (fold()'s optional overrides) [--device cuda]
                                                             predictions: mode off = the stock caller in a clean subprocess (upstream API only;
                                                             --backend shipped, the default = the model as loaded; fused = its fused fast path); kit modes = levers on through
                                                             the kit server's configure(), same fold loop; writes the kit driver's own file set
                                                             <out_dir>/cif_all/<id>__<fast|full>__s<seed>_x<k>.cif + _pae.npz, <out_dir>/<tag>_rows.jsonl
                                                             (the kit's writer: upstream's outputs); --det 1|2: the kit's recipe / level 1 + config-level lm_encoder.lm_dropout=0
  check   --variant V [--mode M] [--json]                    print the activation line (what would be active); runs nothing
  warm    --variant V [--mode M] [--log FILE] [--quiet] [--json] [--keep]
                                                             one public-input prediction through pred: Triton JIT + graph capture
exit codes: 0 ok | 1 check/warm failed | 2 usage | 3 levers not active (reason printed) | otherwise the stock/server exit code
"""


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


# ----------------------------------------------------------------------------------------------------------------- helpers
def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise CliError(f"unknown --mode {mode!r} (choose from {', '.join(MODES)})")
    return mode


def check_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    if variant not in VARIANTS:
        raise CliError(f"unknown --variant {variant!r} (choose from {', '.join(VARIANTS)})")
    return variant


def split_flags(argv: list[str], names=("--mode", "--variant")) -> tuple[dict, list[str]]:
    """Remove ``--name X`` / ``--name=X`` for the given names from argv (last one wins); the rest is returned unchanged and in order."""
    found: dict = {}
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        hit = next((n for n in names if a == n or a.startswith(n + "=")), None)
        if hit is None:
            rest.append(a); i += 1; continue
        if a == hit:
            if i + 1 >= len(argv):
                raise CliError(f"{hit} needs a value")
            found[hit] = argv[i + 1]; i += 2
        else:
            found[hit] = a.split("=", 1)[1]; i += 1
    return found, rest


def activate(mode: str, variant: str | None, dry_run: bool = False, line: str | None = None) -> dict:
    """``enable(mode, variant, line=)`` and the activation report; exceptions become a NOT ACTIVE report (never a traceback without a reason).
    The core logs the activation line itself (``rep["logged"]``); the CLI never prints a second copy."""
    import esmfold2_opt as core
    from . import stack
    from opt_core.oom import is_oom
    if dry_run:
        stack.set_weights_afresh(True)                                          # `check` digests the weights files afresh and refreshes the digest memo
    try:
        rep = stack.activate(mode, variant, dry_run=True, line=line) if dry_run else core.enable(mode, variant, line=line)
    except Exception as e:  # noqa: BLE001
        if is_oom(e): raise                                                     # a GPU out-of-memory error is not an activation verdict: it propagates
        return {"active": False, "mode": mode, "variant": variant, "line": line, "reason": f"esmfold2_opt.enable({mode!r}, {variant!r}, line={line!r}) raised {e!r}"}
    if not isinstance(rep, dict):
        return {"active": False, "mode": mode, "variant": variant, "reason": f"enable() returned {type(rep).__name__}, not the activation report dict"}
    rep.setdefault("mode", mode)
    return rep


def _resolve_settings(a):
    """The fold settings: the library's defaults with the stock flags the caller passed laid over them (settings.resolve)."""
    from . import settings, stack
    p = stack.pins() if os.path.isfile(stack.pins_path()) else None
    return settings.resolve(a, pins=p)


def _samples_gate(a):
    """The run's one diffusion-sample count and its source (inputs.run_samples): ``--num_diffusion_samples`` when given, else the inputs' own
    ``num_diffusion_samples`` key (the same on every input), else the library's default. A disagreement — an input against the flag, or two
    inputs against each other — is a usage error named before anything loads (exit 2): the levers, the accounting and the `settings` line
    all plan for one count."""
    from . import inputs                                                                   # import-free of the library: nothing of upstream loads before _pred_run activates the mode
    try:
        return inputs.run_samples(inputs.load(a.input), flag=a.num_diffusion_samples)         # (count, "flag"|"inputs") or (None, "default": the library's, resolved with the settings)
    except inputs.InputError as e:
        raise CliError(str(e), EXIT_USAGE) from e


def _run_settings(a):
    """_resolve_settings + the run's sample count: when it came from the inputs' own key, the settings carry it (so every reader — the kit's
    configure(), the fold accounting, the `settings` line, the stock subprocess — sees the count that runs) and their source says so."""
    st = _resolve_settings(a)
    n, source = getattr(a, "run_samples", (None, "default"))
    if source == "inputs":
        st.num_diffusion_samples = int(n)
        st.source += f" + inputs: num_diffusion_samples {int(n)}"
    return st


def _seeds_gate(a, st, det_level: int = 0) -> None:
    """The seeds: ``--seeds``, else an input's own ``seeds`` key, else upstream's default — ``fold(seed=None)``, one unseeded fold per input.
    The det recipe seeds the fold, so ``--det 1|2`` without a seed anywhere is refused by name."""
    if a.seeds or st.seeds or not det_level:
        return
    from . import inputs
    missing = [item["id"] for item in inputs.load(a.input) if not item.get("seeds")]
    if missing:
        raise CliError(f"--det {det_level} seeds the fold: give --seeds (input {missing[0]!r} carries no seeds; {len(missing)} of the inputs)", EXIT_USAGE)


# ----------------------------------------------------------------------------------------------------------------- pred
def pred_parser() -> argparse.ArgumentParser:
    from .settings import add_stock_flags
    p = argparse.ArgumentParser(prog=f"{PROG} pred", allow_abbrev=False)
    p.add_argument("--mode", default=DEFAULT_MODE, choices=MODES)
    p.add_argument("--variant", default=None, choices=VARIANTS, help="default: $ESMFOLD2_VARIANT")
    p.add_argument("--n_gpu", default=None, help=f"the resource axis: P GPUs (default 1; explicit, never auto-detected). This kit ships P in {{{','.join(map(str, tp.N_GPU_SHIPPED))}}}: "
                                                 f"P > 1 runs under {tp.MODE} only (P rank processes on one machine: the row-sharded pair stack, rowpair.py) and is refused by name under exact/fast/off (exit 2); a P outside the shipped set is refused by name (exit 3) (tp.py)")
    p.add_argument("--input", required=True, help="upstream prediction-input JSON, one input or a list (inputs.py)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", default=None, help="seed list S[,S...]; default: an input\'s own seeds key, else upstream\'s default (seed=None: one unseeded fold per input); --det 1|2 needs a seed")
    add_stock_flags(p)
    p.add_argument("--det", type=int, default=0, choices=(0, 1, 2), help="1 = the kit's deterministic recipe, 2 = level 1 + config-level lm_encoder.lm_dropout=0 (det.py; stock-side switches, never a mode)")
    p.add_argument("--backend", default=None, choices=("fused", "shipped"), help="--mode off only: the model calls after loading (shipped, the default = none: the model exactly as loaded; fused = set_kernel_backend('fused') + set_chunk_size(None), the library's documented fast path)")
    p.add_argument("--device", default="cuda")
    return p


def stock_command(a, st, pins: dict) -> tuple[list[str], dict]:
    """The stock caller's command line and environment: ``python -s -m esmfold2_opt.stock_fold ...`` (no user site; PYTHONPATH kept so a
    source install of upstream keeps working) with every forbidden variable stripped (stock/PINS.json must_be_absent_prefixes + the
    package's own switches); the library's data paths are its own names (``HF_HOME``, ``ESMCFOLD_CCD_PATH``); the caller proves the
    result itself (stock_fold.env_proof)."""
    from . import outputs, stack
    absent = stack.stock_env_absent(pins)
    env = {k: v for k, v in os.environ.items() if not any((k.startswith(s) if s.endswith("_") else k == s) for s in absent)}
    cmd = [sys.executable, "-s", "-m", "esmfold2_opt.stock_fold", "--variant", a.variant, "--hf-repo", stack.variant_repo(a.variant, pins),
           "--input", os.path.abspath(a.input), "--out_dir", os.path.abspath(a.out_dir), "--settings-json", json.dumps(st.as_dict()),
           "--det", str(int(a.det)), "--env-absent", ",".join(absent), "--device", a.device,
           "--stage-dir", os.path.join(os.path.abspath(a.out_dir), outputs.STAGE_DIRNAME)]
    if a.seeds:
        cmd += ["--seeds", a.seeds]
    if getattr(a, "backend", None):
        cmd += ["--backend", a.backend]
    return cmd, env


def n_gpu_census(requested: int, *, adapter: dict | None, world: int) -> str | None:
    """The --n_gpu the run was asked for against every place that carries it once the model is installed: this process's reported value
    (report.N_GPU, the ACTIVE / EXIT lines' token), the launcher's rank count (``world``) and, under n_gpu > 1, the adapter's own record
    (``rowpair.report()['n_gpu']`` + ``installed``). Returns the NOT ACTIVE reason ``n_gpu_mismatch requested=P active=Q world=W adapter=A``
    on any disagreement, else None — a request the run does not honour is refused, never folded on fewer cards."""
    active = int(report.N_GPU["P"])
    adapter_p = int((adapter or {}).get("n_gpu", 1)); installed = bool((adapter or {}).get("installed", requested == 1))
    if active == requested == int(world) == adapter_p and (requested == 1 or installed):
        return None
    return (f"n_gpu_mismatch requested={requested} active={active} world={int(world)} adapter={adapter_p}"
            + ("" if requested == 1 or installed else " adapter_installed=False"))


def _refused(e: BaseException) -> int:
    """A refusal raised inside `pred` (a lever, the row-sharded family or its launcher refusing BY NAME): the kit's NOT ACTIVE line with the
    refusal's class and words, and the not-active exit code — never a traceback."""
    print(report.not_active_line(f"{type(e).__name__}: {e}"), file=sys.stderr, flush=True)
    return EXIT_NOT_ACTIVE


def cmd_pred(argv: list[str]) -> int:
    from . import stack
    a = pred_parser().parse_args(argv)
    if a.mode != tp.MODE:
        check_mode(a.mode)
    a.line = None                                                    # big has one composition; `line` stays the resolver's internal key
    if not os.path.isfile(a.input):
        raise CliError(f"--input {a.input}: no such file")
    try:
        variant = stack._effective_variant(a.variant)
    except ValueError as e:
        raise CliError(str(e)) from e
    if variant is None:
        raise CliError(f"--variant is required ({'|'.join(VARIANTS)}) or set {stack.ENV_VARIANT}")
    a.variant = variant
    if not os.path.isfile(stack.pins_path()):
        raise CliError(f"stock pins not found at {stack.pins_path()}", EXIT_NOT_ACTIVE)
    pins = stack.pins()
    from . import det as _det, outputs
    det_level = _det.level(a.det)
    if a.backend and a.mode != "off":
        raise CliError("--backend is a stock-side switch (--mode off): under a kit mode the kit's own configure() selects the backend")
    stack._disarm_autoload()                                                               # `pred`'s mode is the command line's (activated by name in _pred_run): the ESMFOLD2_OPT import
    a.run_samples = _samples_gate(a)                                                      # hook is withdrawn before anything below can import the library. ONE sample count per run
    try:
        n_gpu = report.set_n_gpu(tp.precheck(a.mode, a.n_gpu))                              # --n_gpu read once, before any gate: 1 (absent or explicit) passes; P > 1 passes only under big
    except tp.TpError as e:                                                               # AND inside the shipped set (tp.N_GPU_SHIPPED) — refused by name otherwise (tp.py)
        raise CliError(str(e), e.code) from e
    refusals: tuple = ()                                                                    # refusal classes surfaced as NOT ACTIVE (exit 3) from inside the fold body (none can arise at n_gpu=1)
    if n_gpu > 1:                                                                          # the row-sharded pair stack (rowpair.py over opt_core.mem.rowpair): this same command on P rank
        from . import rowpair                                                             # processes; the parent launches them and returns their verdict, each rank runs the fold below
        from opt_core.mem.primitives import MemLeverRefused                               # the memory family's refusal class (RowpairRefused ⊂ NGpuRefused ⊂ MemLeverRefused)
        from opt_core.modes import ModeError                                              # the release tree's refusal class for a lever a mode needs (arch refusals)
        refusals = (ModeError, MemLeverRefused)
        if not rowpair.is_rank_process():
            try:
                rowpair.refuse_unless_visible(n_gpu)                                       # fewer GPUs visible than --n_gpu: refused by name (`refused: n_gpu=P visible=K`,
            except refusals as e:                                                          # exit 3) before the weights gate and before any process starts — never shrunk, never a traceback
                return _refused(e)
            os.environ[stack.ENV_WEIGHTS_MEMO_DIR] = stack.weights_memo_dir()                # the ranks inherit the memo location
            why, _notes = stack.data_path_gate(None, variant=variant, pins=pins)              # the launching process digests the weights ONCE (or reads the memo) before the
            if why:                                                                           # ranks start; every rank reads these entries (digest_memo.rank_digest), never hashes
                raise CliError(why, EXIT_NOT_ACTIVE)
            stack.weights_memo_relay_for_ranks(os.environ.get("HF_HOME", ""), variant)       # an unwritable configured memo dir: the digests reach the ranks through a private writable one
            try:
                return rowpair.launch(list(argv), a.mode, n_gpu)
            except refusals as e:                                                          # a refusal of the launcher itself, in the family's words: by name, exit 3
                return _refused(e)
        if rowpair.RL.world_size() != n_gpu:                                              # a rank process whose family environment disagrees with the command line: usage error by name
            raise CliError(f"--n_gpu {n_gpu}: this rank process's environment says world={rowpair.RL.world_size()}", EXIT_USAGE)
    t0 = time.time()
    try:
        return _pred_run(a, argv, variant, pins, det_level, n_gpu, refusals, rowpair if n_gpu > 1 else None, t0)
    finally:
        if n_gpu > 1:                                                                      # EVERY exit path of a rank process — refused, failed or done — leaves the process group here
            rowpair.finish()                                                               # (idempotent), so no rank idles in interpreter teardown while its peers wait on a collective


def _pred_run(a, argv, variant, pins, det_level, n_gpu, refusals, rowpair, t0) -> int:
    """The fold body of ``pred`` once the axis is settled (``rowpair`` is the adapter module under n_gpu > 1, else None)."""
    from . import det as _det, outputs, stack
    if a.mode == "off":
        rep = activate("off", variant)                                 # first: the explicit mode wins over any ESMFOLD2_OPT in the environment
        report.log_activation(rep)
        os.environ.pop(stack.ENV_MODE, None)
        st = _run_settings(a)
        print(f"{report.PREFIX} settings " + " ".join(f"{k}={v}" for k, v in st.as_dict().items() if k not in ("source", "what")), file=sys.stderr, flush=True)   # the resolved dims
        ok, pin_detail, why = stack.pins_gate(pins, force=bool(os.environ.get(stack.ENV_FORCE)))   # the commit-level pin check holds for the stock arm too
        if not ok:
            raise CliError(why, EXIT_NOT_ACTIVE)
        cmd, env = stock_command(a, st, pins)
        why, _notes = stack.data_path_gate(env, variant=variant, pins=pins)                     # the weights root unset/absent or a file missing: refuse before the subprocess, by name; an unknown checkpoint: WARN and proceed
        if why:
            print(report.not_active_line(why), file=sys.stderr, flush=True)
            return EXIT_NOT_ACTIVE
        _seeds_gate(a, st, det_level)
        print(f"{report.PREFIX} stock subprocess: {' '.join(cmd[:5])} ... (env stripped of {','.join(stack.stock_env_absent(pins))})", file=sys.stderr, flush=True)
        rc = subprocess.call(cmd, env=env)
        stage_dir = os.path.join(a.out_dir, outputs.STAGE_DIRNAME)
        from . import stock_fold
        summary = stock_fold.pass_summary(stock_fold.read_outcome(os.path.join(stage_dir, outputs.PROGRESS_NAME)), rc)
        if os.path.isdir(stage_dir):                                   # every completed fold, whatever the exit code: the driver's file set, through the kit's own writer (outputs.py)
            try:
                server = outputs.kit_server()
                outputs.finalise_staged(server, stage_dir, a.out_dir, outputs.ROWS_TAG)
            except Exception as e:  # noqa: BLE001
                print(f"{report.PREFIX} pred FAILED writing the driver's file set: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                rc = EXIT_FAIL
        if summary["pass_status"] != "complete":
            print(f"{report.PREFIX} {stock_fold.pass_line(summary)}", file=sys.stderr, flush=True)
        v = report.verdict(rc, None, stock_fold.incomplete_of(summary))                # the stock arm has no lever: incomplete only
        return v["exit_code"]
    if det_level:
        _det.apply_env()                                               # before torch is imported (activation imports it)
    rep = activate(a.mode, variant, line=getattr(a, "line", None))
    if not rep.get("active"):
        report.log_activation(rep)
        return EXIT_NOT_ACTIVE
    import torch  # noqa: F401  (loaded by activation)
    if det_level:
        _det.apply_torch()
    try:
        st = _run_settings(a)
    except CliError:
        raise
    except Exception as e:  # noqa: BLE001
        raise CliError(f"fold settings: {e}", EXIT_NOT_ACTIVE) from e
    _seeds_gate(a, st, det_level)
    from . import inputs, stock_fold
    for note in (stack.mk_plan_note(rep.get("levers_planned"), st.num_diffusion_samples), stack.guards_plan_note(rep.get("levers_planned"), st.num_diffusion_samples)):
        if note:
            print(f"{report.PREFIX} {note}", file=sys.stderr, flush=True)
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    rc = 0
    refused_upfront = False                                                                # the partial refusal before the first fold (its line printed there, not again at exit)
    rows_all: list = []
    outcome = None
    try:
        server = outputs.kit_server()                                                  # the kit's writer (already imported by activation)
        gpu_name = stock_fold._gpu_name()
        model = stock_fold.load_model(stack.variant_repo(variant, pins), a.device, det_level=det_level)   # upstream's load as the stock caller's; the kit's configure() makes the model calls
        builder = ESMFold2InputBuilder()
        stack.apply_to(model, builder, trigger="pred", samples=st.num_diffusion_samples, out_dir=a.out_dir)   # the kit server's configure(), eagerly (a memory mode: its add-on first)
        if n_gpu > 1:
            rowpair.install_rank(model, n_gpu, builder)                                    # the pair-stack call sites rebound onto the sharded primitives on this rank and rank 0's featurisation for every rank on the builder (nothing at n_gpu=1); a refusal is a ModeError (NOT ACTIVE below)
        pv = report.verdict(EXIT_OK, stack.status())                                   # a mode is all of its levers on a GPU class: a lever of the set the kit's records show not
        if pv["refused_partial"]:                                                         # applied on this device (it could not compile or launch here) refuses the mode by name BEFORE
            print(report.partial_line(pv, stack.status().get("fallback_reasons")), file=sys.stderr, flush=True)   # the first fold — never a run under the mode's name with a subset
            rc, refused_upfront = EXIT_NOT_ACTIVE, True
            return rc
        why = n_gpu_census(n_gpu, adapter=rowpair.report() if n_gpu > 1 else None, world=rowpair.RL.world_size() if n_gpu > 1 else 1)
        if why:                                                                            # fail-closed: the n_gpu this process REPORTS (its ACTIVE line), the run's rank count and the
            print(report.not_active_line(why), file=sys.stderr, flush=True)              # adapter's own record all equal the requested --n_gpu, or the fold never starts (never a silent
            return EXIT_NOT_ACTIVE                                                         # one-GPU pass under a larger request)
        print(report.ready_line(variant, time.time() - t0), file=sys.stderr, flush=True)   # the one 'model loaded' line

        def writer(item, seed, res, depths, feats, wall):
            if n_gpu > 1 and not rowpair.is_output_rank():                                 # every rank holds the same result; the output rank writes (the adapter names the skip in the rank's log)
                return [None] * max(1, int(st.num_diffusion_samples))
            meta = outputs.spi_meta(depths, feats)
            rows = outputs.write_rows(server, a.out_dir, outputs.ROWS_TAG, item, variant, seed, res, meta, wall, gpu_name, mode=rep.get("server_mode") or a.mode)
            rows_all.extend(rows)
            return rows

        items = inputs.load(a.input)
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
        outcome = stock_fold.Outcome(items)
        stock_fold.fold_items(model, builder, items, st, variant, a.out_dir, seeds=seeds, det=det_level, writer=writer, device=a.device,
                              log=lambda s: print(f"{report.PREFIX} {s}", file=sys.stderr, flush=True), outcome=outcome)
        folds = outcome.n_predictions // max(1, st.num_diffusion_samples)              # one fold per (item, seed): the rows are samples per fold
        if folds * max(1, st.num_diffusion_samples) != outcome.n_predictions:
            raise CliError(f"prediction count {outcome.n_predictions} is not a whole number of folds at {st.num_diffusion_samples} samples")
        gate = stack.xl_gate(expected_folds=folds, replaced=(rowpair.XL_LEVERS_REPLACED if n_gpu > 1 else ()))   # an xl mode: the carry's fallback census per fold (raises by
        #   name); under n_gpu > 1 the levers the row-sharded line re-issues on rows are named in the verdict (replaced_by_rowpair), not counted
        if gate.get("checked"):
            x4 = gate.get("x4")
            x4w = f" x4={x4['state']}(threshold={x4['threshold']} max_tokens={x4['max_tokens']} events={x4['events']}/{x4['expected']})" if x4 else ""
            print(f"{report.PREFIX} xl gate: ok levers={stack.XL_STATE['levers']}{x4w} replaced_by_rowpair={gate.get('replaced_by_rowpair') or []} stats={json.dumps(gate['stats'], sort_keys=True)}", file=sys.stderr, flush=True)
    except refusals as e:                                                              # a lever the mode needs refused by name on this rank / card (arch: ModeError; rowpair: MemLeverRefused)
        rc = _refused(e)
    except Exception as e:  # noqa: BLE001
        print(f"{report.PREFIX} pred FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        rc = EXIT_FAIL
        raise
    finally:
        final = stack.settle_mk(stack.status(), replaced_by=(rowpair.KIT_LEVERS_REPLACED.get("mk") if n_gpu > 1 else None))   # mk: applied only if the sampler's
        #   counters say so; under n_gpu > 1 the row-sharded diffusion forward takes its place by name (levers_replaced, gated)
        planned = set((final or {}).get("levers_planned") or [])
        if final.get("mk") and "mk" in planned:
            print(f"{report.PREFIX} mk after run: {final['mk']['note']}", file=sys.stderr, flush=True)
        final = stack.settle_guards(final)                                             # the other levers with data-dependent guards (ef2_dit's scope, the atom layouts, the hoists'
        if not refused_upfront:                                                        # fall-through counters, the transition kernels' self-checks): judged from their modules' own counters, named
            for lever, gd in sorted((final.get("guards") or {}).items()):
                if lever in planned:
                    print(f"{report.PREFIX} {lever} after run: {gd['kind']}: {gd['note']}", file=sys.stderr, flush=True)
        summary = stock_fold.pass_summary(outcome.as_dict() if outcome is not None else None, rc)   # the rows of completed folds are already on disk
        if summary["pass_status"] != "complete":
            print(f"{report.PREFIX} {stock_fold.pass_line(summary)}", file=sys.stderr, flush=True)
        v = report.verdict(rc, final, stock_fold.incomplete_of(summary))    # the exit rule: incomplete -> 1; a lever the run's own counters show not applied (mk) -> the partial refusal, 3
        rc = v["exit_code"]
        line = None if refused_upfront else report.partial_line(v, final.get("fallback_reasons"))   # the partial refusal's line (None when nothing is partial, the run failed on its own, or pred refused before folding)
        if line:
            print(line, file=sys.stderr, flush=True)
        for g in v["gated"]:
            print(f"{report.PREFIX} gated: {g}", file=sys.stderr, flush=True)
        if n_gpu > 1:                                                                  # the process group is torn down on every rank; rc is this rank's verdict
            rowpair.finish(exit_code=rc)                                               # (kept by the core if the teardown has to force the exit)
    return rc


# ----------------------------------------------------------------------------------------------------------------- check / warm
def cmd_check(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog=f"{PROG} check", allow_abbrev=False)
    p.add_argument("--mode", default=DEFAULT_MODE)
    p.add_argument("--variant", default=None)
    p.add_argument("--json", action="store_true", help="also print the full activation report as JSON")
    p.add_argument("--n_gpu", default=None, help="the resource axis (pred_parser's --n_gpu): the same refusals, no launch")
    a = p.parse_args(argv)
    try:
        report.set_n_gpu(tp.precheck(a.mode, a.n_gpu))
    except tp.TpError as e:
        raise CliError(str(e), e.code) from e
    mode = check_mode(a.mode)
    variant = check_variant(a.variant)
    a.line = None                                                    # big has one composition; `line` stays the resolver's internal key
    rep = activate(mode, variant, dry_run=True, line=a.line)    # the core prints the DRY-RUN / NOT ACTIVE line (one formatter: report.py)
    report.log_activation(rep)                                  # only when the core could not (rep["logged"] unset: the call raised)
    if a.json:
        print(json.dumps({k: v for k, v in rep.items() if k != "resolution"}, indent=1, default=str), flush=True)
    if mode == "off":
        return EXIT_OK
    if not rep.get("server_mode") or rep.get("would_refuse"):
        return EXIT_NOT_ACTIVE
    return EXIT_OK


def _run_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--log", default=None, help="transcript file")
    p.add_argument("--quiet", action="store_true", help="do not echo the transcript to stderr")
    p.add_argument("--json", action="store_true", help="print the parsed result as JSON")


def cmd_warm(argv: list[str]) -> int:
    from . import stack
    p = argparse.ArgumentParser(prog=f"{PROG} warm", allow_abbrev=False, description="one public-input prediction through pred: Triton JIT + graph capture")
    p.add_argument("--mode", default=DEFAULT_MODE)
    p.add_argument("--variant", default=None)
    p.add_argument("--keep", action="store_true", help="keep the warm-up outputs")
    _run_common(p)
    a = p.parse_args(argv)
    mode = check_mode(a.mode)
    try:
        variant = stack._effective_variant(check_variant(a.variant))
    except ValueError as e:
        raise CliError(str(e)) from e
    if variant is None:
        raise CliError(f"--variant is required ({'|'.join(VARIANTS)}) or set {stack.ENV_VARIANT}")
    a.line = None                                                    # big has one composition; `line` stays the resolver's internal key
    res = warm.run(mode, variant, log_path=a.log, echo=not a.quiet, keep=a.keep, line=a.line)
    print(warm.summary_line(res), flush=True)
    if a.json:
        print(json.dumps(res, indent=1, default=str), flush=True)
    if res["status"] == "PASS":
        return EXIT_OK
    return EXIT_NOT_ACTIVE if res.get("exit_code") == EXIT_NOT_ACTIVE else EXIT_FAIL    # the running verb's rule: pred's NOT ACTIVE exit is warm's


# ----------------------------------------------------------------------------------------------------------------- entry
COMMANDS = {"pred": cmd_pred, "check": cmd_check, "warm": cmd_warm}


OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}   # frozen weights: huggingface_hub / transformers' documented switches — set before the stock subprocess and before activation unless the caller set them (run.sh / configs/<gpu>.env set the same)



KIT_SWITCH_PREFIX = "EF2_"                                                  # the optimized modules' switch namespace (stock_fold.DEFAULT_ENV_ABSENT names the same prefix)


def declared_switches() -> set:
    """Every ``EF2_*`` environment name this package itself declares — its ``ENV_*`` constants (the mode table's overrides, the row-sharding
    adapters' block sizes, x4's threshold). Any other ``EF2_*`` name reaches
    an internal setting of the optimized modules that no mode sets."""
    from . import modes, big, rowpair, rowpair_msa, rowpair_heads, atom_swa, stack
    out = set()
    for m in (modes, big, rowpair, rowpair_msa, rowpair_heads, atom_swa, stack):
        for k, v in vars(m).items():
            if (k.startswith("ENV_") or k.endswith("_ENV")) and isinstance(v, str) and v.startswith(KIT_SWITCH_PREFIX):
                out.add(v)
    return out


def stray_switches(environ=None) -> list:
    """The ``EF2_*`` names set in ``environ`` (default ``os.environ``) that the package does not declare (:func:`declared_switches`), sorted."""
    env = os.environ if environ is None else environ
    declared = declared_switches()
    return sorted(k for k in env if k.startswith(KIT_SWITCH_PREFIX) and k not in declared)


def stray_switch_note(environ=None) -> str | None:
    """One NOTE line naming the undeclared ``EF2_*`` names in the environment (None when there are none): they reach internal settings of
    the optimized modules outside every mode's composition."""
    names = stray_switches(environ)
    if not names:
        return None
    return (f"{report.PREFIX} NOTE environment names outside the package's declared switches are set: {', '.join(names)} — "
            "they reach internal settings of the optimized modules that no mode sets; unset them for the modes' own behaviour")


def main(argv: list[str] | None = None) -> int:
    core_gate(kit_anchor(__file__))                                         # statement one: the importable opt_core is the one [tool.opt_core] pins, else `NOT ACTIVE: reason=core_missing|core_mismatch …` exit 3
    from ._producers import refusal as _producers_refusal
    why = _producers_refusal()                                             # statement two: every core module this package imports is on disk under the pinned core
    if why:                                                                 # (producer_missing:<modules>, exit 3, no traceback) BEFORE anything resolves
        print(report.not_active_line(why), file=sys.stderr, flush=True)
        return EXIT_NOT_ACTIVE
    argv = list(sys.argv[1:] if argv is None else argv)
    for k, v in OFFLINE_ENV.items():
        os.environ.setdefault(k, v)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, end="")
        return EXIT_OK if argv else EXIT_USAGE
    cmd = argv[0]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"{report.PREFIX} unknown command {cmd!r}\n{USAGE}", end="", file=sys.stderr)
        return EXIT_USAGE
    note = stray_switch_note()                                              # undeclared EF2_* names in the caller's environment: named once, up front
    if note:
        print(note, file=sys.stderr, flush=True)
    try:
        return fn(argv[1:])
    except CliError as e:
        print(f"{report.PREFIX} ERROR: {e}", file=sys.stderr, flush=True)
        return e.code

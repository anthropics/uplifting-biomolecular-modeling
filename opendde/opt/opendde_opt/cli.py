"""python -m opendde_opt {pred,check,warm} [--mode off|exact|fast|big] ...

A thin command layer over the kit code as shipped and the upstream CLI. It never re-implements a lever, a mode or a test:

* ``pred``   — upstream's action: `opendde pred` on an input JSON with upstream's own flags passed through (settings.py), `opt_manifest.json` beside the outputs.
               ``--mode off``: the stock caller ``stock_pred.py`` in a clean subprocess (every kit variable stripped and proved absent, no
               kit directory on the path — nothing from the kits on the path). Kit modes/lines: ``opendde_opt.enable(...)`` in this process
               (the line's shim armed, route ``cli``), then the stock click group in-process with the arguments unchanged; the hook installs
               the levers on the runner the CLI builds. ``--det 1`` applies the deterministic recipe (det.py) in either case. ``--n_gpu P``
               (default 1) is the memory mode's resource axis (tp.py): ``P > 1`` under ``--mode big`` shards the pair stacks over ``P``
               cards — this process launches the ``P`` rank processes and runs no model itself; under ``off``, ``exact`` and ``fast``, or with
               fewer than ``P`` visible cards, ``P > 1`` is refused by name before anything runs (exit 3). Upstream's own multi-GPU options are
               upstream's: under ``--mode off`` they pass after ``--`` like any other upstream flag.
* ``check``  — the activation line for a mode/line without applying anything: resolved exports, path order, levers by route, gates
               (kit files, version pin, late activation), GPU; ``--json`` prints the full report. Exit 0 = would activate, 3 = would not.
* ``warm``   — warm.py: one prediction of upstream's smallest documented example through ``pred`` (Triton JIT, tile cache).

Exit codes: 0 ok · 1 the action failed · 2 usage · 3 not active / not stock (nothing ran).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
import sys
import tempfile
import time

from . import ENV_MODE
from . import lncensus as _lncensus
from . import stockknob as _stockknob
from . import phase as _phase
from . import alloc as _alloc
from . import det as _det
from . import frozen as _frozen
from . import inputs as _inputs
from . import manifest as _manifest
from . import modes, outputs, settings, stack
from . import nostdin as _nostdin
from . import report as _report
from . import smalln as _smalln
from . import chunklift as _chunklift
from . import stepgraph as _stepgraph
from . import _producers
from . import big
from . import templates as _templates
from . import tp as _tp

PROG = "python -m opendde_opt"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3
EXIT_KERNELS = _lncensus.EXIT_KERNELS                                          # 5: the KERNELS census REQUIRE guard refused the route (an expected accelerator absent / fell back)
STOCK_PROOF_NAME = "stock_env_proof.json"                   # the stock caller's environment proof: written in a private temporary directory, read once, removed
from .stock_pred import STOCK_ENTRY                       # the `opendde` console script's entry point: one definition, the stock caller's
USAGE = f"""usage: {PROG} <command> [--mode off|exact|fast|big] ...
  pred    -i <query.json> -o <out_dir> [upstream flags: --seeds|-s --cycle|-c --step|-p --sample|-e --dtype --model_name|-n --use_msa --use_template --use_rna_msa --need_atom_confidence --trimul_kernel --triatt_kernel] [--det 0|1] [--n_gpu P] [--root <OPENDDE_ROOT_DIR>] [--template_mmcif_dir D] [--allow-partial] [-- <extra opendde pred args>]   (stated flags pass to the engine verbatim, absent ones keep upstream's defaults; --n_gpu P>1 only with --mode big)
  check   [--json]
  warm    [-o <out_dir>] [--keep]
modes: {', '.join(modes.MODES)} (default {modes.DEFAULT_MODE})
exit codes: 0 ok · 1 the command failed · 2 usage · 3 not active — the mode/line refused by name, or a PARTIAL activation (a lever of the mode on
the stock path) without `--allow-partial`: the outputs and the manifest stay, the manifest names the fallbacks; `--allow-partial` is recorded there ·
5 kernels refused — the KERNELS census's REQUIRE guard: an accelerator the route expects (fused LayerNorm, cuEquivariance triangle kernels) is
absent or fell back in the model process (lncensus; the pass's `KERNELS route=` line names it).
"""


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


# ----------------------------------------------------------------------------------------------------------------- selection
def split_selection(argv: list[str]) -> tuple[str, list[str]]:
    """Take ``--mode M`` (or OPENDDE_OPT) out of argv; returns (mode, rest). A --mode that disagrees with OPENDDE_OPT is refused (the two
    routes must not silently differ)."""
    rest, mode = [], None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1]
            i += 2
            continue
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1]; i += 1; continue
        rest.append(a)
        i += 1
    env_mode = (os.environ.get(ENV_MODE) or "").strip().lower()
    if mode and env_mode and mode.lower() != env_mode:
        raise CliError(f"--mode {mode} disagrees with {ENV_MODE}={env_mode}: unset one of them")
    if mode:
        return modes.check_mode(mode), rest
    if env_mode:
        return modes.check_mode(env_mode), rest
    return modes.DEFAULT_MODE, rest


def resolve_or_refuse(sel: str, tree: str) -> modes.Resolution:
    try:
        return modes.resolve(sel, tree, os.environ)
    except modes.OpenModeError as e:
        raise CliError(str(e), EXIT_NOT_ACTIVE) from e
    except (ValueError, KeyError) as e:
        raise CliError(str(e), EXIT_USAGE) from e


# ----------------------------------------------------------------------------------------------------------------- stock route
def stock_command(tree: str, stock_args: list[str], proof_json: str, det_level: int = 0) -> tuple[list[str], dict]:
    """The stock subprocess: command + environment (forbidden prefixes stripped, kit directories off PYTHONPATH, the package's variables
    off). Under the deterministic recipe the names in PINS.json "allowed_under_det" are exempt (the caller exports them after)."""
    pins = stack.pins(tree)
    se = pins["stock_environment"]
    prefixes = list(se["must_be_absent_prefixes"])
    if det_level:
        prefixes = [p for p in prefixes if p not in se.get("allowed_under_det", [])]
    reads = list(se.get("reads", []))
    kit_dirs = [modes.kit_dir(tree, k) for k in modes.KIT_DIRS]
    env = {k: v for k, v in os.environ.items() if not any(k.startswith(p) for p in prefixes)}
    env.pop(ENV_MODE, None)
    roots = [os.path.realpath(d) for d in kit_dirs]
    pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and not any(os.path.realpath(p).startswith(r) for r in roots)]
    if pp:
        env["PYTHONPATH"] = os.pathsep.join(pp)
    else:
        env.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-s", "-m", "opendde_opt.stock_pred", "--proof-json", proof_json, "--env-absent", ",".join(prefixes),
           "--env-reads", ",".join(reads), "--kit-dirs", os.pathsep.join(kit_dirs), "--", *stock_args]
    return cmd, env


def _call_relay(cmd: list[str], env: dict) -> tuple[int, str]:
    """Run ``cmd``, relaying its merged stdout/stderr to this process's stdout line by line; returns (rc, output text)."""
    lines = []
    p = subprocess.Popen(cmd, env=env, stdin=_nostdin.CHILD_STDIN, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            lines.append(line)
        rc = p.wait()
    finally:
        if p.poll() is None:
            p.kill()
    return rc, "".join(lines)


def input_refusal(path: str) -> str | None:
    """The named reason an `-i` query cannot be used (`input_missing:<path>` | `input_unreadable:<path> (<error>)`), or None when
    `_inputs.load_query` reads it: `pred` refuses on it before the stack assertion and the gates (exit 2), never a traceback."""
    if not os.path.isfile(path):
        return f"input_missing:{path}"
    try:
        _inputs.load_query(path)
    except Exception as e:  # noqa: BLE001 — any parse / shape error is the refusal's text
        return f"input_unreadable:{path} ({type(e).__name__}: {e})"
    return None


def run_stock_cli(args: list[str]) -> int:
    """The stock click group in this process (kit modes: the hook is armed first). Returns the exit code."""
    import importlib
    from . import phase as _phase
    entry = getattr(importlib.import_module(STOCK_ENTRY[0]), STOCK_ENTRY[1])
    _phase.install()                                                             # per-item PHASE timing lines: the same boundaries as the stock caller's (phase.py)
    try:
        entry.main(args=list(args), prog_name="opendde", standalone_mode=True)
    except SystemExit as e:
        code = e.code
        return 0 if code is None else (code if isinstance(code, int) else 1)
    return 0


# ----------------------------------------------------------------------------------------------------------------- commands
def _common_pred(p: argparse.ArgumentParser):
    p.add_argument("-i", "--input", required=True, help="upstream query JSON (a list of jobs)")
    p.add_argument("-o", "--out_dir", required=True)
    p.add_argument("--det", default="0", help="deterministic recipe level (0|1)")
    p.add_argument("--root", default=None, help="OPENDDE_ROOT_DIR (weights root); default: the environment's")
    p.add_argument("--template_mmcif_dir", default=None, help="under --use_template true: the directory holding <pdb>.cif for every hit the query's templatesPath files name "
                                                             "(default: the hits files' directory, or its mmcif/ subdirectory when present); the model process reads it as $OPENDDE_ROOT_DIR/search_database/mmcif (templates.py)")
    p.add_argument("--allow-partial", action="store_true", help="accept a run whose activation is PARTIAL (a lever of the mode on the stock path); recorded in the manifest. Without it a PARTIAL run exits 3 (the outputs and the manifest stay)")
    settings.add_arguments(p, stack.tree_root())                                # upstream's own knobs, stock names and defaults, passed through as stated (settings.FLAGS)
    p.add_argument("--n_gpu", type=int, default=1, help="cards: --mode big's row-sharded pair stack over P (default 1; P>1 is refused by name under off, exact and fast, "
                                                          "and when fewer than P cards are visible)")


def rank_argv(a, n_gpu: int, out_dir: str, extra: list[str], templates=None, seed_args=()):
    """``argv_of(r)``: the rank processes' command — this command again, rank ``r`` writing under ``tp.rank_out_dir(out_dir, r)``. Under
    ``--use_template true`` (``templates`` = pred_parent's tuple) every rank is handed the WEIGHTS root and the one cif directory by flag, so each
    builds its own overlay from the same two facts the parent resolved (never the parent's overlay through the environment). ``seed_args`` = the
    run seed the parent drew, stated to every rank as a caller would (``['--seeds', v]``, settings.run_seed; empty when the caller or the query gives the seeds)."""
    def argv_of(r: int) -> list[str]:
        cmd = [sys.executable, "-m", "opendde_opt", "pred", "--mode", "big", "--n_gpu", str(int(n_gpu)), "-i", os.path.abspath(a.input),
               "-o", _tp.rank_out_dir(out_dir, r), *settings.stated_args(a), *seed_args, "--det", str(a.det)]   # every rank states what the caller stated (each adds the stock base itself) + the one drawn run seed, where the caller's own --seeds would stand
        if templates is not None and templates[2]:
            cmd += ["--root", templates[2]]                                   # the weights root itself (the parent's overlay is this process's, not the ranks')
        elif a.root:
            cmd += ["--root", a.root]
        if templates is not None and templates[1]:
            cmd += ["--template_mmcif_dir", templates[1]]                   # the cif directory the parent resolved (stated, or the hits files' directory / its mmcif/)
        if a.allow_partial:
            cmd.append("--allow-partial")
        return cmd + (["--", *extra] if extra else [])
    return argv_of


def pred_parent(a, res: modes.Resolution, n_gpu: int, out_dir: str, extra: list[str], tree: str, inp: dict, started: float, templates=None) -> int:
    """``--mode big --n_gpu P>1``: launch the ``P`` rank processes (tp.run_ranks), relay their lines, write the parent's manifest. Rank 0's
    outputs and manifest are the run's (in ``out_dir``); this process runs no model. ``templates`` = ``(chains, mmcif_dir, weights_root,
    model_root)`` under ``--use_template true``: rank 0's featurizer reads the hits and every rank bears THIS RANK'S ROWS of the template
    pair features (tp_feats; ranks > 0 from the broadcast template coordinates); the census reads the ranks' transcripts
    (templates.census_ranks). The run's seeds are decided ONCE here
    (settings.run_seed: the caller's ``--seeds``, else every job's ``modelSeeds``, else one seed drawn here and stated to every rank; a query
    seeded in part is refused by name) — the LAUNCH line's ``runseed=<v|-> source=<cli|json|drawn>``; rank 0 featurises every item for
    every rank (``data_form=rank0_bcast``, tp._rank0_item)."""
    from opt_core.mem.rowpair import launch as _launch
    from opt_core.mem.rowpair import evidence as _evidence
    jobs = _inputs.load_query(a.input)
    try:
        seed = settings.run_seed(settings.stated(a, extra), jobs, n_gpu)          # P rank processes never draw P seeds: cli | json | drawn once here; a query seeded in part is refused
    except ValueError as e:
        print(f"{_report.PREFIX} PRED refused: exit {EXIT_USAGE} reason={e}", flush=True)
        return EXIT_USAGE
    print(f"{_report.PREFIX} LAUNCH mode=big {_tp.fields(n_gpu)} line={_tp.LINE_NAME} {settings.run_seed_fields(seed)} ranks={n_gpu} {_tp.data_form_fields()} "
          f"out={out_dir} rank_dirs={os.path.join(out_dir, _tp.RANK_SUBDIR)}", flush=True)
    def on_line(text):                                                       # rank 0's output (its ACTIVE / LEVER / EXIT lines), as it arrives
        print(f"[r0] {text}", end="" if text.endswith("\n") else "\n", flush=True)
    try:
        recs = _tp.run_ranks(n_gpu, rank_argv(a, n_gpu, out_dir, extra, templates=templates, seed_args=seed.args), out_dir, on_line=on_line)
    except _launch.RankFailed as e:
        print(_evidence.rank_failed_line(_report.TAG, e), flush=True)
        print(f"{_report.PREFIX} PRED big {_tp.fields(n_gpu)} rc={EXIT_FAIL} reason={str(e)!r}", flush=True)
        return EXIT_FAIL
    rcs = {int(x["rank"]): (EXIT_FAIL if x.get("rc") is None else int(x["rc"])) for x in recs}
    logs = {int(x["rank"]): open(x["log"], errors="replace").read() for x in recs if x.get("log") and os.path.isfile(x["log"])}   # the ranks' transcripts (<out>/.rowpair/rank<r>.log)
    for pl in _phase.peak_summary(logs):                                         # their per-item PEAK lines reduced: rank-max + rank 0
        print(pl, flush=True)
    tcensus = _templates.census_ranks(logs, templates[0]) if templates is not None else None   # rank 0's `Found <n>` lines: one census (a receiver reporting its own is a NOTE)
    rc = max(rcs.values()) if rcs else EXIT_FAIL
    r0m = {}
    try:
        with open(os.path.join(out_dir, _manifest.NAME)) as fh:
            r0m = json.load(fh)
    except (OSError, ValueError):
        pass
    r0_n = int(((r0m.get("activation") or {}).get("n_gpu")) or 0)
    mm = _tp.mismatch(n_gpu, len(rcs), "rank processes the launcher ran") or (_tp.mismatch(n_gpu, r0_n, "rank 0's activation report") if rc == 0 else None)
    if mm:                                                                       # fail-closed: the launcher's rank count and rank 0's reported axis == the request
        print(f"{_report.PREFIX} NOT ACTIVE mode=big reason={mm}", flush=True)
        print(f"{_report.PREFIX} PRED big {_tp.fields(n_gpu)} rc={EXIT_NOT_ACTIVE} reason={mm!r}", flush=True)
        return EXIT_NOT_ACTIVE
    idx = outputs.index_cli(out_dir)
    m = _manifest.build(command="pred", mode="big", line=_tp.LINE_NAME, activation={"active": rc == 0, "n_gpu": int(n_gpu), "sharding": "rowpair",
                        "ranks": rcs, "rank_walls_s": {int(x["rank"]): x.get("wall_s") for x in recs}, "rank0_manifest": os.path.join(out_dir, _manifest.NAME)},
                        tree=tree, stock_flags=settings.describe(settings.effective(a, extra, tree), settings.stated(a, extra), tree), det=_det.describe(_det.check_level(a.det)), inputs=inp,
                        outputs={"n_files": idx["n_files"]}, gpu=stack.gpu_info(),
                        stack_key=os.environ.get("MODEL_OPT_STACK_KEY"), root=os.environ.get("OPENDDE_ROOT_DIR"), started=started, exit_code=rc,
                        extra={"launcher": "opt_core.mem.rowpair.launch.run_rank_processes", "rank_dirs": {r: _tp.rank_out_dir(out_dir, r) for r in range(int(n_gpu))},
                               "runseed": {"value": seed.value, "source": seed.source, "ranks": int(n_gpu)},   # value: the stated --seeds text (cli) | null (json: the jobs' modelSeeds) | the drawn integer (drawn)
                               "templates": _templates_record(True, *templates, tcensus) if templates is not None else None})
    mp = _manifest.write(os.path.join(out_dir, _tp.RANK_SUBDIR), m)
    print(f"{_report.PREFIX} PRED big {_tp.fields(n_gpu)} rc={rc} ranks={','.join(f'r{k}:{v}' for k, v in sorted(rcs.items()))} files={idx['n_files']} "
          f"launcher_manifest={mp}", flush=True)
    if tcensus is not None:                                                      # the template census (as on every route): an information line, never an exit code
        _templates_line(tcensus)
    return EXIT_OK if rc == 0 else (EXIT_NOT_ACTIVE if rc == EXIT_NOT_ACTIVE else (EXIT_KERNELS if rc == EXIT_KERNELS else EXIT_FAIL))


def _templates_record(use_template: bool, rows: list, mmcif_dir, weights_root, model_root, census) -> dict | None:
    """The manifest's `templates` block (None when the run reads no templates)."""
    if not use_template:
        return None
    return {"use_template": True, "mmcif_dir": mmcif_dir, "weights_root": weights_root, "model_root": model_root, "overlay": model_root != weights_root,
            "chains": [{k: r[k] for k in ("task", "chain", "path", "pdb_ids")} for r in rows], "census": census}


def _templates_line(census: dict) -> None:
    """Prints the TEMPLATES census line: per templated sequence, the hits the query names and the engine's own `Found <n>` counts, `real=<r>/<T>`
    (template slots holding a real hit, of the T the featurizer assembles) and `form=` (dense: upstream's featurizer as shipped, one process;
    row_born: under --n_gpu P every rank bore only its rows of the template pair features), and one NOTE
    line for every count that differs (a hit upstream's parser or prefilter dropped, chains of one sequence naming different hits files) —
    information only; the run's own exit code stands."""
    seqs = census["sequences"]
    found = ";".join(f"{(k if len(k) <= 12 else k[:12] + '…')}:named={v['named'][0] if len(v['named']) == 1 else v['named']},found={v['found']}" for k, v in sorted(seqs.items()))
    print(f"{_report.PREFIX} PRED TEMPLATES census: sequences={len(seqs)} all_found={census['ok']} real={census['real']}/{census['slots']} form={census['form']} "
          f"{found or 'no templated chain'}", flush=True)
    for note in census["notes"]:
        print(f"{_report.PREFIX} PRED TEMPLATES NOTE {note}", flush=True)


def cmd_pred(argv: list[str]) -> int:
    sel, rest = split_selection(argv)
    extra = []
    if "--" in rest:
        i = rest.index("--"); extra = rest[i + 1:]; rest = rest[:i]
    p = argparse.ArgumentParser(prog=f"{PROG} pred", description="opendde pred with upstream's own flags passed through, with opt_manifest.json")
    _common_pred(p)
    a = p.parse_args(rest)
    bad = input_refusal(a.input)                                                 # an unreadable query is a named refusal before any gate, never a traceback
    if bad:
        print(f"{_report.PREFIX} PRED refused: exit {EXIT_USAGE} reason={bad}", flush=True)
        return EXIT_USAGE
    tree = stack.tree_root()
    det_level = _det.check_level(a.det)
    st, eff = settings.stated(a, extra), settings.effective(a, extra, tree)       # upstream's knobs: stated on `pred` or after `--` (passed verbatim) / in force (stated, else the base's dtype, else upstream's default)
    knobs = _stockknob.plan(settings.stock_knobs(st))                            # --triatt_kernel / --trimul_kernel stated other than auto: run as stated; the kit levers of that site step aside by name
    kw = f"stock_knobs={_stockknob.word(knobs)} " if knobs else ""              # the PRED line's token (absent when nothing is stated)
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    if a.root:
        os.environ["OPENDDE_ROOT_DIR"] = a.root
    root = os.environ.get("OPENDDE_ROOT_DIR")
    ckpt = os.path.join(root, "checkpoint", "opendde.pt") if root else None
    use_template = _frozen._truthy(eff["use_template"])                         # stated, passed after `--`, or upstream's default: one rule on every route
    tmpl_rows, tmpl_dir, model_root = [], None, root
    if use_template:                                                             # templates.py: the hits the query names, the ONE cif directory, the model process's root
        try:
            tmpl_rows = _templates.hits_of(_inputs.load_query(a.input))
            tmpl_dir = _templates.mmcif_dir_of(tmpl_rows, a.template_mmcif_dir)
        except _templates.TemplateInputError as e:
            raise CliError(f"templates: {e}", EXIT_USAGE) from None
        if root:
            model_root = _templates.model_root(root, tmpl_dir, out_dir)              # the weights root itself, or <out_dir>/opendde_root (symlinks + search_database/mmcif -> the cif directory)
            os.environ["OPENDDE_ROOT_DIR"] = model_root                             # before any upstream import in this process (the kit lines) and inherited by the stock caller (off)
        n_named = sum(len(r["pdb_ids"]) for r in tmpl_rows)
        print(f"{_report.PREFIX} TEMPLATES use_template=true protein_chains={len(tmpl_rows)} templated={sum(1 for r in tmpl_rows if r['pdb_ids'])} hits_named={n_named} "
              f"mmcif_dir={tmpl_dir} root={model_root}" + (f" (overlay of {root})" if model_root != root else ""), flush=True)
    res = resolve_or_refuse(sel, tree)
    print(f"{_report.PREFIX} {stack.stack_line(tree)}", flush=True)              # the box-start stack assertion, one greppable line (OK or MISMATCH; the gates refuse on MISMATCH)
    try:                                                                         # --n_gpu: P>1 only under --mode big with P cards visible (the core's sentences)
        n_gpu = _tp.check(a.n_gpu, res.mode)
    except _tp.TpRefused as e:
        print(f"{_report.PREFIX} NOT ACTIVE mode={res.name} n_gpu={a.n_gpu} reason={str(e)!r}", flush=True)
        raise CliError(str(e), EXIT_NOT_ACTIVE) from None
    modeword = "off" if res.line is None else (res.mode or res.name)
    note = settings.dtype_note(eff["dtype"], modeword)                          # a caller's --dtype other than the stock base's: named once, run as requested (never a refusal)
    if note:
        print(f"{_report.PREFIX} {note}", flush=True)                             # the testing base named once; the run proceeds
    inp = _inputs.describe_query(a.input)

    gate = _frozen.problems(model_root, a.input, eff, extra, tree=tree)         # a frozen-weights run never reaches a stock download (absent OR wrong-size assets;
                                                                                 # under --use_template true: no search, no cif fetch, kalign present — on the model process's root)
    if gate:
        raise CliError("frozen-weights gate REFUSED (" + str(len(gate)) + " named): " + " || ".join(gate), EXIT_NOT_ACTIVE)
    print(_manifest.weights_line(_manifest.weights(root, tree=tree), _report.PREFIX), flush=True)   # equality=pinned | unknown (a WARNING by name; the run proceeds)
    if n_gpu > 1 and res.line is not None and not _tp.in_rank_process():        # big P>1: this process is the launcher's parent
        return pred_parent(a, res, n_gpu, out_dir, extra, tree, inp, started, templates=(tmpl_rows, tmpl_dir, root, model_root) if use_template else None)
    if n_gpu > 1 and res.line is not None:                                       # a rank process of the launcher: big resolves to its row-sharded line
        mm = _tp.mismatch(n_gpu, _tp.rank_world(), "the launcher's world of this rank process")
        if mm:                                                                   # fail-closed: a rank started for another P never runs the model
            print(f"{_report.PREFIX} NOT ACTIVE mode={res.name} reason={mm}", flush=True)
            raise CliError(mm, EXIT_NOT_ACTIVE)
        _tp.enter_rank_process(n_gpu)
        res = resolve_or_refuse(sel, tree)
    ckpt_args = ["--load_checkpoint_path", ckpt] if ckpt else []                # the weights root's checkpoint, by upstream's own flag
    stock_args = ["pred", "-i", os.path.abspath(a.input), "-o", out_dir, *settings.stated_args(a), *settings.base_args(a, extra), *ckpt_args, *_det.cli_args(det_level), *extra]   # the stated upstream flags verbatim (+ the stock base's --dtype when none is stated); unstated = upstream's defaults
    det_env = _det.env(det_level)
    if res.line is None:                                                         # ---- off: the stock caller, clean subprocess
        proof_dir = tempfile.mkdtemp(prefix="stock_proof_")                 # the proof is the parent's handshake with its stock child, not an output: read below, then removed
        proof_json = os.path.join(proof_dir, STOCK_PROOF_NAME)
        cmd, env = stock_command(tree, stock_args, proof_json, det_level)
        env.update(det_env)
        activation = stack.activate("off", tree=tree, n_gpu=n_gpu)               # a torch / cuEquivariance stack other than the pin's is named on its line (stack_mismatch=…), never refused
        if use_template:                                                         # the engine's `Found <n> templates` lines are on the child's stderr: relayed and kept for the census
            rc, transcript = _call_relay(cmd, env)
        else:
            rc = subprocess.call(cmd, env=env, stdin=_nostdin.CHILD_STDIN)
        tcensus = _templates.census(transcript.splitlines(), tmpl_rows) if use_template else None
        proof = json.load(open(proof_json)) if os.path.isfile(proof_json) else {"ok": False, "reason": "no proof file written"}
        shutil.rmtree(proof_dir, ignore_errors=True)
        activation["stock_proof"] = proof
        if rc == 3 or not proof.get("ok"):
            activation["reason"] = f"stock caller did not run stock: {proof}"
        idx = outputs.index_cli(out_dir)
        nonfinite = outputs.nonfinite_cli(idx)                                   # the finiteness census (its own sentence below; the manifest's nonfinite_outputs)
        activation["structures"] = {"found": outputs.found_structures(idx)}
        m = _manifest.build(command="pred", mode="off", line=None, activation=activation, tree=tree, stock_flags=settings.describe(eff, st, tree),
                            det=_det.describe(det_level), inputs=inp, outputs={"n_files": idx["n_files"]},
                            gpu=stack.gpu_info(), stack_key=os.environ.get("MODEL_OPT_STACK_KEY"), root=root, started=started, exit_code=rc,
                            extra={"stock_cmd": cmd, "stock_proof": proof, "stock_args": stock_args, "nonfinite_outputs": nonfinite,
                                   "templates": _templates_record(use_template, tmpl_rows, tmpl_dir, root, model_root, tcensus)})
        mp = _manifest.write(out_dir, m)
        kblock = proof.get("kernels") or {}
        kernels_ok = bool(kblock.get("line")) and not kblock.get("problems")   # fail-closed: a proof WITHOUT the census block (or without its printed line) is not a proven route
        print(f"{_report.PREFIX} PRED mode=off route={modes.route_word(None, None, n_gpu)} {settings.fields(eff, st)} {kw}rc={rc} stock_proof_ok={proof.get('ok')} files={idx['n_files']} "
              f"nonfinite={len(nonfinite)} kernels_ok={kernels_ok} manifest={mp}", flush=True)
        if rc == EXIT_KERNELS or (rc == 0 and not kernels_ok):                 # the KERNELS census refused the stock route in the model process (its line above), or the model
            why = kblock.get("problems") or ["kernels_census_missing_from_stock_proof"]   # process returned 0 without a census record: exit 5 by name, never 0, never 1
            print(f"{_report.PREFIX} PRED KERNELS refused: exit {EXIT_KERNELS} route={modes.route_word(None, None, n_gpu)} " + " ".join(why), flush=True)
            return EXIT_KERNELS
        if nonfinite and not (rc == 3 or not proof.get("ok")):                   # stock's own NaN is never a success either: the gate's one sentence and token
            print(f"{_report.PREFIX} PRED NONFINITE refused: exit {EXIT_NOT_ACTIVE} {outputs.NONFINITE}={','.join(nonfinite)} (the outputs and {mp} stay)", flush=True)
            return EXIT_NOT_ACTIVE
        if tcensus is not None:                                                  # the template census: named vs found per sequence, an information line (never an exit code)
            _templates_line(tcensus)
        return EXIT_NOT_ACTIVE if (rc == 3 or not proof.get("ok")) else (EXIT_OK if rc == 0 else EXIT_FAIL)
    # ---- kit mode / line: the shim armed in this process, then the stock click group in-process
    try:
        big.plan(res, a.input)                                                 # big lines: the size-gate decisions (the offload unit + sample_chunk, the DiT hoist's removal) from the
                                                                                 # query's residue tokens, in force before activation composes the line ({} for every other line;
                                                                                 # recorded as activation.big.<lever>_policy)
        _smalln.plan(res, a.input)                                               # exact / fast: the small-input floor's decision from the same count — every item below
                                                                                 # MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS composes the floor's levers out of the line (activation.small_input_floor)
        _chunklift.plan(res, a.input)                                            # the chunk lever's ceiling from the same count: composed in only when every item <= the table's un-chunked gate
        _stepgraph.plan(res, a.input)                                            # the sampler step-graph's size gate from the same count (composed in only inside [min, max] residue tokens)
    except modes.OpenModeError as e:                                             # a malformed size-gate / floor variable: refused by name (exit 3), nothing applied
        print(f"{_report.PREFIX} NOT ACTIVE mode={res.name} reason={e}", flush=True)
        raise CliError(str(e), EXIT_NOT_ACTIVE) from None
    rep = stack.activate(sel, tree=tree, overrides=det_env,                            # the recipe's env AFTER the line's exports
                         jobs=_inputs.load_query(a.input), n_gpu=n_gpu)                             # the query items: alloc_auto's token-floor decision
    if not rep.get("active"):
        raise CliError(f"not active: {rep.get('reason')}", EXIT_NOT_ACTIVE)
    _stockknob.export()                                                          # the stated stock knobs as the kit process's fact (ODDE_STOCK_KNOBS): the ARM add-on's own attention /
                                                                                 # TriMul sites read it at install and stay on the stock op; the rank processes inherit it
    mm = _tp.mismatch(n_gpu, rep.get("n_gpu", 1), "the activation report")     # fail-closed: the axis the model process reports == the --n_gpu it accepted
    if mm:
        print(f"{_report.PREFIX} NOT ACTIVE mode={res.name} reason={mm}", flush=True)
        raise CliError(mm, EXIT_NOT_ACTIVE)
    try:                                                                         # the LayerNorm backend upstream actually bound (the pinned stock's fused kernel), before any weights load
        rep["layernorm"] = _lncensus.census(strict=True, stream=sys.stdout)
    except _lncensus.NotLoaded as e:
        print(f"{_report.PREFIX} NOT ACTIVE mode={res.name} reason={str(e)!r}", flush=True)
        raise CliError(str(e), EXIT_NOT_ACTIVE) from None
    route = modes.route_word(res.line, res.mode, n_gpu)
    try:                                                                         # the triangle-kernel census (the same reader): presence, counters, device probe, resolved selection at
        _lncensus.arm(route, modes.kernel_expectations(res.line, ln_requested=settings.ln_requested(), n_gpu=n_gpu, knobs=knobs),   # runner init; REFUSES (exit 5); a stated kernel flag's site is the caller's word
                      tag=_report.PREFIX, strict=True, stream=sys.stdout, layernorm=rep["layernorm"])                               # by name before any weights load
    except _lncensus.KernelsRefused as e:
        print(f"{_report.PREFIX} PRED KERNELS refused: exit {EXIT_KERNELS} route={route} " + " ".join(e.problems), flush=True)
        return EXIT_KERNELS
    tcensus = None
    if _tp.in_rank_process():                                                  # FAIL-FAST in a rank process: an exception ends THIS process at once (os._exit; no atexit
        try:                                                                     # group teardown that would wait on peers mid-collective) so the core launcher sees the death
            rc = run_stock_cli(stock_args)                                          # and tears the sibling ranks down — never a 30-minute NCCL watchdog wait
        except BaseException as _e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{_report.PREFIX} ROWPAIR event=rank_failed rank={_tp.rank()} failfast=os_exit {type(_e).__name__}: {str(_e)[:400]}", flush=True)   # fail-LOUD (traceback above, this line), THEN fast
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(EXIT_KERNELS if isinstance(_e, _lncensus.KernelsRefused) else 1)   # the KERNELS guard's refusal keeps its status through the launcher
    else:
        collector = _templates.FoundCollector().attach() if use_template else None   # the featurizer's own logger; upstream's handlers untouched
        try:
            rc = run_stock_cli(stock_args)
        finally:
            if collector is not None:
                collector.detach()
        tcensus = _templates.census(collector.lines, tmpl_rows) if collector is not None else None
    rc = _lncensus.finish(rc)                                                    # THE KERNELS line of this pass (final counters); exit 5 when the library took its reference path where its rules say served
    facts = {"det": bool(det_level), "n_gpu": n_gpu, "rank": _tp.rank(), "token_floors": _alloc.token_floor(_inputs.load_query(a.input)), "stock_knobs": dict(knobs)}   # known before the run: the engagement predicates' input (rank: this process's index among the P rank processes, 0 at P=1)
    rep = stack.refresh(predicted=True, facts=facts)                             # the stock CLI ran the query: every applied lever proves it executed — lever_never_ran:<lever>
    rep["det"] = det_level                                                       # (PARTIAL) inside its domain, lever_inert_by_design:<lever>(<reason>) outside it (complete)
    rep["kernels"] = _lncensus.record()                                          # the KERNELS census block (expectations, presence, probe, resolved selection, counters, words, the line)
    ran_world = (_tp.STATS["n_gpu"] if _tp.STATS.get("group") else _tp.rank_world()) if n_gpu > 1 else 1
    mm = _tp.mismatch(n_gpu, ran_world, "the process group the sharded stacks ran on")   # a run that folded on fewer cards is never a pass
    if mm:
        print(f"{_report.PREFIX} NOT ACTIVE mode={res.name} reason={mm}", flush=True)
        raise CliError(mm, EXIT_NOT_ACTIVE)
    idx = outputs.index_cli(out_dir)
    owed, found = outputs.expected_structures(stock_args, _inputs.load_query(a.input)), outputs.found_structures(idx)
    nonfinite = outputs.nonfinite_cli(idx)                                       # the finiteness census: a NaN structure / summary is never a complete row (its own sentence below)
    rep["structures"] = {"owed": owed, "found": found}
    m = _manifest.build(command="pred", mode=res.mode, line=res.line.name, activation=rep, tree=tree, stock_flags=settings.describe(eff, st, tree),
                        det=_det.describe(det_level), inputs=inp,
                        outputs={"n_files": idx["n_files"]},
                        gpu=stack.gpu_info(), stack_key=os.environ.get("MODEL_OPT_STACK_KEY"), root=root, started=started, exit_code=rc,
                        extra={"stock_args": stock_args, "applied": stack.applied(), "allow_partial": bool(a.allow_partial), "nonfinite_outputs": nonfinite,
                               "templates": _templates_record(use_template, tmpl_rows, tmpl_dir, root, model_root, tcensus)})
    mp = _manifest.write(out_dir, m)
    refused_pin = ",".join(f"{k}:{v}" for k, v in sorted((rep.get("levers_refused_on_pin") or {}).items()))
    fw = _report.floor_word(rep)                                                 # floor=below_gate:<tokens>/<gate> when the small-input floor composed levers out of this call's line
    cw = _report.ceiling_word(rep)                                               # ceiling=above_gate:<tokens>/<gate> when the chunk lever's ceiling composed chunk_lift out
    print(f"{_report.PREFIX} PRED {res.name} {_tp.fields(n_gpu)} {settings.fields(eff, st)} rc={rc} levers={','.join(rep.get('levers_applied') or [])} partial={rep.get('partial')} "
          + (f"floor={fw} " if fw else "") + (f"ceiling={cw} " if cw else "") + (f"ablated={','.join(rep['ablated'])} " if rep.get("ablated") else "") + kw +
          (f"refused_on_pin={refused_pin} " if refused_pin else "") +
          f"files={idx['n_files']} structures={found['total']}/{owed['total']} nonfinite={len(nonfinite)} manifest={mp}", flush=True)
    if nonfinite:                                                                # the finiteness gate: its own sentence and token, exit 3, whatever rc and the levers say
        print(f"{_report.PREFIX} PRED NONFINITE refused: exit {EXIT_NOT_ACTIVE} {outputs.NONFINITE}={','.join(nonfinite)} (the outputs and {mp} stay)", flush=True)
        return EXIT_NOT_ACTIVE
    if rc == 0 and rep.get("partial") and not a.allow_partial:               # a degraded row never reads as success on rc (FAIL-LOUD fallbacks)
        print(f"{_report.PREFIX} PRED PARTIAL refused: exit {EXIT_NOT_ACTIVE} fallbacks={','.join(rep.get('levers_fallback') or [])} "
              f"(the outputs and {mp} stay; pass --allow-partial to accept the degraded row, recorded in the manifest)", flush=True)
        return EXIT_NOT_ACTIVE
    if found["total"] < owed["total"]:                                        # a missing structure is never a success here, whatever the stock's exit code (the pinned opendde exits non-zero on a failed sample)
        missing = {k: f"{found['per_item'].get(k, 0)}/{v}" for k, v in owed["per_item"].items() if found["per_item"].get(k, 0) < v}
        print(f"{_report.PREFIX} PRED INCOMPLETE exit {EXIT_FAIL}: structures {found['total']}/{owed['total']} missing={missing} "
              f"(the stock's per-seed errors are in the transcript above; the outputs and {mp} stay)", flush=True)
        return EXIT_FAIL
    if rc == EXIT_KERNELS:
        print(f"{_report.PREFIX} PRED KERNELS refused: exit {EXIT_KERNELS} route={route} " + " ".join((rep.get('kernels') or {}).get('problems') or []), flush=True)
        return EXIT_KERNELS
    if tcensus is not None:                                                      # the template census (as on the stock route): an information line
        _templates_line(tcensus)
    return EXIT_OK if rc == 0 else EXIT_FAIL


def cmd_check(argv: list[str]) -> int:
    sel, rest = split_selection(argv)
    p = argparse.ArgumentParser(prog=f"{PROG} check", description="dry run: resolve and gate a mode/line on this box; apply nothing")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(rest)
    tree = stack.tree_root()
    rep = stack.activate(sel, dry_run=True, tree=tree)
    rep["stack"] = stack.stack_line(tree)                                        # the box-start stack assertion's one line (STACK OK … | STACK MISMATCH …)
    facts = {"tree": tree, "weights": _manifest.weights(os.environ.get("OPENDDE_ROOT_DIR"), tree=tree, refresh=True),   # `check` hashes afresh and rewrites the memo entry
             "stack_key": os.environ.get("MODEL_OPT_STACK_KEY"), "target_gpu": os.environ.get("MODEL_OPT_TARGET_GPU"), "versions": _manifest.versions()}
    rep["facts"] = facts
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        for k in ("stack", "line", "tier", "hook", "levers_planned", "levers_unavailable", "levers_untested", "pythonpath", "exports", "unset", "reason"):
            if k in rep and rep[k] not in (None, [], {}):
                v = rep[k]
                print(f"  {k}: {json.dumps(v) if isinstance(v, (list, dict)) else v}")
        print(f"  gpu: {rep.get('gpu')}  opendde: {rep.get('opendde_version')}  checkpoint: {facts['weights'].get('checkpoint_present')} at {facts['weights'].get('root')} match={facts['weights'].get('checkpoint_match')}")
        print(_manifest.weights_line(facts["weights"], _report.PREFIX))
    resolvable = rep.get("dry_run") and rep.get("reason", "").startswith("dry run")
    return EXIT_OK if (resolvable or (sel == "off" and "refused" not in rep.get("reason", ""))) else EXIT_NOT_ACTIVE


def cmd_warm(argv: list[str]) -> int:
    from . import warm as _warm
    sel, rest = split_selection(argv)
    p = argparse.ArgumentParser(prog=f"{PROG} warm", description="one prediction of upstream's smallest documented example through pred")
    p.add_argument("-o", "--out_dir", default=None)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(rest)
    r = _warm.run(sel, a.out_dir, keep=a.keep)
    print(_warm.summary_line(r), flush=True)
    if a.json:
        print(json.dumps(r, indent=1, default=str))
    return EXIT_OK if r["status"] == "PASS" else (EXIT_NOT_ACTIVE if r.get("rc") == 3 else EXIT_FAIL)


# ----------------------------------------------------------------------------------------------------------------- entry
COMMANDS = {"pred": cmd_pred, "check": cmd_check, "warm": cmd_warm}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from ._core_gate import gate
    gate(__file__, _report.TAG)                                                  # THE pin gate (statement one): absent / mismatched opt_core -> NOT ACTIVE line, exit 3
    r = _producers.refusal()                                                     # then the finer words: a producer module this package imports is absent -> exit 3
    if r:
        print(r, file=sys.stderr, flush=True)
        return EXIT_NOT_ACTIVE
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE, end="")
        return EXIT_OK if argv else EXIT_USAGE
    for note in settings.stock_env_notes():                                     # a caller's own LAYERNORM_TYPE runs as requested, named once
        print(f"{_report.PREFIX} {note}", file=sys.stderr, flush=True)
    _nostdin.detach()                                                            # every verb runs off the caller's stdin (nostdin: kalign under --use_template blocks on an open non-tty fd 0)
    settings.apply_stock_env()                                                   # the stock base's environment (LAYERNORM_TYPE) before anything imports upstream — every command, `off` included
    cmd = argv[0]
    fn = COMMANDS.get(cmd)
    if fn is None:
        print(f"{_report.PREFIX} unknown command {cmd!r}\n{USAGE}", end="", file=sys.stderr)
        return EXIT_USAGE
    try:
        return fn(argv[1:])
    except CliError as e:
        print(f"{_report.PREFIX} ERROR: {e}", file=sys.stderr, flush=True)
        return e.code

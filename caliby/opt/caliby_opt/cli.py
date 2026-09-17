"""python -m caliby_opt {design,check,warm} [--mode off|fast|exact] [--variant single|ensemble32] ...

A thin command layer over the kit code and the upstream API. It never re-implements a lever or a mode:

* ``design``  — sequence design from structure files (inputs.py) at upstream's settings or the knobs given (settings.py), written by the tree's design
               writer (the kit's ``tests/xcaliby_design.py``) in its one file set — ``seq_des_outputs.csv`` (the table upstream's scripts write),
               ``raw/samples/``, ``cleaned/``, ``timing.json`` — plus ``opt_manifest.json`` and, for stock, ``stock_env_proof.json``.
               ``--mode fast`` (single) / ``--mode exact`` (ensemble32), the variant's one kit mode: the design runs in a child
               process (design_run.py: enable() there — the installed tree proven to be
               the pinned upstream, the kits' files proven present and loaded by import hook (overlay.py), the row
               exported, the activation line, the exit tally); nothing is written into site-packages.
               ``--mode off``: the stock caller ``stock_design.py`` in a clean subprocess (every kit and package switch stripped and
               proven absent, the tree digest asserted equal to the pin; ``NOT STOCK`` refuses).
               Both children start from the caller's environment with every kit and package switch stripped (the ENV-CLEAN line
               names what was stripped); the kit child then gets CALIBY_OPT / CALIBY_VARIANT and the row, nothing else
               (``--clean_workers`` selects the row: modes.py). Upstream's own keywords — ``--model_name``,
               ``--num_seqs_per_pdb``, ``--batch_size``, ``--omit_aas``, ``--temperature``, ``--num_workers``, ``--verbose``,
               ``--sampling_overrides``, ``--pos_constraint_csv``, ``--clean_workers`` and, on ensemble32, ``--num_samples_per_pdb``,
               ``--pp_batch_size``, ``--sampling_yaml_path``, ``--max_num_conformers``, ``--include_primary_conformer``,
               ``--use_primary_res_type`` — are forwarded to the writer only when given, so upstream's default applies otherwise
               (settings.py); ``--seed`` / ``--det`` are the seed and upstream's scripts' deterministic recipe. The DESIGNS line is the
               design count against the request.
* ``check``   — the activation line for (mode, variant) without applying anything, gated on the environment the design child would
               see: the row and its switches, the installed tree state, the kit files' digests, pins, weights, GPU (and the MODEL_OPT_TARGET_GPU comparison),
               compiler, the other literals the kits state for the row; ``--json`` prints the full report. Exit 0 when the mode
               would activate here, 3 when it would refuse (the code ``design`` gives for the same box).
* ``warm``    — one design of the smallest public example structure in the mode (warm.py): the Triton JIT of the fused LCP kernel lands
               in the persistent cache directory of configs/<gpu>.env once; the model load and the sampler's graph capture recur in
               every ``design`` invocation (warm.py).

Exit codes: 0 ok · 1 the run failed, or its outputs are short of the request (``incomplete: <n>/<m>`` in the manifest) · 2 usage ·
3 the mode could not be activated (``NOT ACTIVE``) — including a run in which a lever of the row could not run at call time: a mode is
all of its levers, so that run ends NOT ACTIVE by name (``levers_fallback`` / ``partial`` in the manifest, ``partial=`` on the EXIT
line), never under the mode's name with a subset. Mode: ``--mode``, else
CALIBY_OPT, else ``fast`` on both variants (``modes.default_mode``; never exact); ``exact`` on single (no bit-identical row there) and
an unknown name are refused by name with exit 2 before any work (``modes.mode_refusal``); ``fast`` on
ensemble32 runs the ensemble row ``exact`` names.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

from . import __version__, inputs as _inputs, manifest, modes, report as _report, settings as _settings, stack, warm as _warm
from .report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE      # the shared core's exit table (opt_core.report), one place

PROG = stack.TAG                                                       # the console script (pyproject [project.scripts]) carries the kit's tag as its name: one spelling


class CliError(Exception):
    def __init__(self, msg: str, code: int = EXIT_USAGE):
        super().__init__(msg)
        self.code = code


def _variant_from(a) -> str:
    v = (getattr(a, "variant", None) or os.environ.get(stack.ENV_VARIANT) or modes.DEFAULT_VARIANT).strip().lower()
    if v not in modes.VARIANTS:
        raise CliError(f"unknown variant {v!r} (expected {'|'.join(modes.VARIANTS)})", EXIT_USAGE)
    return v


def _mode_variant(a) -> Tuple[str, str]:
    """(mode, variant): --mode / CALIBY_OPT, else the variant's kit mode; a pair the mode table refuses (modes.mode_refusal — an unknown
    name, exact on single, fast on ensemble32) is exit 2 with its words before any work."""
    v = _variant_from(a)
    m = (getattr(a, "mode", None) or os.environ.get(stack.ENV_MODE) or modes.default_mode(v)).strip().lower()
    why = modes.mode_refusal(m, v)
    if why:
        raise CliError(why, EXIT_USAGE)
    return m, v


def _child_env(mode: str, variant: str, row_env: dict) -> dict:
    """The design child's environment: the caller's with every kit and package switch stripped (both modes), plus, for exact, the
    package's two switches and the row — so the composition is the row and nothing the caller's shell added."""
    from opt_core.process import child_env
    export = {stack.ENV_TREE: stack.tree_home()}
    if mode != "off":
        export.update({stack.ENV_MODE: mode, stack.ENV_VARIANT: variant, **row_env})
    env = child_env(os.environ, strip_prefixes=stack.STOCK_ABSENT_PREFIXES, export=export)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _run_child(argv: List[str], env: dict) -> int:
    """The design child, waited for without a deadline (a design runs to its end or fails on its own)."""
    proc = subprocess.Popen(argv, env=env, start_new_session=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        import signal
        os.killpg(proc.pid, signal.SIGTERM)
        raise


# ----------------------------------------------------------------------------------------------------------------- design / warm
def designs_written(timing: Optional[dict]) -> int:
    """The writer's own count of designs written (``timing.json`` n_designs; no timing.json = 0) — the one count behind ``incomplete`` and
    the DESIGNS line."""
    return int((timing or {}).get("n_designs") or 0)


def incomplete(files: List[str], n_seqs: int, timing: Optional[dict]) -> Optional[str]:
    """``<n>/<m>`` when the writer's ``timing.json`` counts fewer designs than the request (``len(files) * n_seqs``; no timing.json =
    0 designs), else None. An output shortfall is the named state ``incomplete`` (EXIT_FAIL), never ``partial``."""
    want = len(files) * int(n_seqs)
    got = designs_written(timing)
    return f"{got}/{want}" if got < want else None


def usage_refusal(variant: str, knobs: dict) -> Optional[str]:
    """The design verb's one usage rule beyond argparse (exit 2 by name): the conformer-generation and ensemble knobs
    (settings.ENSEMBLE_ONLY) belong to the ensemble32 variant — single generates no conformers. Every upstream value passes through."""
    only = _settings.ensemble_only_given(knobs)
    if only and variant != _settings.ENSEMBLE_VARIANT:
        return f"{', '.join(only)}: ensemble32 knobs (conformer generation / ensemble members); refused on {variant}"
    return None


def run_design(mode: str, variant: str, files: List[str], out_dir: str, *, seed: Optional[int] = None, det: int = 0,
               command: str = "design", **knobs) -> int:
    """``knobs``: upstream's keywords the command gave (settings.PASS_THROUGH / ENSEMBLE_ONLY names; None = not given)."""
    knobs = {k: v for k, v in knobs.items() if v is not None}
    cw = int(knobs.get("clean_workers") or 1)                           # clean_pdbs(num_workers=1) is upstream's default; N>1 selects the parallel-clean row on fast/single
    try:
        res = modes.resolve(mode, variant, cw)
    except ValueError as e:
        raise CliError(str(e), EXIT_USAGE)
    why = usage_refusal(variant, knobs)
    if why:
        raise CliError(why, EXIT_USAGE)
    if det and seed is None:
        raise CliError("--det 1 needs --seed (the deterministic recipe is upstream's scripts': a seed plus cuDNN's deterministic flags)", EXIT_USAGE)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    _, stripped = stack.strip_env(dict(os.environ))
    _report.say(_report.env_clean_line(stripped, mode, variant))   # what the caller's shell carried that the design child will not see (the child starts from the stripped environment)
    wargs = _settings.writer_args(variant, files, out_dir, seed=seed, det=det, **knobs)
    sdesc = _settings.describe(variant, seed=seed, det=det, **knobs)
    t0 = time.time()
    if mode == "off":
        argv = [sys.executable, "-s", "-m", "caliby_opt.stock_design", "--variant", variant, "--out_dir", out_dir, "--"] + wargs
        env = _child_env("off", variant, {})
    else:
        argv = [sys.executable, "-m", "caliby_opt.design_run", "--mode", mode, "--variant", variant, "--clean_workers", str(cw),
                "--model_name", sdesc["model_name"], "--out_dir", out_dir]
        argv += ["--"] + wargs
        env = _child_env(mode, variant, res.env)
    rc = _run_child(argv, env)                                         # both modes: one child process activates, proves and designs
    timing = None
    tpath = os.path.join(out_dir, "timing.json")
    if os.path.isfile(tpath):
        with open(tpath, encoding="utf-8") as fh:
            timing = json.load(fh)
    short = incomplete(files, sdesc["num_seqs_per_pdb"], timing) if rc == EXIT_OK else None
    if short:
        _report.say(f"{_report.PREFIX} incomplete: {short} designs written for the request (timing.json n_designs vs inputs x num_seqs_per_pdb)")
        rc = EXIT_FAIL
    _report.say(_report.designs_line(designs_written(timing), len(files) * int(sdesc["num_seqs_per_pdb"])))   # the same count as ``incomplete``: the writer's timing.json n_designs against inputs x num_seqs_per_pdb
    manifest.update(out_dir, command=command, exit_code=rc, settings=sdesc, timing=timing, incomplete=short,
                    wall_s=round(time.time() - t0, 1), parent_argv=sys.argv, inputs=files)
    _report.say(f"{_report.PREFIX} {command} mode={mode} variant={variant} rc={rc} out={out_dir} "
                f"designs={timing.get('n_designs') if timing else None} wall={round(time.time() - t0, 1)}s")
    return rc


def cmd_design(a) -> int:
    mode, variant = _mode_variant(a)
    try:
        files = _inputs.resolve(a.input)
    except (FileNotFoundError, ValueError) as e:
        raise CliError(f"--input: {e}", EXIT_USAGE)
    knobs = {k: getattr(a, k, None) for k in list(_settings.PASS_THROUGH) + list(_settings.ENSEMBLE_ONLY)}
    return run_design(mode, variant, files, a.out_dir, seed=a.seed, det=a.det, **knobs)


def cmd_warm(a) -> int:
    mode, variant = _mode_variant(a)
    return _warm.run(mode, variant, run_design, out_dir=a.out_dir, keep=a.keep)


# ----------------------------------------------------------------------------------------------------------------- check
def cmd_check(a) -> int:
    """The dry run gates the environment the design child would see (``_child_env``: the caller's, stripped of every kit and package
    switch; the row is added by the resolver), so ``check`` describes what ``design`` would do; the stripped names are reported."""
    mode, variant = _mode_variant(a)
    from . import activate
    env = _child_env(mode, variant, {})
    _, stripped = stack.strip_env(dict(os.environ))
    rep = activate.check(mode, variant, a.clean_workers, need_gpu=not a.no_gpu, env=env, ckpt=a.model_name)
    rep["stripped_from_caller_env"] = stripped
    rep["table"] = {k: v for k, v in modes.table().items()}
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(_report.check_line(rep))
        if rep.get("switches"):
            print("  switches: " + " ".join(f"{k}={v}" for k, v in rep["switches"].items()) + f"   ({rep.get('row_source')})")
        print("  tree files: " + " ".join(f"{k}={v}" for k, v in (rep.get("tree_files") or {}).items()))
        if stripped:
            print("  stripped from the caller's environment: " + " ".join(stripped))
        for s in rep.get("row_alternatives") or []:
            print("  also stated: " + s)
    return EXIT_OK if not rep.get("would_refuse") else EXIT_NOT_ACTIVE


# ----------------------------------------------------------------------------------------------------------------- parser
def _bool_word(s: str) -> bool:
    """upstream's booleans on the command line: true|false (1|0, yes|no)."""
    v = str(s).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true|false, got {s!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=PROG, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"caliby_opt {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--mode", default=None, help=f"{'|'.join(modes.MODES)} (or {stack.ENV_MODE}); default {modes.DEFAULT_MODE} on both variants (single: row X, tier 2; ensemble32: the ensemble row exact names); exact on single is refused by name")
        p.add_argument("--variant", default=None, help=f"{'|'.join(modes.VARIANTS)} (or {stack.ENV_VARIANT}); default {modes.DEFAULT_VARIANT}")

    d = sub.add_parser("design", help="sequence design from structure files")
    common(d)
    d.add_argument("--input", nargs="+", required=True, help="structure files, directories, or list files")
    d.add_argument("--out_dir", required=True)
    d.add_argument("--seed", type=int, default=None, help="the seed (Lightning's seed_everything before sampling — upstream's API sets none, so no --seed = its unseeded call)")
    d.add_argument("--det", type=int, default=0, choices=(0, 1), help="1 = upstream's scripts' deterministic recipe: --seed (required) plus torch.backends.cudnn.deterministic=True / benchmark=False (seq_des.py:22-24); the same on every mode; default 0")
    up = "upstream's keyword, forwarded only when given — default: upstream's"
    d.add_argument("--model_name", default=None, help=f"load_model(model_name=): a name in upstream's MODEL_REGISTRY (the kit pins caliby | soluble_caliby | soluble_caliby_v1) or a .ckpt path; {up} ({_settings.DEFAULT_MODEL_NAME})")
    d.add_argument("--device", default=None, help=f"load_model(device=); {up} (cuda)")
    d.add_argument("--sampling_cfg_path", default=None, help=f"load_model(sampling_cfg_path=): a sampling YAML in place of upstream's inference.yaml; {up}")
    d.add_argument("--num_seqs_per_pdb", type=int, default=None, help=f"sample(num_seqs_per_pdb=): sequences per structure; {up} ({_settings.DEFAULT_NUM_SEQS_PER_PDB})")
    d.add_argument("--batch_size", type=int, default=None, help=f"sample(batch_size=): structures per sequence-design batch; {up} (4)")
    d.add_argument("--omit_aas", default=None, help=f"sample(omit_aas=): one-letter codes never sampled, comma-separated, e.g. C or C,G; {up} (none)")
    d.add_argument("--temperature", type=float, default=None, help=f"sample(temperature=): Potts sampling temperature; {up} (0.01)")
    d.add_argument("--num_workers", type=int, default=None, help=f"sample(num_workers=): data-loading workers; {up} (2)")
    d.add_argument("--verbose", type=_bool_word, default=None, metavar="true|false", help=f"sample(verbose=); {up} (true)")
    d.add_argument("--sampling_overrides", nargs="+", default=None, metavar="KEY=VALUE", help=f"sample(sampling_overrides=): sampling-config keys, Hydra style, e.g. potts_sampling.n_sweeps=200; {up} (none)")
    d.add_argument("--pos_constraint_csv", default=None, help=f"upstream's positional-constraint CSV (seq_des.py pos_constraint_csv: pdb_key, fixed_pos_seq, fixed_pos_scn, fixed_pos_override_seq, pos_restrict_aatype, ...), read into sample(pos_constraint_df=); {up} (none: every position designed)")
    d.add_argument("--clean_workers", type=int, default=None, help=f"clean_pdbs(num_workers=): structure-cleaning workers; {up} (1); >1 is upstream's parallel clean, and under fast/single the parallel-clean row")
    d.add_argument("--num_samples_per_pdb", type=int, default=None, help=f"ensemble32: generate_ensembles(num_samples_per_pdb=): conformers generated per structure; {up} (32)")
    d.add_argument("--pp_batch_size", type=int, default=None, help=f"ensemble32: generate_ensembles(batch_size=): conformers per Protpardelle-1c batch; {up} (8)")
    d.add_argument("--sampling_yaml_path", default=None, help=f"ensemble32: generate_ensembles(sampling_yaml_path=): the partial-diffusion sampling YAML; {up}")
    d.add_argument("--max_num_conformers", type=int, default=None, help=f"ensemble32: members per ensemble — the input structure first, then the generated conformers in natural order (seq_des_ensemble.py max_num_conformers); {up} (32)")
    d.add_argument("--include_primary_conformer", type=_bool_word, default=None, metavar="true|false", help=f"ensemble32: the input structure is the ensemble's first member (seq_des_ensemble.py include_primary_conformer); {up} (true)")
    d.add_argument("--use_primary_res_type", type=_bool_word, default=None, metavar="true|false", help=f"ensemble32: ensemble_sample(use_primary_res_type=); {up} (true)")
    d.set_defaults(func=cmd_design)

    c = sub.add_parser("check", help="dry run: resolve and gate the mode on this box, apply nothing")
    common(c)
    c.add_argument("--clean_workers", type=int, default=None)
    c.add_argument("--model_name", default=None, help=f"the checkpoint the weights step checks under $MODEL_PARAMS_DIR (default {_settings.DEFAULT_MODEL_NAME})")
    c.add_argument("--json", action="store_true")
    c.add_argument("--no-gpu", action="store_true", help="skip the GPU probe (no torch import)")
    c.set_defaults(func=cmd_check)

    w = sub.add_parser("warm", help="one design of the smallest public example structure in the mode (JIT + graph capture)")
    common(w)
    w.add_argument("--out_dir", default=None, help="keep the outputs here (default: a scratch directory, removed on success)")
    w.add_argument("--keep", action="store_true")
    w.set_defaults(func=cmd_warm)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return EXIT_USAGE
    try:
        return a.func(a)
    except CliError as e:
        _report.say(f"{_report.PREFIX} {'NOT ACTIVE: ' if e.code == EXIT_NOT_ACTIVE else ''}{e}")
        return e.code
    except KeyboardInterrupt:
        return 130

"""The stock caller — mode ``off``: the stock ``opendde`` command line in a clean subprocess, proven before it runs.

``pred --mode off`` execs this module (``python -s -m opendde_opt.stock_pred --proof-json <out_dir>/stock_env_proof.json --env-absent
<prefixes> --env-reads <names> --kit-dirs <dirs> -- pred <stock arguments>``) with every environment name under stock/PINS.json
``stock_environment.must_be_absent_prefixes`` stripped and every PYTHONPATH entry inside a kit directory removed (cli.stock_command).
Before anything of opendde is imported the process proves what it is (``env_proof``): no forbidden environment name, no kit module
loaded (by name or by file), no kit directory on sys.path, the package's autoload finder not armed, no kit sitecustomize in this
process, torch not yet imported; the names upstream reads (``stock_environment.reads``) are recorded at their values. The proof is
written as JSON and read into ``opt_manifest.json`` by the ``pred`` process. A process that fails its proof exits 3 without running
anything. Then the stock click group (``runner.cli:opendde_cli``, the ``opendde`` console script's entry point) runs
with the arguments unchanged. This module imports the standard library, this package's ``settings`` / ``lncensus`` / ``phase`` (standard library only) and, at run
time, the stock package — nothing of the kits.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable, List, Optional

from opt_core import stock_proof as _proof

from . import lncensus as _lncensus
from . import phase as _phase
from . import settings as _settings

PREFIX = "[opendde-opt stock]"
EXIT_NOT_STOCK = 3                                                      # the package's EXIT_NOT_ACTIVE: nothing ran
EXIT_KERNELS = _lncensus.EXIT_KERNELS                                   # 5: the KERNELS census refused the stock route (an accelerator upstream engages here is absent / fell back)
KIT_MODULE_PREFIXES = ("odde_", "fpf_", "fpf", "score_interface", "sitecustomize_chained", "odde_arm_t")   # the kits' module names
STOCK_ENTRY = ("runner.cli", "opendde_cli")                             # the `opendde` console script's entry point (upstream pyproject.toml:75; stock/PINS.json "console_script")
PROG = "opendde"


def kit_modules_loaded(kit_dirs: Iterable[str], modules=None) -> List[str]:
    """The loaded modules that are the kits': by name (``KIT_MODULE_PREFIXES``) or by file (under a kit directory)."""
    return _proof.kit_modules_loaded(kit_dirs, KIT_MODULE_PREFIXES, sys.modules if modules is None else modules)


def env_proof(env_absent: Iterable[str], kit_dirs: Iterable[str], env_reads: Iterable[str] = (), environ=None, modules=None, path=None, meta_path=None) -> dict:
    """The clean-process proof: forbidden names absent, no kit module loaded, no kit directory on sys.path, no autoload finder armed, no
    kit sitecustomize in this process, torch not imported yet. ``ok`` is the conjunction; every list names the violations. ``reads`` records
    the names upstream reads at their values. ``kit_modules_loaded_after`` (the same scan once the stock call has returned) is added by ``run``."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    proof = _proof.env_proof(env_absent=env_absent, kit_dirs=kit_dirs, module_prefixes=KIT_MODULE_PREFIXES, environ=environ, modules=modules,
                             path=path, meta_path=meta_path)
    chained = sorted(m for m in modules if m.startswith("sitecustomize_chained") or m.startswith("sitecustomize_opendde_opt"))
    proof.update({"kit_sitecustomize_chained": chained, "reads": {k: environ.get(k) for k in env_reads},
                  "cublas_workspace_config": environ.get("CUBLAS_WORKSPACE_CONFIG"), "ok": bool(proof["ok"]) and not chained})
    return proof


def run(argv: Optional[List[str]] = None) -> int:
    import argparse
    own, stock = _proof.split_argv(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser(prog="python -s -m opendde_opt.stock_pred", description="OpenDDE stock call: the stock CLI, proven clean first")
    ap.add_argument("--proof-json", required=True, help="write the environment proof here")
    ap.add_argument("--env-absent", required=True, help="comma list of forbidden environment-name prefixes")
    ap.add_argument("--env-reads", default="", help="comma list of environment names upstream reads: recorded at their values")
    ap.add_argument("--kit-dirs", default="", help=f"{os.pathsep}-separated kit directories: none may hold a loaded module or a sys.path entry")
    a = ap.parse_args(own)
    env_absent = [s for s in a.env_absent.split(",") if s]
    env_reads = [s for s in a.env_reads.split(",") if s]
    kit_dirs = [d for d in a.kit_dirs.split(os.pathsep) if d]
    proof = env_proof(env_absent, kit_dirs, env_reads)
    proof["stock_argv"] = list(stock)
    _proof.write_proof(a.proof_json, proof)                              # before the stock call; written again after it
    if not proof["ok"]:
        print(f"{PREFIX} NOT STOCK: forbidden env {proof['forbidden_present']}, kit modules {proof['kit_modules_loaded']}, kit dirs "
              f"{proof['kit_dirs_on_path']}, autoload {proof['autoload_armed']}, kit sitecustomize {proof['kit_sitecustomize']} "
              f"{proof['kit_sitecustomize_chained']}, torch loaded {proof['torch_loaded_before_proof']}", file=sys.stderr, flush=True)
        return EXIT_NOT_STOCK
    print(f"{PREFIX} ENV-CLEAN ok: absent={','.join(env_absent)} kit_modules=none kit_dirs=none autoload=none no_user_site={proof['no_user_site']} "
          f"reads={json.dumps(proof['reads'], sort_keys=True)}", file=sys.stderr, flush=True)
    # --- the LayerNorm backend upstream binds in this clean process (the pinned stock's fused kernel; upstream's own lazy load, forced once)
    _settings.apply_stock_env()                                          # the stock base's environment when this caller runs on its own (cli.main exported it already)
    try:
        ln = _lncensus.census(strict=True)
    except _lncensus.NotLoaded as e:
        print(f"{PREFIX} NOT STOCK: {e}", file=sys.stderr, flush=True)
        return EXIT_NOT_STOCK
    # --- the triangle-kernel census: the same reader — what upstream binds and resolves in THIS process, refused by name (exit 5) when absent / fell back
    from . import modes as _modes                                        # the modes table's expectations for the stock caller (standard library + registry; nothing of the kits)
    route = _modes.route_word(None, None)
    try:
        _lncensus.arm(route, _modes.kernel_expectations(None, ln_requested=_settings.ln_requested(), knobs=_settings.stock_knobs(_settings.stated_in(stock))),   # --triatt_kernel / --trimul_kernel as stated: that site's word is the caller's; LAYERNORM_TYPE as the caller's environment has it
                      tag=PREFIX, strict=True, layernorm=ln)
    except _lncensus.KernelsRefused as e:
        proof["kernels"] = _lncensus.record()
        _proof.write_proof(a.proof_json, proof)
        print(f"{PREFIX} KERNELS refused before the stock call: {' '.join(e.problems)}", file=sys.stderr, flush=True)
        return EXIT_KERNELS
    # --- the stock call: the `opendde` console script's entry point with the arguments unchanged ------------------------------------
    import importlib
    try:
        entry = getattr(importlib.import_module(STOCK_ENTRY[0]), STOCK_ENTRY[1])
    except Exception as e:  # noqa: BLE001
        print(f"{PREFIX} the stock opendde CLI is not importable ({STOCK_ENTRY[0]}.{STOCK_ENTRY[1]}: {type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return EXIT_NOT_STOCK
    _phase.install()                                                     # per-item PHASE timing lines (timing only; wraps upstream's stage callables when runner.inference imports)
    rc = 0
    try:
        entry.main(args=list(stock), prog_name=PROG, standalone_mode=True)
    except SystemExit as e:                                              # the CLI's own exit, or the KERNELS guard's refusal at runner init (KernelsRefused = SystemExit(5))
        code = e.code
        rc = 0 if code is None else (code if isinstance(code, int) else 1)
    finally:
        rc = _lncensus.finish(rc)                                        # THE KERNELS line of this pass (final counters); 5 when the library took its reference path where its rules say served
        proof["kernels"] = _lncensus.record()
        # -------------------------------------------------------------------------------------------------------------------------
        proof["kit_modules_loaded_after"] = kit_modules_loaded(kit_dirs)     # the same scan once the stock call has returned
        try:
            import importlib.metadata as md
            proof["opendde_version"] = md.version("opendde")
        except Exception:  # noqa: BLE001
            proof["opendde_version"] = None
        _proof.write_proof(a.proof_json, proof)
        if proof["kit_modules_loaded_after"]:
            print(f"{PREFIX} KIT MODULES LOADED during the stock call: {proof['kit_modules_loaded_after']}", file=sys.stderr, flush=True)
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())

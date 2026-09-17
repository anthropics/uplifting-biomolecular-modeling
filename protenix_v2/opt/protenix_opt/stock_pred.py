"""The stock caller — mode ``off``: the stock ``protenix`` command line in a clean subprocess, proven before it runs.

``pred --mode off`` execs this module (``python -s -m protenix_opt.stock_pred --proof-json <out_dir>/stock_env_proof.json --env-absent
<prefixes> --kit-dirs <dirs> -- pred <stock arguments>``) with every environment name under stock/PINS.json
``stock_environment.must_be_absent_prefixes`` stripped and every PYTHONPATH entry inside a kit directory removed (cli.stock_command).
Before anything of protenix is imported the process proves what it is (``env_proof``): no forbidden environment name, no kit module
loaded (by name or by file), no kit directory on sys.path, the package's autoload finder not armed, the kit's sitecustomize not this
process's, torch not yet imported; the proof is written as JSON and read into ``opt_manifest.json`` by the ``pred`` process. A process
that fails its proof exits 3 without running anything. Under ``--det 1`` the caller passes the ONE named exception (``--det-env``,
``--det-path``: det.stock_exception()) and the proof allows exactly it — the forbidden names present are exactly the recipe's, with its
values; the kit directories on sys.path are exactly the recipe's one entry; the process's sitecustomize is that entry's (its PTX_DET block
is why the entry is there: it imports torch and sets deterministic algorithms, so torch loaded is expected); the kit modules loaded are
exactly ``sitecustomize``; every lever key is absent — and after the stock call ``ptx_trunk2_levers`` was never imported (no lever could
have applied). Any deviation is a named refusal. Then the stock click group (``runner.batch_inference:protenix_cli``, the
``protenix`` console script's entry point) runs with the arguments unchanged, the stock model's per-item phases timed in place
(``phase_timing``: one PHASE line per item, no value changed). This module imports the standard library, ``phase_timing`` (standard
library only) and, at run time, the stock package — nothing of the kit and nothing else of this package.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, List, Optional

from . import _core  # noqa: F401  (opt_core importable; imports nothing of the core itself)
from . import phase_timing                                              # standard library only: the per-item PHASE line on the stock model
from opt_core import stock_proof as _proof                              # the ONE core module a stock process may hold beyond opt_core itself

PREFIX = "[protenix-opt stock]"
EXIT_NOT_STOCK = 3                                                      # the package's EXIT_NOT_ACTIVE: nothing ran
KIT_MODULE_PREFIXES = (                                                  # the kits' own top-level module/package names (levers, ops
    # providers, graphs): opt_core.stock_proof._under_prefix matches a name only if it EQUALS one of these or starts with one of
    # them plus "." (a dotted-package boundary, never a bare substring -- "fpf" alone would not match "fpf_smalln") -- every entry
    # below is therefore a real, complete top-level name, not a partial prefix like the pre-dotted-boundary "ptx_"/"fpf" were.
    "infopt_graphs",                                                     # a real package: infopt_graphs and infopt_graphs.<anything>
    "fpf",                                                               # a real package: fpf and fpf.<anything> (flashpairformer/src/fpf)
    "fpf_clisampler", "fpf_cueq_pad8exact", "fpf_pad8exact", "fpf_smalln", "fpf_stackgraph", "fpf_trimul", "fpf_trimul_exact",
    "fpf_trunkgraph",
    "ptx_addon_levers", "ptx_drop_bond_mask", "ptx_fpf_v02",
    "ptx_lazy_init", "ptx_msa_adapt",
    "ptx_tp", "ptx_trunk2_levers",
)
STOCK_ENTRY = ("runner.batch_inference", "protenix_cli")                # the `protenix` console script's entry point


def kit_modules_loaded(kit_dirs: Iterable[str], modules=None) -> List[str]:
    """The loaded modules that are the kit's: by name (``KIT_MODULE_PREFIXES``) or by file (under a kit directory)."""
    return _proof.kit_modules_loaded(kit_dirs, KIT_MODULE_PREFIXES, modules)


def env_proof(env_absent: Iterable[str], kit_dirs: Iterable[str], environ=None, modules=None, path=None, meta_path=None,
              det_exception: Optional[dict] = None) -> dict:
    """The clean-process proof (opt_core.stock_proof.env_proof with this package's module prefixes): forbidden names absent, no kit module
    loaded, no kit directory on sys.path, no autoload finder armed, the kit's sitecustomize not this process's, torch not imported yet, no
    core module beyond the proof machinery. ``ok`` is the conjunction; every list names the violations. With ``det_exception``
    (``{"env": {name: value}, "pythonpath": [dir]}``) the proof allows exactly the carve-out and nothing beyond it: ``deviations`` names
    every difference. ``kit_modules_loaded_after`` (the same scan once the stock call has returned) is added by ``run``."""
    return _proof.env_proof(env_absent=env_absent, kit_dirs=kit_dirs, module_prefixes=KIT_MODULE_PREFIXES, environ=environ, modules=modules,
                            path=path, meta_path=meta_path, det_exception=det_exception)


LEVER_MODULE = "ptx_trunk2_levers"                                      # the kit's lever module: never imported by a stock call, det or not

CUEQ_CACHE_ENV = "CUEQ_TRITON_CACHE_DIR"                                 # cuequivariance_ops' cache-directory variable (its cache manager reads it first)
CUEQ_CACHE_APP = "cuequivariance-triton"                                # ... else platformdirs' user cache dir of this application name


def cueq_user_cache_census(environ=None) -> dict:
    """Where cuequivariance_ops' triton cache manager would READ tuned-tile json in THIS process's environment, and how many ``*.json`` are
    there now — resolved as the library resolves it ($CUEQ_TRITON_CACHE_DIR when set, else platformdirs' user cache dir of
    ``cuequivariance-triton``: $XDG_CACHE_HOME/<app> when XDG_CACHE_HOME is set and absolute, else $HOME/.cache/<app>), counted without
    creating or touching the directory. ``{"dir", "source": env|xdg|home, "exists": bool, "json": int}`` — evidence of the stock
    process's cache state, printed on the CUEQ-CACHE line; it changes nothing."""
    env = os.environ if environ is None else environ
    d = (env.get(CUEQ_CACHE_ENV) or "").strip()
    if d:
        source = "env"
    else:
        xdg = (env.get("XDG_CACHE_HOME") or "").strip()
        if xdg and os.path.isabs(xdg):
            d, source = os.path.join(xdg, CUEQ_CACHE_APP), "xdg"
        else:
            home = (env.get("HOME") or "").strip() or os.path.expanduser("~")
            d, source = os.path.join(home, ".cache", CUEQ_CACHE_APP), "home"
    exists = os.path.isdir(d)
    n = 0
    if exists:
        try:
            n = sum(1 for f in os.listdir(d) if f.endswith(".json"))
        except OSError:
            n = -1
    return {"dir": d, "source": source, "exists": exists, "json": n}


def cueq_cache_line(census: dict) -> str:
    """``[protenix-opt stock] CUEQ-CACHE cueq_user_cache=<dir> source=<env|xdg|home> exists=<0|1> json=<n>`` — one whitespace-free token each."""
    return (f"{PREFIX} CUEQ-CACHE cueq_user_cache={census['dir'].replace(' ', '%20')} source={census['source']} exists={int(bool(census['exists']))} "
            f"json={census['json']}")


def after_call_check(kit_dirs: Iterable[str], det_exception: Optional[dict] = None, modules=None) -> dict:
    """The scan once the stock call has returned: the kit modules loaded (``kit_modules_loaded_after``) and, under the det exception, whether
    anything beyond ``sitecustomize`` — the lever module above all — was imported (``after_ok``)."""
    r = _proof.after_call_check(kit_dirs, KIT_MODULE_PREFIXES, lever_modules=(LEVER_MODULE,), det_exception=det_exception, modules=modules)
    return {"kit_modules_loaded_after": r["kit_modules_loaded_after"], "lever_module_imported": LEVER_MODULE in r["lever_modules_imported"],
            "after_ok": r["after_ok"]}


def run(argv: Optional[List[str]] = None) -> int:
    a, stock = _proof.parse_stock_argv(list(sys.argv[1:] if argv is None else argv), prog="python -s -m protenix_opt.stock_pred")
    env_absent, kit_dirs, det_exception = a.env_absent, a.kit_dirs, a.det_exception
    proof = env_proof(env_absent, kit_dirs, det_exception=det_exception)
    proof["stock_argv"] = list(stock)

    def write_proof() -> None:
        _proof.write_proof(a.proof_json, proof)
    write_proof()                                                        # before the stock call; written again after it
    if not proof["ok"]:
        if det_exception:
            print(f"{PREFIX} NOT STOCK (det exception not met): " + "; ".join(proof["det_exception"]["deviations"]), file=sys.stderr, flush=True)
        else:
            print(f"{PREFIX} NOT STOCK: forbidden env {proof['forbidden_present']}, kit modules {proof['kit_modules_loaded']}, kit dirs "
                  f"{proof['kit_dirs_on_path']}, autoload {proof['autoload_armed']}, kit sitecustomize {proof['kit_sitecustomize']}, "
                  f"torch loaded {proof['torch_loaded_before_proof']}", file=sys.stderr, flush=True)
        return EXIT_NOT_STOCK
    if det_exception:
        print(f"{PREFIX} ENV-CLEAN ok under the det exception: {' '.join(f'{k}={v}' for k, v in det_exception['env'].items())} "
              f"PYTHONPATH={os.pathsep.join(det_exception['pythonpath'])} sitecustomize={proof['kit_sitecustomize']} kit_modules=sitecustomize "
              f"autoload=none no_user_site={proof['no_user_site']}", file=sys.stderr, flush=True)
    else:
        print(f"{PREFIX} ENV-CLEAN ok: absent={','.join(env_absent)} kit_modules=none kit_dirs=none autoload=none no_user_site={proof['no_user_site']}",
              file=sys.stderr, flush=True)
    proof["cueq_user_cache"] = cueq_user_cache_census()                   # the stock process's cuequivariance tuned-tile cache state: evidence only, nothing redirected
    print(cueq_cache_line(proof["cueq_user_cache"]), file=sys.stderr, flush=True)
    # --- the stock call: the `protenix` console script's entry point with the arguments unchanged ---------------------------------
    import importlib
    try:
        entry = getattr(importlib.import_module(STOCK_ENTRY[0]), STOCK_ENTRY[1])
    except Exception as e:  # noqa: BLE001
        print(f"{PREFIX} the stock protenix CLI is not importable ({STOCK_ENTRY[0]}.{STOCK_ENTRY[1]}: {type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return EXIT_NOT_STOCK
    rc = 0
    phase_timing.install()                                               # the stock runner is imported: its per-item phases are timed in place (no value changes)
    try:
        entry.main(args=list(stock), prog_name="protenix", standalone_mode=True)
    except SystemExit as e:
        code = e.code
        rc = 0 if code is None else (code if isinstance(code, int) else 1)
    finally:
        # -------------------------------------------------------------------------------------------------------------------------
        proof.update(after_call_check(kit_dirs, det_exception))              # the same scan once the stock call has returned
        write_proof()
        if not proof["after_ok"]:
            print(f"{PREFIX} KIT MODULES LOADED during the stock call: {proof['kit_modules_loaded_after']}"
                  + (f" (lever module {LEVER_MODULE} imported)" if proof["lever_module_imported"] else ""), file=sys.stderr, flush=True)
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())

"""The stock caller — the tree's only stock route: ``run_openfold predict`` in a process that proves it is stock before it imports openfold3.

``pred --mode off`` spawns this module in a fresh interpreter (``python -s -m openfold3_ob0_opt.stock_pred``: no user site; the
command is the shared core's, opt_core.stock_proof.stock_command) with every name under stock/PINS.json's must-be-absent prefixes stripped from the environment and no kit directory on
PYTHONPATH (env.strip); the tree (``--home``), the prefixes and the kit directories come from the parent — the child derives nothing.
Before importing openfold3 the process writes its proof to ``--proof-json`` (env.env_proof: forbidden names, kit modules, kit dirs on
sys.path, kit hooks on sys.meta_path, the autoload finder, core modules beyond the proof machinery, torch already loaded) and refuses
(exit 3, the ``NOT STOCK`` line) unless every list is empty; the proof is written again after the call with ``kit_modules_loaded_after`` /
``kit_hooks_installed_after`` / ``core_modules_loaded_after`` — the same gate: kit or core code loaded during the call sets ``ok`` false and exits 3 with the ``NOT STOCK`` line,
whatever the stock call returned — and the effective eval kernel flags, chunk-size tuner setting and trainer precision resolved through
OpenFold3's own config code (the stock configuration's record: which of upstream's Triton / cuEquivariance / DS4Sci kernels are on). The call itself is the ``run_openfold`` console script's entry point (``openfold3.run_openfold:cli``) with the arguments
unchanged, plus ``--runner-yaml <the stock configuration>`` when the caller gave none (modes.STOCK_YAML; stock/PINS.json "stock_config")
and ``--inference-ckpt-path $OPENFOLD3_OB0_CKPT`` when the caller gave neither a path nor ``--inference-ckpt-name`` (upstream 0.5.0 resolves a
name through ``$OPENFOLD_CACHE/ckpt_root`` and writes that file when it is absent — entry_points/parameters.py get_default_checkpoint_dir: a
frozen-weights run names its checkpoint, so a call with neither and no ``OPENFOLD3_OB0_CKPT`` is refused by name, exit 3). The OpenFold cache
directory in force and a user default ``runner.yml`` in it (which upstream deep-merges under the runner yaml: run_openfold.py:209-228) are
recorded in the proof and named on stderr (``note: upstream merges the user runner.yml …``); the run proceeds exactly as upstream runs it.

``--det 1`` (det.py): the parent exports the recipe's variables and puts the det site first on this interpreter's PYTHONPATH
(``--det-env`` / ``--det-path``: opt_core.stock_proof's det exception); the site's sitecustomize applied the recipe's statements at start-up and
the proof checks exactly that carve-out — those variables with those values, that one directory, that sitecustomize, no finder.

``--upstream-fix <ID>[,<ID>…]`` (opt_core.upstream_fix through the package's upstream_fix_registry; the parent's ``pred --mode off --upstream-fix …``): after the proof and before the call
this process loads each named issue file from ``<tree>/upstream_issues/`` by path and installs its fix (``[openfold3_ob0-opt] UPSTREAM-FIX
<ID> applied …``; the records land in the proof under ``upstream_fix``). The issue files live outside every add-on directory and install no
finder, so the after-call gate reads the arm as it is: stock plus the named fixes. A fix that cannot be installed refuses the run
(``UPSTREAM-FIX NOT APPLIED``, exit 3). Without the flag nothing of this runs.

Observability inside the stock call: before the call the process wraps ``OpenFold3.forward`` with the package's per-item forward-time line
(``report.install_forward_timer("stock")``: ``[openfold3_ob0-opt] Model forward time: <s>s item=<query_id> seed=<seed> tokens=<N> route=stock``
on stderr, the synchronised wall time of the model call alone; no finder, no lever, no numeric touched) and records it in the proof
(``forward_timer``: the wrapped target, installed or the reason not).

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .report import PREFIX, exit_code

EXIT_NOT_STOCK = 3
STOCK_ENTRY = ("openfold3.run_openfold", "cli")             # the console script: stock/src/pyproject.toml / wheel entry_points.txt


def _has(args: List[str], *names: str) -> bool:
    return any(a == n or a.startswith(n + "=") for a in args for n in names)


def _interp_words(proof: dict) -> str:
    """The interpreter facts of the ENV-CLEAN line: sys.flags, every PYTHONPATH entry, every sys.meta_path finder by class name (a stock
    interpreter's own three; anything a caller attached is named here), the count of modules loaded at proof time (the full census is in the proof file)."""
    it = proof.get("interpreter") or {}
    flags = ",".join(f"{k}={v}" for k, v in (it.get("sys_flags") or {}).items())
    pp = it.get("pythonpath_entries") or []
    mp = it.get("meta_path") or []
    return (f"sys.flags={flags} PYTHONPATH_entries={os.pathsep.join(pp) if pp else 'none'} meta_path={','.join(mp) if mp else 'none'} "
            f"modules_at_proof={len(it.get('module_census') or [])}")


def effective_eval_kernel_flags(runner_yaml: str, ckpt: Optional[str]) -> dict:
    """The EFFECTIVE eval kernel flags with the runner yaml, resolved through OpenFold3's own config path (what predict builds)."""
    from openfold3.core.config import config_utils
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    args = config_utils.load_yaml(runner_yaml)
    if ckpt:
        args["inference_ckpt_path"] = ckpt
    cfg = InferenceExperimentConfig(**args)
    mc = OF3ProjectEntry().get_model_config_with_update(cfg.model_update)
    ev = mc.settings.memory.eval
    flags = {k: bool(ev[k]) for k in ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma", "tune_chunk_size")}
    flags["chunk_size"] = ev["chunk_size"] if ev["chunk_size"] is None else int(ev["chunk_size"])
    off = ev["offload_inference"]
    flags["offload_inference"] = {k: (bool(off[k]) if k != "token_cutoff" else int(off[k])) for k in ("msa_module", "template_module", "confidence_heads", "token_cutoff") if k in off}
    return flags


LIGHTNING_DEFAULT_PRECISION = "32-true"          # lightning.Trainer's own default when the runner yaml carries no pl_trainer_args.precision (upstream's predict entry point: fp32)


def effective_precision(runner_yaml: str) -> str:
    """The EFFECTIVE Lightning precision of the stock call: the runner yaml's ``pl_trainer_args.precision`` as given, else Lightning's default
    (``32-true`` — upstream's shipped predict configuration sets none)."""
    import yaml
    with open(runner_yaml, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    p = (doc.get("pl_trainer_args") or {}).get("precision")
    return str(p) if p is not None else f"{LIGHTNING_DEFAULT_PRECISION} (Lightning default: the runner yaml sets no pl_trainer_args.precision)"


def run(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    own, stock = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser(prog="openfold3_ob0_opt.stock_pred", description="stock run_openfold predict in a proven-clean process")
    ap.add_argument("--proof-json", required=True)
    ap.add_argument("--env-absent", default="", help="comma-separated name prefixes that must be absent (stock/PINS.json)")
    ap.add_argument("--kit-dirs", default="", help=f"{os.pathsep}-separated kit directories: none may hold a loaded module or a sys.path entry")
    ap.add_argument("--home", required=True, help="the openfold3_ob0/ tree, from the parent (the child derives nothing: no core module beyond the proof machinery loads here)")
    ap.add_argument("--det-env", default="", help="comma-separated NAME=VALUE the deterministic recipe exports (the stock proof's det exception: exactly these, with these values)")
    ap.add_argument("--det-path", default="", help=f"{os.pathsep}-separated directories the recipe puts on PYTHONPATH (the det site)")
    ap.add_argument("--upstream-fix", default="", help="comma-separated IDs of upstream-issue fixes to install in this process before the stock call (upstream_fix.py); default none")
    a = ap.parse_args(own)
    from . import env as _env
    from . import modes
    from opt_core import stock_proof as _proof
    home = a.home
    env_absent = [s for s in a.env_absent.split(",") if s]
    kit_dirs = [d for d in a.kit_dirs.split(os.pathsep) if d]
    det_exception = None                                                 # the core's det exception shape (opt_core.stock_proof.parse_stock_argv): None without --det-env/--det-path
    if a.det_env or a.det_path:
        det_exception = {"env": dict(kv.split("=", 1) for kv in a.det_env.split(",") if kv), "pythonpath": [d for d in a.det_path.split(os.pathsep) if d]}
    proof = _env.env_proof(env_absent, kit_dirs, home=home, det_exception=det_exception)
    proof["stock_argv"] = list(stock)
    proof["det"] = 1 if det_exception else 0
    fix_ids = [s for s in a.upstream_fix.split(",") if s.strip()]
    proof["upstream_fix"] = []                                           # the records of the upstream-issue fixes installed in this process (filled right before the stock call)
    os.makedirs(os.path.dirname(os.path.abspath(a.proof_json)), exist_ok=True)

    def write_proof() -> None:
        with open(a.proof_json, "w", encoding="utf-8") as fh:
            json.dump(proof, fh, indent=1, default=str)
            fh.write("\n")
    write_proof()                                                        # before the stock call; written again after it
    cache = proof["openfold_cache"]
    if not proof["ok"]:
        print(f"{PREFIX} NOT STOCK: {_proof.violations_sentence(proof)}; kit hooks {proof['kit_hooks_installed'] or 'none'}; core modules beyond "
              f"{proof['core_modules_allowed']}: {proof['core_modules_loaded'] or 'none'}; openfold3 loaded before the proof {proof['openfold3_loaded_before_proof']}", file=sys.stderr, flush=True)
        return EXIT_NOT_STOCK
    if det_exception:
        print(f"{PREFIX} ENV-CLEAN ok under the det exception: {' '.join(f'{k}={v}' for k, v in det_exception['env'].items())} "
              f"PYTHONPATH={os.pathsep.join(det_exception['pythonpath'])} sitecustomize={proof['kit_sitecustomize']} torch_loaded_by_sitecustomize="
              f"{proof['det_exception']['torch_loaded_by_sitecustomize']} kit_hooks=none autoload=none no_user_site={proof['no_user_site']} "
              f"openfold_cache={cache['dir']} upstream_reads={json.dumps(proof['upstream_reads_present'])} {_interp_words(proof)}", file=sys.stderr, flush=True)
    else:
        print(f"{PREFIX} ENV-CLEAN ok: absent={','.join(env_absent)} kit_modules=none kit_dirs=none kit_hooks=none autoload=none no_user_site={proof['no_user_site']} "
              f"openfold_cache={cache['dir']} upstream_reads={json.dumps(proof['upstream_reads_present'])} {_interp_words(proof)}", file=sys.stderr, flush=True)
    if cache["user_runner_yml"]:                                         # upstream deep-merges it under the runner yaml (run_openfold.py:209-228): named, the run proceeds as upstream runs it
        print(f"{PREFIX} note: upstream merges the user runner.yml from {cache['user_runner_yml']} under the runner yaml (OPENFOLD_CACHE "
              f"{'set by the caller' if cache['from_env'] else 'unset: ~/.openfold3'})", file=sys.stderr, flush=True)
    # --- the stock configuration and the checkpoint: the caller's own values win, the tree's are the defaults ------------------------
    if not _has(stock, "--runner-yaml", "--runner_yaml"):
        stock += ["--runner-yaml", os.path.join(home, modes.STOCK_DET_YAML if det_exception else modes.STOCK_YAML)]
    if not _has(stock, "--inference-ckpt-path", "--inference_ckpt_path", "--inference-ckpt-name", "--inference_ckpt_name"):
        if not os.environ.get("OPENFOLD3_OB0_CKPT"):                        # upstream would resolve <OPENFOLD_CACHE>/ckpt_root and WRITE it when absent (entry_points/parameters.py
            proof["not_stock_after"] = "no checkpoint"                       # get_default_checkpoint_dir): a frozen-weights run names its checkpoint — refused by name
            write_proof()
            print(f"{PREFIX} NOT STOCK: no checkpoint on the command (--inference-ckpt-path / --inference-ckpt-name) and OPENFOLD3_OB0_CKPT unset — upstream's "
                  f"default-checkpoint resolution would read and write $OPENFOLD_CACHE/ckpt_root (stock/src/openfold3/entry_points/parameters.py get_default_checkpoint_dir); "
                  f"pass --inference-ckpt-path or set OPENFOLD3_OB0_CKPT", file=sys.stderr, flush=True)
            return EXIT_NOT_STOCK
        stock += ["--inference-ckpt-path", os.environ["OPENFOLD3_OB0_CKPT"]]
    yml = next((stock[i + 1] for i, x in enumerate(stock) if x in ("--runner-yaml", "--runner_yaml") and i + 1 < len(stock)), None)
    ckpt = next((stock[i + 1] for i, x in enumerate(stock) if x in ("--inference-ckpt-path", "--inference_ckpt_path") and i + 1 < len(stock)), None)
    from . import manifest as _manifest
    proof["runner_yaml"] = yml
    proof["runner_yaml_sha256"] = _manifest.sha256_file(yml, home=home) if yml else None
    proof["checkpoint"] = _manifest.checkpoint_info(ckpt, hash_it=False, home=home)
    print(f"{PREFIX} stock: run_openfold predict {' '.join(stock)} (runner yaml {(proof['runner_yaml_sha256'] or '-')[:8]}, det={proof['det']})", file=sys.stderr, flush=True)
    # --- the stock call: the `run_openfold` console script's entry point with the arguments unchanged --------------------------------
    import importlib
    try:
        entry = getattr(importlib.import_module(STOCK_ENTRY[0]), STOCK_ENTRY[1])
    except Exception as e:  # noqa: BLE001
        print(f"{PREFIX} the stock run_openfold CLI is not importable ({STOCK_ENTRY[0]}.{STOCK_ENTRY[1]}: {type(e).__name__}: {e})", file=sys.stderr, flush=True)
        proof["import_error"] = f"{type(e).__name__}: {e}"
        write_proof()
        return EXIT_NOT_STOCK
    try:
        if yml:
            proof["effective_precision"] = effective_precision(yml)
            proof["effective_eval_kernel_flags"] = effective_eval_kernel_flags(yml, ckpt)
            proof["use_tf32"] = next((stock[i + 1] for i, x in enumerate(stock) if x == "--use_tf32" and i + 1 < len(stock)), "true (upstream default: torch.set_float32_matmul_precision('high'))")
            print(f"{PREFIX} stock config: effective eval kernel flags {json.dumps(proof['effective_eval_kernel_flags'])} precision {proof['effective_precision']} use_tf32 {proof['use_tf32']}", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        proof["effective_eval_kernel_flags"] = f"unresolved: {type(e).__name__}: {e}"
    flags = proof.get("effective_eval_kernel_flags")
    if isinstance(flags, dict) and flags.get("use_deepspeed_evo_attention"):      # the configuration turns the DS4Sci evoformer attention on: the op must LOAD here, before the run — never a per-item load error or an eager path timed as stock
        from .stack import DS4SCI_NOT_LOADED, ds4sci_load
        ok, detail = ds4sci_load()
        proof["ds4sci_load"] = {"ok": ok, "detail": detail}
        if not ok:
            print(f"{PREFIX} NOT STOCK: {DS4SCI_NOT_LOADED}: {detail} (the runner yaml sets use_deepspeed_evo_attention true; run on the stack with the pre-built "
                  f"evoformer_attn op, or under the det recipe --det 1 where that kernel is off)", file=sys.stderr, flush=True)
            write_proof()
            return EXIT_NOT_STOCK
        print(f"{PREFIX} DS4SCI evoformer_attn op loaded: {detail}", file=sys.stderr, flush=True)
    from .report import install_forward_timer
    proof["forward_timer"] = install_forward_timer("stock")             # the per-item `Model forward time: … route=stock` line: a wrap on OpenFold3.forward, no finder, numerically inert
    if fix_ids:                                                          # --upstream-fix: the requested upstream-issue fixes, installed HERE — after the proof, before the stock call —
        from opt_core import upstream_fix as _upstream_fix                # by the one process that runs the model (UPSTREAM-FIX <ID> applied); a failure refuses the run by name.
        from . import upstream_fix_registry                              # The one core module this adds to the process is named in the after-call scan (CORE_ALLOWED_WITH_UPSTREAM_FIX).
        try:
            proof["upstream_fix"] = upstream_fix_registry(home).apply(fix_ids, log=lambda words: print(f"{PREFIX} {words}", file=sys.stderr, flush=True))
        except Exception as e:  # noqa: BLE001
            proof["upstream_fix_error"] = f"{type(e).__name__}: {e}"
            write_proof()
            print(f"{PREFIX} {_upstream_fix.NOT_APPLIED} (stock --upstream-fix {','.join(fix_ids)}): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return EXIT_NOT_STOCK
    write_proof()
    rc = 0
    try:
        entry.main(args=["predict"] + list(stock), prog_name="run_openfold", standalone_mode=True)
    except SystemExit as e:
        rc = exit_code(e)
    finally:
        proof.update(_env.after_call(kit_dirs, det_exception, upstream_fix=bool(fix_ids)))   # the same scan once the stock call has returned
        out_dir = next((stock[i + 1] for i, x in enumerate(stock) if x in ("--output-dir", "--output_dir") and i + 1 < len(stock)), None)
        proof["n_cif"] = _manifest.count_structures(out_dir)
        from . import templ_census as _templ_census
        proof["templates_featurised"] = _templ_census.featurised()          # per item: the featurised template slots real / allocated (the parent judges: templ_census.judge)
        proof["exit_code"] = rc
        proof["ok_after"] = bool(proof["after_ok"])
        if not proof["ok_after"]:                                              # the same gate as before the call: the arm was not stock
            proof["ok"] = False
            proof["not_stock_after"] = f"kit code loaded during the stock call: modules {proof['kit_modules_loaded_after']} hooks {proof['kit_hooks_installed_after']} core {proof['core_modules_loaded_after']}"
        write_proof()
        if not proof["ok_after"]:
            print(f"{PREFIX} NOT STOCK: {proof['not_stock_after']}", file=sys.stderr, flush=True)
    if not proof["ok_after"]:
        return EXIT_NOT_STOCK
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())

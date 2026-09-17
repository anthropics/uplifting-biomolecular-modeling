"""The stock caller — mode ``off``: upstream as a user runs it, nothing from the kit on the path.

`design --mode off` execs this file in a clean subprocess (``python -s -m pxdesign_opt.stock_infer --tree … --tasks … --out_dir …
--det … [knobs] [--seeds …]``, built by ``cli.stock_command``; every variable of the kit stripped from the environment,
``opt_core.stock_proof.strip_env``); before importing torch the process proves its environment (``env_proof`` = the shared core's proof,
``opt_core.stock_proof.env_proof``, plus this engine's two rules; every input of the proof is derived from ``--tree`` and ``--det`` — the
one list ``stock/PINS.json`` ``stock_environment.must_be_absent_prefixes``, the kit directories ``opt/forward`` and ``opt/serving``, the
carried lever kit's module prefixes, the recipe's det exception): no variable with a forbidden prefix (``CUBLAS_WORKSPACE_CONFIG`` is allowed
only as the deterministic recipe's own variable — under ``--det 1`` it must hold the recipe's value, under ``--det 0`` its presence is a
violation), no kit module loaded and no core module beyond the proof machinery, no kit directory on ``sys.path``, no autoload finder
armed, torch not yet imported. A violation exits 3 (NOT STOCK) with the proof written first. ``LAYERNORM_TYPE`` is stock's own variable
(Protenix reads it at import, ``protenix/openfold_local/model/primitives.py:49-51``, before upstream's ``configure_runtime_env`` sets it,
``pxdesign/utils/infer.py:492-503``): it reaches this process exactly as the caller's environment holds it — ``configs/<gpu>.env`` exports
``fast_layernorm``; absent or any other value selects OpenFold's LayerNorm, as it does for a user of the bare console script — and is
RECORDED (``layernorm_env``; the KERNELS line's ``layernorm=fused|plain``), never required.

The route follows ``--det``, recorded in ``stock_env_proof.json`` as ``route``:
* ``--det 0`` → route ``cli``: the upstream console script ``pxdesign infer -i <tasks> -o <out> <knobs> [--seeds <s>] --load_checkpoint_dir <dir>``
  — stock, literally, as a child of this proven process (same environment);
* ``--det 1`` → route ``inprocess``: ``infer_loop.run`` — upstream's ``main()`` replayed with ``seed_everything(seed, deterministic=True)``
  (the console script hard-codes ``deterministic=False``, ``runner/inference.py:236``, so the recipe cannot reach it through the script).
The task file is upstream's own (JSON task list or YAML, ``utils/inputs.py:156-181``); ``--seeds`` absent = upstream's one clock-derived seed on
either route; a dump directory that already holds results is upstream's to handle.
This is the only stock caller in the tree (``run.sh design --mode off`` and ``design --mode off`` both reach it). Outputs are the
upstream dump layout, untouched; the proof JSON (``det``, ``route``, ``layernorm_class`` / ``layernorm_kind`` / ``layernorm_source`` —
the built model's first LayerNorm class on the in-process route, the class the proven environment selects on the console-script route —
``layernorm_env``, environment, counts, exit code, the batch / LayerNorm / KERNELS records under ``stamps`` and ``kit_modules_loaded_after`` — the
core's after-call check) is written beside them, listed never judged. One evidence line prints on both routes with this caller's prefix: KERNELS (stamps.py) — on the in-process route once the runner is built,
on the console-script route before the exec, from this proven interpreter and environment (the child's own). The child's streams are its
own: nothing is piped or parsed. The exit status is upstream's; outputs short of tasks x seeds x N_sample print
``INCOMPLETE …`` instead of ``DONE …`` and exit 1 when upstream itself exited 0. The core pin gate is statement one of this process
entry too (``pxdesign_opt.core_gate``: it imports nothing of the core, so the proof's core-module rule is unaffected): an absent or
mismatched ``opt_core`` is one ``[pxdesign-opt] NOT ACTIVE: reason=…`` line and exit status 3, never a traceback from the import below.
"""
from . import TAG, core_gate

core_gate()                                              # [pxdesign-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …  -> exit 3

import argparse  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Optional  # noqa: E402

from opt_core import stock_proof  # noqa: E402

from . import det as _det  # noqa: E402
from . import infer_loop, options, outputs, stamps  # noqa: E402

PREFIX = f"[{TAG} stock]"                              # the stock caller's own lines; TAG is the package's one spelling (import-free)
PROG = "python -s -m pxdesign_opt.stock_infer"
KIT_MODULE_PREFIXES = ("pxd_xattempt",)     # = registry.KIT_MODULE_PREFIXES (registry is not imported in a stock process; the tests lock the two copies)
LEVER_MODULES = ("pxd_xattempt.hoist", "pxdesign_opt.stack")   # named in the after-call check: the lever and the package's activation module
ROUTES = {0: "cli", 1: "inprocess"}                     # the route follows --det: the console script | upstream's main() replayed with the recipe's seeding (infer_loop.run)
EXIT_NOT_STOCK = 3                                      # the environment proof failed, or a kit module was loaded by the call: = the package's NOT ACTIVE code (report.EXIT_NOT_ACTIVE)
EXIT_INCOMPLETE = 1                                     # upstream exited 0 but the outputs fall short of tasks x seeds x N_sample (the INCOMPLETE line): = report.EXIT_FAIL
CONSOLE_SCRIPT, CONSOLE_ENTRY = "pxdesign", "pxdesign.runner.cli:cli"    # upstream's console script and its entry point (stock/src/PXDesign/setup.py:58-60)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True, help="the pxdesign/ tree (stock/PINS.json is read from it)")
    ap.add_argument("--tasks", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_dir", default=None, help=f"default: ${infer_loop.CKPT_ENV}")
    options.add_arguments(ap)                                                    # upstream's knobs, their names and defaults, + --seeds
    ap.add_argument("--det", type=int, choices=_det.LEVELS, default=0, help="0: the console script (route cli) | 1: the deterministic recipe, in-process (route inprocess)")
    ap.add_argument("--proof-json", default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="further pxdesign infer arguments, verbatim")
    return ap


def parse_args(argv=None):
    return _parser().parse_args(argv)


def kit_dirs(tree: str) -> list:
    """The directories a stock process must not have on its path or hold modules from: the carried lever kit under opt/ (= stack.kit_dirs())."""
    return [os.path.join(tree, "opt", "forward"), os.path.join(tree, "opt", "serving")]


def proof_inputs(tree: str, pins: dict, det: int) -> dict:
    """Every input of the core's proof, derived from the tree and --det: the one list (PINS), the kit directories, the module prefixes,
    the recipe's det exception (det.RECIPE_ENV under --det 1, None under --det 0)."""
    se = pins["stock_environment"]
    return {"env_absent": list(se["must_be_absent_prefixes"]), "kit_dirs": kit_dirs(tree), "module_prefixes": list(KIT_MODULE_PREFIXES),
            "det_exception": ({"env": dict(_det.RECIPE_ENV), "pythonpath": []} if det else None)}


def env_proof(tree: str, pins: dict, det: int, proof_path: Optional[str] = None) -> dict:
    """Prove the clean environment before torch is imported (the core's proof + this engine's rules); raises SystemExit(3) on any
    violation (the proof with its violations is written to `proof_path` first when given). ``LAYERNORM_TYPE`` is recorded as found
    (``layernorm_env``: the value, or ``absent``), never required: it is stock's own variable and reaches the child as the caller has it."""
    inputs = proof_inputs(tree, pins, det)
    ln_found = os.environ.get(options.LAYERNORM_ENV)
    p = stock_proof.env_proof(**inputs)
    allowed = set(pins["stock_environment"].get("allowed_exceptions", []))
    recipe_var = list(_det.RECIPE_ENV)[0]
    env_bad = [k for k in p["forbidden_present"] if k not in allowed]
    if not det and recipe_var in os.environ:
        env_bad.append(f"{recipe_var} (set, but --det 0)")
    violations = {"env": env_bad, "modules": list(p["kit_modules_loaded"]) + list(p["core_modules_loaded"]), "sys_path": list(p["kit_dirs_on_path"]),
                  "autoload_finder": bool(p["autoload_armed"]), "torch_loaded": bool(p["torch_loaded_before_proof"]), "kit_sitecustomize": p["kit_sitecustomize"],
                  "det_exception": list((p.get("det_exception") or {}).get("deviations") or [])}
    proof = dict(p, forbidden_prefixes=inputs["env_absent"], allowed_exceptions=sorted(allowed), violations=violations, pxdesign_opt_env=os.environ.get("PXDESIGN_OPT"),
                 layernorm_env=ln_found if ln_found not in (None, "") else stamps.ABSENT,
                 env={k: os.environ.get(k) for k in ("LAYERNORM_TYPE", "PROTENIX_DATA_ROOT_DIR", "TORCH_EXTENSIONS_DIR", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", infer_loop.CKPT_ENV,
                                                                  "PYTORCH_CUDA_ALLOC_CONF", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")})     # the environment the child inherits, key by key (None = absent); torch's TF32 variables pass through as the caller has them — recorded here, reported by the KERNELS line
    proof["clean"] = not (env_bad or violations["modules"] or violations["sys_path"] or violations["autoload_finder"] or violations["torch_loaded"]
                          or violations["kit_sitecustomize"] or violations["det_exception"])
    if not proof["clean"]:
        print(f"{PREFIX} environment not clean: { {k: v for k, v in violations.items() if v} }", file=sys.stderr, flush=True)
        if proof_path:
            stock_proof.write_proof(proof_path, dict(proof, det=int(det), exit_code=EXIT_NOT_STOCK))
        raise SystemExit(EXIT_NOT_STOCK)
    log(f"ENV-CLEAN {stock_proof.clean_sentence(p)} det={det}")
    return proof


def log(msg: str) -> None:
    """One stock-caller line on stderr, on a fresh line: ``[pxdesign-opt stock] <msg>`` (stamps.emit)."""
    stamps.emit(f"{PREFIX} {msg}")


def cli_child_command(argv, *, exe: Optional[str] = None) -> list:
    """The console-script route's child argv (``--det 0``): ``pxdesign infer <argv>`` — upstream's console script as installed, nothing added."""
    return [exe or CONSOLE_SCRIPT, "infer", *argv]


def cli_child_env(environ=None) -> dict:
    """The child's environment: this (proven) process's own."""
    return dict(os.environ if environ is None else environ)


def chunk_in_effect(extra, pins: dict):
    """``infer_setting.sample_diffusion_chunk_size`` as the child will resolve it: ``--sample_diffusion_chunk_size`` among the extra arguments, else
    upstream's default as pinned (``stock/PINS.json`` ``cli_defaults``, ``configs/configs_base.py:60-62``)."""
    v = stamps.option_value(extra, "sample_diffusion_chunk_size")
    return v if v is not None else pins["cli_defaults"].get("sample_diffusion_chunk_size")


def run(a) -> int:
    t0 = time.perf_counter()
    tree = os.path.abspath(a.tree)
    pins = options.read_pins(os.path.join(tree, "stock", "PINS.json"))
    route = ROUTES[int(a.det)]
    pj = a.proof_json or os.path.join(a.out_dir, outputs.STOCK_PROOF_NAME)
    proof = env_proof(tree, pins, a.det, proof_path=pj)                                    # before torch
    proof.update(det=int(a.det), route=route)
    ckpt_dir = a.ckpt_dir or os.environ.get(infer_loop.CKPT_ENV)
    proof["preflight"] = infer_loop.preflight(pins, ckpt_dir, a.out_dir)                    # the deployment gate: the weights directory and the CCD cache hold what upstream requires
    st = options.from_args(a, pins=pins)
    argv = infer_loop.upstream_argv(a.tasks, a.out_dir, ckpt_dir, st.argv(), st.seeds_arg(), a.extra)
    n_seeds = len(st.seeds) or 1                                                            # no --seeds: upstream derives one seed from the clock
    proof.update(options=st.as_dict(), upstream_argv=argv, python=sys.version.split()[0])
    os.makedirs(a.out_dir, exist_ok=True)
    rc = 0
    sm = proof.setdefault("stamps", {})                                                     # the records filed under `stamps` (stamps.py): batch, layernorm, kernels
    try:
        if route == "cli":
            exe = shutil.which(CONSOLE_SCRIPT)
            if not exe:
                print(f"{PREFIX} the upstream console script `{CONSOLE_SCRIPT}` is not on PATH (setup.py entry point {CONSOLE_ENTRY})", file=sys.stderr)
                return 2
            cmd = cli_child_command(argv, exe=exe)
            proof["command"] = cmd
            n_tasks, n_sample = outputs.n_tasks_of(a.tasks), int(st.values["N_sample"])
            chunk = chunk_in_effect(a.extra, pins)
            ln_kind, ln_class, ln_env = stamps.layernorm_of_env(os.environ)                  # the proven environment decides the class at the child's import (primitives.py:49-51): the KERNELS line's layernorm= word
            kernels = stamps.kernels_facts(layernorm_kind=ln_kind, ds4sci=stamps.truthy(stamps.option_value(a.extra, "use_deepspeed_evo_attention") or pins["cli_defaults"].get("use_deepspeed_evo_attention", False)),
                                           source="caller_process")                         # this interpreter and this (proven) environment are the child's
            log(stamps.kernels(kernels))
            sm.update(batch=stamps.batch_facts(n_sample, chunk, n_tasks, n_seeds),
                      layernorm={"kind": ln_kind, "class": ln_class, "source": "env", "env": ln_env}, kernels=kernels)
            log(f"exec {' '.join(cmd)}")
            rc = subprocess.call(cmd, env=cli_child_env())                                 # the child's argv, cwd and environment are this proven process's; its streams are its own
            proof["seeding"] = "upstream: seed_everything(seed, deterministic=False) (runner/inference.py:236)"
        else:
            _det.apply_env(a.det)
            res = infer_loop.run(argv, list(st.seeds), a.det, log=log)
            sm.update(infer_loop.split_stamps(res))
            proof["run"] = res
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 1
        raise
    except BaseException as e:  # noqa: BLE001
        rc = 1; proof["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        ln = sm.get("layernorm") or dict(zip(("kind", "class", "env"), stamps.layernorm_of_env(os.environ)), source="env")   # the built model's class (in-process) | the class the proven environment selects
        proof.update(layernorm_class=ln["class"], layernorm_kind=ln["kind"], layernorm_source=ln["source"])
        proof.update(stock_proof.after_call_check(kit_dirs(tree), KIT_MODULE_PREFIXES, lever_modules=LEVER_MODULES))
        proof.update(exit_code=rc, wall_s=round(time.perf_counter() - t0, 2), outputs=outputs.summarize(a.out_dir, a.tasks, (proof.get("run") or {}).get("seeds") or list(st.seeds), int(st.values["N_sample"])))   # in-process: the seeds run (derived included); console script: the given ones, else unknown (scope dir)
        stock_proof.write_proof(pj, proof)
        o = proof["outputs"]
        log(f"{'DONE' if rc == 0 and o['complete'] else 'INCOMPLETE'} route={route} det={a.det} designs={o['n_designs']}/{o['expected']} rc={rc} out_dir={a.out_dir} "
            f"kit_modules_after={','.join(proof.get('kit_modules_loaded_after') or []) or 'none'} scope={o.get('scope')}")
    if proof.get("kit_modules_loaded_after"):
        return rc or EXIT_NOT_STOCK
    return rc if proof["outputs"]["complete"] else (rc or EXIT_INCOMPLETE)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

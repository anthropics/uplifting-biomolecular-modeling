"""The in-process upstream route — the one implementation both the stock caller (mode off, `--det 1`) and the kit modes use.

It replays `pxdesign.runner.inference.main()` (`runner/inference.py:207-237`) step for step with upstream's own objects
(`get_configs`, `process_input_file`, `download_inference_cache`, `save_config`, `convert_to_bioassembly_dict`, `InferenceRunner`,
`_inference`), the console-script argument list turned into `main()`'s argparse form by upstream's own `build_argv`
(`to_inference_argv`; `runner/cli.py:56-64,193-223`), and one difference: the seeding call is `seed_everything(seed, deterministic=<det>)` (det.py) instead of the
hard-coded `deterministic=False`. With `--det 0` the call is upstream's own. The seed list is upstream's: `--seeds` when given, else one
seed derived from the clock exactly as `main()` does (`derive_seed(time.time_ns())`, `inference.py:233`). A YAML or JSON task file, the
checkpoint directory, a dump directory that already holds results — all behave as upstream's `main()` does, because it is upstream's code
that runs.
One evidence line rides this route (stamps.py; the caller's `log` gives it its prefix): KERNELS, once the runner is built — the stack's
versions and switches and the BUILT model's LayerNorm kind. Its record, the batch facts and the LayerNorm record come back in the result under
`STAMP_KEYS` (`split_stamps`) for the manifest; no per-item line is printed (the DONE line and the manifest carry the job's timings). The
per-task records in the log are upstream's own, unchanged on every route: `[Rank 0 (<i>/<n>)] <task>: N_asym …` before the model call and
`[Rank 0] <task> succeeded. Saved to <dir>` after the dump (`runner/inference.py:160-189`).

Under a kit mode the caller has activated the package before `InferenceRunner(configs)` is built; the lever is installed by the hook
on `load_checkpoint` (stack.py), so this file never touches the kit. Its one use of the activation module — the verb gate,
imports `pxdesign_opt.stack` inside `run`, never at module import: the stock caller (stock_infer.py) imports this
module before its environment proof, and that proof lists `pxdesign_opt.stack` among the modules that must be absent
(`test_stock_caller_process_is_clean`). Preflight (`preflight`) is the deployment gate, before anything is loaded: the weights
directory (`PXDESIGN_CKPT_DIR` / `--ckpt_dir`) must hold the checkpoint and the three files upstream requires beside it, and the CCD cache
directory (`PROTENIX_DATA_ROOT_DIR`) its two files — the tree never downloads weights (README.md, Weights and Variables). Nothing is digested:
the checkpoint is cited once, by name and sha256, in STOCK.md.
"""
import json
import os
import time
from typing import List, Optional

from . import TAG
from . import det as _det

CKPT_ENV, CCD_ENV = "PXDESIGN_CKPT_DIR", "PROTENIX_DATA_ROOT_DIR"




def preflight(pins: dict, ckpt_dir: Optional[str], out_dir: Optional[str] = None) -> dict:
    """The deployment gate: refuse (SystemExit, named) when the weights directory or the CCD cache lacks a file upstream requires. Returns the
    record {checkpoint, required_in_dir, ccd_root, ccd_files} for opt_manifest.json. ``out_dir`` is accepted for the callers' symmetry and not
    inspected: a dump directory that already holds results is upstream's to handle (it skips a completed (task, seed))."""
    ckpt_dir = ckpt_dir or os.environ.get(CKPT_ENV)
    if not ckpt_dir:
        raise SystemExit(f"[{TAG}] {CKPT_ENV} is not set (the directory holding pxdesign_v0.1.0.pt and its three companion checkpoints; README.md, Weights and Variables)")
    w = pins["weights"]["checkpoint"]
    ckpt = os.path.join(ckpt_dir, w["file"])
    if not os.path.isfile(ckpt):
        raise SystemExit(f"[{TAG}] checkpoint missing: {ckpt} (never downloaded by the tree)")
    rec = {"checkpoint": ckpt, "required_in_dir": {}, "ccd_root": os.environ.get(CCD_ENV), "ccd_files": {}}
    for f in (pins["weights"].get("required_in_dir") or {}).get("files", {}):        # upstream downloads any of these when absent: refuse first
        p = os.path.join(ckpt_dir, f)
        rec["required_in_dir"][f] = os.path.isfile(p)
        if not rec["required_in_dir"][f]:
            raise SystemExit(f"[{TAG}] checkpoint directory file missing: {p} (upstream requires it beside {w['file']}; never downloaded by the tree)")
    ccd = os.environ.get(CCD_ENV)
    if not ccd:
        raise SystemExit(f"[{TAG}] {CCD_ENV} is not set (the CCD cache directory; README.md, Weights and Variables)")
    for f in pins["ccd_cache"]["files"]:
        p = os.path.join(ccd, f)
        rec["ccd_files"][f] = os.path.isfile(p)
        if not rec["ccd_files"][f]:
            raise SystemExit(f"[{TAG}] CCD cache file missing: {p} (never downloaded by the tree)")
    return rec


def upstream_argv(tasks: str, out_dir: str, ckpt_dir: str, option_argv: List[str], seeds: Optional[str], extra: Optional[List[str]] = None) -> List[str]:
    """The `pxdesign infer` argument list (console-script form): -i/-o, the knobs, the seeds when given (absent = upstream derives one from
    the clock), the checkpoint directory."""
    return ["-i", tasks, "-o", out_dir, *option_argv, *(["--seeds", seeds] if seeds else []), "--load_checkpoint_dir", ckpt_dir, *(extra or [])]


COMMON_KEYS = {"-i": "input_json_path", "--input": "input_json_path", "-o": "dump_dir", "--dump_dir": "dump_dir", "--dtype": "dtype", "--N_sample": "N_sample",
               "--N_step": "N_step", "--eta_type": "eta_type", "--eta_min": "eta_min", "--eta_max": "eta_max"}   # runner/cli.py:67-148 common_run_options


def to_inference_argv(cli_argv: List[str]) -> List[str]:
    """The console script's forwarding (`runner/cli.py:193-223`): the shared options become the `common` dict, everything else is passed
    through, and upstream's own `build_argv` flattens both into the argparse form `inference.main()` parses."""
    from pxdesign.runner.cli import build_argv
    common, extra, i = {}, [], 0
    while i < len(cli_argv):
        a = cli_argv[i]
        if a in COMMON_KEYS and i + 1 < len(cli_argv):
            common[COMMON_KEYS[a]] = cli_argv[i + 1]; i += 2
        else:
            extra.append(a); i += 1
    return build_argv(common, extra)


STAMP_KEYS = ("batch", "layernorm", "kernels")           # the records in run()'s result that the callers file under `stamps` (split_stamps)


def split_stamps(res: dict) -> dict:
    """Move the stamps' records out of `run`'s result into their own block ({key: record}); the callers file them under ``stamps``."""
    return {k: res.pop(k) for k in STAMP_KEYS if k in res}


def seed_list(seeds, derive, now_ns=None) -> List[int]:
    """upstream's seed rule (`runner/inference.py:233`): the given seeds as ints, or — none given — exactly one, ``derive(time.time_ns())``."""
    if seeds:
        return [int(s) for s in seeds]
    return [int(derive(time.time_ns() if now_ns is None else now_ns))]


def run(argv: List[str], seeds: List[int], det: int, log=print, on_runner=None) -> dict:
    """upstream's main() with the seeding call parameterised. ``seeds`` empty = upstream's own rule (one clock-derived seed, `inference.py:233`).
    Returns timings, the seeds run, the report-only stamps' records and the runner's config paths."""
    layernorm_env = os.environ.get("LAYERNORM_TYPE") or None              # empty = absent (both select OpenFold's LayerNorm); the value Protenix reads at its import below (primitives.py:49-51), before upstream's configure_runtime_env rewrites it
    _det.apply_env(det)                                                   # before torch's first cuBLAS call
    import torch  # noqa: F401  (the recipe's environment is set above)
    from pxdesign.runner import inference as I                           # upstream's module: its own names below
    from . import stamps
    t0 = time.perf_counter()
    argv = to_inference_argv(argv)                                        # the console script's own forwarding of -i/-o and the shared options
    configs = I.get_configs(argv)
    os.makedirs(configs.dump_dir, exist_ok=True)
    configs.input_json_path = I.process_input_file(configs.input_json_path, out_dir=configs.dump_dir)
    I.download_inference_cache(configs)                                   # upstream's own; a no-op when the files exist (preflight, the deployment gate, made sure they do)
    if I.DIST_WRAPPER.rank == 0:
        I.save_config(configs, os.path.join(configs.dump_dir, "config.yaml"))
        with open(configs.input_json_path, "r") as f:
            orig_inputs = json.load(f)
        for x in orig_inputs:
            I.convert_to_bioassembly_dict(x, configs.dump_dir)
        configs.input_json_path = os.path.join(configs.dump_dir, "input_tasks.json")
        with open(configs.input_json_path, "w") as f:
            json.dump(orig_inputs, f, indent=4)
    from . import stack                                                   # here, never at module import: the stock caller imports this module before its environment proof, which lists pxdesign_opt.stack among the modules that must be absent
    runner = I.InferenceRunner(configs)                                   # loads the model and the checkpoint (the hook fires here under a kit mode)
    if on_runner is not None:
        on_runner(runner)
    load_s = time.perf_counter() - t0
    seed_recs, per_seed = [], {}
    seeds = seed_list(seeds, I.derive_seed)                              # runner/inference.py:233 verbatim: no --seeds = one clock-derived seed
    n_items = len(runner.dataset)
    n_sample = int(configs.sample_diffusion.N_sample) if hasattr(configs, "sample_diffusion") else None
    chunk = configs.infer_setting.sample_diffusion_chunk_size if hasattr(configs, "infer_setting") else None   # as upstream resolved it (configs_base.py:60-62, the --sample_diffusion_chunk_size alias)
    ln_kind, ln_class = stamps.layernorm_of_model(runner.model)                                              # the BUILT model's LayerNorm class: the KERNELS line's layernorm= word and the manifest's record
    kernels = stamps.kernels_facts(layernorm_kind=ln_kind, ds4sci=stamps.truthy(getattr(configs, "use_deepspeed_evo_attention", False)), source="process")
    log(stamps.kernels(kernels))
    for s in seeds:
        log(f"----------Infer with seed {s}----------")
        seed_recs.append(_det.seed(s, det))
        t1 = time.perf_counter()
        runner._inference(s)
        per_seed[str(s)] = round(time.perf_counter() - t1, 3)
    return {"load_s": round(load_s, 3), "per_seed_s": per_seed, "n_items": n_items, "seeds": list(seeds), "det": int(det), "seeding": seed_recs,
            "dump_dir": configs.dump_dir, "N_sample": n_sample,
            "batch": stamps.batch_facts(n_sample or 0, chunk, n_items, len(seeds)),
            "layernorm": {"kind": ln_kind, "class": ln_class, "source": "model", "env": layernorm_env},
            "kernels": kernels}

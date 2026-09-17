"""Command-line interface for Genie3.

This module is intentionally thin: it parses invocation options and dispatches
to runner modules. The existing legacy entry points remain unchanged while the
new CLI grows around them.
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from genie3.runtime import RunContext, create_run_context
from genie3.runtime.shards import (
    detect_num_eval_shards,
    detect_num_gen_shards,
    discover_problem_dirs,
    filter_problem_dirs,
    parse_selections,
    shard_done_missing,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="genie3",
        description="Run Genie3 generation and evaluation workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_run_command(subparsers)
    _add_generate_command(subparsers)
    _add_evaluate_command(subparsers)
    _add_train_command(subparsers)
    _add_status_command(subparsers)

    return parser


def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    """Attach shared flags (--config, --verbose, --log-dir, --num-devices) to a subcommand parser."""
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the experiment configuration file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed runtime output in the terminal.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/runs",
        help="Root directory for detailed run logs (default: logs/runs).",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=None,
        metavar="N",
        help="Number of GPU devices to use (overrides runtime.num_devices in config).",
    )


def _add_run_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'run' subcommand (generate + evaluate in one shot)."""
    parser = subparsers.add_parser(
        "run",
        help="Run generation followed by evaluation.",
    )
    _add_shared_options(parser)
    parser.set_defaults(handler=_handle_run)


def _add_generate_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'generate' subcommand with optional sharding flags."""
    parser = subparsers.add_parser(
        "generate",
        help="Run generation only.",
    )
    _add_shared_options(parser)
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        metavar="ID",
        help="Zero-based shard index for multi-node generation (default: 0).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        metavar="N",
        help="Total number of shards for multi-node generation (default: 1 = no sharding).",
    )
    parser.set_defaults(handler=_handle_generate)


def _add_evaluate_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'evaluate' subcommand with optional sharding and reduce flags."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Run evaluation only.",
    )
    _add_shared_options(parser)
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        metavar="ID",
        help="Zero-based shard index for multi-node evaluation (default: 0).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        metavar="N",
        help="Total number of shards for multi-node evaluation (default: 1 = no sharding).",
    )
    parser.add_argument(
        "--reduce",
        action="store_true",
        help="Merge shard outputs and run the reduce step. Run after all shards complete.",
    )
    parser.set_defaults(handler=_handle_evaluate)


def _add_train_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'train' subcommand for model training."""
    parser = subparsers.add_parser(
        "train",
        help="Run model training.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=int,
        required=True,
        help="Number of GPU devices to use.",
    )
    parser.add_argument(
        "-n",
        "--num-nodes",
        type=int,
        default=1,
        help="Number of compute nodes.",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Disable remote logging (W&B) for local runs.",
    )
    parser.add_argument(
        "--mpi-plugin",
        action="store_true",
        help="Enable the MPI environment plugin.",
    )
    parser.add_argument(
        "--memory-snapshot",
        action="store_true",
        help="Enable CUDA memory snapshot collection.",
    )
    parser.add_argument(
        "--reset-dataloader-state",
        action="store_true",
        help="Reset the dataloader checkpoint state before training.",
    )
    parser.set_defaults(handler=_handle_train)


def _add_status_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'status' subcommand for checking shard progress."""
    parser = subparsers.add_parser(
        "status",
        help="Show generation shard progress.",
    )
    _add_shared_options(parser)
    parser.set_defaults(handler=_handle_status)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _resolve_shard_args(args: argparse.Namespace) -> tuple[int, int]:
    """Validate --shard-id / --num-shards args. Returns (shard_id, num_shards)."""
    shard_id = args.shard_id
    num_shards = args.num_shards
    if num_shards < 1:
        print("error: --num-shards must be >= 1.", file=sys.stderr)
        raise SystemExit(2)
    if shard_id < 0 or shard_id >= num_shards:
        print(
            f"error: --shard-id must be in [0, num_shards). "
            f"Got shard_id={shard_id}, num_shards={num_shards}.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return shard_id, num_shards


def _handle_run(args: argparse.Namespace) -> int:
    """Handle 'genie3 run': generate then evaluate, supporting iterative rounds."""
    context: RunContext | None = None
    try:
        with create_run_context(
            command="run",
            config_path=args.config,
            verbose=args.verbose,
            log_dir=args.log_dir,
        ) as context:
            _print_run_header(context, "run")
            child_env = _build_child_run_env(context)

            from genie3.config import load_experiment_config
            config = load_experiment_config(args.config)

            if config.rounds:
                _run_iterative(args, child_env, config)
            else:
                _run_single_shot(args, child_env)

        return 0
    except KeyboardInterrupt:
        _report_interrupt(context)
        return 130
    except Exception as exc:
        _report_failure(context, exc)
        return 1


def _handle_generate(args: argparse.Namespace) -> int:
    """Handle 'genie3 generate': run the generation stage for one shard."""
    context: RunContext | None = None
    try:
        from genie3.generation.workflow import run_generation

        shard_id, num_shards = _resolve_shard_args(args)

        with create_run_context(
            command="generate",
            config_path=args.config,
            verbose=args.verbose,
            log_dir=args.log_dir,
        ) as context:
            if not _is_child_run_stage():
                _print_run_header(context, "generate")
            context.logger.info("🧬 Generation")
            generation_result = run_generation(
                config_path=args.config,
                verbose=args.verbose,
                log_dir=args.log_dir,
                context=context,
                shard_id=shard_id,
                num_shards=num_shards,
                num_devices=args.num_devices,
            )
            context.progress.clear_status()
            context.logger.info("✅ Generation finished")
            context.logger.info("   Output: %s", generation_result.output_dir)
            _print_generation_summary(context, generation_result)
        return 0
    except KeyboardInterrupt:
        _report_interrupt(context)
        return 130
    except Exception as exc:
        _report_failure(context, exc)
        return 1


def _handle_evaluate(args: argparse.Namespace) -> int:
    """Handle 'genie3 evaluate': run evaluation for one shard, or reduce all shards."""
    context: RunContext | None = None
    try:
        from genie3.evaluation.workflow import run_evaluation

        reduce = getattr(args, "reduce", False)
        if not reduce:
            shard_id, num_shards = _resolve_shard_args(args)
        else:
            shard_id, num_shards = 0, 1

        with create_run_context(
            command="evaluate",
            config_path=args.config,
            verbose=args.verbose,
            log_dir=args.log_dir,
        ) as context:
            if not _is_child_run_stage():
                _print_run_header(context, "evaluate")
            context.logger.info("🧪 Evaluation")
            evaluation_result = run_evaluation(
                config_path=args.config,
                verbose=args.verbose,
                log_dir=args.log_dir,
                context=context,
                shard_id=shard_id,
                num_shards=num_shards,
                reduce=reduce,
                num_devices=args.num_devices,
            )
            context.progress.clear_status()
            context.logger.info("✅ Evaluation finished")
            context.logger.info("   Results: %s", evaluation_result.output_dir)
            _print_evaluation_summary(context, evaluation_result)
        return 0
    except KeyboardInterrupt:
        _report_interrupt(context)
        return 130
    except Exception as exc:
        _report_failure(context, exc)
        return 1


def _handle_train(args: argparse.Namespace) -> int:
    """Handle 'genie3 train': launch model training via PyTorch Lightning."""
    try:
        from genie3.generation.workflow import run_training

        run_training(
            config_path=args.config,
            devices=args.devices,
            num_nodes=args.num_nodes,
            test=args.test,
            mpi_plugin=args.mpi_plugin,
            memory_snapshot=args.memory_snapshot,
            reset_dataloader_state=args.reset_dataloader_state,
        )
        return 0
    except KeyboardInterrupt:
        print("⏹️ Training interrupted.", file=sys.stderr)
        return 130
    except Exception:
        print("❌ Training failed.", file=sys.stderr)
        return 1


def _handle_status(args: argparse.Namespace) -> int:
    """Handle 'genie3 status': print generation and evaluation shard progress."""
    from genie3.config import load_experiment_config, to_evaluation_kwargs
    from genie3.config.loader import to_generation_config

    config = load_experiment_config(args.config)

    # --- Iterative mode status ---
    if config.rounds:
        base_outdir = Path(config.rootdir() or ".")
        print(f"📁 {base_outdir}")
        print()
        for i, round_cfg in enumerate(config.rounds):
            round_id = round_cfg.id or f"round_{i}"
            round_dir = base_outdir / round_id
            gen_done = (round_dir / ".generate_done").exists()
            eval_done = (round_dir / ".evaluate_done").exists()

            gen_icon = "✅" if gen_done else "⬜"
            eval_icon = "✅" if eval_done else "⬜"

            if gen_done and eval_done:
                row_icon = "✅"
            elif gen_done:
                row_icon = "🟡"
            else:
                row_icon = "🔵"

            strategy_label = f"[{round_cfg.cond_strategy}]"
            print(
                f"{row_icon} {round_id:<12} {strategy_label:<20} "
                f"generate {gen_icon}   evaluate {eval_icon}"
            )
        return 0

    gen_config = to_generation_config(config)
    outdir = Path(gen_config["io"]["outdir"])

    print(f"📁 {outdir}")
    print()

    # --- Per-problem status: generation + evaluation from unified .shard_markers/ ---
    try:
        eval_kwargs = to_evaluation_kwargs(config)
        eval_rootdir = Path(eval_kwargs["rootdir"])
    except Exception:
        eval_rootdir = outdir

    selections_str = gen_config.get("dataset", {}).get("selections")
    problem_dirs: list[Path] = [
        Path(p) for p in filter_problem_dirs(
            discover_problem_dirs(str(eval_rootdir)),
            parse_selections(selections_str),
        )
    ]

    if not problem_dirs:
        # Generation hasn't produced any problem dirs yet — check outdir-level markers.
        marker_dir = outdir / ".shard_markers"
        print()
        num_shards = detect_num_gen_shards(marker_dir)
        if num_shards is None:
            print("🔵 Generation   no shards started")
            return 0
        done, missing = shard_done_missing(marker_dir, "generate", num_shards)
        bar = "".join("✅" if i not in missing else "⬜" for i in range(num_shards))
        if missing:
            print(f"🟡 Generation   {len(done)}/{num_shards} shards  {bar}")
            print()
            print("   Re-run missing shards:")
            for i in missing:
                print(f"   genie3 generate -c {args.config} --shard-id {i} --num-shards {num_shards}")
        else:
            print(f"✅ Generation   {num_shards}/{num_shards} shards  {bar}")
        return 0

    print()
    all_eval_shards_done = True
    all_reduced = True
    for problem_dir in sorted(problem_dirs):
        problem_name = problem_dir.name if problem_dir != eval_rootdir else eval_rootdir.name
        marker_dir = problem_dir / ".shard_markers"

        # --- Generation ---
        gen_num_shards = detect_num_gen_shards(marker_dir)
        if gen_num_shards is None:
            print(f"🔵 Generation  {problem_name}  no shards started")
            all_eval_shards_done = False
            all_reduced = False
            continue
        gen_done, gen_missing = shard_done_missing(marker_dir, "generate", gen_num_shards)
        gen_bar = "".join("✅" if i not in gen_missing else "⬜" for i in range(gen_num_shards))
        if gen_missing:
            print(f"🟡 Generation  {problem_name}  {len(gen_done)}/{gen_num_shards} shards  {gen_bar}")
            print()
            print("   Re-run missing shards:")
            for i in gen_missing:
                print(f"   genie3 generate -c {args.config} --shard-id {i} --num-shards {gen_num_shards}")
            all_eval_shards_done = False
            all_reduced = False
            continue
        print(f"✅ Generation  {problem_name}  {gen_num_shards}/{gen_num_shards} shards  {gen_bar}")

        # --- Evaluation ---
        if (problem_dir / "results" / "eval.done").exists():
            print(f"✅ Evaluation  {problem_name}  complete (reduce done)")
            continue

        all_reduced = False
        eval_num_shards = detect_num_eval_shards(problem_dir)
        if eval_num_shards is None:
            print(f"🔵 Evaluation  {problem_name}  no shards started")
            all_eval_shards_done = False
            continue
        eval_done, eval_missing = shard_done_missing(marker_dir, "evaluate", eval_num_shards)
        if eval_missing:
            bar = "".join("✅" if i not in eval_missing else "⬜" for i in range(eval_num_shards))
            print(f"🟡 Evaluation  {problem_name}  {len(eval_done)}/{eval_num_shards} shards  {bar}")
            print()
            print("   Re-run missing shards:")
            for i in eval_missing:
                print(
                    f"   genie3 evaluate -c {args.config}"
                    f" --shard-id {i} --num-shards {eval_num_shards}"
                )
            all_eval_shards_done = False
        else:
            bar = "✅" * eval_num_shards
            print(f"✅ Evaluation  {problem_name}  {eval_num_shards}/{eval_num_shards} shards  {bar}")

    if all_eval_shards_done and not all_reduced:
        print()
        print(f"   Run reduce step:  genie3 evaluate --reduce -c {args.config}")

    return 0


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

_ITER_STRATEGIES = frozenset({"iter_common", "iter_common_prob"})


def _run_single_shot(args: argparse.Namespace, child_env: dict[str, str]) -> None:
    """Run generate → evaluate → reduce as three sequential child processes."""
    _run_child_stage(
        stage="generate",
        config_path=args.config,
        verbose=args.verbose,
        child_env=child_env,
        num_devices=args.num_devices,
    )
    _cleanup_generation_runtime()
    _run_child_stage(
        stage="evaluate",
        config_path=args.config,
        verbose=args.verbose,
        child_env=child_env,
        num_devices=args.num_devices,
    )
    _run_child_stage(
        stage="evaluate",
        config_path=args.config,
        verbose=args.verbose,
        child_env=child_env,
        num_devices=args.num_devices,
        extra_args=["--reduce"],
    )


def _run_iterative(args, child_env: dict[str, str], config) -> None:
    """Execute multi-round iterative design, skipping rounds that have already completed."""
    import yaml

    base_outdir = config.rootdir()
    if base_outdir is None:
        raise RuntimeError("paths.rootdir must be set for iterative design.")

    completed_round_outdirs: list[str] = []

    for i, round_cfg in enumerate(config.rounds):
        round_id = round_cfg.id or f"round_{i}"
        round_outdir = os.path.join(base_outdir, round_id)
        generate_sentinel = os.path.join(round_outdir, ".generate_done")
        evaluate_sentinel = os.path.join(round_outdir, ".evaluate_done")

        generate_done = os.path.exists(generate_sentinel)
        evaluate_done = os.path.exists(evaluate_sentinel)

        if generate_done and evaluate_done:
            print(f"  ✅ {round_id}  [{round_cfg.cond_strategy}]  already complete, skipping")
            completed_round_outdirs.append(round_outdir)
            continue

        print(f"  ▶  {round_id}  [{round_cfg.cond_strategy}]")

        original_datadir = config.generation.dataset.get("datadir", "")
        effective_datadir = original_datadir
        resolved_cond_strategy = round_cfg.cond_strategy

        if round_cfg.cond_strategy in _ITER_STRATEGIES and not generate_done:
            resolved_cond_strategy = _compute_and_cache_iter_interface(
                base_outdir=base_outdir,
                original_datadir=original_datadir,
                cond_strategy=round_cfg.cond_strategy,
                completed_round_outdirs=completed_round_outdirs,
                config=config,
            )
            effective_datadir = base_outdir

        raw_override = _build_round_config(
            original_raw=config.raw,
            cond_strategy=resolved_cond_strategy,
            datadir=effective_datadir,
            round_outdir=round_outdir,
            n_sample=round_cfg.n_sample,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix=f"genie3_round_{round_id}_"
        ) as tmp:
            yaml.dump(raw_override, tmp)
            tmp_config_path = tmp.name

        try:
            os.makedirs(round_outdir, exist_ok=True)

            if not generate_done:
                _run_child_stage(
                    stage="generate",
                    config_path=tmp_config_path,
                    verbose=args.verbose,
                    child_env=child_env,
                    num_devices=args.num_devices,
                )
                _cleanup_generation_runtime()
                Path(generate_sentinel).touch()

            if not evaluate_done:
                _run_child_stage(
                    stage="evaluate",
                    config_path=tmp_config_path,
                    verbose=args.verbose,
                    child_env=child_env,
                    num_devices=args.num_devices,
                )
                _run_child_stage(
                    stage="evaluate",
                    config_path=tmp_config_path,
                    verbose=args.verbose,
                    child_env=child_env,
                    num_devices=args.num_devices,
                    extra_args=["--reduce"],
                )
                Path(evaluate_sentinel).touch()
        finally:
            os.unlink(tmp_config_path)

        completed_round_outdirs.append(round_outdir)


def _build_child_run_env(context: RunContext) -> dict[str, str]:
    """Build the environment dict passed to child stage processes spawned by 'genie3 run'."""
    env = os.environ.copy()
    env["GENIE3_RUN_DIR"] = str(context.run_dir)
    env["GENIE3_RUN_LOG"] = str(context.log_file)
    env["GENIE3_WORKERS_DIR"] = str(context.logging_config.workers_dir)
    env["GENIE3_PARENT_COMMAND"] = "run"
    env["GENIE3_VERBOSE"] = "1" if context.invocation.verbose else "0"
    return env


def _run_child_stage(
    *,
    stage: str,
    config_path: str,
    verbose: bool,
    child_env: dict[str, str],
    num_devices: int | None = None,
    extra_args: list[str] | None = None,
) -> None:
    """Spawn a 'genie3 <stage>' child process with the given config and environment."""
    genie3_executable = shutil.which("genie3")
    if not genie3_executable:
        raise RuntimeError(
            "genie3 executable not found on PATH. Install the package in the "
            "active environment before using `genie3 run`."
        )
    cmd = [
        genie3_executable,
        stage,
        "--config",
        config_path,
    ]
    if verbose:
        cmd.append("--verbose")
    if num_devices is not None:
        cmd.extend(["--num-devices", str(num_devices)])
    if extra_args:
        cmd.extend(str(a) for a in extra_args)
    subprocess.run(cmd, check=True, env=child_env)


def _cleanup_generation_runtime() -> None:
    """Free GPU memory and destroy the distributed process group between pipeline stages."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Iterative design helpers
# ---------------------------------------------------------------------------

def _build_round_config(
    original_raw: dict,
    cond_strategy: str,
    datadir: str,
    round_outdir: str,
    n_sample: int | None,
) -> dict:
    """Build a round-specific config dict by overriding the base config with per-round settings.

    Strips the `rounds` key so the child process does not recurse into iterative logic.
    Sets `paths.rootdir` to `round_outdir` and `paths.dataset` to `datadir`.
    """
    raw = copy.deepcopy(original_raw)

    # Remove rounds: from the round-specific config to avoid recursive iteration
    raw.pop("rounds", None)

    generation = raw.setdefault("generation", {})
    dataset = generation.setdefault("dataset", {})
    dataset["cond_strategy"] = cond_strategy
    if n_sample is not None:
        dataset["n_sample"] = n_sample
    # generation.dataset.datadir is forbidden by the loader; use paths.dataset instead

    # Clear generation.io.outdir — it would conflict with paths.rootdir
    io = generation.get("io", {})
    io.pop("outdir", None)

    paths = raw.setdefault("paths", {})
    paths["rootdir"] = round_outdir
    # Override dataset path so iter_* strategies read from the accumulating problems dir
    paths["dataset"] = datadir

    return raw


def _compute_and_cache_iter_interface(
    base_outdir: str,
    original_datadir: str,
    cond_strategy: str,
    completed_round_outdirs: list[str],
    config,
) -> str:
    """Compute the common binder interface from prior-round successful complexes and cache it.

    For each problem, collects all `v0_success/successful_complexes/*.pdb` files from
    `completed_round_outdirs`, computes the common interface residues, writes the result
    as `{cond_strategy}_round{N}` into the accumulated problem JSON, and returns the
    resolved strategy key (e.g. `iter_common_round0`).
    """
    from genie3.generation.utils.interface.common import (
        compute_common_interface_from_filepaths,
        parse_chain_residue_indices,
        parse_hotspot_tags,
    )

    accum_problems_dir = os.path.join(base_outdir, "problems")
    os.makedirs(accum_problems_dir, exist_ok=True)

    problems = _list_selected_problems(original_datadir, config.generation.dataset)
    resolved_key = None

    for problem in problems:
        accum_path = os.path.join(accum_problems_dir, f"{problem}.json")

        if not os.path.exists(accum_path):
            src = os.path.join(original_datadir, "problems", f"{problem}.json")
            with open(src) as f:
                info = json.load(f)
        else:
            with open(accum_path) as f:
                info = json.load(f)

        existing = [
            k for k in info["target_interface_residues"]
            if k.startswith(cond_strategy + "_round")
        ]
        this_key = f"{cond_strategy}_round{len(existing)}"
        if resolved_key is None:
            resolved_key = this_key

        filepaths: list[str] = []
        for prior_outdir in completed_round_outdirs:
            pattern = os.path.join(
                prior_outdir, problem, "results", "v0_success", "successful_complexes", "*.pdb"
            )
            filepaths.extend(glob.glob(pattern))

        if not filepaths:
            logging.warning(
                "No successes found for %s across prior rounds; %s will be absent from problem JSON",
                problem, this_key,
            )
        else:
            chain_residue_indices = parse_chain_residue_indices(original_datadir, problem)
            hotspot_tags = parse_hotspot_tags(original_datadir, problem)
            tags = compute_common_interface_from_filepaths(filepaths, chain_residue_indices, hotspot_tags)
            info["target_interface_residues"][this_key] = tags
            logging.info("Computed %s for %s: %d residues", this_key, problem, len(tags))

        with open(accum_path, "w") as f:
            json.dump(info, f, indent=2)

    return resolved_key or f"{cond_strategy}_round0"


def _list_selected_problems(datadir: str, dataset: dict) -> list[str]:
    """Return the list of problem names to process, filtered by selections or tags."""
    selections = dataset.get("selections")
    tags = dataset.get("tags")

    all_problem_files = glob.glob(os.path.join(datadir, "problems", "*.json"))

    if selections:
        selected = {s.strip() for s in selections.split(",")}
        return [p for p in selected if os.path.exists(os.path.join(datadir, "problems", f"{p}.json"))]

    if tags:
        tag_set = {t.strip() for t in tags.split(",")}
        problems = []
        for fpath in all_problem_files:
            with open(fpath) as f:
                info = json.load(f)
            if "tag" in info and any(t in tag_set for t in info["tag"]):
                problems.append(os.path.basename(fpath).replace(".json", ""))
        return problems

    return [os.path.basename(f).replace(".json", "") for f in all_problem_files]


# ---------------------------------------------------------------------------
# Process utilities
# ---------------------------------------------------------------------------

def _is_child_run_stage() -> bool:
    """Return True if this process was launched as a child stage of 'genie3 run'."""
    return os.environ.get("GENIE3_PARENT_COMMAND") == "run"


def _is_primary_process() -> bool:
    """Return True if this is the primary (rank-0) process in a distributed run."""
    local_rank = os.environ.get("LOCAL_RANK")
    rank = os.environ.get("RANK")
    if local_rank is not None:
        return local_rank == "0"
    if rank is not None:
        return rank == "0"
    return True


def _report_failure(
    context: RunContext | None,
    exc: BaseException | None = None,
) -> None:
    """Print a failure message to stderr; no-op on non-primary processes."""
    if not _is_primary_process():
        return
    if context is not None:
        context.progress.clear_status()
        print(f"❌ Run failed. See log: {context.log_file}", file=sys.stderr)
        return
    if exc is not None:
        print(f"❌ Run failed: {exc}", file=sys.stderr)
    else:
        print("❌ Run failed.", file=sys.stderr)


def _report_interrupt(
    context: RunContext | None,
    exc: BaseException | None = None,
) -> None:
    """Print an interruption message to stderr; no-op on non-primary processes."""
    if not _is_primary_process():
        return
    if context is not None:
        context.progress.clear_status()
        print(f"⏹️ Run interrupted. See log: {context.log_file}", file=sys.stderr)
        return
    if exc is not None:
        print(f"⏹️ Run interrupted: {exc}", file=sys.stderr)
    else:
        print("⏹️ Run interrupted.", file=sys.stderr)


def _print_run_header(context: RunContext, command: str) -> None:
    """Log the run header (command name, experiment name, rootdir) at the start of a top-level invocation."""
    _log_divider(context)
    context.logger.info("🚀 %s started", command.capitalize())
    _log_kv(context, "Experiment", context.config.experiment.name, level=1, label_width=24)
    _log_kv(context, "Rootdir", context.config.rootdir(), level=1, label_width=24)
    _log_divider(context)


# ---------------------------------------------------------------------------
# Logging / summary
# ---------------------------------------------------------------------------

def _print_generation_summary(context: RunContext, generation_result) -> None:
    """Log structured generation timing and output counts to the run logger."""
    summary = generation_result.summary
    if summary is None:
        return
    _log_divider(context)
    context.logger.info("📊 Generation Summary")
    output_rows = [("Samples generated", str(summary.generated_samples))]
    if summary.search_name == "beam":
        output_rows.append(("Beam-search runs", str(summary.batch_count)))
    if summary.worker_count > 1:
        output_rows.append(("Workers", str(summary.worker_count)))
        if summary.representative_device is not None:
            output_rows.append(("Representative device", summary.representative_device))
    _log_subsection(context, "📦 Outputs", output_rows)

    setup_rows: list[tuple[str, str]] = []
    if summary.compile_warmup_seconds > 0:
        setup_rows.append(("Compile warmup", f"{summary.compile_warmup_seconds:.3f}s"))

    sampling_rows: list[tuple[str, str]] = []
    if summary.per_beam_search_seconds is not None:
        beam_label = "Per beam-search run"
        if summary.beam_width is not None:
            beam_label = f"Per beam-search run (K={summary.beam_width})"
        sampling_rows.append(
            (
                beam_label,
                f"{summary.per_beam_search_seconds:.3f}s",
            )
        )
    if summary.per_sample_seconds is not None:
        sampling_rows.append(
            (
                "Per-sample generation",
                f"{summary.per_sample_seconds:.3f}s",
            )
        )
    if summary.total_batch_seconds > 0:
        sampling_rows.append(("Sampling elapsed", f"{summary.total_batch_seconds:.3f}s"))

    if setup_rows:
        _log_subsection(context, "⚙️ Setup", setup_rows)
    if sampling_rows:
        _log_subsection(context, "🧪 Sampling", sampling_rows)
    _log_divider(context)


def _print_evaluation_summary(context: RunContext, evaluation_result) -> None:
    """Log structured evaluation timing, model details, and success counts to the run logger."""
    summary = evaluation_result.summary
    if summary is None:
        return
    _log_divider(context)
    context.logger.info("📊 Evaluation Summary")
    output_rows = [
        ("Structures generated", str(summary.generated_structures)),
        ("Sequences generated", str(summary.generated_sequences)),
        ("Model outputs written", str(summary.predicted_structures)),
    ]
    if summary.worker_count > 1:
        output_rows.append(("Workers", str(summary.worker_count)))
        if summary.representative_device is not None:
            output_rows.append(("Representative device", summary.representative_device))
    _log_subsection(
        context,
        "📦 Outputs",
        output_rows,
    )

    setup_rows: list[tuple[str, str]] = []
    inverse_fold_init_display = (
        summary.inverse_fold_model_init_wall_seconds
        if summary.worker_count > 1
        else summary.inverse_fold_model_init_seconds
    )
    fold_model_init_display = (
        summary.fold_model_init_wall_seconds
        if summary.worker_count > 1
        else summary.fold_model_init_seconds
    )
    startup_display = (
        summary.startup_wall_seconds
        if summary.worker_count > 1
        else summary.startup_seconds
    )

    if inverse_fold_init_display > 0:
        setup_rows.append(
            ("Inverse-fold model init", f"{inverse_fold_init_display:.3f}s elapsed")
        )
    if fold_model_init_display > 0:
        setup_rows.append(("Fold model preload", f"{fold_model_init_display:.3f}s elapsed"))
    if startup_display > 0:
        setup_rows.append(("Startup elapsed", f"{startup_display:.3f}s"))

    inverse_fold_rows: list[tuple[str, str]] = []
    inverse_fold_total_display = (
        summary.inverse_fold_wall_seconds
        if summary.worker_count > 1
        else summary.inverse_fold_seconds
    )
    inverse_fold_per_sequence_display = (
        summary.inverse_fold_wall_seconds_per_sequence
        if summary.worker_count > 1
        else summary.inverse_fold_seconds_per_sequence
    )
    if inverse_fold_total_display > 0:
        inverse_fold_rows.append(("Inverse fold elapsed", f"{inverse_fold_total_display:.3f}s"))
    if inverse_fold_per_sequence_display is not None:
        inverse_fold_rows.append(
            ("Per-sequence inverse fold", f"{inverse_fold_per_sequence_display:.3f}s")
        )

    fold_details = f"model={summary.fold_model_name}"
    if summary.fold_num_models is not None:
        fold_details += f", models={summary.fold_num_models}"
    if summary.fold_num_recycles is not None:
        fold_details += f", recycles={summary.fold_num_recycles}"

    fold_rows: list[tuple[str, str]] = []
    fold_total_display = (
        summary.fold_wall_seconds
        if summary.worker_count > 1
        else summary.fold_seconds
    )
    fold_per_sequence_display = (
        summary.fold_wall_seconds_per_sequence
        if summary.worker_count > 1
        else summary.fold_seconds_per_sequence
    )
    fold_per_model_output_display = (
        summary.fold_wall_seconds_per_model_output
        if summary.worker_count > 1
        else summary.fold_seconds_per_model_output
    )
    if fold_total_display > 0:
        fold_rows.append(("Fold elapsed", f"{fold_total_display:.3f}s ({fold_details})"))
    if fold_per_sequence_display is not None:
        fold_rows.append(
            ("Per-sequence fold", f"{fold_per_sequence_display:.3f}s ({fold_details})")
        )
    if fold_per_model_output_display is not None:
        fold_rows.append(
            (
                "Per-model-output fold",
                f"{fold_per_model_output_display:.3f}s ({fold_details})",
            )
        )

    if not getattr(evaluation_result, "is_reduce", False):
        if setup_rows:
            _log_subsection(context, "⚙️ Setup", setup_rows)
        if inverse_fold_rows:
            _log_subsection(context, "🧬 Inverse Fold", inverse_fold_rows)
        if fold_rows:
            _log_subsection(context, "🧪 Fold", fold_rows)

    for success in summary.success_summaries:
        _log_subsection(
            context,
            "🏁 Successes",
            [
                (
                    success.label.replace("_", " ").capitalize(),
                    f"{success.successes} successes, "
                    f"{success.unique_tm_0_5} unique @ tm_0_5, "
                    f"{success.unique_tm_0_6} unique @ tm_0_6",
                )
            ],
        )
    _log_divider(context)


def _log_divider(context: RunContext) -> None:
    """Log a horizontal divider line."""
    context.logger.info("────────────────────────────────────────")


def _log_kv(
    context: RunContext,
    label: str,
    value,
    *,
    level: int = 1,
    label_width: int = 30,
) -> None:
    """Log a key-value pair with consistent indentation and column alignment."""
    indent = "   " * (level + 1)
    context.logger.info("%s%-*s %s", indent, label_width, f"{label}:", value)


def _log_subsection(
    context: RunContext,
    title: str,
    rows: list[tuple[str, str]],
) -> None:
    """Log a titled subsection with a list of (label, value) rows."""
    context.logger.info("   %s", title)
    for label, value in rows:
        _log_kv(context, label, value, level=2, label_width=32)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

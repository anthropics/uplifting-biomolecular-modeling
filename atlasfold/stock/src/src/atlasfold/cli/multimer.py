"""Command-line inference pipeline for AtlasFold multimer models."""

import argparse
import json
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np

from atlasfold.cli import multigpu


def create_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run AtlasFold-Multimer inference.",
    )
    required = parser.add_argument_group("required arguments")
    inference = parser.add_argument_group("inference options")
    runtime = parser.add_argument_group("runtime and batching options")
    output = parser.add_argument_group("output options")

    required.add_argument(
        "-i",
        "--input-fasta",
        type=Path,
        required=True,
        help=(
            "Input FASTA file. Each FASTA record is treated as one complex, "
            "with ':' separating chains."
        ),
    )
    required.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where predictions will be written.",
    )

    inference.add_argument(
        "--num-recycles",
        type=int,
        default=10,
        help="Number of recycling iterations.",
    )
    inference.add_argument(
        "--mlm-prob",
        type=float,
        default=0.20,
        help="LM masking probability used during recycling.",
    )
    inference.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of diffusion samples to generate.",
    )
    inference.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[1],
        help="Random seed(s) for inference. Example: --seed 1 2 3.",
    )
    inference.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="Number of diffusion sampling steps.",
    )

    runtime.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local model weights.",
    )
    runtime.add_argument(
        "--lm-path",
        type=Path,
        default=None,
        help="Optional local AtlasLM weights.",
    )
    runtime.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for AtlasFold and AtlasLM weights.",
    )
    runtime.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    runtime.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="Visible CUDA indices for target-parallel inference; excludes --device.",
    )
    runtime.add_argument(
        "--kernel",
        choices=["auto", "torch", "cuequiv"],
        default="auto",
        help=(
            "Triangle kernel backend. By default, selects cuequiv when available, "
            "otherwise torch."
        ),
    )
    runtime.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help=(
            "Maximum bucketed residue tokens per model call. Lower this if "
            "CUDA runs out of memory."
        ),
    )
    runtime.add_argument(
        "--length-buckets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit residue buckets. Defaults to the runner buckets: "
            "32, 64, 128, 192, 256, 384, 512, 640, 768, ..."
        ),
    )

    output.add_argument(
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format for sample and ranked outputs.",
    )
    output.add_argument(
        "--save-confidence-arrays",
        action="store_true",
        help="Save raw pLDDT, PAE, and PDE arrays for each sample as NPZ files.",
    )
    output.add_argument(
        "--save-distogram",
        action="store_true",
        help="Save raw distogram logits and boundaries for each seed as NPZ files.",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute targets even when outputs already exist.",
    )
    return parser


def run(args: argparse.Namespace, inputs=None) -> None:
    multigpu.validate_args(args)
    if args.num_recycles < 0:
        raise ValueError(f"num_recycles must be non-negative, got {args.num_recycles}.")
    if not 0.0 < args.mlm_prob <= 1.0:
        raise ValueError(
            f"mlm_prob must be greater than 0 and at most 1, got {args.mlm_prob}."
        )
    if args.num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {args.num_samples}.")
    if args.num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {args.num_steps}.")
    if args.max_tokens_per_batch <= 0:
        raise ValueError(
            f"max_tokens_per_batch must be positive, got {args.max_tokens_per_batch}."
        )
    if args.length_buckets is not None and any(
        bucket <= 0 for bucket in args.length_buckets
    ):
        raise ValueError(
            "length_buckets must contain only positive values, "
            f"got {args.length_buckets}."
        )

    import torch

    from atlasfold.data.fasta import read_fasta
    from atlasfold.model import AtlasFold_Multimer, SamplingConfig
    from atlasfold.pretrained import load_model as load_pretrained_model
    from atlasfold.runner_multimer import (
        MultimerFoldingOutput,
        MultimerFoldingRunner,
        MultimerInput,
    )

    def setup_logger() -> logging.Logger:
        logger = logging.getLogger("atlasfold.multimer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(message)s",
                datefmt="%y/%m/%d %H:%M:%S",
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

    def split_chain_sequences(sequence: str) -> list[str]:
        sequences = sequence.split(":")
        if len(sequences) == 0 or any(not part.strip() for part in sequences):
            raise ValueError("FASTA target contains an empty chain.")
        return sequences

    def load_inputs(args: argparse.Namespace) -> list[MultimerInput]:
        if not args.input_fasta.exists():
            raise FileNotFoundError(
                f"Input FASTA file does not exist: {args.input_fasta}"
            )

        inputs: list[MultimerInput] = []
        for header, sequence in read_fasta(args.input_fasta):
            header_fields = header.split()
            if not header_fields:
                raise ValueError("FASTA target has an empty header.")
            name = header_fields[0]
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError(
                    f"FASTA target name must not contain path components: {name!r}."
                )
            chains = split_chain_sequences(sequence)
            inputs.append(MultimerInput(name, chains))
        if len(inputs) == 0:
            raise ValueError(f"No sequences found in FASTA file: {args.input_fasta}")

        seen: set[str] = set()
        duplicates: list[str] = []
        for item in inputs:
            name = item.name
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                "FASTA target names must be unique after normalization. "
                f"Duplicates: {sorted(set(duplicates))}"
            )

        return inputs

    def load_model(
        model_path: str | Path | None = None,
        lm_path: str | Path | None = None,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        kernel: str = "auto",
    ) -> AtlasFold_Multimer:
        device_str = device
        if device_str is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        logger.info("Using device: %s", device)

        if model_path is not None:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Local weight file does not exist: {model_path}")
            logger.info("Loading local weight path=%s", model_path)
        else:
            logger.info(
                "Loading `SeonghwanSeo/atlasfold-m-260725` from Hugging Face, "
                "cache_dir=%s",
                cache_dir,
            )

        if lm_path is not None:
            lm_path = Path(lm_path)
            if not lm_path.exists():
                raise FileNotFoundError(
                    f"Local AtlasLM weight file does not exist: {lm_path}"
                )
            logger.info("Loading local AtlasLM weight path=%s", lm_path)
        else:
            logger.info(
                "Loading `SeonghwanSeo/atlaslm-3b-base` from Hugging Face, cache_dir=%s",
                cache_dir,
            )

        model = load_pretrained_model(
            "atlasfold-m-260725",
            device=device,
            cache_dir=cache_dir,
            lm_path=lm_path,
            model_path=model_path,
            kernel=kernel,
        )
        if not isinstance(model, AtlasFold_Multimer):
            raise TypeError(f"Expected AtlasFold_Multimer, got {type(model)!r}.")
        return model

    def write_outputs(
        out_dir: Path,
        output: MultimerFoldingOutput,
        format: str,
        save_confidence_arrays: bool = False,
        save_distogram: bool = False,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_records = []

        # Remove the "done.txt" file if it exists, since we are overwriting the outputs.
        done_path = out_dir / "done.txt"
        if done_path.exists():
            done_path.unlink()

        best_sample_idx = output.best_key
        for (seed, sample_idx), sample in sorted(output.outputs.items()):
            target_name = output.name
            sample_name = f"{target_name}_seed-{seed}_sample-{sample_idx}"

            sample_text = (
                sample.to_mmcif(model="multimer")
                if format == "cif"
                else sample.to_pdb(model="multimer")
            )
            confidence_scores = sample.confidence_scores

            with open(out_dir / f"{sample_name}_model.{format}", "w") as f:
                f.write(sample_text)
            with open(out_dir / f"{sample_name}_confidence.json", "w") as f:
                json.dump(confidence_scores, f, indent=2)
            if save_confidence_arrays:
                np.savez(
                    out_dir / f"{sample_name}_confidence.npz",
                    plddt=sample.plddt,
                    pae=sample.pae,
                    pde=sample.pde,
                )

            record = {
                "seed": seed,
                "sample_index": sample_idx,
                "sample_name": sample_name,
                "score": sample.ranking_score,
                "ranking_score": confidence_scores["complex"]["ranking_score"],
                "ptm": confidence_scores["complex"]["ptm"],
                "iptm": confidence_scores["complex"]["iptm"],
                "fraction_disordered": confidence_scores["complex"][
                    "fraction_disordered"
                ],
                "avg_plddt": confidence_scores["complex"]["avg_plddt"],
                "avg_pde": confidence_scores["complex"]["avg_pde"],
            }

            sample_records.append(record)

            if (seed, sample_idx) == best_sample_idx:
                rank_name = f"{target_name}_ranked"
                best_record = record
                with open(out_dir / f"{rank_name}_model.{format}", "w") as f:
                    f.write(sample_text)
                with open(out_dir / f"{rank_name}_confidence.json", "w") as f:
                    json.dump(confidence_scores, f, indent=2)

        if save_distogram:
            if len(output.distogram_logits) == 0 or output.distogram_boundaries is None:
                raise ValueError(
                    f"No distograms are available for target {output.name!r}."
                )
            for seed, logits in sorted(output.distogram_logits.items()):
                np.savez(
                    out_dir / f"{output.name}_seed-{seed}_distogram.npz",
                    logits=logits,
                    boundaries=output.distogram_boundaries,
                )

        with open(out_dir / f"{output.name}_summary.csv", "w") as f:
            f.write(
                "seed,sample_index,ranking_score,ptm,iptm,"
                "fraction_disordered,avg_pde,avg_plddt\n"
            )
            for record in sample_records:
                f.write(
                    f"{record['seed']},"
                    f"{record['sample_index']},"
                    f"{record['ranking_score']},"
                    f"{record['ptm']},"
                    f"{record['iptm']},"
                    f"{record['fraction_disordered']},"
                    f"{record['avg_pde']},"
                    f"{record['avg_plddt']}\n"
                )

        # Create a "done.txt" file to indicate that the target is complete.
        done_path.touch()

        return best_record

    # Set up logging
    logger = setup_logger()

    # Set torch matmul precision to highest for better performance.
    torch.set_float32_matmul_precision("highest")
    inputs = load_inputs(args) if inputs is None else list(inputs)
    num_residues = [item.length for item in inputs]
    logger.info(
        "Loaded %d multimer target(s). Residue range: %d-%d.",
        len(inputs),
        min(num_residues),
        max(num_residues),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter out completed targets unless --overwrite is set.
    if not args.overwrite:
        filtered = [
            item for item in inputs if not (out_dir / item.name / "done.txt").exists()
        ]
        num_skipped = len(inputs) - len(filtered)
        if num_skipped > 0:
            logger.info("Skipping %d completed targets.", num_skipped)
        inputs = filtered

    if len(inputs) == 0:
        logger.info("All targets are complete. Nothing to do.")
        return

    if getattr(args, "gpu_ids", None) is not None:
        multigpu.run(args, inputs, "multimer", logger)
        return

    # Load the model
    model = load_model(
        model_path=args.model_path,
        lm_path=args.lm_path,
        device=args.device,
        cache_dir=args.cache_dir,
        kernel=args.kernel,
    )
    # Set up the runner
    runner = MultimerFoldingRunner(model)
    logger.info("Using triangle kernel backend: %s", model.kernel_backend)
    sampling_config = SamplingConfig(num_steps=args.num_steps)

    logger.info(
        "Starting multimer inference: seeds=%s, num_samples=%d, "
        "num_recycles=%d, mlm_prob=%s, num_steps=%d, format=%s",
        args.seed,
        args.num_samples,
        args.num_recycles,
        args.mlm_prob,
        args.num_steps,
        args.format,
    )

    num_finished = 0
    num_total = len(inputs)
    start = timer()
    batch_start = timer()
    for outputs in runner.fold_iter_batch(
        inputs,
        num_samples=args.num_samples,
        seeds=args.seed,
        num_recycles=args.num_recycles,
        mlm_prob=args.mlm_prob,
        sampling_config=sampling_config,
        length_buckets=args.length_buckets,
        max_tokens_per_batch=args.max_tokens_per_batch,
        return_distogram=args.save_distogram,
    ):
        batch_elapsed = timer() - batch_start
        time_per_target = batch_elapsed / len(outputs)
        batch_start = timer()

        for output in outputs:
            target_dir = out_dir / output.name
            best_record = write_outputs(
                target_dir,
                output,
                args.format,
                save_confidence_arrays=args.save_confidence_arrays,
                save_distogram=args.save_distogram,
            )
            num_finished += 1
            logger.info(
                "Completed %s (length=%d): "
                "pTM=%.3f, iPTM=%.3f, pLDDT=%.3f time=%.2f (%d/%d)",
                output.name,
                output.length,
                best_record["ptm"],
                best_record["iptm"],
                best_record["avg_plddt"],
                time_per_target,
                num_finished,
                num_total,
            )
    elapsed = timer() - start
    logger.info(
        "Finished %d target(s) in %.1fs.",
        len(inputs),
        elapsed,
    )


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

"""
PyTorch Lightning runner for Genie training and sampling.

This module provides the GenieRunner class which orchestrates the training
and inference processes for Genie models using PyTorch Lightning. It handles:
- Model and diffusion setup
- Training step with loss computation
- Test/sampling step with postprocessing
- Checkpoint loading and optimizer configuration
"""

from __future__ import annotations

import os
import csv
import glob
import json
import torch
import time
import shutil
import logging
from torch.optim import Adam
from lightning import LightningModule
from collections import OrderedDict
from ml_collections import ConfigDict
from typing import Dict, Tuple, List, Any

from genie3.generation.diffusion.registry import get_diffusion
from genie3.generation.diffusion.sampler.registry import get_diffusion_sampler
from genie3.generation.diffusion.search.registry import get_search
from genie3.generation.diffusion.reward.registry import get_reward
from genie3.generation.model.registry import get_model
from genie3.generation.runner.postprocess import postprocess
from genie3.generation.utils.config_utils import parse_batch_config
from genie3.generation.utils.feat_utils import (
    prepare_tensor_features,
    pad_tensor_features_to_n_token,
)

from genie3.generation.utils.loss.fape import update_fape_loss
from genie3.generation.utils.loss.sequence import update_sequence_loss
from genie3.generation.utils.loss.mse import compute_mse_weight_from_plddt, mse
from genie3.runtime.progress import get_active_reporter


class GenieRunner(LightningModule):
    """
    PyTorch Lightning module for Genie training and sampling.

    This class manages the complete lifecycle of Genie models including:
    - Setup of model, diffusion, and sampler
    - Training with MSE, FAPE, and sequence losses
    - Sampling/inference with postprocessing
    - Checkpoint management
    """

    def __init__(self, config: ConfigDict):
        """
        Initialize Genie runner.

        Args:
            config: Configuration dictionary containing model, diffusion,
                   training, and sampling parameters
        """
        super(GenieRunner, self).__init__()
        self.config = config
        self._compile_warmup_seconds = 0.0
        self._test_batch_start_time = None
        self._test_batch_seconds_total = 0.0
        self._test_sample_count = 0
        self._test_output_count = 0
        self._first_batch_seconds = None
        self._first_batch_sample_count = 0
        self._fixed_binder_max_length = None

    def setup(self, stage: str):
        """
        Setup model, diffusion, and sampler for training or testing.

        Args:
            stage: 'fit' for training or 'test' for sampling/inference
        """

        # Load configuration
        config = self.config
        if stage == "test":
            config = self.config.model

        # Set up model and diffusion process
        self.model = get_model(config.model)
        self.diffusion = get_diffusion(config.diffusion)
        self.diffusion.setup(self.device)

        # Additional set up for test stage
        if stage == "test":

            # Load checkpoint
            self._load_checkpoint(self.config.base.checkpoint)

            # Compile (if necessary)
            if self.config.compile:
                self.model = torch.compile(self.model)

            # Set up sampler
            inference = self.config.inference
            self.sampler = get_diffusion_sampler(inference.sampler)
            self.sampler.set_n_timestep(self.diffusion.n_timestep)
            self.sampler.set_betas(self.diffusion.betas)

            # Wire search and reward if configured
            if hasattr(inference, "search") and hasattr(inference, "reward"):
                inference.reward.reward.device = str(self.device)
                reward = get_reward(inference.reward.name, inference.reward.reward)
                search = get_search(inference.search.name, inference.search.search, reward)
                self.sampler.set_search(search)

            if self.sampler.verbose:
                logging.warning(
                    """
                    Verbose logging is enabled to save intermediate generated structures. This should
                    be used with cautions since it is disk-intensive.
                """
                )

    def _configure_fixed_shape_inference(self) -> None:
        """Enable fixed-shape padding for single-problem binder generation."""
        self._fixed_binder_max_length = None
        if not self.config.compile or self.config.dataset.source != "target":
            return
        if getattr(self.sampler, "_beam_search", None) is not None:
            logging.info(
                "[GenieRunner] Deferring fixed-shape padding to beam search "
                "for torch.compile"
            )
            return

        dm = getattr(self.trainer, "datamodule", None)
        dataset = getattr(dm, "dataset", None)
        tasks = getattr(dataset, "tasks", None)
        datadir = getattr(dataset, "datadir", None)
        if not tasks or datadir is None:
            return

        problem_names = list(dict.fromkeys(tasks))
        if len(problem_names) != 1:
            logging.info(
                "[GenieRunner] Skipping fixed-shape padding for torch.compile "
                f"because {len(problem_names)} problems are present"
            )
            return

        problem_path = os.path.join(datadir, "problems", f"{problem_names[0]}.json")
        with open(problem_path) as file:
            problem_info = json.load(file)
        binder_max_length = problem_info.get("binder_max_length", None)
        if binder_max_length is None:
            logging.info(
                "[GenieRunner] Skipping fixed-shape padding for torch.compile "
                f"because binder_max_length is missing for {problem_names[0]}"
            )
            return

        self._fixed_binder_max_length = int(binder_max_length)
        logging.info(
            "[GenieRunner] Enabled fixed-shape denoiser padding for torch.compile "
            f"(problem={problem_names[0]}, binder_max_length={self._fixed_binder_max_length})"
        )

    def _get_fixed_n_token(self, batch: Dict[str, torch.Tensor]) -> int | None:
        """Compute padded token length for single-problem binder generation."""
        if self._fixed_binder_max_length is None:
            return None

        binder_lengths = (
            ((1 - batch["cond_seq_mask"]) * batch["gt_atom_mask"])
            .sum(dim=1)
            .to(dtype=torch.int)
        )
        if torch.unique(binder_lengths).numel() != 1:
            raise RuntimeError(
                "Fixed-shape padding expects a shared binder length per batch, "
                f"but found {binder_lengths.tolist()}"
            )

        binder_length = int(binder_lengths[0].item())
        if binder_length > self._fixed_binder_max_length:
            raise RuntimeError(
                "Observed binder length exceeds configured binder_max_length: "
                f"{binder_length} > {self._fixed_binder_max_length}"
            )

        return batch["token_mask"].shape[1] + (
            self._fixed_binder_max_length - binder_length
        )

    def _maybe_pad_batch_for_model(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pad denoiser-facing tensors to a fixed per-problem shape when configured."""
        fixed_n_token = self._get_fixed_n_token(batch)
        if fixed_n_token is None:
            return batch
        return pad_tensor_features_to_n_token(batch, fixed_n_token)

    def _crop_sample_outputs(
        self,
        outputs: Dict[str, torch.Tensor] | None,
        real_n_token: int,
    ) -> Dict[str, torch.Tensor] | None:
        """Crop sampled outputs back to the real token length after padded inference."""
        if outputs is None:
            return None
        cropped = {}
        for key, value in outputs.items():
            if key in {"xl", "ai"} and isinstance(value, torch.Tensor):
                cropped[key] = value[:, :real_n_token]
            else:
                cropped[key] = value
        return cropped

    def _prepare_test_batch(
        self,
        raw_batch: Tuple[Dict[str, torch.Tensor], List[Tuple[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        """Prepare one test batch for model execution, including sample metadata."""
        batch_data = prepare_tensor_features(
            raw_batch[0], device=self.device, reduce_dim=0
        )
        if len(raw_batch) > 1 and raw_batch[1]:
            sample_name = raw_batch[1][0][0]
            batch_data["name"] = sample_name.rsplit("_", 1)[0]
            batch_data["sample_name"] = sample_name
        return batch_data

    def _warmup_compiled_sampling(
        self,
        batch_data: Dict[str, torch.Tensor],
        *,
        label: str,
        skip_sequence_decode: bool = False,
    ) -> None:
        """Warm up torch.compile on a representative inference batch."""
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if self.device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(device=self.device)

        step_size = self.sampler.n_timestep // self.sampler.n_sample_step
        first_step = 1 + ((self.sampler.n_timestep - 1) // step_size) * step_size
        batch_size = batch_data["gt_atom_positions"].shape[0]
        t = torch.full((batch_size,), first_step, dtype=torch.int, device=self.device)

        logging.info(
            "[GenieRunner] Starting torch.compile warmup "
            f"({label}, batch_size={batch_size}, n_token={batch_data['token_mask'].shape[1]})"
        )
        get_active_reporter().set_status("compile warmup")
        warmup_start_time = time.perf_counter()
        try:
            xl = torch.randn_like(batch_data["gt_atom_positions"])
            logging.info("[GenieRunner] torch.compile warmup phase: structure step")
            xl = self.sampler._step(
                model=self.model,
                batch=batch_data,
                xs=xl,
                s=t,
                step_size=step_size,
            )

            if (
                getattr(self.sampler, "predict_sequence", False)
                and xl is not None
                and not skip_sequence_decode
            ):
                seq_batch = dict(batch_data)
                for mutable_key in ("gt_restype", "cond_seq_mask"):
                    if mutable_key in seq_batch and isinstance(
                        seq_batch[mutable_key], torch.Tensor
                    ):
                        seq_batch[mutable_key] = seq_batch[mutable_key].clone()
                logging.info("[GenieRunner] torch.compile warmup phase: sequence decode")
                self.sampler._sample_sequence(self.model, seq_batch, xl)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device=self.device)

        warmup_elapsed = time.perf_counter() - warmup_start_time
        logging.info(
            "[GenieRunner] torch.compile warmup runtime "
            f"({label}, batch_size={batch_size}, n_token={batch_data['token_mask'].shape[1]}) = "
            f"{warmup_elapsed:.3f}s"
        )
        self._compile_warmup_seconds = warmup_elapsed
        next_status = (
            "sidechain"
            if getattr(self.config.dataset, "source", None) == "sidechain"
            else "generation"
        )
        get_active_reporter().set_status(next_status)

    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Execute single training step with loss computation.

        Computes MSE, FAPE, and sequence losses based on configuration.

        Args:
            batch: Tuple of (batch_config, batch_data)

        Returns:
            dict: Loss dictionary with 'loss' and component losses
        """
        batch_config, batch = batch[0], batch[1]
        batch_config = parse_batch_config(batch_config)
        batch = prepare_tensor_features(batch, reduce_dim=0)
        out = self.diffusion.training_step(self.model, batch)

        per_residue_mse_losses = mse(
            out["zl_out"], out["zl"], batch["gt_atom_mask"], aggregate=None
        )
        unweighted_mse_losses = torch.sum(
            per_residue_mse_losses * batch["gt_atom_mask"], dim=-1
        ) / torch.sum(batch["gt_atom_mask"], dim=-1)

        weighted_mse_losses = unweighted_mse_losses.clone()
        if self.config.loss.mse.reweight_by_plddt:
            w = compute_mse_weight_from_plddt(batch["plddt"])
            weighted_mse_losses = torch.sum(
                per_residue_mse_losses * w * batch["gt_atom_mask"], dim=-1
            ) / torch.sum(batch["gt_atom_mask"], dim=-1)

        loss = {
            "losses": weighted_mse_losses.clone(),
            "unweighted_mse_loss": torch.mean(unweighted_mse_losses),
            "weighted_mse_loss": torch.mean(weighted_mse_losses),
        }

        if self.config.diffusion.name == "ddpm":
            if batch_config["predict_sequence"]:
                loss = update_sequence_loss(
                    batch=batch,
                    out=out,
                    loss=loss,
                    **self.config.loss.auxiliary.sequence
                )

            loss = update_fape_loss(
                batch=batch, out=out, loss=loss, **self.config.loss.auxiliary.fape
            )

        # Postprocess loss dictionary
        loss["loss"] = torch.mean(loss["losses"])
        del loss["losses"]
        self._log(loss)

        return loss

    def test_step(
        self, batch: Tuple[Dict[str, torch.Tensor], List[Tuple[str, Any]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Execute single test/sampling step.

        Args:
            batch: Batch data for sampling

        Returns:
            dict: Sampling outputs including generated structures
        """
        batch_data = prepare_tensor_features(batch[0], reduce_dim=0)
        real_n_token = batch_data["token_mask"].shape[1]

        # Inject names so downstream components can use them.
        # 'name'        → problem name (e.g. 'il7ra'), used by ColabFoldReward
        #                 to look up problem JSON / target sequence.
        # 'sample_name' → full sample name (e.g. 'il7ra_0'), used by BeamSearch
        #                 for output paths so multiple samples of the same problem
        #                 do not overwrite each other across devices.
        if len(batch) > 1 and batch[1]:
            sample_name = batch[1][0][0]
            batch_data['name'] = sample_name.rsplit('_', 1)[0]
            batch_data['sample_name'] = sample_name

        search = getattr(self.sampler, "_beam_search", None)
        model_batch = (
            batch_data if search is not None else self._maybe_pad_batch_for_model(batch_data)
        )
        out = self.diffusion.p_sample(
            sampler=self.sampler, model=self.model, batch=model_batch
        )
        out = self._crop_sample_outputs(out, real_n_token)
        return out

    def on_test_start(self):
        """
        Prime expensive per-problem inference resources before the first test
        batch runs. This keeps template preparation / JAX warmup out of the
        per-sample scoring path when beam-search rewards are active.
        """
        self._configure_fixed_shape_inference()
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return

        search = getattr(self.sampler, "_beam_search", None)
        reward = getattr(search, "reward", None)
        prime_problem = getattr(reward, "prime_problem", None)
        if search is None or prime_problem is None:
            if self.config.compile:
                self._warmup_compiled_test_step(dm)
            return

        dataset = getattr(dm, "dataset", None)
        tasks = getattr(dataset, "tasks", None)
        if not tasks:
            return

        problem_names = list(dict.fromkeys(tasks))
        if self.config.compile and len(problem_names) != 1:
            raise RuntimeError(
                "Beam search with torch.compile requires exactly one resolved "
                "binder-design problem in the test dataset, but found "
                f"{len(problem_names)}: {problem_names}"
            )
        if len(problem_names) != 1:
            raise RuntimeError(
                "Beam search with in-process ColabFold requires exactly one "
                f"problem in the test dataset, but found {len(problem_names)}: "
                f"{problem_names}"
            )
        logging.info(
            f"[GenieRunner] Priming ColabFold for {len(problem_names)} problem(s) before test start"
        )
        for problem_name in problem_names:
            prime_problem(problem_name)

        if self.config.compile:
            logging.info("[GenieRunner] Beam search torch.compile warmup is enabled")
            set_fixed_token_padding = getattr(search, "set_fixed_token_padding", None)
            load_problem_context = getattr(reward, "_load_problem_context", None)
            if set_fixed_token_padding is not None and load_problem_context is not None:
                _, problem_info = load_problem_context(problem_names[0])
                binder_max_length = problem_info.get("binder_max_length", None)
                if binder_max_length is None:
                    raise RuntimeError(
                        "Beam search with torch.compile requires binder_max_length in "
                        f"the selected problem config, but it is missing for {problem_names[0]}"
                    )
                set_fixed_token_padding(int(binder_max_length))
                logging.info(
                    "[GenieRunner] Enabled fixed-shape denoiser padding for torch.compile "
                    f"(problem={problem_names[0]}, binder_max_length={int(binder_max_length)})"
                )
            self._warmup_compiled_beam_search(dm, search)

    def _warmup_compiled_test_step(self, dm) -> None:
        """Warm up compiled inference for the regular non-beam test path."""
        loader = dm.test_dataloader()
        try:
            warmup_batch = next(iter(loader))
        except StopIteration:
            return

        batch_data = self._prepare_test_batch(warmup_batch)
        batch_data = self._maybe_pad_batch_for_model(batch_data)
        self._warmup_compiled_sampling(batch_data, label="regular")

    def _warmup_compiled_beam_search(self, dm, search) -> None:
        """
        Warm up torch.compile on the actual beam-search inference shape before
        test_step so compilation cost is paid outside the timed per-sample path.
        """
        loader = dm.test_dataloader()
        try:
            warmup_batch = next(iter(loader))
        except StopIteration:
            return

        batch_data = self._prepare_test_batch(warmup_batch)
        maybe_pad_batch_for_model = getattr(search, "_maybe_pad_batch_for_model", None)
        if maybe_pad_batch_for_model is not None:
            batch_data = maybe_pad_batch_for_model(batch_data)

        beam_width = search.beam_width
        beam_batch = search._tile_batch(batch_data, beam_width)
        skip_sequence_decode = False
        should_skip_sequence_sampling = getattr(
            search, "_should_skip_sequence_sampling", None
        )
        if should_skip_sequence_sampling is not None:
            skip_sequence_decode = bool(should_skip_sequence_sampling())
            if skip_sequence_decode:
                logging.info(
                    "[GenieRunner] Skipping beam-search sequence compile warmup "
                    "because the reward redesigns sequences with ProteinMPNN"
                )
        self._warmup_compiled_sampling(
            beam_batch,
            label=f"beam_width={beam_width}",
            skip_sequence_decode=skip_sequence_decode,
        )

    def on_test_batch_start(self, batch, batch_idx: int, dataloader_idx: int = 0):
        del batch, batch_idx, dataloader_idx
        self._test_batch_start_time = time.perf_counter()

    def on_test_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Tuple[Dict[str, torch.Tensor], List[Tuple[str, Any]]],
        batch_idx: int,
    ):
        """
        Postprocess and save outputs after each test batch.

        Skipped when beam search is active — beam search writes its own
        structured output (pdbs/, index.csv, search/) during run().

        Args:
            outputs: Sampling outputs from test_step
            batch: Original batch data
            batch_idx: Batch index
        """
        batch_elapsed = 0.0
        if self._test_batch_start_time is not None:
            batch_elapsed = time.perf_counter() - self._test_batch_start_time
        batch_sample_count = len(batch[1]) if len(batch) > 1 else 0
        batch_output_count = batch_sample_count
        if outputs is not None and isinstance(outputs, dict):
            xl = outputs.get("xl")
            if isinstance(xl, torch.Tensor):
                batch_output_count = int(xl.shape[0])
        self._test_batch_seconds_total += batch_elapsed
        self._test_sample_count += batch_sample_count
        self._test_output_count += batch_output_count
        if self._first_batch_seconds is None:
            self._first_batch_seconds = batch_elapsed
            self._first_batch_sample_count = batch_sample_count

        if outputs is not None and not hasattr(self.config.inference, 'search'):
            postprocess(
                source=self.config.dataset.source,
                outputs=outputs,
                batch=batch,
                outdir=self.config.io.outdir,
            )

    def on_test_end(self):
        """
        Cleanup and consolidate outputs after all test batches complete.

        For motif scaffolding, creates scaffold_info.tsv and removes config dirs.
        """
        self._write_generation_stats()

        if self.trainer is not None and self.trainer.is_global_zero:
            if self.config.dataset.source == "motif":
                for config_dir in glob.glob(
                    os.path.join(self.config.io.outdir, "*", "configs")
                ):

                    # Parse configuration files
                    basedir = config_dir.rsplit("/", 1)[0]
                    problem = basedir.rsplit("/", 1)[-1]
                    rows_by_name = {}

                    # Merge with existing TSV so multi-job runs don't lose earlier samples
                    tsv_path = os.path.join(basedir, "scaffold_info.tsv")
                    if os.path.exists(tsv_path):
                        with open(tsv_path) as f:
                            header = f.readline()
                            for line in f:
                                parts = line.rstrip("\n").split("\t")
                                if len(parts) == 3:
                                    rows_by_name[parts[1]] = parts

                    for filepath in glob.glob(os.path.join(config_dir, "*.txt")):
                        name = filepath.split("/")[-1].split(".")[0]
                        with open(filepath) as file:
                            config = file.readlines()[0].strip()
                            rows_by_name[name] = [problem, name, config]

                    # Save
                    columns = ["problem", "name", "config"]
                    with open(tsv_path, "w") as file:
                        file.write("\t".join(columns) + "\n")
                        for row in sorted(rows_by_name.values(), key=lambda r: r[1]):
                            file.write("\t".join(row) + "\n")

                    # Clean up
                    shutil.rmtree(config_dir)

            if hasattr(self.config.inference, 'search'):
                outdir = self.config.io.outdir
                # Build one aggregate rewards.csv per problem from per-sample summary.json files.
                # summary.json is written by each beam search run; by on_test_end all devices
                # have finished so all files are present on the shared filesystem.
                for jobs_dir in sorted(glob.glob(os.path.join(outdir, '*', 'jobs'))):
                    problem_dir = os.path.dirname(jobs_dir)
                    rows = []
                    for summary_file in sorted(glob.glob(os.path.join(jobs_dir, '*', 'summary.json'))):
                        with open(summary_file) as f:
                            rows.append(json.load(f))
                    if rows:
                        # Each summary.json is a list of K dicts; flatten and sort by global index
                        flat_rows = [entry for sample_rows in rows for entry in sample_rows]
                        flat_rows.sort(key=lambda r: r['index'])
                        csv_path = os.path.join(problem_dir, 'rewards.csv')
                        with open(csv_path, 'w', newline='') as f:
                            writer = csv.DictWriter(
                                f, fieldnames=['index', 'sample', 'rank', 'reward', 'i_pae', 'iptm', 'ptm', 'plddt'])
                            writer.writeheader()
                            writer.writerows(flat_rows)

    def _write_generation_stats(self) -> None:
        stats_dir = os.path.join(self.config.io.outdir, "generation_stats")
        os.makedirs(stats_dir, exist_ok=True)
        stage_name = "sidechain" if self.config.dataset.source == "sidechain" else "main"
        filepath = os.path.join(stats_dir, f"{stage_name}_rank_{int(self.global_rank)}.json")
        payload = {
            "stage": stage_name,
            "rank": int(self.global_rank),
            "sample_count": int(self._test_sample_count),
            "output_count": int(self._test_output_count),
            "batch_seconds_total": float(self._test_batch_seconds_total),
            "first_batch_seconds": float(self._first_batch_seconds or 0.0),
            "first_batch_sample_count": int(self._first_batch_sample_count),
            "compile_warmup_seconds": float(self._compile_warmup_seconds),
        }
        search = getattr(self.sampler, "_beam_search", None)
        if search is not None:
            payload["search_name"] = "beam"
            payload["beam_width"] = int(search.beam_width)
            payload["n_output"] = int(search.n_output)
        with open(filepath, "w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure Adam optimizer for training.

        Returns:
            torch.optim.Adam: Optimizer with learning rate from config
        """
        return Adam(self.model.parameters(), lr=self.config.optimization["lr"])

    def _log(self, loss: Dict[str, torch.Tensor]):
        """
        Log all loss components to Lightning logger.

        Args:
            loss: Dictionary of loss values to log
        """
        for key in loss:
            self.log(key, loss[key], on_step=True, on_epoch=True)

    def _load_checkpoint(self, filepath: str):
        """
        Load model checkpoint with backward compatibility handling.

        Handles legacy Genie 2 checkpoints by removing _orig_mod prefix
        and updating weight names.

        Args:
            filepath: Path to checkpoint file
        """

        # Load state dict
        checkpoint = torch.load(filepath, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        # For compatiability with Genie 2 (legacy)
        # Replace _orig_mod prefix (due to model compilation) and weight name
        updated_state_dict = OrderedDict()
        for k in state_dict:
            updated_k = k.replace("_orig_mod.", "").replace(
                ".linear_motif_template.", ".linear_cond_template."
            )
            updated_state_dict[updated_k] = state_dict[k]

        # Load model checkpoint
        self.load_state_dict(updated_state_dict)

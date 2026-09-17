import gc
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MeanMetric, MetricCollection

from atlasfold.model.model_multimer import AtlasFoldMultimerConfig
from atlasfold.model.network.diffusion_head import SamplingConfig
from atlasfold.train import losses
from atlasfold.train.monomer import train_module as monomer_train_module
from atlasfold.train.multimer import train_alignment, validation_metrics
from atlasfold.train.multimer.model_train import AtlasFoldForTrain
from atlasfold.train.utils.ema import ExponentialMovingAverage
from atlasfold.utils.geometry import metrics as structure_metrics


def to_dict(config: DictConfig) -> dict:
    """Convert a DictConfig to a standard Python dictionary."""
    return OmegaConf.to_container(config, resolve=True)


class TrainingModule(monomer_train_module.TrainingModule):
    """Lightning module for multimer training and validation."""

    def __init__(self, config: DictConfig):
        pl.LightningModule.__init__(self)
        self.global_config: DictConfig = config
        self.config = config.train
        self.training_config = self.config.training
        self.validation_config = self.config.validation
        self.optimizer_config = self.config.optimizer
        self.loss_config = self.config.loss
        self.compile_config = self.config.compile

        self.save_hyperparameters(to_dict(self.global_config))

        self.train_trunk = self.training_config.train_trunk
        self.train_diffusion_head = self.training_config.train_diffusion_head
        self.train_confidence_head = self.training_config.train_confidence_head
        self.train_pde_head = self.training_config.train_pde_head
        self.train_pae_head = self.training_config.train_pae_head

        model_cfg = OmegaConf.to_object(
            OmegaConf.merge(AtlasFoldMultimerConfig, self.global_config.model)
        )
        self.model: AtlasFoldForTrain = AtlasFoldForTrain(model_cfg)
        self.model.kernel_backend = "cuequiv"

        self.freeze_submodules()
        self.setup_losses()
        self.setup_metrics()

        if self.compile_config.enabled:
            self.model.compile_train(
                mode=self.compile_config.mode, dynamic=self.compile_config.dynamic
            )

        self.ema: ExponentialMovingAverage = ExponentialMovingAverage(
            self.model,
            decay=self.optimizer_config.ema_decay,
            submodules_to_ignore=self.optimizer_config.ema_ignore_params,
            submodules_to_update=self.optimizer_config.ema_update_params,
        )
        self.stored_params: dict[str, torch.Tensor] | None = None

        rng = np.random.default_rng(seed=42)
        self.recycles_per_step: np.ndarray = rng.integers(
            0, self.training_config.num_recycles + 1, size=1000_000
        )
        self.last_lr_step: int = -1

    def freeze_submodules(self):
        additional_groups = ("pde_head",) if self.train_pde_head is False else ()
        super().freeze_submodules(additional_groups)

    def setup_losses(self):
        super().setup_losses()
        confidence_loss_config = self.loss_config.confidence_loss
        self.pde_loss = losses.confidence.PDELoss(**confidence_loss_config["pde_loss"])

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if not isinstance(batch, dict) or "full_label" not in batch:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        return {
            "feat": super().transfer_batch_to_device(
                batch["feat"], device, dataloader_idx
            ),
            "label": super().transfer_batch_to_device(
                batch["label"], device, dataloader_idx
            ),
            "loss_mask": super().transfer_batch_to_device(
                batch["loss_mask"], device, dataloader_idx
            ),
            "full_label": super().transfer_batch_to_device(
                batch["full_label"], device, dataloader_idx
            ),
        }

    def setup_metrics(self):
        val_metrics = {"distogram/loss": MeanMetric()}
        metric_names = [
            "complex/rmsd",
            "complex/lddt",
            "complex/lddt-ca",
            "chain/rmsd",
            "chain/lddt",
            "chain/lddt-ca",
            "interface/lddt",
            "interface/lddt-ca",
        ]
        for prefix in ["top", "avg", "rank"]:
            for name in metric_names:
                val_metrics[f"{prefix}/{name}"] = MeanMetric()
        self.val_metrics = MetricCollection(val_metrics, prefix="val/")

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        idx = self.global_step % len(self.recycles_per_step)
        num_recycles = int(self.recycles_per_step[idx])

        model_out: dict[str, Any] = self(
            batch=batch["feat"],
            label=batch["label"],
            num_recycles=num_recycles,
            mode="train",
        )

        device_type = batch["feat"]["aatype"].device.type
        with torch.autocast(
            device_type, dtype=torch.float32, enabled=(device_type == "cuda")
        ):
            loss, metrics = self.compute_losses(
                model_out=model_out,
                batch=batch["feat"],
                label=batch["label"],
                loss_mask=batch["loss_mask"],
                full_label=batch["full_label"],
            )

        for k, v in metrics.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"), sync_dist=False)

        return loss

    def compute_losses(
        self,
        model_out: dict[str, Any],
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        loss_mask: dict[str, torch.Tensor],
        full_label: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute multimer losses with full-label confidence alignment."""
        loss = torch.zeros(1, device=batch["aatype"].device, dtype=torch.float32)
        loss_weights = self.loss_weights

        metrics: dict[str, torch.Tensor] = {}
        if self.train_trunk:
            distogram_loss, distogram_metrics = self.compute_distogram_loss(
                model_out["distogram"], batch, label
            )
            loss += loss_weights["distogram"] * distogram_loss
            metrics |= {f"distogram/{k}": v for k, v in distogram_metrics.items()}

        if self.train_diffusion_head:
            diffusion_loss, diffusion_metrics = self.compute_diffusion_loss(
                model_out["diffusion"], batch, label
            )
            loss += loss_weights["diffusion"] * diffusion_loss
            metrics |= {f"diffusion/{k}": v for k, v in diffusion_metrics.items()}

        if self.train_confidence_head:
            confidence_out = model_out["confidence"]
            x_pred = confidence_out["mini_rollout"]["sample_coords"]
            do_permutation = loss_mask["confidence"].tolist()

            aligned_coords_list = []
            aligned_mask_list = []
            for b_i in range(x_pred.shape[0]):
                x_pred_i = x_pred[b_i]
                feat_i = {k: v[b_i] for k, v in batch.items()}
                label_i = {k: v[b_i] for k, v in label.items()}
                x_gt, mask = train_alignment.get_aligned_gt_structure(
                    x_pred=x_pred_i,
                    feat=feat_i,
                    label=label_i,
                    full_label=full_label[b_i],
                    permutation=do_permutation[b_i],
                )
                aligned_coords_list.append(x_gt)
                aligned_mask_list.append(mask)

            x_gt = torch.stack(aligned_coords_list, dim=0)
            mask = torch.stack(aligned_mask_list, dim=0)
            aligned_label = {"coordinates": x_gt, "resolved_mask": mask}

            confidence_loss, confidence_metrics = self.compute_confidence_loss(
                confidence_out, batch, aligned_label, loss_mask["confidence"]
            )
            loss += loss_weights["confidence"] * confidence_loss
            metrics |= {f"confidence/{k}": v for k, v in confidence_metrics.items()}

            with torch.no_grad():
                rmsd = structure_metrics.compute_rmsd_atom14(x_pred, x_gt, mask)
                lddt_ca = structure_metrics.compute_lddt_ca(x_pred, x_gt, mask)
                metrics |= {"mini_rollout/rmsd": rmsd.mean()}
                metrics |= {"mini_rollout/lddt-ca": lddt_ca.mean()}

        metrics["loss"] = loss.detach()
        return loss, metrics

    def compute_confidence_loss(
        self,
        pred: dict[str, Any],
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        L_confidence, metrics = super().compute_confidence_loss(
            pred, batch, label, loss_mask
        )
        if self.train_pde_head:
            w = loss_mask.float()
            n_valid_samples = w.sum().clamp(1)
            L_pde = self.pde_loss(
                logits=pred["pde"]["logits"],
                bin_centers=pred["pde"]["bin_centers"],
                x_pred=pred["mini_rollout"]["sample_coords"],
                x_gt=label["coordinates"],
                mask=label["resolved_mask"],
                cbeta_idx=batch["pseudo_beta"],
            )
            L_pde = (L_pde * w).sum() / n_valid_samples
            metrics["pde_loss"] = L_pde.detach()
            L_confidence = L_confidence + L_pde
        metrics["loss"] = L_confidence.detach()
        return L_confidence, metrics

    def validation_step(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
        batch_idx: int,
    ):
        feat, label = batch["feat"], batch["label"]
        val_config = self.validation_config
        seed = batch_idx
        try:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                sample_out: dict[str, torch.Tensor] = self(
                    batch=feat,
                    label=None,
                    num_recycles=val_config.num_recycles,
                    mode="validation",
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("**WARNING**: ran out of memory, skipping batch")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            raise e

        distogram_loss = self._compute_validation_distogram_loss(sample_out, feat, label)
        self.val_metrics["distogram/loss"].update(distogram_loss)

        x_pred = sample_out["sample_coords"]  # [N, L, 14, 3]
        sanity_error = self._validation_rollout_sanity_error(sample_out, feat)
        failed = sanity_error is not None
        self.log(
            "val/rollout_sanity_failed",
            torch.tensor(float(failed), device=feat["seq_mask"].device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        if sanity_error is not None:
            print(f"**WARNING**: validation rollout sanity failed: {sanity_error}")
            return

        rank_idx = self._select_validation_rank_idx(sample_out, feat)

        metrics: dict[str, float] = validation_metrics.compute_validation_metric(
            x_pred, feat, label, rank_idx
        )

        val_metrics: MetricCollection = self.val_metrics
        for k, v in metrics.items():
            if k in val_metrics:
                val_metrics[k].update(v)

    def _compute_validation_distogram_loss(
        self,
        sample_out: dict[str, torch.Tensor],
        feat: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute distogram loss for an unbatched validation example."""
        loss = self.distogram_loss(
            logits=sample_out["distogram.logits"].unsqueeze(0),
            boundaries=sample_out["distogram.boundaries"],
            x_gt=label["coordinates"].unsqueeze(0),
            mask_gt=label["resolved_mask"].unsqueeze(0),
            cbeta_idx=feat["pseudo_beta"].unsqueeze(0),
        )
        return loss.mean()

    def _validation_rollout_sanity_error(
        self,
        sample_out: dict[str, torch.Tensor],
        feat: dict[str, torch.Tensor],
    ) -> str | None:
        if "sample_coords" not in sample_out:
            return "missing sample_coords"
        if "plddt" not in sample_out:
            return "missing plddt"

        x_pred = sample_out["sample_coords"]
        plddt = sample_out["plddt"]
        seq_mask = feat["seq_mask"].bool()
        num_samples = int(self.validation_config.num_diffusion_samples)
        length = int(seq_mask.shape[0])

        if x_pred.ndim != 4 or x_pred.shape[-2:] != (14, 3):
            return f"bad sample_coords shape {tuple(x_pred.shape)}"
        if plddt.ndim != 2:
            return f"bad plddt shape {tuple(plddt.shape)}"
        if x_pred.shape[:2] != plddt.shape:
            return (
                "sample_coords and plddt shape mismatch: "
                f"{tuple(x_pred.shape[:2])} != {tuple(plddt.shape)}"
            )
        if x_pred.shape[0] != num_samples:
            return f"expected {num_samples} samples, got {x_pred.shape[0]}"
        if x_pred.shape[1] != length:
            return f"expected length {length}, got {x_pred.shape[1]}"
        if seq_mask.sum() < 4:
            return "fewer than 4 valid residues"
        if not torch.isfinite(x_pred[:, seq_mask]).all():
            return "non-finite coordinates in valid residues"
        if not torch.isfinite(plddt[:, seq_mask]).all():
            return "non-finite plddt in valid residues"
        if torch.allclose(x_pred[:, seq_mask], torch.zeros_like(x_pred[:, seq_mask])):
            return "all valid coordinates are zero"
        if "pde" in sample_out:
            pde = sample_out["pde"]
            if pde.shape != (x_pred.shape[0], x_pred.shape[1], x_pred.shape[1]):
                return f"bad pde shape {tuple(pde.shape)}"
            if not torch.isfinite(pde[:, seq_mask][:, :, seq_mask]).all():
                return "non-finite pde in valid residues"
        return None

    def _select_validation_rank_idx(
        self,
        sample_out: dict[str, torch.Tensor],
        feat: dict[str, torch.Tensor],
    ) -> int:
        seq_mask = feat["seq_mask"].bool()
        if (
            "pde" in sample_out
            and "distogram.logits" in sample_out
            and "distogram.boundaries" in sample_out
        ):
            prob_contact = validation_metrics.compute_distogram_contact_probability(
                sample_out["distogram.logits"], sample_out["distogram.boundaries"]
            )
            global_pde = validation_metrics.compute_global_pde(
                sample_out["pde"], prob_contact, seq_mask
            )
            if torch.isfinite(global_pde).all():
                rank_idx = int(torch.argmin(global_pde))
                self.log(
                    "val/rank_global_pde",
                    global_pde[rank_idx],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=1,
                )
                return rank_idx

        plddt = sample_out["plddt"]
        mask = seq_mask.float()
        avg_plddt = (plddt * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        self.log(
            "val/rank_fallback_plddt",
            torch.tensor(1.0, device=plddt.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        return int(torch.argmax(avg_plddt))

    def on_validation_epoch_end(self):
        torch.backends.cudnn.benchmark = True
        if not self.trainer.sanity_checking:
            avg_values = self.val_metrics.compute()
            for k, v in avg_values.items():
                self.log(
                    k,
                    v,
                    prog_bar=(k == "val/rank/complex/lddt-ca"),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        self.val_metrics.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor] | None,
        num_recycles: int,
        mode: str,
    ) -> Any:
        if mode == "train":
            assert label is not None, "Label cannot be None in training mode"
            training_config = self.training_config
            sampling_config = SamplingConfig(
                num_steps=training_config.num_mini_rollout_steps
            )
            return self.model.forward_train(
                batch=batch,
                label=label,
                num_recycles=num_recycles,
                diffusion_batch_size=training_config.diffusion_batch_size,
                train_trunk=self.train_trunk,
                train_diffusion_head=self.train_diffusion_head,
                train_confidence_head=self.train_confidence_head,
                sampling_config=sampling_config,
            )

        val_config = self.validation_config
        sampling_config = SamplingConfig(num_steps=val_config.num_steps)
        return self.model.inference(
            batch,
            num_recycles=num_recycles,
            num_samples=val_config.num_diffusion_samples,
            sampling_config=sampling_config,
            return_representations=False,
        )

"""DDIM (Denoising Diffusion Implicit Model) sampler."""

import logging
import time
import torch
from typing import Dict, Optional

from genie3.generation.model.implementation.base import Denoiser
from genie3.generation.diffusion.sampler.sampler import Sampler
from genie3.generation.np.protein_constants import PROTEIN_RESTYPES


class DDIMSampler(Sampler):

    def __init__(
        self,
        n_sample_step: int,
        direction_scale: float,
        noise_scale: float,
        eta: float,
        verbose: bool,
        predict_sequence: bool = False,
        predict_sidechain: bool = False,
        sequence_decoding_order: str = "all",
    ):
        """
        Initialize DDIM sampler.

        Args:
            n_sample_step: Number of sampling steps (can be < n_timestep)
            direction_scale: Scale factor for predicted direction
            noise_scale: Scale factor for added noise
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
            verbose: Whether to save intermediate structures
            predict_sequence: Whether to predict amino acid sequence
            predict_sidechain: Whether to predict sidechain atoms
            sequence_decoding_order: Order for sequence decoding ('all', 'max_entropy', 'min_entropy', 'random')
        """
        super().__init__()
        self.n_sample_step = n_sample_step
        self.direction_scale = direction_scale
        self.noise_scale = noise_scale
        self.eta = eta
        self.verbose = verbose
        self.predict_sequence = predict_sequence
        self.predict_sidechain = predict_sidechain
        self.sequence_decoding_order = sequence_decoding_order
        self._beam_search = None

    def set_search(self, search) -> None:
        """Attach a search strategy. Called by the runner after construction."""
        self._beam_search = search

    def _setup_schedule(self, betas: torch.Tensor):
        """
        Setup DDIM sampling schedule from beta values.

        Args:
            betas: Noise schedule tensor
        """
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample(
        self, model: Denoiser, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a structure, optionally using beam search.

        Delegates to BeamSearch.run() when search_config was provided at
        construction time; otherwise falls back to the standard single-trajectory
        loop inherited from Sampler.
        """
        if self._beam_search is not None:
            return self._beam_search.run(
                sampler=self,
                model=model,
                batch=batch,
                n_timestep=self.n_timestep,
                n_sample_step=self.n_sample_step,
            )
        return super().sample(model, batch)

    def _step(
        self,
        model: Denoiser,
        batch: Dict[str, torch.Tensor],
        xs: torch.Tensor,
        s: torch.Tensor,
        step_size: int,
    ) -> torch.Tensor:
        """
        Execute single DDIM reverse step.

        Args:
            model: Denoiser model
            batch: Batch data
            xs: Current noisy sample
            s: Current timestep
            step_size: Step size for sampling

        Returns:
            torch.Tensor: Denoised sample at previous timestep
        """

        model_start_time = time.perf_counter()
        with torch.inference_mode():
            out = model(batch=batch, xl=xs, t=s / self.n_timestep)
            zs_out = xs - out["xl"]
            xs_out = (
                (xs - self.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1) * zs_out)
                / self.sqrt_alphas_cumprod[s].view(-1, 1, 1)
            ) * batch["gt_atom_mask"].unsqueeze(-1)
        model_elapsed = time.perf_counter() - model_start_time
        logging.info(
            '[DDIMSampler] _step model runtime '
            f'(batch_size={xs.shape[0]}, n_token={xs.shape[1]}, step={int(s[0].item())}) = '
            f'{model_elapsed:.3f}s'
        )

        # Sample
        if (s - step_size <= 0).all():

            xt = xs_out

        else:

            t = s - step_size
            sigma = (
                torch.sqrt(
                    (
                        self.one_minus_alphas_cumprod[t]
                        / self.one_minus_alphas_cumprod[s]
                    )
                    * (1.0 - self.alphas_cumprod[s] / self.alphas_cumprod[t])
                )
                * self.eta
            )
            direction = (
                torch.sqrt(1 - self.alphas_cumprod[t] - sigma**2).view(-1, 1, 1)
                * zs_out
            )
            xt = (
                self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * xs_out
                + self.direction_scale * direction
                + self.noise_scale * sigma.view(-1, 1, 1) * torch.randn_like(xs)
            )

        return xt

    def _sample_sequence(
        self,
        model: Denoiser,
        batch: Dict[str, torch.Tensor],
        xl: torch.Tensor,
        eps: float = 1e-10,
        T: float = 0.1,
    ) -> torch.Tensor:
        """
        Sample amino acid sequence using autoregressive or non-autoregressive decoding.

        Args:
            model: Denoiser model
            batch: Batch data
            xl: Generated coordinates
            eps: Small constant for numerical stability
            T: Temperature for softmax sampling

        Returns:
            torch.Tensor: Predicted amino acid sequence (one-hot)
        """

        # Sample time step
        batch_size = xl.shape[0]
        n_token = xl.shape[1]
        device = xl.device
        t = torch.ones((batch_size,), device=device).int()

        # Sample noise
        zl = torch.randn_like(xl) * batch["gt_atom_mask"][..., None]

        # Apply noise
        xl = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * xl
            + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * zl
        ) * batch["gt_atom_mask"][..., None]

        # Compute one-step prediction
        with torch.inference_mode():

            if self.sequence_decoding_order == "all":
                out = model(batch=batch, xl=xl, t=t / self.n_timestep)
                prob = torch.softmax(out["ai"] / T, dim=-1)
                batch["gt_restype"] = (
                    torch.nn.functional.one_hot(
                        torch.argmax(prob, dim=-1), num_classes=len(PROTEIN_RESTYPES)
                    )
                    * (1 - batch["cond_seq_mask"])[..., None]
                    + batch["gt_restype"] * batch["cond_seq_mask"][..., None]
                ) * batch["gt_atom_mask"][..., None]
                return batch["gt_restype"]

            for _ in range(n_token):
                out = model(batch=batch, xl=xl, t=t / self.n_timestep)
                prob = torch.softmax(out["ai"] / T, dim=-1)
                entropy = -(prob * (prob + eps).log()).sum(dim=-1)
                entropy_mask = (1 - batch["cond_seq_mask"]) * batch["gt_atom_mask"]
                entropy = entropy * entropy_mask

                selected_index = self._select_residue_index(entropy, entropy_mask)
                selected_mask = torch.nn.functional.one_hot(
                    selected_index, num_classes=n_token
                )
                selected_mask = selected_mask * entropy_mask
                selected_restype = (
                    torch.nn.functional.one_hot(
                        torch.argmax(prob, dim=-1), num_classes=len(PROTEIN_RESTYPES)
                    )
                    * selected_mask[..., None]
                )
                batch["gt_restype"] = batch["gt_restype"] + selected_restype
                batch["cond_seq_mask"] = batch["cond_seq_mask"] + selected_mask

        return batch["gt_restype"]

    def _select_residue_index(
        self, entropy: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Select next residue index for autoregressive sequence decoding.

        Args:
            entropy: Entropy values for each position
            mask: Valid position mask

        Returns:
            torch.Tensor: Selected residue indices
        """
        if self.sequence_decoding_order == "max_entropy":
            return torch.argmax(entropy, dim=-1)
        elif self.sequence_decoding_order == "min_entropy":
            max_entropy = entropy.max()
            entropy = entropy * mask + 10 * max_entropy * (1 - mask)
            return torch.argmin(entropy, dim=-1)
        elif self.sequence_decoding_order == "random":
            entropy = torch.randn_like(entropy)
            max_entropy = entropy.max()
            entropy = entropy * mask + 10 * max_entropy * (1 - mask)
            return torch.argmin(entropy, dim=-1)
        else:
            logging.error(
                f"Invalid sequence decoding order: {self.sequence_decoding_order}"
            )
            exit(0)

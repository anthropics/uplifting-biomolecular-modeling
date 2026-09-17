"""Folding Trunk."""

from typing import Any

import torch

from atlasfold.model.model_multimer import AtlasFold_Multimer
from atlasfold.model.network.diffusion_head import DiffusionHead, SamplingConfig
from atlasfold.utils.geometry.random_augment import center_random_augmentation_atom14
from atlasfold.utils.torch_utils import expand_dim, get_context_dtype


class AtlasFoldForTrain(AtlasFold_Multimer):
    def compile_train(self, **kwargs) -> None:
        """Compile the functions used in the training step."""
        self.__forward_trunk = torch.compile(self.__forward_trunk, **kwargs)
        self.__forward_distogram = torch.compile(self.__forward_distogram, **kwargs)
        self.__forward_diffusion = torch.compile(self.__forward_diffusion, **kwargs)
        self.__forward_confidence = torch.compile(self.__forward_confidence, **kwargs)
        self.__forward_denoise_step = torch.compile(self.__forward_denoise_step, **kwargs)

    def get_module_groups(self) -> dict[str, list[torch.nn.Module | torch.nn.Parameter]]:
        return {
            "lm": [self.lm],
            "trunk": [
                # Initialization
                self.s_init,
                self.z_init,
                self.z_rel_pos,
                # Recycling
                self.recycle_s,
                self.recycle_z,
                # LM stack
                self.lm_layer_weights,
                self.layernorm_lm_emb,
                self.lm_emb_to_s_lm,
                self.proj_lm_attn,
                self.lm_attn_to_z_lm,
                self.lm_stack,
                self.proj_s_lm,
                self.proj_z_lm,
                self.template_module,
                self.main_stack,
            ],
            "distogram_head": [self.distogram_head],
            "diffusion_head": [self.diffusion_head],
            "confidence_head": [self.confidence_head],
            "pae_head": [self.confidence_head.pae_head],
            "pde_head": [self.confidence_head.pde_head],
        }

    def get_module_group_names(self) -> dict[str, list[str]]:
        return {
            "lm": ["lm"],
            "trunk": [
                # Initialization
                "s_init",
                "z_init",
                "z_rel_pos",
                # Recycling
                "recycle_s",
                "recycle_z",
                # LM stack
                "lm_layer_weights",
                "layernorm_lm_emb",
                "lm_emb_to_s_lm",
                "proj_lm_attn",
                "lm_attn_to_z_lm",
                "lm_stack",
                "proj_s_lm",
                "proj_z_lm",
                # Template module
                "template_module",
                # Main stack
                "main_stack",
            ],
            "distogram_head": ["distogram_head"],
            "diffusion_head": ["diffusion_head"],
            "confidence_head": ["confidence_head"],
            "pae_head": ["confidence_head.pae_head"],
            "pde_head": ["confidence_head.pde_head"],
        }

    # ==================================================
    # Forward functions for training step.
    # ==================================================
    def forward_train(
        self,
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        num_recycles: int,
        diffusion_batch_size: int,
        train_trunk: bool,
        train_diffusion_head: bool,
        train_confidence_head: bool,
        sampling_config: SamplingConfig,
    ) -> dict[str, Any]:
        bs = batch["aatype"].shape[0]

        # Return dictionary for training outputs.
        out: dict[str, Any] = {}

        mask = batch["seq_mask"]
        B, L = mask.shape
        device = mask.device
        dtype = get_context_dtype()

        # Compute positional encodings
        self.compute_rel_pos_encoding(batch)

        # Recycling iterations with stochastic masking during training.
        s_prev = torch.zeros(B, L, self.channel_s, device=device, dtype=dtype)
        z_prev = torch.zeros(B, L, L, self.channel_z, device=device, dtype=dtype)
        for i in range(0, num_recycles + 1):
            enable_grad = train_trunk and self.training and i == num_recycles
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()
                s, z = self.__forward_trunk(
                    batch, s_prev, z_prev, mask, train=enable_grad
                )
            s_prev, z_prev = s, z
        s, z = s.float(), z.float()

        # Return distogram logits
        if train_trunk:
            distogram_out = self.__forward_distogram(z)
            out["distogram"] = distogram_out

        # Return diffusion head outputs
        if train_diffusion_head:
            # Diffusion conditioning.
            p = torch.rand(bs, device=s.device)
            use_cond = p < 0.8
            _s = use_cond.view(bs, 1, 1) * s
            _z = use_cond.view(bs, 1, 1, 1) * z
            with torch.autocast(s.device.type, dtype=torch.float32, enabled=True):
                diffusion_out = self.__forward_diffusion(
                    batch, label, _s, _z, diffusion_batch_size
                )
            out["diffusion"] = diffusion_out

        # Return confidence head outputs
        if train_confidence_head:
            s, z = s.detach(), z.detach()
            with (
                torch.no_grad(),
                torch.autocast(s.device.type, dtype=torch.float32, enabled=True),
            ):
                # Sample the structure for confidence head training.
                sample_coords = self.__forward_mini_rollout(batch, s, z, sampling_config)

            # Select the confidence head conditioning.
            p = torch.rand(bs, device=s.device)
            use_cond = p < 0.8
            _s = s
            _z = use_cond.view(bs, 1, 1, 1) * z

            confidence_out = self.__forward_confidence(batch, _s, _z, sample_coords)
            confidence_out["mini_rollout"] = {"sample_coords": sample_coords}

            # Remove the sample dimension
            # [B, 1, ...] -> [B, ...]
            for k, v in confidence_out.items():
                for kk, vv in v.items():
                    if kk in ("logits", "sample_coords"):
                        assert vv.shape[1] == 1, (
                            f"Expected sample dimension to be 1, but got {vv.shape}"
                            f"for key {k}/{kk}"
                        )
                        confidence_out[k][kk] = vv.squeeze(1)

            out["confidence"] = confidence_out

        return out

    def __forward_trunk(
        self,
        batch: dict[str, torch.Tensor],
        s_prev: torch.Tensor,
        z_prev: torch.Tensor,
        mask: torch.Tensor,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        s = self.s_init(batch["aatype"])
        a, b = self.z_init(batch["aatype"]).chunk(2, dim=-1)
        z = a[..., :, None, :] + b[..., None, :, :]
        z += self.z_rel_pos(batch["seq_rel_pos"])

        # For training stability.
        s, z = s.float(), z.float()

        # Recycling embedding
        s = s + self.recycle_s(s_prev)
        z = z + self.recycle_z(z_prev)

        # Run LM module with stochastic masking.
        mlm_mask = self.sample_mlm_mask(batch, 0.15)
        s_lm, z_lm = self.run_lm_embedder(batch, mlm_mask, train)
        s = s + self.proj_s_lm(s_lm)
        z = z + self.proj_z_lm(z_lm)

        z = z + self.template_module(batch, z, mask, self.kernel_backend)

        # Run main trunk
        s, z = self.main_stack(s, z, mask, self.kernel_backend)
        return s, z

    def sample_mlm_mask(
        self,
        batch: dict[str, torch.Tensor],
        prob: float | torch.Tensor,
        synchronized: bool = True,
    ) -> torch.Tensor:
        """Sample a random MLM mask for the input batch.
        NOTE: synchronized masking is used for inference to ensure that
        the prediction is not changed by the batch size or the order of
        the sequences in the batch.
        """
        input_ids = batch["lm.input_ids"]  # [B, S]
        B, S = input_ids.shape
        shape = (1, S) if synchronized else (B, S)
        return torch.rand(shape, device=input_ids.device) < prob

    def __forward_distogram(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.distogram_head(z)

    def __forward_diffusion(
        self,
        batch: dict[str, torch.Tensor],
        label: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step for the structure module.
        See Section 5 of EDM paper.
        """
        B, N = s.shape[0], diffusion_batch_size

        model: DiffusionHead = self.diffusion_head

        # Sample noise levels
        nt = torch.randn((B, N), dtype=torch.float32, device=s.device)
        t_hat = model.sigma_data * torch.exp(-1.2 + 1.5 * nt)

        # Repeat label coords with random augmentation for each diffusion training input.
        label_coords = label["coordinates"]  # [B, L, 14, 3]
        resolved_mask = label["resolved_mask"]  # [B, L, 14]
        x_gt = expand_dim(label_coords, N, dim=1)  # [B, N, L, 14, 3]
        mask = expand_dim(resolved_mask, N, dim=1)  # [B, N, L, 14]

        # Random augmentation
        x_gt = center_random_augmentation_atom14(x_gt, mask)

        # Add noise
        noise = torch.randn_like(x_gt)  # [B, N, L, 14, 3]
        x_noisy = x_gt + t_hat.view(B, N, 1, 1, 1) * noise
        x_noisy = x_noisy * mask[..., None]

        # Forward pass through the score model
        _t_hat = t_hat.view(B, N, 1, 1, 1)  # [B, N, 1, 1, 1]
        c_in = model.c_in(_t_hat)
        c_skip = model.c_skip(_t_hat)
        c_out = model.c_out(_t_hat)
        loss_weights = 1 / (model.c_out(t_hat) ** 2 + 1e-10)  # [B, N]

        # DiT Conditioning
        c_noise = model.c_noise(t_hat)
        single_cond = model.single_conditioning(batch, s, c_noise)  # [B, N, L, c_s]
        pair_bias = model.pair_conditioning(batch, z)  # [B, L, L, Nblock, Nhead]

        # Input EDM conditioning
        r_noisy = c_in * x_noisy  # [B, N, L, 14, 3]
        # Forward pass through the score model
        r_update = model.score_model(batch, r_noisy, single_cond, pair_bias)
        # Output EDM conditioning
        x_out = c_skip * x_noisy + c_out * r_update

        return {
            "x_noisy": x_noisy,
            "x_out": x_out,
            "x_gt": x_gt,
            "resolved_mask": resolved_mask,
            "loss_weights": loss_weights,
        }

    def __forward_mini_rollout(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        config: SamplingConfig,
    ) -> torch.Tensor:
        """Perform a mini rollout for the confidence head training."""

        device = s.device

        B, L = batch["aatype_int"].shape
        N = 1
        mask = batch["atom14_mask"].unsqueeze(1)  # (B, 1, L, 14)

        def sample_noise() -> torch.Tensor:
            """Sample noise with synchronized randomness across different inputs.
            This ensures that the resulting coordinates are the same regardless of
            batch size.
            """
            return torch.randn(
                size=(1, N, L, 14, 3), device=device, dtype=torch.float32
            ).expand(B, -1, -1, -1, -1)  # (B, N, L, 14, 3)

        # Get noise schedule
        sigmas: list[float] = config.get_sigmas()
        sigma_0 = sigmas[0]
        x = sigma_0 * sample_noise()  # (B, N, L, 14, 3)
        x.masked_fill_(~mask[..., None], 0.0)  # apply atom mask

        diffusion_head = self.diffusion_head

        # Compute time-independent variables
        # Algorithm 20 Line 1: DiffusionConditioning
        pair_bias = diffusion_head.pair_conditioning(batch, z)
        del z

        # Gradually denoise
        for step in range(1, config.num_steps + 1):
            # Apply centering and random augmentation.
            x = diffusion_head.random_augmentation(x, mask)

            sigma_tm, sigma_t = sigmas[step - 1], sigmas[step]
            gamma = config.gamma_0 * (sigma_t > config.gamma_min)
            t_hat = torch.tensor(sigma_tm * (1 + gamma), device=device)

            # Add noise
            noise_var = config.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = noise_var.sqrt() * sample_noise()  # (B, N, L, 14, 3)
            eps.masked_fill_(~mask[..., None], 0.0)  # apply atom mask
            x_noisy = x + eps

            # Denoise
            x_denoised = self.__forward_denoise_step(batch, s, pair_bias, x_noisy, t_hat)
            delta = (x_noisy - x_denoised) / t_hat
            dt = sigma_t - t_hat
            x = x_noisy + config.step_scale * dt * delta
        return x  # [B, 1, L, 14, 3]

    def __forward_denoise_step(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        pair_bias: torch.Tensor,
        x_noisy: torch.Tensor,
        t_hat: torch.Tensor,
    ) -> torch.Tensor:
        c_noise = self.diffusion_head.c_noise(t_hat).view(1, 1)
        single_cond = self.diffusion_head.single_conditioning(batch, s, c_noise)
        return self.diffusion_head.inference_step(
            batch, x_noisy, t_hat, single_cond, pair_bias
        )

    def __forward_confidence(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        sample_coords: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return self.confidence_head(batch, s, z, sample_coords, self.kernel_backend)

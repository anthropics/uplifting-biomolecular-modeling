import dataclasses
import math
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn

from atlasfold.model.network.atom_attention import AtomDecoder, AtomEncoder
from atlasfold.model.network.diffusion_transformer import (
    DiffusionTransformerStack,
    PairConditioning,
    SingleConditioning,
)
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias
from atlasfold.utils.geometry.random_augment import do_centering, random_rotations_torch

SIGMA_DATA = 16.0

_ScalarOrTensor = TypeVar("_ScalarOrTensor", float, torch.Tensor)


@dataclasses.dataclass(kw_only=True)
class SamplingConfig:
    num_steps: int = 200
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    rho: float = 7
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5

    # Optional configurations
    chunk_size: int | None = 5  # If None, no chunking is applied.

    def get_sigmas(self) -> list[float]:
        """Get the noise schedule."""
        steps = np.linspace(0, 1, self.num_steps + 1)
        inv_rho = 1 / self.rho
        sigma_max_pow = self.sigma_max**inv_rho
        sigma_min_pow = self.sigma_min**inv_rho
        sigmas = sigma_max_pow + steps * (sigma_min_pow - sigma_max_pow)
        sigmas = (sigmas**self.rho) * SIGMA_DATA
        return sigmas.tolist()


class DiffusionModule(nn.Module):
    def __init__(
        self,
        channel_a: int = 768,
        channel_atom: int = 96,
        channel_atompair: int = 14,
        channel_cond: int = 384,
        num_heads: int = 16,
        num_blocks: int = 12,
        num_atom_heads: int = 2,
        num_atom_blocks: int = 2,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_atom : int
            The atom representation dimension.
        channel_atompair : int
            The atom relative position encoding dimension.
        channel_s : int
            The single conditioning dimension.
        num_heads : int
            The number of attention heads.
        num_blocks : int
            The number of transformer blocks.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint for gradient checkpointing,
            by default None.
        """
        super().__init__()
        self.channel_a: int = channel_a
        self.num_blocks: int = num_blocks
        self.channel_cond: int = channel_cond
        self.num_heads: int = num_heads

        # Atom attention encoder
        self.atom_encoder = AtomEncoder(
            channel_atom, channel_atompair, num_atom_heads, num_atom_blocks
        )

        # Global transformer stack
        self.proj_q_to_a = nn.Sequential(
            LinearNoBias(channel_atom, channel_a, init="default"),
            nn.ReLU(),
        )

        self.proj_s_to_a = nn.Sequential(
            LayerNorm(channel_cond, create_offset=False),
            LinearNoBias(channel_cond, channel_a, init="final", precision=32),
        )

        self.diffusion_transformer = DiffusionTransformerStack(
            channel_a=channel_a,
            channel_cond=channel_cond,
            num_blocks=num_blocks,
            num_heads=num_heads,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        # Atom attention decoder
        self.proj_a_to_q = nn.Sequential(
            LayerNorm(channel_a, create_offset=False),
            LinearNoBias(channel_a, 14 * channel_atom, init="default"),
        )

        self.atom_decoder = AtomDecoder(
            channel_atom, channel_atompair, num_atom_heads, num_atom_blocks
        )

    # === Main forward function for training === #
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        r_noisy: torch.Tensor,
        single_cond: torch.Tensor,
        pair_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Single diffusion step forward pass

        See Section 3.7 Algorithm 20 Diffusion Module in AlphaFold3 SI

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch.
        r_noisy: torch.Tensor
            The noisy atom positions, shape [B, N, L, 14, 3],
            where N is number of diffusion samples.
        single_cond: torch.Tensor
            The single conditioning, shape [B, N, L, c_s].
        pair_bias: torch.Tensor
            The pair bias for the diffusion transformer, shape [Nblock, B, L, L, Nhead].

        Returns
        -------
        r_update : torch.Tensor
            The scaled updated atom positions, shape [B, N, L, 14, 3].
        """
        # Line 3: Atom encoder
        with torch.autocast(r_noisy.device.type, enabled=False):
            q, c = self.atom_encoder(batch, r_noisy)

        # Algorithm 5 Line 16
        num_atoms = batch["atom14_mask"].float().sum(-1)  # [B, L]
        a = self.proj_q_to_a(q).sum(dim=-2)  # [B, N, L, c_a]
        a = a / num_atoms[:, None, :, None].clamp(1.0)

        # Global transformer stack
        mask = batch["seq_mask"]
        a = a.float()
        # Line 4: Full self-attention on token level
        a = a + self.proj_s_to_a(single_cond)  # [B, N, L, c_a]

        # Line 5: Diffusion transformer stack
        a = self.diffusion_transformer(
            a,  # [B, N, L, c_a]
            mask=mask.unsqueeze(1),  # [B, 1, L]
            single_cond=single_cond,  # [B, N, L, c_s]
            pair_bias=pair_bias.unsqueeze(2),  # [Nblock, B, 1, L, L, Nhead]
        )  # [B, N, L, c_a]

        # Atom decoder
        # Algorithm 6 Line 1
        q = q + self.proj_a_to_q(a).unflatten(-1, (14, -1))

        # Line 7: Atom decoder
        with torch.autocast(r_noisy.device.type, enabled=False):
            r_update = self.atom_decoder(batch, q, c)

        return r_update


class DiffusionHead(nn.Module):
    def __init__(
        self,
        channel_a: int = 768,
        channel_atom: int = 96,
        channel_cond: int = 384,
        channel_s: int = 768,
        channel_z: int = 192,
        num_heads: int = 16,
        num_blocks: int = 12,
        num_atom_heads: int = 2,
        num_atom_blocks: int = 2,
        seq_rel_pos_bins: int = 73,
        atom_rel_pos_bins: int = 14,
        # For training
        blocks_per_ckpt: int | None = None,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_atom : int
            The atom representation dimension.
        channel_cond : int
            The conditioning dimension for the diffusion module.
        channel_s : int
            The single conditioning dimension.
        channel_z : int
            The pair conditioning dimension.
        num_heads : int
            The number of attention heads.
        num_blocks : int
            The number of transformer blocks.
        num_atom_enc_heads : int
            The number of attention heads in the atom encoder.
        num_atom_enc_blocks : int
            The number of blocks in the atom encoder.
        num_atom_dec_heads : int
            The number of attention heads in the atom decoder.
        num_atom_dec_blocks : int
            The number of blocks in the atom decoder.
        seq_rel_pos_bins : int
            The number of bins for the sequence relative positional encoding.
        atom_rel_pos_bins : int
            The number of bins for the atom relative positional encoding.

        # For training
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint for gradient checkpointing,
            by default None.
        """
        super().__init__()
        # Diffision pre-conditioning
        self.sigma_data: float = SIGMA_DATA

        # Diffusion conditioning
        self.single_conditioning = SingleConditioning(
            channel_s, channel_cond, dim_fourier=256
        )
        self.pair_conditioning = PairConditioning(
            channel_z, (num_blocks, num_heads), channel_rel_pos=seq_rel_pos_bins
        )
        # Diffusion module
        self.score_model = DiffusionModule(
            channel_a=channel_a,
            channel_atom=channel_atom,
            channel_atompair=atom_rel_pos_bins,
            channel_cond=channel_cond,
            num_heads=num_heads,
            num_blocks=num_blocks,
            num_atom_heads=num_atom_heads,
            num_atom_blocks=num_atom_blocks,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    # ============================================================
    # EDM Pre-conditioning
    # ============================================================
    def c_skip(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        return (self.sigma_data**2) / (t_hat**2 + self.sigma_data**2)

    def c_out(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        _sqrt = lambda x: math.sqrt(x) if isinstance(x, float) else torch.sqrt(x)  # noqa
        return t_hat * self.sigma_data / _sqrt(self.sigma_data**2 + t_hat**2)  # type: ignore

    def c_in(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        _sqrt = lambda x: math.sqrt(x) if isinstance(x, float) else torch.sqrt(x)  # noqa
        return 1 / _sqrt(t_hat**2 + self.sigma_data**2)  # type: ignore

    def c_noise(self, t_hat: _ScalarOrTensor) -> _ScalarOrTensor:
        _log = lambda x: math.log(x) if isinstance(x, float) else torch.log(x)  # noqa
        _clip = lambda x, v: max(x, v) if isinstance(x, float) else x.clamp(v)  # noqa
        return _log(_clip(t_hat / self.sigma_data, 1e-20)) * 0.25  # type: ignore

    # ============================================================
    # For inference
    # ============================================================
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        s: torch.Tensor,
        z: torch.Tensor,
        num_samples: int = 1,
        config: SamplingConfig | None = None,
    ) -> torch.Tensor:
        """Sample structures via diffusion sampling.
        See Section 3.7: Algorithm 18 of AlphaFold3 paper.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        s : torch.Tensor
            The single representation, shape [B, L, c_s].
        z : torch.Tensor
            The pair representation, shape [B, L, L, c_z].
        num_samples : int
            The number of samples to generate for each input in the batch.
        config : SamplingConfig
            The sampling configuration.

        Returns
        -------
        coords : torch.Tensor
            The sampled atom coordinates, shape [B, N, L, 14, 3].
        """
        config = SamplingConfig() if config is None else config
        device = s.device

        B, L = batch["aatype_int"].shape
        N = num_samples
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

        # Compute time-independent variables
        # Algorithm 20 Line 1: DiffusionConditioning
        pair_bias = self.pair_conditioning(batch, z)
        del z

        def run_step(x_noisy: torch.Tensor, t_hat: float) -> torch.Tensor:
            c_noise = torch.tensor(self.c_noise(t_hat), device=device).view(1, 1)
            # Algorithm 20 Line 1: DiffusionConditioning
            single_cond = self.single_conditioning(batch, s, c_noise)
            return self.inference_step(
                batch,
                x_noisy,
                t_hat,
                single_cond,
                pair_bias,
                chunk_size=config.chunk_size,
            )

        # Gradually denoise
        for step in range(1, config.num_steps + 1):
            # Apply centering and random augmentation.
            x = self.random_augmentation(x, mask)

            sigma_tm, sigma_t = sigmas[step - 1], sigmas[step]
            gamma = config.gamma_0 * (sigma_t > config.gamma_min)
            t_hat: float = sigma_tm * (1 + gamma)

            # Add noise
            noise_var: float = config.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = math.sqrt(noise_var) * sample_noise()  # (B, N, L, 14, 3)
            eps.masked_fill_(~mask[..., None], 0.0)  # apply atom mask
            x_noisy = x + eps

            # Denoise
            x_denoised = run_step(x_noisy, t_hat)
            delta = (x_noisy - x_denoised) / t_hat
            dt = sigma_t - t_hat
            x = x_noisy + config.step_scale * dt * delta

        return x

    @staticmethod
    def random_augmentation(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, L, _, _ = coords.shape
        coords = coords.view(B, N, L * 14, 3)
        mask = mask.view(B, -1, L * 14)

        with torch.autocast(coords.device.type, enabled=False):
            coords = do_centering(coords, mask, mask_to_zero=False)

            R = random_rotations_torch((N,), coords.device)  # [N, 3, 3]
            coords = coords @ R.unsqueeze(0)  # [B, N, L*14, 3]

            noise = torch.randn((1, N, 1, 3), device=coords.device)  # [1, N, 1, 3]
            coords.add_(noise)

            # Mask out
            coords.masked_fill_(~mask[..., None], 0.0)
        return coords.view(B, N, L, 14, 3)

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        x_noisy: torch.Tensor,
        t_hat: float,
        single_cond: torch.Tensor,
        pair_bias: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Forward pass through the score model.
        See Section 3.7: Algorithm 5 Diffusion Module in AlphaFold3 SI.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            The input batch.
        x_noisy : torch.Tensor
            Noisy atom coordinates. Shape (B, N, L, 14, 3).
        t_hat : float
            Diffusion noise level (or sigmas of EDM).
        single_cond : torch.Tensor
            The single conditioning, shape [B, N, L, c_s].
        pair_bias : torch.Tensor
            The pair bias for the token transformer, shape [B, Nblock, H, L, L].

        Returns
        -------
        x_out : torch.Tensor
            Denoised atom coordinates. Shape (B, N, L, 14, 3).
        """
        B, N, L, _, _ = x_noisy.shape

        c_in, c_skip, c_out = self.c_in(t_hat), self.c_skip(t_hat), self.c_out(t_hat)

        # Input conditioning
        r_noisy = c_in * x_noisy

        if chunk_size is None:
            r_update = self.score_model(batch, r_noisy, single_cond, pair_bias)
        else:
            r_update = torch.zeros_like(r_noisy)
            for st in range(0, r_noisy.shape[1], chunk_size):
                end = min(st + chunk_size, r_noisy.shape[1])
                _r_noisy = r_noisy[:, st:end]
                _single_cond = (
                    single_cond if single_cond.shape[1] == 1 else single_cond[:, st:end]
                )
                r_update[:, st:end] = self.score_model(
                    batch, _r_noisy, _single_cond, pair_bias
                )

        # Output conditioning
        mask = batch["atom14_mask"]
        x_out = c_skip * x_noisy + c_out * r_update
        x_out = x_out * mask.view(B, 1, L, 14, 1)  # apply atom mask

        return x_out

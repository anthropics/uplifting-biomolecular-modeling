from functools import partial

import torch

from atlasfold.model.network.block import PairBlock
from atlasfold.utils.checkpointing import checkpoint_blocks


class PairformerStack(torch.nn.Module):
    """Main trunk of the AtlasFold model"""

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads: int = 16,
        num_tri_heads: int = 4,
        dropout_z: float = 0.25,
        num_blocks: int = 48,
        num_pair_to_single_blocks: int = 8,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        pair_to_single_start = num_blocks - num_pair_to_single_blocks
        self.blocks = torch.nn.ModuleList(
            [
                PairBlock(
                    channel_s=channel_s,
                    channel_z=channel_z,
                    num_heads_attn=num_heads,
                    num_heads_tri_attn=num_tri_heads,
                    dropout_z=dropout_z,
                    single_to_pair=False,
                    pair_to_pair=True,
                    pair_to_single=i >= pair_to_single_start,
                )
                for i in range(num_blocks)
            ]
        )
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        kernel_backend: str = "torch",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The single representations of shape (B, L, C_s)
        z : torch.Tensor
            The pair representations of shape (B, L, L, C_z)
        mask : torch.Tensor
            The token mask of shape (B, L)
        kernel_backend : str
            Triangle operation backend: "torch" or "cuequiv".
            Defaults to "torch".

        Returns
        -------
        z: torch.Tensor
            The updated pair representations
        """
        pair_mask = mask[..., :, None] & mask[..., None, :]
        blocks = [
            partial(
                b,
                mask=mask,
                pair_mask=pair_mask,
                kernel_backend=kernel_backend,
            )
            for b in self.blocks
        ]
        if self.blocks_per_ckpt is None or not self.training:
            for b in blocks:
                s, z = b(s, z)
        else:
            s, z = checkpoint_blocks(
                blocks,
                (s, z),
                self.blocks_per_ckpt,
            )
        return s, z


class LMStack(torch.nn.Module):
    """LM Module of the AtlasFold model"""

    def __init__(
        self,
        channel_s: int = 768,
        channel_z: int = 128,
        num_heads: int = 16,
        num_tri_heads: int = 4,
        dropout_z: float = 0.25,
        num_blocks: int = 4,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                PairBlock(
                    channel_s=channel_s,
                    channel_z=channel_z,
                    num_heads_attn=num_heads,
                    num_heads_tri_attn=num_tri_heads,
                    dropout_z=dropout_z,
                    single_to_pair=True,
                    pair_to_pair=True,
                    pair_to_single=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        kernel_backend: str = "torch",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The single representations of shape (B, L, C_s)
        z : torch.Tensor
            The pair representations of shape (B, L, L, C_z)
        mask : torch.Tensor
            The token mask of shape (B, L)
        kernel_backend : str
            Triangle operation backend: "torch" or "cuequiv".
            Defaults to "torch".

        Returns
        -------
        s : torch.Tensor
            The updated single representations
        z: torch.Tensor
            The updated pair representations
        """
        pair_mask = mask[..., :, None] & mask[..., None, :]
        blocks = [
            partial(
                b,
                mask=mask,
                pair_mask=pair_mask,
                kernel_backend=kernel_backend,
            )
            for b in self.blocks
        ]
        if self.blocks_per_ckpt is None or not self.training:
            for b in blocks:
                s, z = b(s, z)
        else:
            s, z = checkpoint_blocks(
                blocks,
                (s, z),
                self.blocks_per_ckpt,
            )
        return s, z

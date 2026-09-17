"""Template embedding module for multimer trunk inputs."""

from functools import partial

import torch

from atlasfold.model.network.block import PairStack
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias
from atlasfold.utils.torch_utils import add


def compute_distogram(
    coords: torch.Tensor,
    mask: torch.Tensor,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    """Compute one-hot pseudo-beta distogram bins.

    Parameters
    ----------
    coords
        Tensor of shape [B, T, L, 3].
    mask
        Boolean tensor of shape [B, T, L].
    boundaries
        Distogram lower breaks of shape [num_bins].


    Returns
    -------
    distogram
        Tensor of shape [B, T, L, L, num_bins]
    """
    with torch.autocast(coords.device.type, enabled=False):
        coords, boundaries = coords.float(), boundaries.float()
        lower_breaks = torch.square(boundaries)
        upper_breaks = torch.cat(
            [lower_breaks[1:], lower_breaks.new_tensor([1e8])],
            dim=-1,
        )
        diff = coords[..., None, :, :] - coords[..., :, None, :]
        dist2 = torch.sum(torch.square(diff), dim=-1, keepdim=True)
        distogram = (dist2 > lower_breaks) * (dist2 < upper_breaks)
        pair_mask = mask[..., :, None] & mask[..., None, :]
        distogram = distogram * pair_mask[..., None]
    return distogram


def compute_unit_vector(
    frame_coords: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute local-frame backbone unit vectors.

    Parameters
    ----------
    frame_coords
        Tensor of shape [B, T, L, 3, 3] containing backbone
        coordinates in the order of N, CA, C.
    mask
        Boolean tensor of shape [B, T, L] indicating valid residues.
    eps
        Small value to avoid division by zero in normalization.

    Returns
    -------
    unit_vector
        Tensor of shape [B, T, L, L, 3] containing unit vectors
    """
    with torch.autocast(frame_coords.device.type, enabled=False):
        x_n, x_ca, x_c = frame_coords.float().unbind(dim=-2)

        def normalize(x: torch.Tensor) -> torch.Tensor:
            norm2 = torch.sum(torch.square(x), dim=-1, keepdim=True)
            norm = torch.sqrt(torch.clamp(norm2, min=eps**2))
            return x / norm

        e0 = normalize(x_n - x_ca)
        e1 = x_c - x_ca
        e1 = normalize(e1 - torch.sum(e1 * e0, dim=-1, keepdim=True) * e0)
        e2 = torch.linalg.cross(e0, e1, dim=-1)
        basis = torch.stack([e0, e1, e2], dim=-2)

        ca_vec = x_ca.unsqueeze(-3) - x_ca.unsqueeze(-2)
        local_vec = (basis.unsqueeze(-3) @ ca_vec.unsqueeze(-1)).squeeze(-1)
        unit_vector = normalize(local_vec)
        pair_mask = mask[..., :, None] & mask[..., None, :]
        unit_vector = unit_vector * pair_mask[..., None]
    return unit_vector


class TemplateModule(torch.nn.Module):
    """Embed template features and return a pair-representation update.

    Inputs follow the AF3-style residue contract from the dataloader:
    one-hot restypes, pseudo-beta coordinates, backbone coordinates, and
    residue-level masks. Pair features are derived inside this module on the
    accelerator.
    """

    def __init__(
        self,
        channel_z: int,
        channel_template: int = 64,
        num_blocks: int = 2,
        num_tri_heads: int = 4,
        dropout_z: float = 0.25,
        num_distogram_bins: int = 39,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        super().__init__()
        self.channel_z = channel_z
        self.channel_template = channel_template

        boundaries = torch.linspace(min_dist, max_dist, num_distogram_bins)
        self.register_buffer("distogram_boundaries", boundaries, persistent=False)

        input_dim = (
            num_distogram_bins + 1 + 3 + 1 + 2 * 21
        )  # distogram, pseudo_beta_mask, restype_i, restype_j, unit_vector, backbone_mask
        self.linear_template = LinearNoBias(input_dim, channel_template, init="relu")
        self.proj_z = torch.nn.Sequential(
            LayerNorm(channel_z),
            LinearNoBias(channel_z, channel_template, init="relu"),
        )
        self.stack = PairStack(
            channel_z=channel_template,
            num_heads_tri_attn=num_tri_heads,
            dropout_z=dropout_z,
            pair_transition_factor=2,
            single_to_pair=False,
            pair_to_pair=True,
            pair_to_single=False,
            num_blocks=num_blocks,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_out = LayerNorm(channel_template)
        self.linear_out = LinearNoBias(channel_template, channel_z, init="relu")

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        z: torch.Tensor,
        mask: torch.Tensor,
        kernel_backend: str = "torch",
    ) -> torch.Tensor:
        """Return a template-derived update.

        kernel_backend must be "torch" or "cuequiv" and defaults to "torch".
        """
        _add = partial(add, inplace=not self.training)

        required = (
            "template.mask",
            "template.aatype",
            "template.pseudo_beta",
            "template.pseudo_beta_mask",
            "template.backbone_coords",
            "template.backbone_frame_mask",
        )
        if not all(k in batch for k in required):
            return torch.zeros_like(z)

        template_mask = batch["template.mask"]  # [B, T]
        template_aatype = batch["template.aatype"]
        template_pseudo_beta = batch["template.pseudo_beta"]
        pseudo_beta_mask = batch["template.pseudo_beta_mask"]
        backbone_coords = batch["template.backbone_coords"]
        backbone_frame_mask = batch["template.backbone_frame_mask"]

        B, T, L, _ = template_aatype.shape
        if T == 0:
            return torch.zeros_like(z)

        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]  # [B, 1, L, L]
        same_chain = (
            batch["asym_id"][:, None, :, None] == batch["asym_id"][:, None, None, :]
        )  # [B, 1, L, L]
        same_chain &= pair_mask
        same_chain_f = same_chain.float()

        b_backbone_frame_mask = (
            backbone_frame_mask[..., :, None] & backbone_frame_mask[..., None, :]
        )
        b_backbone_frame_mask &= same_chain
        b_backbone_frame_mask = b_backbone_frame_mask.float()

        b_pseudo_beta_mask = (
            pseudo_beta_mask[..., :, None] & pseudo_beta_mask[..., None, :]
        )
        b_pseudo_beta_mask &= same_chain
        b_pseudo_beta_mask = b_pseudo_beta_mask.float()

        f_distogram = compute_distogram(
            template_pseudo_beta,
            pseudo_beta_mask,
            self.distogram_boundaries,
        ).float()
        f_distogram *= same_chain_f[..., None]

        f_unit_vector = compute_unit_vector(
            backbone_coords,
            backbone_frame_mask,
        ).float()
        f_unit_vector *= same_chain_f[..., None]

        restype_i = template_aatype[..., :, None, :].expand(B, T, L, L, 21)
        restype_j = template_aatype[..., None, :, :].expand(B, T, L, L, 21)

        a = torch.cat(
            [
                f_distogram,
                b_pseudo_beta_mask[..., None],
                f_unit_vector,
                b_backbone_frame_mask[..., None],
                restype_i,
                restype_j,
            ],
            dim=-1,
        )
        a = a.permute(1, 0, 2, 3, 4)  # [T, B, L, L, C]

        u = torch.zeros(B, L, L, self.channel_template, device=z.device, dtype=z.dtype)
        z = self.proj_z(z)
        for i in range(T):
            a_i = a[i]
            v = self.linear_template(a_i) + z
            _, v = self.stack(None, v, mask, kernel_backend=kernel_backend)
            v = self.layernorm_out(v)
            v = v * template_mask[:, i, None, None, None].float()
            u = _add(u, v)

        n_templates = template_mask.sum(-1).float()  # [B]

        u = u / n_templates[:, None, None, None].clamp(min=1.0)
        return self.linear_out(torch.relu(u))

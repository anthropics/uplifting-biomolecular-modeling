"""Folding Trunk."""

import dataclasses
from functools import partial
from pathlib import Path
from typing import Literal

import torch

from atlasfold.model.model import AtlasFoldConfig
from atlasfold.model.network import (
    confidence_head,
    diffusion_head,
    distogram_head,
    template,
    trunk,
)
from atlasfold.model.network.diffusion_head import SamplingConfig
from atlasfold.model.network.primitives import LayerNorm, LinearNoBias
from atlasfold.model.network.rel_pos_encoding import (
    AtomRelativePositionEncoding,
    RelativePositionEncoding,
)
from atlasfold.model.utils import confidence_metrics
from atlasfold.utils import torch_utils
from atlasfold.utils.kernel import select_kernel_backend
from atlaslm.model import Alphabet, AtlasLM


@dataclasses.dataclass(kw_only=True)
class TrunkConfig:
    num_heads: int = 16
    num_tri_heads: int = 4
    dropout_z: float = 0.25
    num_lm_blocks: int = 4
    num_blocks: int = 48
    num_pair_to_single_blocks: int = 12
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class TemplateModuleConfig:
    channel_template: int = 64
    num_blocks: int = 2
    num_tri_heads: int = 4
    dropout_z: float = 0.25
    num_distogram_bins: int = 39
    min_dist: float = 3.25
    max_dist: float = 50.75
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class DiffusionHeadConfig:
    channel_a: int = 768
    channel_atom: int = 96
    channel_cond: int = 384
    num_heads: int = 16
    num_blocks: int = 12
    num_atom_heads: int = 2
    num_atom_blocks: int = 3
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class DistogramHeadConfig:
    num_bins: int = 64
    min_dist: float = 2.0
    max_dist: float = 22.0


@dataclasses.dataclass(kw_only=True)
class ConfidenceHeadConfig:
    num_heads: int = 16
    num_tri_heads: int = 4
    dropout_z: float = 0.25
    num_blocks: int = 4
    num_bins: int = 39
    min_dist: float = 3.25
    max_dist: float = 50.75
    # heads
    num_plddt_bins: int = 50
    max_pae_error: float = 32.0
    num_pae_bins: int = 64
    max_pde_error: float = 32.0
    num_pde_bins: int = 64
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class AtlasFoldMultimerConfig(AtlasFoldConfig):
    name: str = "atlasfold-m-260725"
    lm_name: str = "atlaslm-3b"
    lm_path: str | None = None

    channel_s: int = 384
    channel_s_lm: int = 768
    channel_z: int = 128
    trunk: TrunkConfig = dataclasses.field(default_factory=TrunkConfig)
    template_module: TemplateModuleConfig = dataclasses.field(
        default_factory=TemplateModuleConfig
    )
    distogram_head: DistogramHeadConfig = dataclasses.field(
        default_factory=DistogramHeadConfig
    )
    diffusion_head: DiffusionHeadConfig = dataclasses.field(
        default_factory=DiffusionHeadConfig
    )
    confidence_head: ConfidenceHeadConfig = dataclasses.field(
        default_factory=ConfidenceHeadConfig
    )


class AtlasFold_Multimer(torch.nn.Module):
    def __init__(
        self,
        cfg: AtlasFoldMultimerConfig,
        lm: AtlasLM | None = None,
    ) -> None:
        """Initialize the AtlasFold model."""
        super().__init__()
        self.cfg: AtlasFoldMultimerConfig = cfg
        # Trunk dimensions
        self.channel_s: int = cfg.channel_s
        self.channel_s_lm: int = cfg.channel_s_lm
        self.channel_z: int = cfg.channel_z

        # Diffusion head dimensions
        self.channel_a: int = cfg.diffusion_head.channel_a
        self.channel_atom: int = cfg.diffusion_head.channel_atom

        # Relative positional encoding
        self.seq_rel_pos_encoding = RelativePositionEncoding(r_max=32, s_max=2)
        self.atom_rel_pos_encoding = AtomRelativePositionEncoding(max_r=4)

        # === Language model === #
        if lm is None:
            lm_source = Path(cfg.lm_path) if cfg.lm_path is not None else cfg.lm_name
            lm = AtlasLM.from_pretrained(lm_source, dtype=torch.bfloat16)
        self.lm: AtlasLM = lm
        # Freeze LM parameters
        self.lm.requires_grad_(False)
        self.alphabet: Alphabet = self.lm.alphabet

        # === Representation initialization === #
        self.s_init = LinearNoBias(21, self.channel_s)
        self.z_init = LinearNoBias(21, 2 * self.channel_z)
        self.z_rel_pos = LinearNoBias(self.seq_rel_pos_encoding.dim, self.channel_z)

        # === Recycling embedding === #
        self.recycle_s = torch.nn.Sequential(
            LayerNorm(self.channel_s),
            LinearNoBias(self.channel_s, self.channel_s, init="final"),
        )
        self.recycle_z = torch.nn.Sequential(
            LayerNorm(self.channel_z),
            LinearNoBias(self.channel_z, self.channel_z, init="final"),
        )

        # === LM Stack === #
        self.lm_layer_weights = torch.nn.Parameter(torch.zeros(self.lm.n_layers + 1))
        self.layernorm_lm_emb = LayerNorm(self.lm.d_model)
        self.lm_emb_to_s_lm = torch.nn.Sequential(
            LinearNoBias(self.lm.d_model, self.channel_s_lm, init="relu"),
            torch.nn.ReLU(),
            LinearNoBias(self.channel_s_lm, self.channel_s_lm, init="default"),
        )
        self.proj_lm_attn = torch.nn.ModuleList(
            [
                LinearNoBias(self.lm.n_heads, self.channel_z)
                for _ in range(self.lm.n_layers)
            ]
        )
        self.lm_attn_to_z_lm = torch.nn.Sequential(
            LayerNorm(self.channel_z),
            LinearNoBias(self.channel_z, self.channel_z, init="relu"),
            torch.nn.ReLU(),
            LinearNoBias(self.channel_z, self.channel_z, init="default"),
        )
        self.lm_stack = trunk.LMStack(
            channel_s=self.channel_s_lm,
            channel_z=self.channel_z,
            num_heads=cfg.trunk.num_heads,
            num_tri_heads=cfg.trunk.num_tri_heads,
            dropout_z=cfg.trunk.dropout_z,
            num_blocks=cfg.trunk.num_lm_blocks,
            blocks_per_ckpt=cfg.trunk.blocks_per_ckpt,
        )
        self.proj_s_lm = torch.nn.Sequential(
            LayerNorm(self.channel_s_lm),
            LinearNoBias(self.channel_s_lm, self.channel_s),
        )
        self.proj_z_lm = torch.nn.Sequential(
            LayerNorm(self.channel_z),
            LinearNoBias(self.channel_z, self.channel_z),
        )

        # === Trunk body === #
        self.main_stack = trunk.PairformerStack(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            num_heads=cfg.trunk.num_heads,
            num_tri_heads=cfg.trunk.num_tri_heads,
            dropout_z=cfg.trunk.dropout_z,
            num_blocks=cfg.trunk.num_blocks,
            num_pair_to_single_blocks=cfg.trunk.num_pair_to_single_blocks,
            blocks_per_ckpt=cfg.trunk.blocks_per_ckpt,
        )

        # === Template module === #
        template_cfg = dataclasses.asdict(cfg.template_module)
        self.template_module = template.TemplateModule(
            channel_z=self.channel_z, **template_cfg
        )

        # === Distogram head === #
        self.distogram_head = distogram_head.DistogramHead(
            channel_z=self.channel_z,
            **dataclasses.asdict(cfg.distogram_head),
        )

        # === Structure prediction heads === #
        self.diffusion_head = diffusion_head.DiffusionHead(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            seq_rel_pos_bins=self.seq_rel_pos_encoding.dim,
            atom_rel_pos_bins=self.atom_rel_pos_encoding.dim,
            **dataclasses.asdict(cfg.diffusion_head),
        )

        # === Confidence prediction head === #
        self.confidence_head = confidence_head.ConfidenceHead_Multimer(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            **dataclasses.asdict(cfg.confidence_head),
        )

        # Triangle kernel backend.
        self.kernel_backend = "torch"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_kernel_backend(
        self,
        backend: Literal["auto", "torch", "cuequiv"],
    ) -> str:
        """Select the "auto", "torch", or "cuequiv" kernel backend."""
        self.kernel_backend = select_kernel_backend(backend, self.device.type)
        return self.kernel_backend

    # ==================================================
    # Inference
    # ==================================================
    def fold(
        self,
        sequence: str,
        mode: str = "full",
        num_recycles: int = 10,
        num_samples: int = 1,
        sampling_config: SamplingConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fold a single sequence."""
        raise NotImplementedError("Multimer folding is not implemented yet.")

    @torch.inference_mode()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        mode: str = "full",
        num_samples: int = 3,
        return_representations: bool = False,
        # Advanced options
        mlm_prob: float | None = None,
        num_recycles: int = 3,
        sampling_config: SamplingConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        batch: dict[str, torch.Tensor]
            The input batch with the following keys:
            # Folding trunk inputs
            - "aatype"          : [B, L, 21] float
                One-hot encoded amino acid types.
            - "aatype_int"      : [B, L] int
                Amino acid type indices.
            - "entity_id"       : [B, L] int
                Entity IDs for multimer structures.
            - "asym_id"         : [B, L] int
                Asymmetry IDs (chain IDs) for multimer structures.
            - "sym_id"          : [B, L] int
                Symmetry IDs for multimer structures.
            - "res_idx"         : [B, L] int
                1-based residue indices.
            - "seq_tok_idx"     : [B, L] int
                LM sequence indices corresponding to each residue.
            - "seq_mask"        : [B, L] bool
                Boolean mask for valid sequence positions (True for valid).
            - "atom14_mask"     : [B, L, 14] bool
                Boolean mask for valid atom14 positions.
            - "atom37_mask"     : [B, L, 37] bool
                Boolean mask for valid atom37 positions.
            - "pseudo_beta"     : [B, L] int
                Pseudo-beta index for each residue (C-beta, or C-alpha for glycine).
            # LM inputs
            - "lm.input_ids"    : [B, S] int
                Input token IDs, including special tokens.
            - "lm.pos_id"       : [B, S] int
                Positional IDs. Same to 1-based residue indices for valid positions,
                where 0 for CLS and L+1 for SEP.
            - "lm.seq_id"       : [B, S] int
                Sequence IDs for attention masking. 1 for valid tokens, 0 for padding.
            - "lm.mlm_mask"     : [B, S] bool
                Optional boolean mask for stochastic LM feature extraction.
            # Optional template inputs, indexed by residues rather than LM tokens
            - "template.mask" : [B, T] bool
                Boolean mask for valid per-chain template slots.
            - "template.aatype" : [B, T, L, 21] float
                One-hot encoded template amino acid types.
            - "template.pseudo_beta_mask" : [B, T, L] bool
                Pseudo-beta coordinate mask (C-beta, or C-alpha for glycine).
            - "template.backbone_frame_mask" : [B, T, L] bool
                Mask indicating residues with valid N, CA, and C coordinates.
            - "template.pseudo_beta" : [B, T, L, 3] float
                Raw pseudo-beta coordinates.
            - "template.backbone_coords" : [B, T, L, 3, 3] float
                Raw backbone coordinates ordered as N, CA, C. Used for
                local-frame unit-vector features inside the template module.
        num_recycles : int
            The number of recycling steps.
        sampling_config : SamplingConfig
            The configuration for the diffusion sampling process.
        return_representations : bool
            Whether to return the single and pair representations.

        # Advanced options
        mlm_prob : float | None
            The probability of masking input tokens for the language model.
            If > 0, a random MLM mask will be sampled for each forward pass.

        Returns
        -------
        out: dict[str, torch.Tensor]
            The output dictionary containing the predicted coordinates and optionally
            the distogram logits and representations.
        """
        if mode not in ["base", "full"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of 'base' or 'full'.")

        # Set default MLM probability to 0.15 for full mode if not specified
        mlm_prob = mlm_prob if mlm_prob is not None else 0.15
        if mlm_prob <= 0.0:
            raise ValueError("mlm_prob must be greater than 0 for full mode.")

        is_batched = batch["aatype"].dim() == 3
        if not is_batched:
            # Add batch dimension if the input is not already batched
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        out: dict[str, torch.Tensor] = {}

        # Compute positional encodings
        self.compute_rel_pos_encoding(batch)

        # Run trunk
        s, z = self.run_trunk(batch, num_recycles, mlm_prob)
        s, z = s.float(), z.float()
        if return_representations:
            out["trunk.s"] = s
            out["trunk.z"] = z

        # Run distogram head
        distogram_out = self.distogram_head(z)
        out["distogram.logits"] = distogram_out["logits"]
        out["distogram.boundaries"] = distogram_out["boundaries"]

        # Run diffusion heads
        with torch.autocast(self.device.type, enabled=False):
            sample_coords = self.diffusion_head.sample(
                batch, s, z, num_samples, sampling_config
            )
        out["sample_coords"] = sample_coords

        # Run confidence head
        confidence_out = self.confidence_head(
            batch, s, z, sample_coords, self.kernel_backend
        )
        del s, z

        # Compute confidence metrics
        with torch.autocast(self.device.type, enabled=False):
            mask = batch["seq_mask"].unsqueeze(1)  # [B, 1, L]
            out["plddt"] = confidence_metrics.compute_plddt(
                **confidence_out["plddt"], mask=mask
            )
            out["pde"] = confidence_metrics.compute_pde(
                **confidence_out["pde"], mask=mask
            )
            pae_logits = confidence_out["pae"]["logits"]
            pae_bin_centers = confidence_out["pae"]["bin_centers"]
            pae_probs = torch.softmax(pae_logits, dim=-1)
            out["pae"] = confidence_metrics.compute_pae_from_probs(
                pae_probs,
                pae_bin_centers,
                mask,
            )
            out["ptm"] = confidence_metrics.compute_ptm_from_probs(
                pae_probs,
                pae_bin_centers,
                mask,
            )
            out["iptm"] = confidence_metrics.compute_iptm_from_probs(
                pae_probs,
                pae_bin_centers,
                batch["asym_id"],
                mask,
            )
            out["chain_ptm"], out["interface_iptm"] = (
                confidence_metrics.compute_chain_tm_scores_from_probs(
                    pae_probs,
                    pae_bin_centers,
                    batch["asym_id"],
                    mask,
                )
            )
        del confidence_out, mask, pae_logits, pae_probs

        # Remove batch dimension if the input was not originally batched
        if not is_batched:
            for k, v in out.items():
                out[k] = v.squeeze(0)

        return out

    # ==================================================
    # Forward pass components
    # ==================================================
    def compute_rel_pos_encoding(self, batch: dict[str, torch.Tensor]) -> None:
        """Compute the relative positional encodings"""
        seq_rel_pos = self.seq_rel_pos_encoding(batch)  # [B, L, L, bins]
        atom_rel_pos = self.atom_rel_pos_encoding(batch)  # [B, W, Lq, 14, Lk, 14, bins]
        batch["seq_rel_pos"] = seq_rel_pos
        batch["atom_rel_pos"] = atom_rel_pos

    def run_trunk(
        self,
        batch: dict[str, torch.Tensor],
        num_recycles: int,
        mlm_prob: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk."""
        mask = batch["seq_mask"]
        B, L = mask.shape
        device = mask.device
        dtype = torch_utils.get_context_dtype(device.type)

        # Recycling iteration with stochastic LM features
        s_prev = torch.zeros(B, L, self.channel_s, device=device, dtype=dtype)
        z_prev = torch.zeros(B, L, L, self.channel_z, device=device, dtype=dtype)

        for _ in range(0, num_recycles + 1):
            s = self.s_init(batch["aatype"])
            a, b = self.z_init(batch["aatype"]).chunk(2, dim=-1)
            z = a[..., :, None, :] + b[..., None, :, :]
            z += self.z_rel_pos(batch["seq_rel_pos"])

            # Recycling embedding
            s += self.recycle_s(s_prev)
            z += self.recycle_z(z_prev)

            # Run LM module with stochastic masking
            mlm_mask = self.sample_mlm_mask(batch, mlm_prob)
            s_lm, z_lm = self.run_lm_embedder(batch, mlm_mask)
            s += self.proj_s_lm(s_lm)
            z += self.proj_z_lm(z_lm)

            if self.template_module is not None:
                z += self.template_module(batch, z, mask, self.kernel_backend)

            # Run main trunk
            s, z = self.main_stack(s, z, mask, self.kernel_backend)
            s_prev, z_prev = s, z
        return s, z

    def sample_mlm_mask(
        self,
        batch: dict[str, torch.Tensor],
        prob: float,
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

        if prob <= 0.0:
            return torch.zeros(shape, dtype=torch.bool, device=input_ids.device)
        else:
            return torch.rand(shape, device=input_ids.device) < prob

    def run_lm_embedder(
        self,
        batch: dict[str, torch.Tensor],
        mlm_mask: torch.Tensor | None = None,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract LM features and project them to single and pair representations."""
        _add = partial(torch_utils.add, inplace=not train)

        # === Prepare LM inputs === #
        input_ids = batch["lm.input_ids"]  # [B, S]
        pos_id = batch["lm.pos_id"]  # [B, S]
        seq_id = batch["lm.seq_id"]  # [B, S]. 1 for valid, 0 for padding

        # Apply MLM mask to input IDs for stochastic feature extraction
        if mlm_mask is not None:
            assert mlm_mask.shape[1] == input_ids.shape[1], (
                f"MLM mask must have the same length dimension as input_ids. "
                f"Got {mlm_mask.shape} and {input_ids.shape}."
            )
            aa_idxs = torch.tensor(self.alphabet.aa_idxs, device=input_ids.device)
            mlm_mask = mlm_mask & torch.isin(input_ids, aa_idxs)
            input_ids = input_ids.masked_fill(mlm_mask, self.alphabet.mask_idx)

        # === Accumulate LM features === #
        B, S = input_ids.shape
        device = input_ids.device
        lm_emb = torch.zeros((B, S, self.lm.d_model), device=device, dtype=torch.float32)
        lm_attn = torch.zeros(
            (B, S, S, self.channel_z), device=device, dtype=torch.float32
        )
        w_layers = self.lm_layer_weights.softmax(dim=0)  # [n_layers+1,]

        with torch.no_grad():
            x = self.lm.embed(input_ids)
        lm_emb = _add(lm_emb, w_layers[0] * self.layernorm_lm_emb(x))

        for i, block in enumerate(self.lm.transformer.blocks):
            with torch.no_grad():
                x, attn = block(x, seq_id, pos_id, return_attn_logits=True)
                # neginf will be set to 0 at the end of this function.
                attn = attn.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                attn = attn.clamp_(-100.0, 100.0).div_(100)
                attn = attn.moveaxis(1, -1)  # [B, S, S, n_heads]
            lm_emb = _add(lm_emb, w_layers[i + 1] * self.layernorm_lm_emb(x))
            lm_attn = _add(lm_attn, self.proj_lm_attn[i](attn))

        # Extract the single and pair representations for the valid sequence positions
        # [B, S, c_s], [B, S, S, c_z] -> [B, L, c_s], [B, L, L, c_z]
        b_i = torch.arange(input_ids.shape[0], device=input_ids.device)
        b_i_s, b_i_z = b_i[:, None], b_i[:, None, None]
        row_s = batch["seq_tok_idx"]  # [B, L]
        row_z, col_z = row_s[:, :, None], row_s[:, None, :]
        lm_emb = lm_emb[b_i_s, row_s]  # [B, L, c_s]
        lm_attn = lm_attn[b_i_z, row_z, col_z]  # [B, L, L, c_z]

        s_lm = self.lm_emb_to_s_lm(lm_emb)
        z_lm = self.lm_attn_to_z_lm(lm_attn)
        del lm_emb, lm_attn, row_s, row_z, col_z, b_i_s, b_i_z

        # Mask the LM features for invalid sequence positions
        mask = batch["seq_mask"]  # [B, L]
        pair_mask = mask[:, None, :] & mask[:, :, None]
        intra_mask = batch["asym_id"][:, None, :] == batch["asym_id"][:, :, None]
        intra_mask &= pair_mask
        s_lm = s_lm * mask[:, :, None]
        z_lm = z_lm * intra_mask[:, :, :, None]  # [B, L, L, c_z]

        # Run LM stack
        s_lm, z_lm = self.lm_stack(s_lm, z_lm, mask, self.kernel_backend)
        return s_lm, z_lm

    # TODO: current implementation is for development.
    # I may want to remove this method in the future.
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "SeonghwanSeo/atlasfold-m-260725",
        *,
        config: AtlasFoldMultimerConfig | None = None,
        lm: AtlasLM | None = None,
        device: str | torch.device = "cpu",
        cache_dir: str | Path | None = None,
    ) -> "AtlasFold_Multimer":
        """Create an AtlasFold-M model from pretrained weights."""
        if config is None:
            config = AtlasFoldMultimerConfig()

        source = pretrained_model_name_or_path
        if isinstance(source, Path):
            if not source.is_file():
                raise FileNotFoundError(f"Model checkpoint does not exist: {source}")
            model_path = source
        elif Path(source).is_file():
            model_path = Path(source)
        else:
            from huggingface_hub import snapshot_download

            repo_path = snapshot_download(
                repo_id=source,
                repo_type="model",
                cache_dir=cache_dir,
            )
            model_name = source.rsplit("/", maxsplit=1)[-1]
            model_path = Path(repo_path) / "weights" / f"{model_name}.pth"

        device = torch.device(device)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        if lm is None:
            lm_source = (
                Path(config.lm_path) if config.lm_path is not None else config.lm_name
            )
            lm = AtlasLM.from_pretrained(
                lm_source,
                device=device,
                dtype=dtype,
                cache_dir=cache_dir,
            )
        else:
            lm = lm.to(device=device, dtype=dtype)

        # Create the model on meta device
        with torch.device("meta"):
            model = cls(config, lm=lm)
            # Remove the LM, which will be loaded separately
            del model.lm

        # Load the state dict onto the target device
        model = model.to_empty(device=device)

        # Load the state dict with the specified strictness
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)

        # Finally, attach the separately loaded, potentially shared LM
        model.lm = lm

        if dtype is torch.bfloat16:
            model.lm = model.lm.bfloat16()
            model.lm_stack = model.lm_stack.to(dtype)
            model.main_stack = model.main_stack.to(dtype)

        # Freeze the model parameters and set to eval mode
        model.requires_grad_(False)
        model.eval()
        return model

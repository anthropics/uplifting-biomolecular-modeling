import contextlib
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from atlaslm.alphabet import Alphabet
from atlaslm.layers import RegressionHead, TransformerStack

if TYPE_CHECKING:
    from atlaslm.pretrained import AtlasLMConfig


@dataclasses.dataclass(slots=True)
class PLMOutput:
    embeddings: torch.Tensor
    sequence_logits: torch.Tensor | None = None
    hidden_states: list[torch.Tensor] | None = None
    attentions: list[torch.Tensor] | None = None


class AtlasLM(torch.nn.Module):
    """
    Protein language model for AtlasFold.

    Reference: "https://github.com/evolutionaryscale/esm"

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
    """

    def __init__(
        self,
        d_model: int = 2304,
        n_heads: int = 36,
        n_layers: int = 48,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers

        self.alphabet: Alphabet = Alphabet()
        self.vocab_size: int = len(self.alphabet)

        self.embed = torch.nn.Embedding(64, d_model)
        self.transformer = TransformerStack(d_model, n_heads, n_layers)
        self.sequence_head = RegressionHead(d_model, 64)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = ("SeonghwanSeo/atlaslm-3b-base"),
        *,
        config: "AtlasLMConfig | None" = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        cache_dir: str | Path | None = None,
    ) -> "AtlasLM":
        """Create an AtlasLM model from pretrained weights."""
        from atlaslm.pretrained import load_model

        return load_model(
            pretrained_model_name_or_path,
            config=config,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def embed_sequences(
        self,
        sequences: list[str],
        return_logits: bool = False,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
    ) -> PLMOutput:
        """
        Performs forward pass through the PLM.

        Args:
            sequences (list[str]): The amino acid sequences.
            return_logits (bool): Whether to return logits. (default: False)
            return_hidden_states (bool): Whether to return list of hidden states.
                hidden_states shape: (batch_size, seq_len, d_model)
            return_attentions (bool): Whether to return list of attentions.
                attentions shape: (batch_size, n_heads, seq_len, seq_len)

        Returns:
            PLMOutput: The output of the PLM.
        """
        device = self.device
        with (
            torch.autocast(device.type, dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext(),
        ):
            input_ids = self.encode(sequences)
            attn_mask = input_ids != self.alphabet.pad_idx
            out = self(
                input_ids,
                seq_id=attn_mask,
                return_logits=return_logits,
                return_hidden_states=return_hidden_states,
                return_attentions=return_attentions,
            )
        return out

    def encode(self, sequences: list[str]) -> torch.Tensor:
        """
        Encode a batch of sequences into token IDs.

        Args:
            sequences (list[str]): List of amino acid sequences.

        Returns:
            input_ids (torch.Tensor): Tensor of shape (batch_size, max_seq_len)
        """
        batch_size = len(sequences)
        max_len = max(len(seq) + 2 for seq in sequences)
        out = np.full((batch_size, max_len), self.alphabet.pad_idx, dtype=np.int64)
        for i, seq in enumerate(sequences):
            out[i, : len(seq) + 2] = self.alphabet.encode(sequences[i])
        return torch.from_numpy(out).to(device=self.device, non_blocking=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_id: torch.Tensor | None = None,
        pos_id: torch.Tensor | None = None,
        return_logits: bool = False,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
    ) -> PLMOutput:
        """
        Performs forward pass through the PLM.

        Parameters
        ----------
        input_ids: torch.Tensor
            The amino acid tokens of shape (batch_size, seq_len).
        seq_id: torch.Tensor | None
            Optional tensor of shape (batch_size, seq_len) identifying sequence boundaries
            for masking within a packed batch.
        pos_id: torch.Tensor | None
            Optional tensor of shape (batch_size, seq_len) providing specific position
            indices for rotary embeddings.
        return_logits: bool
            Whether to return logits. (default: False)
        return_hidden_states: bool
            Whether to return list of hidden states.
            hidden_states shape: (batch_size, seq_len, d_model)
        return_attentions: bool
            Whether to return list of attentions.
            attentions shape: (batch_size, n_heads, seq_len, seq_len)

        Return
        ------
        out: PLMOutput
            The output of the PLM containing embeddings, sequence logits, hidden states,
        """
        if seq_id is None:
            seq_id = input_ids != self.alphabet.pad_idx
        else:
            if input_ids.shape != seq_id.shape:
                raise ValueError(
                    "input_ids and seq_id must have the same shape. "
                    f"Got input_ids shape {input_ids.shape} and "
                    f"seq_id shape {seq_id.shape}."
                )
        if pos_id is not None:
            if pos_id.shape != input_ids.shape:
                raise ValueError(
                    "input_ids and pos_id must have the same shape. "
                    f"Got input_ids shape {input_ids.shape} and "
                    f"pos_id shape {pos_id.shape}."
                )

        # Forward pass
        x = self.embed(input_ids)
        x, hiddens, attentions = self.transformer(x, seq_id, pos_id, return_attentions)

        if not return_hidden_states:
            hiddens = []

        if not return_attentions:
            attentions = []

        if return_logits:
            sequence_logits = self.sequence_head(x)
        else:
            sequence_logits = None

        return PLMOutput(
            embeddings=x,
            sequence_logits=sequence_logits,
            hidden_states=hiddens if hiddens else None,
            attentions=attentions if attentions else None,
        )

    @torch.inference_mode()
    def pseudo_perplexity(
        self,
        sequence: str,
        *,
        max_batch_tokens: int = 4096,
    ) -> dict[str, torch.Tensor]:
        """Compute pseudo-perplexity for one protein sequence.

        Each residue is masked once and scored as ``p(x_i | x_without_i)``. The
        pseudo-perplexity is ``exp(-sum(log p_i) / length)``. Masked examples are
        processed in batches; this changes throughput but not the definition of the
        score.

        Parameters
        ----------
        sequence : str
            Protein sequence to score.
        max_batch_tokens : int, optional
            Maximum total number of tokens evaluated in one forward pass. CUDA
            out-of-memory errors automatically reduce the derived batch size.

        Returns
        -------
        dict
            Scalar pseudo-perplexity and residue log-probabilities with shape
            ``(length,)``.
        """
        if max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be positive")

        tokens = self.encode([sequence])
        length = len(sequence)
        batch_size = min(length, max(1, max_batch_tokens // tokens.shape[1]))
        residue_log_probabilities = torch.empty(
            length, device=self.device, dtype=torch.float32
        )
        start = 0
        device = self.device
        mask_id = self.alphabet.mask_idx
        while start < length:
            end = min(start + batch_size, length)
            bs = end - start
            rows = torch.arange(bs, device=device)
            positions = rows + start + 1  # Skip <cls>.

            masked_tokens = tokens.expand(bs, -1).clone()
            masked_tokens[rows, positions] = mask_id
            try:
                with (
                    torch.autocast(device.type, dtype=torch.bfloat16)
                    if self.device.type == "cuda"
                    else contextlib.nullcontext(),
                ):
                    output = self(masked_tokens, return_logits=True)
                if output.sequence_logits is None:
                    raise RuntimeError("AtlasLM did not return sequence logits")
                logits = output.sequence_logits[rows, positions].float()
                targets = tokens[0, positions]
                residue_log_probabilities[start:end] = -torch.nn.functional.cross_entropy(
                    logits, targets, reduction="none"
                )
                del masked_tokens, output, logits
                start = end
            except torch.cuda.OutOfMemoryError:
                del masked_tokens
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                batch_size = max(1, bs // 2)

        pseudo_perplexity = (-residue_log_probabilities.mean()).exp()
        return {
            "pseudo_perplexity": pseudo_perplexity,
            "residue_log_probabilities": residue_log_probabilities,
        }

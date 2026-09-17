"""Featurization utilities for AtlasFold inference."""

from collections.abc import Iterable, Sequence

import numpy as np
import torch

from atlasfold.common import residue_constants
from atlaslm.alphabet import VOCAB

VOCAB_TO_IDX: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
BOS_IDX: int = VOCAB_TO_IDX["<cls>"]
EOS_IDX: int = VOCAB_TO_IDX["<eos>"]
MASK_IDX: int = VOCAB_TO_IDX["<mask>"]
PAD_IDX: int = VOCAB_TO_IDX["<pad>"]

# NOTE: This default bucket list is designed for uncompiled model inference.
# For compiled inference, we recommend using a smaller set of buckets to
# reduce the number of recompilations.
DEFAULT_BUCKETS = [
    32, 64, 128, 192, 256, 384, 512, 640, 768, 896, 1024,
    1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
]  # fmt: skip


def featurize(
    sequence: str,
    *,
    entity_id: int = 1,
    asym_id: int = 1,
    sym_id: int = 1,
    residue_index: Sequence[int] | None = None,
    pad_to_multiple_of: int | None = None,
) -> dict[str, np.ndarray]:
    """Featurize the input sequence for the model."""
    assert entity_id > 0, "entity_id must be a positive integer starting from 1."
    assert asym_id > 0, "asym_id must be a positive integer starting from 1."
    assert sym_id > 0, "sym_id must be a positive integer starting from 1."

    # Sanitize the input sequence
    sequence = sequence.upper()
    sequence = "".join(
        [aa if aa in residue_constants.restype_orders else "X" for aa in sequence]
    )
    length = len(sequence)

    # === Static features (chain-related) === #
    entity_id_arr = np.full(length, entity_id, dtype=np.int64)
    asym_id_arr = np.full(length, asym_id, dtype=np.int64)
    sym_id_arr = np.full(length, sym_id, dtype=np.int64)

    # === Residue index === #
    if residue_index is not None:
        raise NotImplementedError("Custom residue_index is not supported yet.")
    else:
        res_idx = np.arange(1, length + 1, dtype=np.int64)

    # === Prepare the input for the language model trunk === #
    input_ids = np.array(
        [BOS_IDX] + [VOCAB_TO_IDX[aa] for aa in sequence] + [EOS_IDX], dtype=np.int64
    )

    pos_id = np.concatenate(([0], res_idx, [res_idx[-1] + 1]))
    seq_id = np.full_like(input_ids, asym_id, dtype=np.int64)

    lm_input = {
        "lm.input_ids": input_ids,  # [L+2]
        "lm.pos_id": pos_id,  # [L+2]
        "lm.seq_id": seq_id,  # [L+2]
    }

    # === Prepare the input for the folding trunk === #
    # Convert amino acid sequence to integer indices
    length = len(sequence)
    aatype = np.array([residue_constants.restype_orders[aa] for aa in sequence])
    aatype_onehot = np.eye(21, dtype=np.float32)[aatype]

    # Create masks
    seq_mask = np.ones(length, dtype=bool)
    atom14_mask = residue_constants.restype_atom14_mask[aatype]
    atom37_mask = residue_constants.restype_atom37_mask[aatype]

    # Get the pseudo-beta index (CB for most residues, CA for glycine)
    cbeta_idx = np.full((length,), 4, dtype=np.int64)  # CB: index=4
    cbeta_idx[aatype == residue_constants.restype_orders["G"]] = 1

    # Add lm input to folding input mapping
    seq_tok_idx = np.arange(1, length + 1, dtype=np.int64)

    folding_input = {
        "entity_id": entity_id_arr,  # [L]
        "asym_id": asym_id_arr,  # [L]
        "sym_id": sym_id_arr,  # [L]
        "aatype": aatype_onehot,  # [L, 21]
        "aatype_int": aatype,  # [L]
        "res_idx": res_idx,  # [L]
        "seq_tok_idx": seq_tok_idx,  # [L]
        "seq_mask": seq_mask,  # [L]
        "atom14_mask": atom14_mask,  # [L, 14]
        "atom37_mask": atom37_mask,  # [L, 37]
        "pseudo_beta": cbeta_idx,  # [L]
    }
    feat = {**lm_input, **folding_input}

    if pad_to_multiple_of is not None:
        for k, v in feat.items():
            pad_len = (-len(v)) % pad_to_multiple_of
            pad_width = ((0, pad_len),) + ((0, 0),) * (v.ndim - 1)
            feat[k] = np.pad(v, pad_width, constant_values=0)

    return feat


def featurize_complex(
    sequences: list[str],
    entity_ids: list[int],
    asym_ids: list[int],
    sym_ids: list[int],
    pad_to_multiple_of: int | None = None,
) -> dict[str, np.ndarray]:
    """Featurize the input sequence for the model."""
    num_chains = len(sequences)
    if not (len(entity_ids) == len(asym_ids) == len(sym_ids) == num_chains):
        raise ValueError(
            f"Number of sequences ({num_chains}) must match "
            f"number of entity_ids ({len(entity_ids)}), "
            f"asym_ids ({len(asym_ids)}), and sym_ids ({len(sym_ids)})."
        )
    feats = [
        featurize(seq, entity_id=eid, asym_id=aid, sym_id=sid)
        for seq, eid, aid, sid in zip(
            sequences, entity_ids, asym_ids, sym_ids, strict=True
        )
    ]
    lm_offset = 0
    for feat in feats:
        feat["seq_tok_idx"] = feat["seq_tok_idx"] + lm_offset
        lm_offset += len(feat["lm.input_ids"])

    # Concatenate features from all chains
    concatenated_feats = {}
    for k in feats[0].keys():
        concatenated_feats[k] = np.concatenate([feat[k] for feat in feats], axis=0)
    if pad_to_multiple_of is not None:
        for k, v in concatenated_feats.items():
            pad_len = (-len(v)) % pad_to_multiple_of
            pad_width = ((0, pad_len),) + ((0, 0),) * (v.ndim - 1)
            concatenated_feats[k] = np.pad(v, pad_width, constant_values=0)
    return concatenated_feats


# ============================================================
# TODO: I do not use following methods and class yet.
# In future, we have to use these methods and class.
# ============================================================


def featurize_batch(
    sequence_list: list[str],
    residue_index_list: list[Sequence[int] | None] | None = None,
    buckets: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Featurize the input sequence for the model."""
    bs = len(sequence_list)
    maxlen = max(len(seq) for seq in sequence_list)

    if buckets is not None:
        buckets = sorted(buckets)
        if buckets[-1] < maxlen:
            # If the longest sequence exceeds the largest bucket,
            # pad to the next multiple of 8.
            maxlen = ((maxlen + 7) // 8) * 8
        else:
            for bucket in buckets:
                if maxlen <= bucket:
                    maxlen = bucket
                    break

    batch = {
        "lm.input_ids": np.full((bs, maxlen + 2), PAD_IDX, dtype=int),
        "lm.pos_id": np.zeros((bs, maxlen + 2), dtype=int),
        "lm.seq_id": np.zeros((bs, maxlen + 2), dtype=int),
        "entity_id": np.zeros((bs, maxlen), dtype=np.int64),
        "asym_id": np.zeros((bs, maxlen), dtype=np.int64),
        "sym_id": np.zeros((bs, maxlen), dtype=np.int64),
        "aatype": np.zeros((bs, maxlen, 21), dtype=np.float32),
        "aatype_int": np.zeros((bs, maxlen), dtype=int),
        "res_idx": np.zeros((bs, maxlen), dtype=int),
        "seq_mask": np.zeros((bs, maxlen), dtype=bool),
        "atom14_mask": np.zeros((bs, maxlen, 14), dtype=bool),
        "atom37_mask": np.zeros((bs, maxlen, 37), dtype=bool),
        "pseudo_beta": np.zeros((bs, maxlen), dtype=int),
    }

    for i in range(bs):
        seq = sequence_list[i]
        length = len(seq)
        residue_index = residue_index_list[i] if residue_index_list else None
        featurized = featurize(seq, residue_index)
        for k, v in featurized.items():
            if k.startswith("lm."):
                batch[k][i, : length + 2] = v
            else:
                batch[k][i, :length] = v

    return batch


class InputFeaturizer:
    """
    Featurizes input amino acid sequences into batched tensor representations.

    This module acts as the data pipeline entry point for AtlasFold inference,
    processing raw protein sequences (and optional residue indices) into the
    structured tensor dictionaries required by the AtlasLM language model trunk
    and the downstream diffusion module.

    A critical feature of this class is its handling of conformational diversity
    through deterministic stochasticity:

    * Deterministic Mode: If `seeds` is None, sequences are featurized normally
      without any Masked Language Modeling (MLM) masking.
    * Stochastic Mode: If a list of `seeds` is provided, the featurizer pre-computes
      reproducible MLM masking patterns. This introduces controlled uncertainty into
      the language model trunk, which allows the downstream diffusion model to sample
      multiple diverse conformational states for the exact same input sequence.

    I note that the deterministicity is only applied to the trunk featurization,
    while the diffusion module is always **stochastic**.

    Parameters
    ----------
    seeds : list of int or None, optional
        A list of master seeds used to generate reproducible stochastic masks for
        sampling multiple conformations. If None or empty, trunk featurization
        is deterministic.
    max_length : int, optional
        The maximum sequence length supported for the pre-computed shared masking
        patterns. Default is 20000.
    """

    def __init__(
        self,
        max_length: int = 20000,
        buckets: list[int] | None = None,
    ) -> None:
        super().__init__()
        if buckets is None:
            buckets = DEFAULT_BUCKETS

        self.max_length: int = max_length
        self.buckets: list[int] = sorted(buckets)

    def iter_batched_inputs(
        self,
        input_list: list[tuple[str, str]],
        max_tokens_per_batch: int = 1024,
    ) -> Iterable[dict]:
        """Featurize the input sequence for the model."""
        for batch in self.iter_batched_sequences(input_list, max_tokens_per_batch):
            feat = featurize_batch(
                batch["sequences"],
                residue_index_list=None,
            )
            feat = {k: torch.from_numpy(v) for k, v in feat.items()}
            batch["feat"] = feat
            yield batch

    def iter_batched_sequences(
        self,
        input_list: list[tuple[str, str]],
        max_tokens_per_batch: int = 1024,
    ) -> Iterable[dict]:
        """Featurize the input sequence for the model."""

        def get_bucketed_length(length: int) -> int:
            """Finds the appropriate bucket length for the folding trunk."""
            if self.buckets and length <= self.buckets[-1]:
                for bucket in self.buckets:
                    if length <= bucket:
                        return bucket
            return ((length + 15) // 16) * 16

        def get_empty_batch() -> dict[str, list]:
            """Returns an empty batch dictionary."""
            return dict(headers=[], seeds=[], sequences=[])

        batch = get_empty_batch()
        max_len_in_batch = 0

        for header, seq in input_list:
            seq_len = len(seq)
            new_max_len = max(max_len_in_batch, seq_len)
            bucketed_len = get_bucketed_length(new_max_len)

            new_tokens = (len(batch["sequences"]) + 1) * bucketed_len
            if new_tokens > max_tokens_per_batch and len(batch["sequences"]) > 0:
                yield batch
                batch = get_empty_batch()
                max_len_in_batch = seq_len
            else:
                max_len_in_batch = new_max_len

            batch["headers"].append(header)
            batch["sequences"].append(seq)

        if len(batch["sequences"]) > 0:
            yield batch

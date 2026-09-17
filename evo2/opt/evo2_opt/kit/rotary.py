"""R1: the attention blocks' rotary cos/sin tables reused across scoring forwards. vortex's ``_update_cos_sin_cache`` rebuilds the tables on
every call of a model left in training mode once they were built under inference mode (``self.training and self._cos_cached.is_inference()``),
which is every scoring forward after the first; under inference mode those tables are usable as they are, and a table built for a longer
length holds the same leading rows (arange, outer product, cos/sin and the dtype cast are elementwise), so the rebuild is skipped exactly when
its result would be the cached rows: inference mode on, no xPos scale, same device and dtype, a length within the cached one. Training-mode
autograd use, a longer length, a device or dtype change rebuild as the stock does."""
from __future__ import annotations

import torch
from vortex.model.positional_embeddings import LinearlyScaledRotaryEmbedding

LEVER = "R1_rotary_table_reuse"
ORIG = {"update": LinearlyScaledRotaryEmbedding._update_cos_sin_cache}


def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
    cos = self._cos_cached
    if (cos is not None and self.scale is None and torch.is_inference_mode_enabled() and seqlen <= self._seq_len_cached
            and cos.device == device and cos.dtype == dtype and self._sin_cached is not None):
        return
    return ORIG["update"](self, seqlen, device=device, dtype=dtype)

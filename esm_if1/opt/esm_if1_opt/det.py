"""``--det``: the kit modes' deterministic-recipe switch, default 0. ``--det 1`` has nothing to switch on this engine and is named on one
NOT APPLIED line under either mode: ``fast`` sets no deterministic-algorithm switch (the graph network's scatter aggregation has no
deterministic implementation to select) and needs none for its records — ``torch.manual_seed(--seed)`` once after the model load, before
the first batch (``batched.run``), makes the sampled FASTA records repeat run to run at a fixed ``--batch_size`` and input order (floats are
not bitwise: scatter aggregation uses atomics; the sampled records are); ``off`` is upstream's script as shipped, unseeded.
"""
from __future__ import annotations

DEFAULT_SEED = 37                              # --seed's default

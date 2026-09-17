"""pipeline_tf — the Borzoi kit: the stock ``borzoi_sad.py`` (calico/borzoi @5c93582 on baskerville @544073b) with the kit's
blocks inserted at anchored stock lines, rendered into ``v17/`` by ``build.py`` from the pinned stock bytes (sha256 asserted),
plus ``v17/kitlib/``: the lookup-table one-hot, the column-chunked threaded host post-processing and its statistics writer, and
the forward call (the traced graph from the second call on, the copy-free return). ``v17/`` is self-contained (stdlib +
numpy/scipy/h5py + the stock packages) and keeps the stock CLI. Tier 1: every dataset of sad.h5 byte-identical to the stock's
(tests/ compare the one-hot and the chunked post-processing with the stock functions on synthetic inputs)."""
KIT = "pipeline_tf"
FROZEN_DIRS = ("v17",)

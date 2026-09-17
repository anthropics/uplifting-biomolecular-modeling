"""kernels.apb.ditfast — the fused DiT block's Triton ROW kernels, carried byte-identically from the producing kit's package
(``kernels.py``: token-stream rows at c_a — adaln, gate, swiglu, resgate, resgate_adaln; ``atom_kernels.py``: atom-stream 2-D row tiles at
c_atom — adaln2, resgate_adaln2, resgate (via resgate_adaln2 ln=False), gate2d, swiglu2d; ``CELLS.json`` / ``NOTICE``: the package's own).
The package's model-bound schedules (the fused 24-block token stack, the fused atom stacks, the conditioning de-duplication and their
install seams) are NOT carried: they walk an engine's module tree and stay in the kits, which bind these kernels through
``opt_core.kernels.apb`` (row ``dit_fast``; ``dit_block_rows()``).  Nothing is imported here: the two modules import torch and triton at
their own module level and are imported only when the row is selected."""

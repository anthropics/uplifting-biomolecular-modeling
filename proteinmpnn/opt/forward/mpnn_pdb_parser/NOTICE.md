# NOTICE — mpnn_pdb_parser

`kit/fast_parse.py` is a modified copy of `helper_scripts/parse_multiple_chains.py` from ProteinMPNN
(https://github.com/dauparas/ProteinMPNN, commit 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57), MIT License, Copyright (c) 2022 Justas Dauparas —
the upstream licence text is reproduced verbatim in `licenses/dauparas__ProteinMPNN__LICENSE`; the modification reads each PDB once and buckets
its ATOM lines by chain. No ProteinMPNN source tree and no model weights are included here:
both come from the user's own checkout of the pinned commit (`MPNN_DIR`).

protenix_fpf_atom_attn_exact — the protenix_v2 kit's EXACT-class atom local-attention kernel (lever atom_attn_exact of that kit; here served to
protenix 1.1.0 through the adapter `ptxfpf/ditfast_ptx1.py`, the same lever word, exact tier only).

PROVENANCE: carried byte-identical (every file but this README) from the kit package protenix_fpf_atom_attn_exact
(package version 1.0.0) where it was first written. Same import name on purpose: a kit that lifts it into the shared core swaps an import.
Licence / notice: NOTICE beside this file (first-party Triton code written for the kit; no third-party source included).

install.py replaces protenix.model.modules.primitives._local_attention (a module global Attention.forward resolves per call; identical in protenix
1.1.0 and 2.0.0) after determining the cuBLAS GEMM numerics of the pinned image per process and a 4-case digest load check (vectors.json); a
mismatch raises LeverRefused by name, which the adapter turns into the lever stepping aside (the stock statement serves, exit 0). The package reads
its switch word PTX_ATOM_ATTN_EXACT, which the arm word sets in-process.

# Sources — `tests/inputs/`

Two OpenFold3 0.4.1 example queries, byte-identical to the upstream files (Apache License, Version 2.0; "Copyright 2026 AlQuraishi Laboratory"
as printed in the upstream LICENSE), renamed by content; nothing in them is edited:

| file | upstream file (`stock/src/`) | what it describes | terms of the described data |
|---|---|---|---|
| `query_7cnx_multimer_590tok.json` | `examples/example_inference_inputs/query_multimer.json` | query `7cnx`: the two protein entity sequences of wwPDB entry 7CNX (chains A/C and B/D), typed as an inference query; no coordinates | PDB archive data: CC0 1.0 (wwPDB) |
| `query_leucine_zipper_68tok.json` | `examples/example_inference_inputs/query_homomer.json` | query `leucine_zipper`: one 34-residue leucine-zipper peptide as a homodimer (chains A/B); upstream names no database entry for it | as the upstream file (Apache-2.0) |

Both are run single-sequence by `warm` and `check` (`--use-msa-server false --use-templates false`), so no alignment or structure file accompanies them.

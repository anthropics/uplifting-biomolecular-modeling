# tests/ — example inputs shipped with the kit (`tests/inputs/`)

`tests/inputs/` holds two ColabFold complex inputs in a3m form, query-only: each file holds the paired query row and one unpaired query row per chain and no
other sequence — no alignment database record (UniRef, BFD, MGnify or other) is in them.

| file | chains | sequence source |
|---|---|---|
| `inputs/1BRS_AD.a3m` | barnase (110 residues) and barstar (89 residues) | wwPDB entry 1BRS, chains A and D |
| `inputs/1A3N_hemoglobin.a3m` | human haemoglobin alpha (141 residues, two copies) and beta (146 residues, two copies) | wwPDB entry 1A3N, chains A and B |

The sequences are those of the named wwPDB entries (wwPDB archive data, CC0 1.0; cite the entries by identifier). The files themselves are
part of the kit's original code and text (Apache License 2.0, `LICENSE` in the kit directory).

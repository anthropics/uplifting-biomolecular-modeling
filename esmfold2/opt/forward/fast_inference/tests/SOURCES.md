# SOURCES — `tests/w4_public_slice.json`

The one data file in this directory, `w4_public_slice.json`, holds three ESMFold2 prediction inputs in upstream's input schema
(`id`, `sequences`, `seeds`, `num_diffusion_samples`). Every sequence in it is one of the two protein chains of PDB entry 1BRS
(barnase / barstar), obtained from the RCSB PDB (https://www.rcsb.org/structure/1BRS):

| input id | chains | content |
|---|---|---|
| `1BRS_barstar1_barnase1` | A (89 residues), B (110 residues) | one barstar chain, one barnase chain |
| `1BRS_barstar3_barnase4` | A–C (89), D–G (110) | three copies of the barstar chain, four of the barnase chain |
| `1BRS_barstar4_barnase4` | A–D (89), E–H (110) | four copies of each chain |

Relation to the entry: the file holds the amino-acid sequences of the entry's two distinct chains as plain one-letter strings, repeated
to form the larger assemblies; it includes no coordinates, no experimental data, no alignment (every `msa` field is `null`) and no
other field of the entry. `seeds` and `num_diffusion_samples` are run settings, not entry data.

Licence / terms: PDB archive data are released by the wwPDB under CC0 1.0. The file is the input of `run.sh warm` and of the kit
README's example; the kit-level list of third-party material is `esmfold2/THIRD_PARTY_NOTICES.md`.

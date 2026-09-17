# Public binder–target complexes — the kit's example inputs

All coordinates are experimental structures from the RCSB PDB (files.rcsb.org), ATOM records of standard residues only (first alt-loc;
waters, ligands and modified residues dropped), re-chained so that the chain playing the BINDER role is chain A and comes first, followed by
the target chain(s) B, C, … (the dl_binder_design convention: binder = the first chain), residues renumbered from 1 per chain.
Barnase–barstar (1BRS) is tiled k:k (copies displaced 70 Å along x, no contacts between copies) to give inputs of ~200–800 residues. The
files contain no designed sequences; PDB entries are available under the wwPDB usage policy (CC0, https://www.rcsb.org/pages/usage-policy).

`index.csv` lists the full set of eight; six ship under `pdbs/` (the two largest 1BRS tilings, `1brs_bb_2to2` and `1brs_bb_4to4`, are left
out to keep the tree small). `python ../../tools/fetch_public_inputs.py --out DIR` rebuilds all eight from the RCSB with the standard library
(retries, deterministic processing, each file's sha256 checked against the table in the script, one provenance line per file).

| id | PDB | binder (chain A) | target chain(s) | binder len | target chains | total residues | ships |
|---|---|---|---|---|---|---|---|
| 1brs_bb_1to1 | 1BRS | barstar chain D (first copy) | barnase ×1 | 87 | 1 | 195 | yes |
| 5jds_nb_pdl1 | 5JDS | nanobody KN035 (chain B) | PD-L1 IgV domain (chain A) | 127 | 1 | 242 | yes |
| 1brs_bb_2to2 | 1BRS | barstar chain D (first copy) | barnase ×2 + barstar ×1 | 87 | 3 | 390 | no |
| 6m0j_rbd_ace2n | 6M0J | SARS-CoV-2 RBD (chain E) | ACE2 residues 19–330 (chain A) | 194 | 1 | 506 | yes |
| 1brs_bb_3to3 | 1BRS | barstar chain D (first copy) | barnase ×3 + barstar ×2 | 87 | 5 | 585 | yes |
| 1yy9_egfrd3_fab | 1YY9 | EGFR domain III residues 310–501 (chain A) | cetuximab Fab light (C) + heavy (D) | 192 | 2 | 623 | yes |
| 1brs_bb_4to4 | 1BRS | barstar chain D (first copy) | barnase ×4 + barstar ×3 | 87 | 7 | 780 | no |
| 6m0j_rbd_ace2 | 6M0J | SARS-CoV-2 RBD (chain E) | ACE2 peptidase domain (chain A) | 194 | 1 | 791 | yes |

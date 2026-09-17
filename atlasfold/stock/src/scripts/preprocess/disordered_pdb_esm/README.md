# Construct the Monomer Disordered-PDB Dataset

This dataset augments experimentally resolved monomer structures with ESMFold predictions for chains that contain substantial unresolved regions. It is distinct from the standard RCSB training dataset.

The ESMFold-based workflow was restored from the Git history before commit `c8f9541`. That implementation, rather than the later AlphaFold-Multimer replacement, defines the intended monomer dataset construction:

1. Read the preprocessed monomer RCSB LMDB and select chains with at least 40 residues whose Cα coordinates are unresolved.
2. Deduplicate the selected sequences before structure prediction.
3. Predict one structure per unique sequence with ESMFold.
4. Align each ESMFold prediction to every experimental chain with the same sequence and retain predictions with GDT-TS at least 60 over resolved residues.
5. Cluster the retained sequences at 40% sequence identity with 80% coverage.
6. Convert the retained ESMFold PDB files to AtlasFold monomer NPZ records and construct the LMDB and manifest. Prediction metadata records `ESMFold` as the source model.

The scripts use the following intermediate layout under the supplied data root:

```text
disordered_pdb_esm/
├── rcsb_sequences.fasta     # RCSB chains with at least 40 unresolved Cα atoms
├── unique_sequences.fasta   # Deduplicated ESMFold inputs
├── pdb_all/                 # All ESMFold predictions
├── pdb/                     # Predictions passing the GDT-TS filter
├── rcsb_clusters.csv        # MMseqs2 clusters
├── npz/                     # Processed ESMFold structures
├── structure.lmdb
├── manifest.msgpack
└── manifest.json
```

The pipeline stages are `a1_extract_sequences.py`, `a2_prepare_apo_input.py`, `a3_check_gdt.py`, `b1_cluster.py`, `c1_process.py`, and `c2_construct_lmdb.py`. Run ESMFold on `unique_sequences.fasta` between the `a2` and `a3` stages and place its PDB outputs in `pdb_all/`. AlphaFold2 preprocessing is maintained separately in `scripts/preprocess/disordered_pdb_af2/`.

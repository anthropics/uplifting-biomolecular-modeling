# Construct the Multimer Disordered-PDB Dataset

This directory contains the preprocessing pipeline for the multimer disordered-PDB training source. It operates on the PDB entries already selected in `rcsb_multimer/manifest.msgpack`, expands their first biological assemblies from the original RCSB mmCIF files, and writes a separate `disordered_pdb_multimer` dataset. It is not the standard RCSB preprocessing workflow.

## Prerequisites

- The raw RCSB divided mmCIF archive.
- A completed `rcsb_multimer/manifest.msgpack` under the same data root.
- MMseqs2 for sequence clustering.

## Preprocessing Workflow

Run the scripts from this directory in order.

### 1. Extract entity sequences

Read the PDB IDs from the multimer RCSB manifest, extract unique protein-entity sequences from their mmCIF files, and write `disordered_pdb_multimer/rcsb_sequences.fasta`.

```bash
python a1_extract_sequences.py \
    --cif_dir /raw_data/RCSB/mmCIF \
    --data_dir /path/to/data/root \
    --num_workers 16
```

### 2. Cluster sequences

Cluster the extracted entities with MMseqs2 at 40% sequence identity and 80% coverage. The resulting entity-to-cluster mapping is written to `rcsb_clusters.csv`.

```bash
python a2_cluster.py \
    --data_dir /path/to/data/root \
    --mmseqs "$(which mmseqs)"
```

### 3. Process biological assemblies

Expand the first biological assembly for each selected PDB entry, retain valid protein chains, remove severe clashes, detect interfaces, and save one complex-level NPZ and JSON metadata record per entry. Chains must contain at least four resolved residues, and no residue type may occupy more than 80% of a chain. The output metadata records `AlphaFold-Multimer` as the prediction source.

```bash
python b1_process.py \
    --cif_dir /raw_data/RCSB/mmCIF \
    --data_dir /path/to/data/root \
    --num_workers 16
```

### 4. Construct the LMDB

Pack the processed NPZ records into `structure.lmdb`, combine their JSON metadata into `manifest.msgpack`, and add chain- and interface-cluster sizes.

```bash
python b2_construct_lmdb.py \
    --data_dir /path/to/data/root
```

## Dataset Structure

```text
disordered_pdb_multimer/
├── rcsb_sequences.fasta
├── rcsb_clusters.csv
├── npz/
│   ├── 1abc.npz
│   ├── 1abc.json
│   └── ...
├── structure.lmdb
└── manifest.msgpack
```

The `npz/` directory is an intermediate representation and can be removed after the LMDB and manifest have been verified.

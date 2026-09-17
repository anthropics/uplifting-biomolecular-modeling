# Construct the AlphaFold2 Distillation Dataset

Following AlphaFold 3, we use AlphaFold 2 predicted structures from the **MGnify** sequences as distillation datasets. These are available at the [OpenFold Portal](https://portal.openfold.omsf.io/).

The dataset version we used contains **430k short sequences (< 200 residues) and 16M long sequences (≥ 200 residues)**.

## Preprocessing Workflow

After downloading the dataset, you can preprocess it by executing the scripts in this directory sequentially.

### 1. Extract Sequences and Structures
Process the raw PDB/CIF files (can be `.zst` compressed) and save them as compressed NumPy (`.npz`) files. This script expects the input directory to be organized into subdirectories (shards).

```bash
python a1_process.py \
    --pdb_dir /path/to/raw/pdb_files \
    --data_dir /path/to/data/root \
    --name MGnify_AF2 \
    --num_workers 16
```

### 2. Construct LMDB Database
Merge the processed `.npz` files into a single LMDB file for efficient training access. This step also generates a `manifest.msgpack` file containing metadata (including pLDDT) for all entries.

```bash
python a2_construct_lmdb.py \
    --data_dir /path/to/data/root \
    --name MGnify_AF2 \
    --size_gb 1000
```

### 3. Filter by Confidence
Filter the dataset to remove low-confidence predictions. This script creates a new manifest file containing only entries that meet the pLDDT threshold.

```bash
python a3_filter.py \
    --data_dir /path/to/data/root \
    --name MGnify_AF2 \
    --out_prefix manifest_plddt70 \
    --threshold 70.0
```

> [!NOTE]
> Following pLDDT > 70 filtering, the processed dataset contains approximately **360k short sequences and 15.4M long sequences**.

## Dataset Structure
After preprocessing, your data directory will look like this:
```
MGnify_AF2/
├── npz/                   # Intermediate processed files (can be deleted after Step 2)
│   ├── shard_0/
│   │   ├── MGY000....npz
│   │   └── ...
│   └── ...
├── structure.lmdb         # Combined structure database
├── manifest.msgpack       # Metadata for all entries
└── manifest_plddt70.msgpack # Filtered metadata for training
```

> [!TIP]
> The `npz/` directory contains intermediate files and can be safely removed once `structure.lmdb` and `manifest.msgpack` are successfully created to reclaim disk space.


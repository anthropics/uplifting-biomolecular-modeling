# Construct the CAMEO Validation Dataset

This directory contains scripts to preprocess the CAMEO validation dataset, which is derived from the RCSB PDB.
This workflow assumes that you have already downloaded the raw mmCIF files during training data preparation.

## Preprocessing Workflow

Execute the scripts sequentially to prepare the dataset.

### 1. Process Structures
Perform detailed processing of mmCIF files, including geometry validation and filtering (resolution ≤ 9.0Å, length ≥ 16). This step uses the cluster information and saves individual chains as `.npz` files.

```bash
python a1_process.py \
    --cif_dir /raw_data/RCSB/mmCIF \
    --data_dir /path/to/data/root/ \
    --target_id /assets/cameo_val_ids.txt
```

### 2. Construct LMDB Database
Merge the processed `.npz` files into a single LMDB database and generate a `manifest.msgpack` containing all metadata and cluster size information.

```bash
python a2_construct_lmdb.py \
    --data_dir /path/to/data/root/ \
    --size_gb 1
```

## Dataset Structure
After preprocessing, your data directory will look like this:
```
cameo_val/
├── npz/                    # Intermediate processed files (can be deleted after Step 4)
│   ├── 1abc_A.npz
│   └── 1abc_A.json
│   └── ...
├── structure.lmdb          # Combined structure database
└── manifest.msgpack        # Metadata
```

> [!TIP]
> The `npz/` directory can be safely removed once `structure.lmdb` and `manifest.msgpack` are successfully created to reclaim disk space.

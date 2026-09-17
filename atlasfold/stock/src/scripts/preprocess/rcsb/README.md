# Construct the RCSB Dataset

This directory contains scripts to preprocess the RCSB PDB dataset for training. The workflow includes sequence extraction, clustering, structure processing, and database construction.

## Raw Data Download and Preparation

You can download the raw data files from the RCSB PDB website using the following commands:

```bash
# Create your own raw data directory
mkdir -p /raw_data/RCSB/mmCIF

# mmCIF files for each structure (divided into subdirectories)
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ /raw_data/RCSB/mmCIF/
```

## Preprocessing Workflow

Execute the scripts sequentially to prepare the dataset.

### 1. Extract Sequences
Extract protein sequences from mmCIF files to prepare for clustering. This script filters entries by release date (default: ≤ 2020-05-01).

```bash
python a1_extract_sequences.py \
    --cif_dir /raw_data/RCSB/mmCIF \
    --data_dir /path/to/data/root/ \
    --num_workers 16
```

### 2. Cluster Sequences
Cluster the extracted sequences using MMseqs2 to handle redundancy. This generates `rcsb_clusters.csv`.

```bash
python a2_cluster.py \
    --data_dir /path/to/data/root/ \
    --mmseqs $(which mmseqs)
```

### 3. Process Structures
Perform detailed processing of mmCIF files, including geometry validation and filtering (resolution ≤ 9.0Å, length ≥ 16). This step uses the cluster information and saves individual chains as `.npz` files.

```bash
python b1_process.py \
    --cif_dir /raw_data/RCSB/mmCIF \
    --data_dir /path/to/data/root/ \
    --num_workers 16
```

### 4. Construct LMDB Database
Merge the processed `.npz` files into a single LMDB database and generate a `manifest.msgpack` containing all metadata and cluster size information.

```bash
python b2_construct_lmdb.py \
    --data_dir /path/to/data/root/ \
    --size_gb 500
```

### 5. Filter for Confidence Training
Create a specialized manifest for confidence-only model training by filtering entries based on high-resolution experimental data (default: 0.1Å to 3.0Å).

```bash
python b3_filter.py \
    --data_dir /path/to/data/root/ \
    --out_prefix manifest_confidence \
    --min_resolution 0.1 \
    --max_resolution 3.0
```

## Dataset Structure
After preprocessing, your data directory will look like this:
```
rcsb/
├── rcsb_sequences.fasta    # Extracted sequences
├── rcsb_clusters.csv       # Clustering results
├── npz/                    # Intermediate processed files (can be deleted after Step 4)
│   ├── ab/
│   │   ├── 1abc/
│   │   │   ├── 1abc_A.npz
│   │   │   └── 1abc_A.json
│   │   └── ...
│   └── ...
├── structure.lmdb          # Combined structure database
├── manifest.msgpack        # Metadata for all entries
└── manifest_confidence.msgpack # Filtered manifest for confidence training
```

> [!TIP]
> The `npz/` directory can be safely removed once `structure.lmdb` and `manifest.msgpack` are successfully created to reclaim disk space.

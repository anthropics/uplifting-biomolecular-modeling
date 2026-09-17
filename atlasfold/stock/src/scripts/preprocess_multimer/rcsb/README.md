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
Extract protein entity sequences from mmCIF files to prepare for clustering. This script filters entries by release date (default: ≤ 2021-09-30).

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
Perform detailed processing of mmCIF files, including biological assembly expansion, geometry validation, all-atom clash filtering, interface detection, and filtering (resolution ≤ 9.0Å for train). Assemblies with more than 20 valid protein chains are reduced to the closest 20 chains around a sampled interface/contact seed, following the AF3 SI 2.5.4 subcomplex extraction idea. This step uses the cluster information for train and saves one complex-level `.npz` and `.json` metadata file per PDB entry.

```bash
python b1_process.py \
    --cif_dir /raw_data/RCSB/mmCIF \
    --data_dir /path/to/data/root/ \
    --split train \
    --num_workers 16
```

### 4. Construct LMDB Database
Merge the processed `.npz` files into a single LMDB database and generate a `manifest.msgpack` containing all metadata and cluster size information.

```bash
python b2_construct_lmdb.py \
    --data_dir /path/to/data/root/ \
    --size_gb 500
```

### 5. Filter for Confidence Head Training
Create a specialized manifest for confidence-head training by filtering complex-level metadata to high-resolution experimental structures (default: 0.1Å to 3.0Å). The script also keeps lightweight AF3/RCSB manifest guards for release date, chain count, and minimum chain length; the expensive bioassembly filters are already applied during Step 3.

```bash
python b3_filter.py \
    --data_dir /path/to/data/root/ \
    --out_prefix manifest_confidence \
    --min_resolution 0.1 \
    --max_resolution 3.0
```

### 6. Process Template Structures
Convert raw template chain `.npz` files into AtlasFold monomer atom14 `.npz` files that can be loaded by `MonomerDataPipeline`.

```bash
python c1_process_templates.py \
    --template_dir /path/to/templates \
    --data_dir /path/to/data/root/ \
    --num_workers 16
```

### 7. Construct Template LMDB
Pack processed template structures into an LMDB database and write a template manifest.

```bash
python c2_construct_template_lmdb.py \
    --data_dir /path/to/data/root/ \
    --size_gb 256
```

### 8. Create Entry-Template Mapping
Convert per-entry template-hit metadata into an LMDB mapping keyed by `{pdb_id}_{entity_id}` from `manifest.msgpack`. Only templates released at least 60 days before the entry release date are kept. The source `idx_map` is treated as 1-based and exposed as explicit `entry_indices` and `template_indices` fields.

```bash
python c3_create_template_mapping.py \
    --metadata_dir /path/to/template/metadata \
    --data_dir /path/to/data/root/ \
    --num_workers 16 \
    --size_gb 64
```

### 9. Construct Validation Set
Create the protein-multimer validation set in `rcsb_multimer_val`. Validation follows the AF3 validation date split (2021-10-01 through 2023-01-12), uses at most 20 chains, and keeps complexes up to 1536 protein residues, matching the AF2-multimer crop length. Templates and cluster-size recomputation are not required for validation.

```bash
# Step 9-1: Process AF3 validation-window structures.
python b1_process.py \
    --cif_dir /raw_data/RCSB/mmCIF \
    --data_dir /path/to/data/root/ \
    --split val \
    --num_workers 16

# Step 9-2: Select validation IDs and write manifest.{msgpack,json}.
python d1_get_val_ids.py \
    --data_dir /path/to/data/root/ \
    --mmseqs $(which mmseqs) \
    --num_workers 16

# Step 9-3: Pack selected validation structures into LMDB.
python d2_construct_val_lmdb.py \
    --data_dir /path/to/data/root/ \
    --size_gb 64
```

## Dataset Structure
After preprocessing, your data directory will look like this:
```
rcsb_multimer/
├── rcsb_sequences.fasta    # Extracted sequences
├── rcsb_clusters.csv       # Clustering results
├── npz/                    # Intermediate processed files (can be deleted after Step 4)
│   ├── ab/
│   │   ├── 1abc/
│   │   │   ├── 1abc.npz
│   │   │   └── 1abc.json
│   │   └── ...
│   └── ...
├── structure.lmdb          # Combined structure database
├── manifest.msgpack        # Metadata for all entries
├── manifest_confidence.msgpack # Filtered manifest for confidence training
├── template.lmdb           # Combined template structure database
├── template_manifest.msgpack
├── template_mapping.lmdb   # Entry-to-template hit metadata database
└── templates/
    └── npz/                # Processed template atom14 NPZ files
rcsb_multimer_val/
├── validation_ids.txt      # Selected validation PDB IDs
├── npz/                    # Validation candidate processed files
├── structure.lmdb          # Selected validation structures
├── manifest.msgpack        # Selected validation metadata
└── manifest.json
```

> [!TIP]
> The `npz/` directory can be safely removed once `structure.lmdb` and `manifest.msgpack` are successfully created to reclaim disk space.

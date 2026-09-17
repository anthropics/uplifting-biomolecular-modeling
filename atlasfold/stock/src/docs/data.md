# AtlasFold structural datasets

The processed AtlasFold monomer and multimer training data are distributed from the [AtlasFold structural-data folder](https://drive.google.com/drive/folders/1EiRTKSUL3iD_MQ_0qmj5Sb-2KMh-3kmS).

## Dataset inventory

| Directory | Description | Entries |
| --- | --- | ---: |
| `rcsb` | Experimentally determined monomer chains from the RCSB Protein Data Bank | 490,703 |
| `disordered_pdb_af2` | AlphaFold2-predicted monomer distillation structures derived from disordered-PDB sequences | 11,828 |
| `disordered_pdb_esm` | ESMFold-predicted monomer distillation structures derived from disordered-PDB sequences | 59,181 |
| `mgnify_long` | Long MGnify monomer distillation structures | 16,099,404 |
| `mgnify_short` | Short MGnify monomer distillation structures | 430,418 |
| `cameo_val` | CAMEO monomer validation targets | 362 |
| `rcsb_multimer` | Experimentally determined protein complexes from the RCSB Protein Data Bank | 177,363 complexes |
| `disordered_pdb_afm` | AlphaFold-Multimer-predicted multimer distillation structures | 29,431 complexes |
| `rcsb_multimer_val` | Protein-complex validation targets | 512 complexes |

Multimer complexes can provide multiple effective samples through chain and interface sampling. `rcsb_multimer` provides 1,125,363 effective samples and `disordered_pdb_afm` provides 176,472 effective samples.

`disordered_pdb_esm` records an intermediate monomer dataset used during early development. The released final monomer stage configurations use `disordered_pdb_af2`.

## Directory layout

```text
atlasfold_data/
├── cameo_val/
├── disordered_pdb_af2/
├── disordered_pdb_afm/
├── disordered_pdb_esm/
├── mgnify_long/
├── mgnify_short/
├── rcsb/
├── rcsb_multimer/
└── rcsb_multimer_val/
```

Each dataset directory contains a `manifest.msgpack` metadata file and a `structure.lmdb/` structure database. Some datasets also include sequence files, cluster assignments, preprocessing metadata, or template resources. `rcsb_multimer` includes `template.lmdb/`, `template_mapping.lmdb/`, and `template_manifest.msgpack` for template-based training.

## Training configuration

Set `train.data.data_root` to the extracted root:

```yaml
train:
  data:
    data_root: /path/to/atlasfold_data
```

The released monomer configurations use `rcsb`, `disordered_pdb_af2`, `mgnify_long`, and `mgnify_short`, with `cameo_val` for validation. The multimer configurations use `rcsb_multimer`, `disordered_pdb_afm`, `mgnify_long`, and `mgnify_short`, with `rcsb_multimer_val` for validation. Exact sampling weights differ by stage and are recorded in `configs/monomer/` and `configs/multimer/`.

## Preprocessing

The repository contains preprocessing workflows under `scripts/preprocess/` and `scripts/preprocess_multimer/`. Follow the README in each source directory and execute its numbered scripts in order. These scripts document how RCSB, distillation, validation, and template resources were converted to the released LMDB format.

## Data terms

The AtlasFold source code, model weights, and released datasets are distributed under the MIT License. Underlying source records may remain subject to their original providers' terms; consult the RCSB PDB, MGnify, AlphaFold, ESMFold, and other upstream data providers before redistribution or commercial use.

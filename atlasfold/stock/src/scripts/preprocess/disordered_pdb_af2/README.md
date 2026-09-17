# Construct the Monomer Disordered-PDB AlphaFold2 Dataset

This directory contains only training-data preprocessing. MSA generation and AlphaFold2 inference orchestration are intentionally outside this pipeline.

The input/output dataset directory is `<data_dir>/disordered_pdb_af2/`. Before running `a3`, place the completed ColabFold outputs in `pdb/seq_<id>.pdb` and `scores/seq_<id>.json`.

Run the stages in order:

```bash
python scripts/preprocess/disordered_pdb_af2/a1_extract_sequences.py --data_dir /cache/wykim_lab/icl_shwan/atlasfold_data --num_workers 32
python scripts/preprocess/disordered_pdb_af2/a2_prepare_apo_input.py --data_dir /cache/wykim_lab/icl_shwan/atlasfold_data
python scripts/preprocess/disordered_pdb_af2/a3_check_gdt.py --data_dir /cache/wykim_lab/icl_shwan/atlasfold_data --threshold 60
python scripts/preprocess/disordered_pdb_af2/b1_cluster.py --data_dir /cache/wykim_lab/icl_shwan/atlasfold_data
python scripts/preprocess/disordered_pdb_af2/c1_process.py --data_dir /cache/wykim_lab/icl_shwan/atlasfold_data --num_workers 32
python scripts/preprocess/disordered_pdb_af2/c2_construct_lmdb.py --data_dir /cache/wykim_lab/icl_shwan/atlasfold_data --size_gb 20
```

The final dataset contains `structure.lmdb`, `manifest.msgpack`, and `manifest.json`. The manifest records `AlphaFold2` as the prediction model and includes cluster IDs and cluster sizes for cluster-aware training sampling.

`ground_truth_npz/` stores experimental coordinates used only for GDT-TS filtering. `npz/` stores accepted AF2 predictions; keeping these directories separate prevents experimental coordinates from entering the distillation LMDB.

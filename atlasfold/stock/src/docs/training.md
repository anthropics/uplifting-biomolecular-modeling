# Training AtlasFold

This guide covers AtlasFold monomer training and AtlasFold-M fine-tuning. AtlasLM model code and pretrained/intermediate weights are included in the release, but the AtlasLM pretraining pipeline is not part of this repository.

## Requirements

Install the repository with folding, training, and optional cuEquivariance dependencies:

```bash
pip install -e ".[fold,train,cuequiv]"
```

Training requires Python 3.10 or later, CUDA GPUs, and enough aggregate devices to divide the configured global batch size exactly. The released configurations use mixed BF16 precision, per-device batch size 1, and global batch size 256.

The structure models use `atlaslm-3b-base`. Its final weights are downloaded from Hugging Face on the first run unless `model.lm_path` is set to a local file.

## Training data

Download and extract the processed datasets from the [AtlasFold structural-data folder](https://drive.google.com/drive/folders/1EiRTKSUL3iD_MQ_0qmj5Sb-2KMh-3kmS). See the [data guide](data.md) for dataset counts, directory layout, and provenance.

Point the configuration at the extracted root either by editing the stage YAML or with a command-line override:

```bash
python scripts/train_monomer.py configs/monomer/train_stage1.yaml \
    --debug \
    --override train.data.data_root=/path/to/atlasfold_data
```

The directory must contain the dataset names referenced by the selected stage configuration. In particular, monomer stages use `rcsb`, `disordered_pdb_af2`, `mgnify_long`, `mgnify_short`, and `cameo_val`; multimer stages use `rcsb_multimer`, `disordered_pdb_afm`, `mgnify_long`, `mgnify_short`, and `rcsb_multimer_val`.

## Released checkpoints

Final inference weights are hosted on Hugging Face. Stage handoff checkpoints are in the [intermediate-checkpoint folder](https://drive.google.com/drive/folders/1TDz2Ng4-zYfpTxlqg5wD3MkzwJyL2gTm).

| Model | Released intermediate checkpoint | Training position |
| --- | --- | ---: |
| AtlasLM-3B | `atlaslm-stage1-1500k.pth` | 1.5M updates, before the long-context stage |
| AtlasFold | `atlasfold-stage1-45k.pth` | End of monomer stage 1 |
| AtlasFold | `atlasfold-stage2-90k.pth` | End of monomer stage 2 |
| AtlasFold | `atlasfold-stage3-98k.pth` | End of monomer stage 3 |
| AtlasFold-M | `atlasfold-m-stage1-21k.ckpt` | End of multimer stage 1 |
| AtlasFold-M | `atlasfold-m-stage2-23k.ckpt` | End of multimer stage 2 |

The final AtlasFold monomer checkpoint follows stage 4 at 100k total updates, and the final AtlasFold-M checkpoint follows stage 3 at 25k multimer fine-tuning updates. Use the Hugging Face weights for inference and the intermediate checkpoints for stage-to-stage training continuation.

## Check a configuration

Run a short debug job before a full distributed job. Debug mode uses one GPU, zero data-loader workers, 50 training batches, and 10 validation batches.

```bash
python scripts/train_monomer.py configs/monomer/train_stage1.yaml \
    --debug \
    --override train.data.data_root=/path/to/atlasfold_data

python scripts/train_multimer.py configs/multimer/train_stage1.yaml \
    --debug \
    --init_weight /path/to/atlasfold-260703.pth \
    --override train.data.data_root=/path/to/atlasfold_data
```

The debug flag shortens the run but does not reduce the configured crop length or diffusion batch size. Apply explicit overrides if the debug job exceeds device memory.

## Monomer training

AtlasFold monomer training uses four progressively longer crop stages:

| Stage | Updates | `max_length` | `max_seq_length` | Initialization |
| --- | ---: | ---: | ---: | --- |
| 1 | 45k | 256 | 512 | Random structure-model initialization |
| 2 | 45k | 384 | 768 | Stage 1 checkpoint |
| 3 | 8k | 512 | 1,024 | Stage 2 checkpoint |
| 4 | 2k | 640 | 1,280 | Stage 3 checkpoint; trunk initialized from EMA and frozen |

AtlasLM remains frozen throughout structure training. Start stage 1 with:

```bash
python scripts/train_monomer.py configs/monomer/train_stage1.yaml \
    --out_dir ./experiments \
    --num_nodes "$NUM_NODES" \
    --num_gpus "$NUM_GPUS" \
    --override train.data.data_root=/path/to/atlasfold_data
```

Start stage 2 from the released stage 1 checkpoint without restoring the previous optimizer state:

```bash
python scripts/train_monomer.py configs/monomer/train_stage2.yaml \
    --out_dir ./experiments \
    --num_nodes "$NUM_NODES" \
    --num_gpus "$NUM_GPUS" \
    --resume_from_checkpoint /path/to/atlasfold-stage1-45k.pth \
    --override \
        train.data.data_root=/path/to/atlasfold_data \
        train.load_opt_state=false
```

Use `configs/monomer/train_stage3.yaml` with `atlasfold-stage2-90k.pth` for stage 3, then `configs/monomer/train_stage4.yaml` with `atlasfold-stage3-98k.pth` for stage 4. Stage 4's configuration freezes the trunk, initializes it from EMA weights, and uses a fixed maximum learning rate of `1e-3`.

## Multimer fine-tuning

AtlasFold-M fine-tuning uses three crop stages:

| Stage | Updates | `max_length` | `max_seq_length` | `max_contiguous_chains` | Initialization |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 21k | 384 | 768 | 6 | Final AtlasFold monomer weights |
| 2 | 2k | 640 | 1,280 | 6 | Multimer stage 1 checkpoint |
| 3 | 2k | 768 | 1,536 | 8 | Multimer stage 2 checkpoint; trunk initialized from EMA and frozen |

Start stage 1 from the final monomer state dict:

```bash
python scripts/train_multimer.py configs/multimer/train_stage1.yaml \
    --out_dir ./experiments \
    --init_weight /path/to/atlasfold-260703.pth \
    --num_nodes "$NUM_NODES" \
    --num_gpus "$NUM_GPUS" \
    --override train.data.data_root=/path/to/atlasfold_data
```

Continue with stage 2:

```bash
python scripts/train_multimer.py configs/multimer/train_stage2.yaml \
    --out_dir ./experiments \
    --num_nodes "$NUM_NODES" \
    --num_gpus "$NUM_GPUS" \
    --resume_from_checkpoint /path/to/atlasfold-m-stage1-21k.ckpt \
    --override \
        train.data.data_root=/path/to/atlasfold_data \
        train.load_opt_state=false
```

Use `configs/multimer/train_stage3.yaml` with `atlasfold-m-stage2-23k.ckpt` for stage 3. Stage 3 freezes the trunk and initializes it from EMA weights.

## Distributed batch size

The training scripts set gradient accumulation so that:

```text
global_batch_size = batch_size * num_nodes * num_gpus * accumulate_grad_batches
```

`global_batch_size` must be divisible by `batch_size * num_nodes * num_gpus`. The released configuration uses `global_batch_size=256` and `batch_size=1`. Change `train.data.global_batch_size` only when intentionally changing the optimization schedule.

## Configuration overrides

`--override` accepts one or more OmegaConf dot-list assignments. Put all assignments after one `--override` flag:

```bash
python scripts/train_monomer.py configs/monomer/train_stage1.yaml \
    --num_gpus 8 \
    --override \
        train.data.data_root=/path/to/atlasfold_data \
        train.data.num_workers=4 \
        train.wandb.use=false
```

Command-line options such as `--out_dir`, `--num_nodes`, `--num_gpus`, and `--num_workers` override the corresponding configuration values. `--wandb` and `--compile` enable their respective features.

## Memory tuning

### Activation checkpointing

`blocks_per_ckpt` controls activation checkpointing for the trunk, diffusion head, confidence head, and multimer template module. `null` disables checkpointing for that stack. A positive integer groups that many consecutive blocks into each checkpointed region; `1` checkpoints every block and minimizes peak activation memory at the cost of recomputation. Larger values create fewer checkpointed regions and generally trade more memory for speed.

```yaml
model:
  trunk:
    blocks_per_ckpt: 1
  diffusion_head:
    blocks_per_ckpt: 1
  confidence_head:
    blocks_per_ckpt: 1
  # Multimer only:
  template_module:
    blocks_per_ckpt: 1
```

Checkpointing is applied only while a stack is training with gradients enabled. It does not affect inference or a frozen trunk.

### Smooth-lDDT loss chunking

The smooth-lDDT loss constructs pairwise atom-distance tensors for each diffusion sample. `train.loss.diffusion_loss.smooth_lddt_loss.chunk_size` controls how many diffusion samples are evaluated in each checkpointed loss chunk. `null` evaluates all samples together without loss checkpointing; a smaller positive integer lowers peak memory but increases backward recomputation. `1` gives the lowest-memory setting.

```yaml
train:
  loss:
    diffusion_loss:
      smooth_lddt_loss:
        chunk_size: 1
```

This option matters only when `train.loss.weights.smooth_lddt` is greater than zero. Chunking operates over diffusion samples, not residues, so it does not remove the quadratic dependence on crop length.

### Pairwise-distance kernel

`train.loss.diffusion_loss.smooth_lddt_loss.use_kernel` enables the optimized pairwise-distance kernel used by the smooth-lDDT loss. Set it to `false` when the kernel is unavailable; this uses the equivalent PyTorch calculation with higher peak memory.

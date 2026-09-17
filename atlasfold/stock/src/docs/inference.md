# AtlasFold inference

AtlasFold provides separate monomer and multimer runners for single-target folding, streaming and batched inference.

Install Python 3.10 or later. A CUDA GPU is strongly recommended:

```bash
pip install "atlasfold[fold,cuequiv]"
```

## Command-line inference

Use the CLI for FASTA-driven jobs that should write complete output directories, summaries, confidence files, and completion markers automatically:

```bash
atlasfold monomer --input-fasta monomers.fasta --out-dir predictions/monomers
atlasfold multimer --input-fasta multimers.fasta --out-dir predictions/multimers
```

### Output layout

Each FASTA record is written to a separate directory named after the first whitespace-delimited token in its header. With the default one seed and five samples, the `protein_a` monomer example produces:

```text
predictions/monomers/protein_a/
├── protein_a_seed-1_sample-0_model.cif
├── protein_a_seed-1_sample-0_confidence.json
├── ...
├── protein_a_seed-1_sample-4_model.cif
├── protein_a_seed-1_sample-4_confidence.json
├── protein_a_ranked_model.cif
├── protein_a_ranked_confidence.json
├── protein_a_summary.csv
└── done.txt
```

The ranked files duplicate the highest-confidence sample for convenient access. Use `--format pdb` to write PDB instead of mmCIF, `--save-confidence-arrays` to add one confidence NPZ per sample, and `--save-distogram` to add one distogram NPZ per seed. A completed target is skipped when `done.txt` exists unless `--overwrite` is supplied.

See [Running your first prediction](../README.md#running-your-first-prediction) for the basic FASTA examples, and use `atlasfold monomer --help` or `atlasfold multimer --help` for the complete current option list.

Measured GPU memory and runtime across sequence lengths are available in the [performance guide](performance.md).

## Multi-GPU inference

Use `--gpu-ids` with either CLI subcommand to process multiple FASTA targets across GPUs on one machine:

```bash
atlasfold monomer --input-fasta monomers.fasta --out-dir predictions/monomers --gpu-ids 0 1
atlasfold multimer --input-fasta multimers.fasta --out-dir predictions/multimers --gpu-ids 0 1
```

- Without `--gpu-ids`, inference uses one GPU when CUDA is available, otherwise CPU.
- `--gpu-ids` cannot be combined with `--device`.
- GPU IDs are indices within `CUDA_VISIBLE_DEVICES` when it is set; for example, `CUDA_VISIBLE_DEVICES=2,3` maps `--gpu-ids 0 1` to physical GPUs 2 and 3.
- Targets are distributed across GPUs, with each target processed entirely on one GPU. `--max-tokens-per-batch` applies per GPU.

## Loading a model and runner

`load_model()` downloads published weights when necessary and returns a model on the requested device. `get_runner()` accepts either a loaded pretrained model or a compatible fine-tuned model instance and returns the matching runner.

| Model | Accepted names | Runner |
| --- | --- | --- |
| AtlasFold monomer | `atlasfold`, `atlasfold-260703` | `FoldingRunner` |
| AtlasFold-M multimer | `atlasfold-m`, `atlasfold-m-260725` | `MultimerFoldingRunner` |

```python
from atlasfold.pretrained import get_runner, load_model

model = load_model("atlasfold", device="cuda", kernel="auto")
runner = get_runner(model)
```

If `device` is omitted, AtlasFold uses CPU. Use `cache_dir` to choose the download cache, `model_path` to load local AtlasFold weights, and `lm_path` to load local AtlasLM weights. A local `model_path` must match the architecture selected by `model_name`.

The CLI chooses CUDA automatically when available, whereas the Python `load_model()` API defaults to CPU. Pass `device="cuda"` explicitly for GPU inference.

Pass `kernel="auto"`, `kernel="cuequiv"`, or `kernel="torch"` to `load_model()` to select the triangle operation backend.
The default is `auto`, which uses cuEquivariance when it is installed and the model is loaded on CUDA, and otherwise uses native Torch.
The CLI provides the same selection through `--kernel`.

`AtlasFold.from_pretrained()` and `AtlasFold_Multimer.from_pretrained()` only construct and load a model; they do not perform automatic kernel selection.
Prefer `load_model()` for inference.
If a model is constructed directly or loaded through `from_pretrained()`, it uses the Torch backend unless `set_kernel_backend()` is called explicitly:

```python
model = AtlasFold.from_pretrained("atlasfold", device="cuda")
model.set_kernel_backend("cuequiv") # auto | cuequiv | torch
```


Custom models can be passed directly without going through `load_model()` again:

```python
custom_model.eval()
runner = get_runner(custom_model)
```

## Single-target prediction

### Monomer

Call `fold()` with a target name and one amino-acid sequence:

```python
from atlasfold.pretrained import get_runner, load_model

runner = get_runner(load_model("atlasfold", device="cuda"))
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPD",
    num_samples=5,
    seeds=[1, 2],
)

print(result.best_key)  # (seed, sample_index)
print(result.best.avg_plddt, result.best.ptm)
```

### Multimer

Call `fold()` with one sequence per chain. Chain order is preserved in the output, and repeated sequences represent repeated copies in a homomer.

```python
from atlasfold.pretrained import get_runner, load_model

runner = get_runner(load_model("atlasfold-m", device="cuda"))
result = runner.fold(
    "complex_a",
    ["MKTAYIAKQRQISFVKSHFS", "GGHVDHGKSTTTGHLIYK"],
    num_samples=5,
    seeds=[1, 2],
)

print(result.best.iptm, result.best.ptm)
print(result.best.confidence_scores["complex"])
```

The Python multimer API requires pre-split chain sequences and does not split strings on `:`. Colon-delimited chains are supported only by the FASTA input used by the multimer CLI.

AtlasFold-M contains a template module used during training, but the released runner and CLI are sequence-only and do not accept templates. The reported FoldBench results were generated without templates.

## Inputs

Monomer iterators accept `FoldingInput` objects or equivalent `(name, sequence)` tuples:

```python
from atlasfold.runner import FoldingInput

monomer_inputs = [
    FoldingInput("protein_a", "MKTAYIAKQRQISFVKSHFS"),
    ("protein_b", "GILGYTEHQVVSSDFQKAA"),
]
```

Multimer iterators accept `MultimerInput` objects or equivalent `(name, chains)` tuples:

```python
from atlasfold.runner_multimer import MultimerInput

multimer_inputs = [
    MultimerInput("complex_a", ["MKTAYIAK", "GGHVDHGK"]),
    ("complex_b", ["GILGYTEH", "QQLLQYFQ"]),
]
```

The runners remove sequence whitespace, convert residues to uppercase, and replace nonstandard residues with `X` while emitting a warning. Empty monomer sequences and empty multimer chains raise `ValueError`.

## Batched inference

Both runners expose three execution methods:

| Method | Use |
| --- | --- |
| `fold()` | Return one target result. |
| `fold_iter()` | Stream one target result at a time without exposing model batch boundaries. |
| `fold_iter_batch()` | Yield one list of target results for each model batch. |

`fold_iter()` is the usual choice for processing many targets because outputs can be consumed and saved incrementally:

```python
from atlasfold.pretrained import get_runner, load_model
from atlasfold.runner import FoldingInput

runner = get_runner(load_model("atlasfold", device="cuda"))
inputs = [
    FoldingInput("protein_a", "MKTAYIAKQRQISFVKSHFS"),
    FoldingInput("protein_b", "GILGYTEHQVVSSDFQKAA"),
    FoldingInput("protein_c", "MQDRVKRPMNAFIVWSRDQRRKMALEN"),
]

for result in runner.fold_iter(inputs, max_tokens_per_batch=1024):
    result_path = f"{result.name}.cif"
    with open(result_path, "w") as handle:
        handle.write(result.best.to_mmcif())
```

Use `fold_iter_batch()` when downstream work needs the actual model batches:

```python
for batch_results in runner.fold_iter_batch(inputs, max_tokens_per_batch=1024):
    print([result.name for result in batch_results])
```

### Length bucketing and token budget

Targets are assigned to the smallest length bucket that can contain them and only targets in the same bucket are batched together. `max_tokens_per_batch` controls the batch size as `max(1, max_tokens_per_batch // bucket_length)`; it is a batching budget rather than a maximum supported sequence length.

The default buckets cover common lengths from 32 through 2048 residues, and longer inputs use buckets rounded up to a multiple of 128. Custom `length_buckets` are sorted and deduplicated, must contain positive values, and must include a bucket large enough for every input.

Bucketing can change output order. Match outputs by `result.name` instead of zipping an iterator with the original input list.

If CUDA runs out of memory during multi-target inference, reduce `max_tokens_per_batch`. A target larger than the token budget still runs by itself, so for a single large target also consider reducing `num_samples` or the diffusion `chunk_size`.

## Inference options

The main runner options and defaults are:

| Argument | Monomer default | Multimer default | Description |
| --- | --- | --- | --- |
| `num_samples` | `5` | `5` | Number of diffusion samples generated for each seed. |
| `seeds` | `[1]` | `[1]` | One integer seed or a sequence of seeds. |
| `num_recycles` | `4` | `10` | Number of recycling iterations. |
| `mlm_prob` | `0.15` | `0.20` | Probability of masking LM input residues during inference. |
| `sampling_config` | Length-dependent | 200 diffusion steps | Optional `SamplingConfig` override. |
| `length_buckets` | Automatic | Automatic | Optional explicit residue-length buckets. |
| `max_tokens_per_batch` | `1024` | `1024` | Bucketed token budget accepted by iterator methods. |
| `return_distogram` | `False` | `False` | Return raw distogram logits and boundaries. |

These are Python runner defaults. The CLI generates five diffusion samples per seed by default and exposes its current defaults through `--help`.

### Seeds and samples

Each target produces `len(seeds) * num_samples` structures. Outputs are keyed by `(seed, sample_index)`, so the two sampling axes remain explicit rather than being flattened into one model number.

Inference runs inside a forked Torch RNG context. Reusing the same model, input, seed, and options reproduces the same random choices without advancing the caller's Torch RNG state.

Both monomer and multimer inference require `0 < mlm_prob <= 1`. LM masking is sampled again for each recycling iteration.

### Diffusion configuration

When `sampling_config` is omitted, monomer inference selects the number of diffusion steps from the bucketed length: 20 steps through 512 residues, 30 steps through 1024 residues, and 100 steps for longer inputs. Multimer inference uses 200 steps by default.

Pass `SamplingConfig` to override the diffusion schedule. `chunk_size` controls how many diffusion samples pass through the score model at once; lowering it can reduce peak memory when `num_samples` is large.

```python
from atlasfold.model import SamplingConfig

sampling_config = SamplingConfig(num_steps=50, chunk_size=5)
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFS",
    num_samples=8,
    sampling_config=sampling_config,
)
```

The remaining `SamplingConfig` fields expose the diffusion noise schedule and solver parameters. Change them only when reproducing an evaluated sampling configuration.

## Results and ranking

`FoldingOutput` and `MultimerFoldingOutput` share the following target-level interface:

| Attribute | Description |
| --- | --- |
| `name` | Target name. |
| `length` | Number of residues in the target. |
| `outputs` | All predicted structures keyed by `(seed, sample_index)`. |
| `ranking` | Sample keys ordered from best to worst. |
| `best_key` | Key of the highest-ranked sample. |
| `best` | Highest-ranked structure object. |
| `distogram_logits` | Raw distogram logits keyed by seed when requested. |
| `distogram_boundaries` | Shared distogram bin boundaries when requested. |

Monomer samples are `ProteinOutput` objects containing coordinates, per-residue pLDDT, PAE, and pTM. They are ranked by mean pLDDT, exposed on a 0–100 scale as `avg_plddt`.

Multimer samples are `ProteinMultimerOutput` objects that additionally contain PDE, ipTM, fraction disordered, per-chain pTM, and pairwise interface ipTM. They are ranked by `0.8 * ipTM + 0.2 * pTM + 0.5 * fraction_disordered`, where `fraction_disordered` is the fraction of residues with window-smoothed RASA above 0.581. The `confidence_scores` property groups values under `complex`, `chains`, and `interfaces`.

```python
for key in result.ranking:
    sample = result.outputs[key]
    print(key, sample.ranking_score)
```

## Saving structures and confidence data

Every sample supports `to_mmcif()` and `to_pdb()`. These methods return text and do not write to disk themselves.

```python
best = result.best

with open(f"{result.name}.cif", "w") as handle:
    handle.write(best.to_mmcif())

confidence = best.confidence_scores
```

The Python API keeps confidence arrays as NumPy values on each sample. Monomer samples expose `plddt` with shape `(L,)` on a 0–1 scale and `pae` with shape `(L, L)` in angstroms; multimer samples additionally expose `pde` with shape `(L, L)` in ångströms. `avg_plddt` and structure B-factors use a 0–100 scale; pTM, ipTM, and fraction disordered use a 0–1 scale; and the multimer ranking score ranges from 0 to 1.5. Scalar scores are available as properties or through `confidence_scores`.

## Distogram output

Distogram output is disabled by default because its memory use grows quadratically with sequence length. Enable it only when raw distance-bin logits are needed:

```python
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFS",
    seeds=[1, 2],
    return_distogram=True,
)

seed_1_logits = result.distogram_logits[1]
boundaries = result.distogram_boundaries
```

Distogram logits are stored once per seed because they come from the folding trunk, while coordinates and confidence arrays are stored for every diffusion sample.

## Reproducing released benchmarks

The [benchmark release](benchmarks.md) documents the CAMEO22, CASP14, CASP15, and FoldBench sampling and selection protocols. Large prediction structures and evaluation CSV files are available from the linked Google Drive folder.

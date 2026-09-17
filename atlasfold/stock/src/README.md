# AtlasFold

[\[Paper\]](https://www.biorxiv.org/content/10.64898/2026.09.04.749352v2), [\[PDF\]](docs/atlasfold.pdf)

The **Atlas family** is a collection of open protein models for sequence representation and structure prediction.
AtlasLM is a protein language model (PLM), while AtlasFold and AtlasFold-M are trainable PLM-based models for protein folding and co-folding, respectively.

![AtlasFold predictions (blue) overlaid with experimental structures (gray) for CASP15 target T1183 (8IFX_B) and complex 8OI4.](docs/images/atlasfold-predictions.png)

AtlasFold achieves state-of-the-art accuracy among protein language model-based folding methods. AtlasFold and AtlasFold-M predict structures without an MSA search.
This repository provides pretrained models, training and inference code, staged training configurations, and preprocessing workflows for monomer and multimer folding.

## Installation

AtlasFold requires Python 3.10 or later. A CUDA GPU is recommended for structure prediction.

Install the inference dependencies from PyPI:

```bash
pip install "atlasfold[fold,cuequiv]"
```

To use AtlasLM as a standalone protein language model:

```bash
pip install atlasfold
```

To install the latest development version from GitHub:

```bash
git clone https://github.com/SeonghwanSeo/atlasfold.git
cd atlasfold
pip install -e ".[fold,cuequiv]"
```

AtlasFold supports [cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance/) kernels for faster inference. For systems without compatible NVIDIA CUDA hardware, install `atlasfold[fold]` instead.

## Running your first prediction

For a monomer, save one protein sequence per FASTA record:

```text
>protein_a
MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPD
>protein_b
FNPVGVAFKGNNGKYLSRIHRSGIDYTEFAKDNTD
```

Then run:

```bash
atlasfold monomer --input-fasta monomer.fasta --out-dir predictions/monomer/
```

For a protein complex, use one FASTA record per complex and separate chains with `:`:

```text
>complex_a
MKTAYIAKQRQISFVKSHFS:GGHVDHGKSTTTGHLIYK
>complex_b
MKEGFYWIQHNGRVQVAYYTHGVTEDLETGQTIIGVWHLTQGDDICHNGEAEILAGPLEPPI:MKEGFYWIQHNGRVQVA
YYTHGVTEDLETGQTIIGVWHLTQGDDICHNGEAEILAGPLEPPI
```

```bash
atlasfold multimer --input-fasta multimer.fasta --out-dir predictions/multimer/
```

Template-assisted inference for AtlasFold-M is not supported by the current runner or CLI.

The repository entry point provides the same interface:

```bash
python run_atlasfold.py monomer --input-fasta monomer.fasta --out-dir predictions/monomer/
python run_atlasfold.py multimer --input-fasta multimer.fasta --out-dir predictions/multimer/
```

Both AtlasFold and AtlasFold-M support batched inference with multiple FASTA records, enabling high-throughput structure prediction.

To distribute FASTA targets across multiple GPUs on one machine, pass `--gpu-ids`:

```bash
atlasfold monomer --input-fasta monomer.fasta --out-dir predictions/monomer/ --gpu-ids 0 1
atlasfold multimer --input-fasta multimer.fasta --out-dir predictions/multimer/ --gpu-ids 0 1
```

See [Multi-GPU inference](docs/inference.md#multi-gpu-inference) for GPU selection and per-GPU batching.

Run `atlasfold monomer --help` or `atlasfold multimer --help` for all options, and see the [inference guide](docs/inference.md) for batching, sampling, confidence values, and output formats.

## Performance and GPU memory

Peak GPU memory grows with total residue length when generating five diffusion samples.

| Total residues | AtlasFold | AtlasFold-M |
| ---: | ---: | ---: |
| 256 | 7.22 GiB | 7.31 GiB |
| 512 | 8.80 GiB | 9.07 GiB |
| 1,024 | 15.02 GiB | 16.05 GiB |
| 1,536 | 25.36 GiB | 27.63 GiB |
| 2,048 | 39.81 GiB | 43.83 GiB |

For multi-target workloads, batched inference substantially increases throughput. With five diffusion samples and up to 4,096 residues processed per batch:

| Workload | Unbatched | Batched | Throughput gain |
| --- | ---: | ---: | ---: |
| AtlasFold, 64-residue monomers | 0.775 sequences/s | 9.663 sequences/s | 12.5× |
| AtlasFold-M, 256-residue complexes | 0.149 complexes/s | 0.314 complexes/s | 2.1× |

Measurements were collected on a single NVIDIA B200 using PyTorch 2.10.0, CUDA 12.8, and cuEquivariance 0.10.0. See the [performance guide](docs/performance.md) for the complete memory and runtime measurements.

## Python API

### Structure prediction

```python
from atlasfold.pretrained import get_runner, load_model

model = load_model("atlasfold", device="cuda")
runner = get_runner(model)
result = runner.fold(
    "protein_a",
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPD",
    num_samples=5,
)

print(result.best.avg_plddt, result.best.ptm)
with open("protein_a.pdb", "w") as handle:
    handle.write(result.best.to_pdb())
with open("protein_a.cif", "w") as handle:
    handle.write(result.best.to_mmcif())
```

Use `load_model("atlasfold-m", device="cuda")` for a complex and pass a list of chain sequences to `runner.fold()`. The Python API defaults to CPU if `device` is omitted; the CLI selects CUDA automatically when available.

### Protein language model

```python
import torch

from atlaslm import load_model

model = load_model("atlaslm-3b", device="cuda", dtype=torch.bfloat16)
output = model.embed_sequences(
    ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPD"],
    return_hidden_states=True,
)

print(output.embeddings.shape)
print(len(output.hidden_states))
```

Pass `return_attentions=True` to return attention maps. Attention tensors grow quadratically with sequence length and can require substantially more memory.

## Available models

| Model | Parameters | Weight download | Use | Weights |
| --- | ---: | ---: | --- | --- |
| AtlasLM-600M | 575M | 1.07 GiB | Protein sequence representations | [`SeonghwanSeo/atlaslm-600m-base`](https://huggingface.co/SeonghwanSeo/atlaslm-600m-base) |
| AtlasLM-3B | 3.06B | 5.71 GiB | Protein sequence representations and AtlasFold backbone | [`SeonghwanSeo/atlaslm-3b-base`](https://huggingface.co/SeonghwanSeo/atlaslm-3b-base) |
| AtlasFold | 215M + AtlasLM-3B | 0.80 GiB + AtlasLM-3B | Monomer structure prediction | [`SeonghwanSeo/atlasfold-260703`](https://huggingface.co/SeonghwanSeo/atlasfold-260703) |
| AtlasFold-M | 220M + AtlasLM-3B | 0.82 GiB + AtlasLM-3B | Protein-complex structure prediction | [`SeonghwanSeo/atlasfold-m-260725`](https://huggingface.co/SeonghwanSeo/atlasfold-m-260725) |

AtlasLM-600M is deprecated now that ESMC-600M is available for commercial use. AtlasLM-3B is the recommended AtlasLM checkpoint.

## Evaluation

Evaluation protocols and results for AtlasFold and AtlasFold-M are provided in the [benchmark documentation](docs/benchmarks.md). The associated prediction structures and evaluation artifacts are available from the release folder below.

## Checkpoints, data, and benchmark artifacts

Large release artifacts are hosted in the [AtlasFold Google Drive folder](https://drive.google.com/drive/folders/1xjSBmbCFqghWj8xKYuEIDISKoQkCX45I?usp=sharing):

| Artifact | Contents |
| --- | --- |
| [Intermediate checkpoints](https://drive.google.com/drive/folders/1TDz2Ng4-zYfpTxlqg5wD3MkzwJyL2gTm) | AtlasLM pretraining checkpoints and AtlasFold/AtlasFold-M staged training checkpoints |
| [Structural datasets](https://drive.google.com/drive/folders/1EiRTKSUL3iD_MQ_0qmj5Sb-2KMh-3kmS) | Processed monomer and multimer training and validation data |
| [Benchmark artifacts](https://drive.google.com/drive/folders/1KjhQe4yvLMSBEJdxXZ6oC9Wi-pw5a439) | CAMEO22, CASP14, CASP15 and FoldBench results |


## Training

Install the training dependencies with `pip install -e ".[fold,train,cuequiv]"`. AtlasFold monomer training uses four progressively longer crop stages, and AtlasFold-M fine-tuning uses three stages initialized from the monomer model. See the [training guide](docs/training.md) for data setup, released intermediate checkpoints, complete commands, and configuration overrides, and the [data guide](docs/data.md) for the released dataset layout and provenance.

## Citation

```bibtex
@article{seo2026atlasfold,
  author = {Seo, Seonghwan and Kim, Hyeongwoo and Moon, Seokhyun and Kim, Woo Youn and {Team KAIST}},
  title = {AtlasFold: Protein structure prediction with metagenomic-scale language models},
  year = {2026},
  doi = {10.64898/2026.09.04.749352},
  URL = {https://www.biorxiv.org/content/10.64898/2026.09.04.749352v2},
  journal = {bioRxiv}
}
```

## Acknowledgements

This project was developed as part of the **K-Fold** initiative supported by the Ministry of Science and ICT (MSIT) of the Republic of Korea. The K-Fold project for biomolecular complex prediction is currently under active development with numerous contributors at KAIST and will be released soon!

---

I would like to thank [Dr. Hyeongwoo Kim](https://scholar.google.com/citations?user=YpiY1q8AAAAJ&hl=en&oi=ao), [Dr. Seokhyun Moon](https://scholar.google.com/citations?hl=en&user=U1j8Ip8AAAAJ), and [Prof. Woo Youn Kim](https://scholar.google.com/citations?user=elJ5KrcAAAAJ&hl=en) for their guidance and support during the development of AtlasFold.

This project is built upon the pioneering works of Google DeepMind, Meta AI, OpenFold Consortium, and EvolutionaryScale in the fields of biomolecular language modeling and structure prediction. I am deeply grateful to the open-source community for advancing the fields of biomolecular language modeling and structure prediction.

**Foundations of AtlasLM:**

- **ESM2**: Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379.6637 (2023): 1123-1130.
- **ESM3**: Hayes, Thomas, et al. "Simulating 500 million years of evolution with a language model." Science 387.6736 (2025): 850-858.
- **ESMC**: ESM Team. "ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning." EvolutionaryScale Website, December 4, 2024. https://evolutionaryscale.ai/blog/esm-cambrian.

**Foundations of AtlasFold:**

- **AlphaFold2**: Jumper, John, et al. "Highly accurate protein structure prediction with AlphaFold." Nature 596.7873 (2021): 583-589.
- **OpenFold**: Ahdritz, Gustaf, et al. "OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization." Nature Methods 21.8 (2024): 1514-1524.
- **ESMFold**: Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379.6637 (2023): 1123-1130.
- **AlphaFold3**: Abramson, Josh, et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature 630.8016 (2024): 493-500.
- **SimpleFold**: Wang, Yuyang, et al. "SimpleFold: Folding proteins is simpler than you think." arXiv preprint arXiv:2509.18480 (2025).
- **ESMFold2**: Candido, Salvatore, et al. "Language Modeling Materializes a World Model of Protein Biology." [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.06.03.729735) (2026).

## License

The source code, model weights, and released datasets are licensed under the [MIT License](LICENSE).

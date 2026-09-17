# ESM-IF1 — stock, as pinned

## Pin

Upstream: `https://github.com/facebookresearch/esm` at commit `2b369911bb5b4b0dda914521b9475cad1656b2ac` (fair-esm 2.0.1, `esm/version.py`;
the repository is archived; PyPI's fair-esm 2.0.0 is not the pin — its `GVPTransformerModel.sample()` cannot run on a GPU), shipped under
`stock/` as the source archive `fair-esm-2b369911.tar.gz` (`git archive` of the commit) and, extracted from it, `stock/src/` — upstream's
example entry `examples/inverse_folding/sample_sequences.py` lives there; commit, version and archive in `stock/PINS.json`. Weights:
`esm_if1_gvp4_t16_142M_UR50.pt` (bytes, sha256 and upstream's URL in PINS.json), fetched by `bash run.sh install --weights DIR` through
`torch.hub.download_url_to_file`, the transfer upstream's loader uses; upstream's licence (MIT) as found under `stock/src/`. `stock/` is never edited.

## Stack

Base image `nvidia/cuda:12.1.1-runtime-ubuntu22.04` (Ubuntu 22.04, CUDA 12.1.1 runtime; host driver ≥ 530), Python 3.11.5, torch 2.4.0+cu121, triton 3.0.0, numpy 1.26.4, scipy 1.11.4, biotite 0.39.0,
torch-geometric 2.6.1 with torch_scatter 2.1.2 / torch_sparse 0.6.18 / torch_cluster 1.6.3 (pt24cu121 wheels; `torch_geometric` is a required
import of `esm.inverse_folding`) — the full list is `environment/requirements.lock`; `environment/Dockerfile` and `apptainer.def` build it,
one image for every configured card. Outside the image — Linux x86-64; the host needs only the NVIDIA driver ≥ 530 (torch's wheels carry the CUDA 12.1
libraries) and `git` only for the git-URL alternative below; ≈3 GB of downloads — use a released CPython 3.11 in a fresh venv, exactly 3.11 (cp311
wheels; another minor version fails at `pip install -r`): uv brings its own (the block's second line installs uv itself when absent);
python.org, conda, or apt/deadsnakes `python3.11` + `python3.11-venv` work too (no `-dev` headers: nothing here compiles), never Ubuntu 22.04's own
`python3.11` package (3.11.0rc1). Or use an environment that already runs fair-esm at the pin. From inside `esm_if1/`, in this order:

```bash
sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl   # bare Ubuntu host (root: no sudo); skip what you have; no compiler needed — every pin is a wheel
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.11 venv && . venv/bin/activate   # uv's own released CPython 3.11
grep -v -E '^(#|fair-esm @)' environment/requirements.lock > /tmp/stack.txt && python -m pip install --no-deps -r /tmp/stack.txt   # the pinned stack
python -m pip install --no-deps stock/fair-esm-2b369911.tar.gz   # upstream at the pin; or: pip install --no-deps 'fair-esm @ git+https://github.com/facebookresearch/esm.git@2b369911bb5b4b0dda914521b9475cad1656b2ac'
bash run.sh install        # kit + shared core editable (pip install -e ../common/opt_core -e opt), then the pin check; = README step 2's install line (README adds `--weights DIR`: the 1.7 GB checkpoint is fetched into DIR if absent, otherwise only checked there) — type it once; step 2 continues at its export line
```

`bash run.sh install` ends with `fair-esm 2.0.1 (2b369911): pinned — <N> files match stock/fair-esm-2b369911.tar.gz at <site-packages>` (exit 0); a
fair-esm at another version (PyPI's 2.0.0 included), another commit or with edited files is refused by name (exit 3).

Precision is PyTorch's defaults on every mode: fp32, matmul TF32 off; upstream has no `torch.compile`, CUDA-graph or flash-attention path.

## How stock is run

`--mode off` executes, once per structure, `python -s -m esm_if1_opt.stock_design --proof-json <out>/stock_env_proof.json --env-absent <prefixes> --kit-dirs "" -- stock/src/examples/inverse_folding/sample_sequences.py <pdbfile> [--chain C] --temperature T --num-samples N --outpath FILE [--multichain-backbone] [--nogpu]`
(`--kit-dirs` is empty: the stock runner is a module of this package, so no path is hidden) in a subprocess whose environment is stripped of `ESM_IF1_OPT*`, `ESM_IF1_KIT*`, `NVIDIA_TF32_OVERRIDE`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`,
`CUBLAS_WORKSPACE_CONFIG`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` and `PYTHONSAFEPATH`. Before torch is imported the child confirms those names are
absent and no kit finder is armed (`[esm_if1-opt] ENV-CLEAN ok (…)`, written to `stock_env_proof.json`; otherwise `NOT STOCK` and exit 3 with
nothing run), then runs the script as `__main__` with exactly its own arguments — `--seed`, `--batch_size` and `--det` are not among them and are
reported `NOT APPLIED`. The script is unseeded as shipped. After it returns the child prints `PEAK` and `KERNELS`, read from the torch the script
imported. `TORCH_HOME`, `CUDA_VISIBLE_DEVICES` and `PYTORCH_CUDA_ALLOC_CONF` pass through to the child unchanged.

Upstream's arguments keep the script's names and defaults on both modes: `PDBFILE` (`.pdb` / `.cif`), `--chain` (none: every chain of the
file), `--temperature` 1.0, `--num-samples` 1, `--outpath` `output/sampled_seqs.fasta`, `--multichain-backbone` | `--singlechain-backbone`
(single), `--nogpu`. `model.sample()`'s optional `confidence` input is not an argument of the script and is not served.

## Stock exceptions

None. Every mode runs upstream's code as shipped; the one known upstream issue is an opt-in fix (README, Known upstream issues).

## Variables

| variable | required | default (`configs/h100.env`) | effect |
|---|---|---|---|
| `ESM_IF1_WEIGHTS` | no | unset | a local copy of the checkpoint; `design` links torch.hub's cache entry to it (an existing regular file there is kept). Unset: upstream's loader reads the cache or downloads |
| `ESM_IF1_OPT` | no | unset | the mode when `--mode` is absent; a `--mode` that disagrees with it is a usage error (exit 2) |
| `MODEL_OPT_STATE` | no | `$HOME/.cache/esm_if1_opt` | state directory; holds `torch_home/` unless `TORCH_HOME` is set |
| `TORCH_HOME` | no | `$MODEL_OPT_STATE/torch_home` | torch.hub's cache root, passed to every child as an absolute path |
| `MODEL_OPT_TARGET_GPU` | no | `H100` (`A100` in `configs/a100.env`, `H200` in `configs/h200.env`) | the GPU class `check` prints `target=` / `match=` against — a report, never a refusal |
| `MODEL_OPT` | no | the `esm_if1/` directory | exported by `run.sh`; read only when the package cannot find the tree beside itself |

`configs/a100.env` and `configs/h200.env` set `MODEL_OPT_TARGET_GPU` (`A100`, `H200`) and source `h100.env`; every variable keeps a value already set in
the environment.
An `ESM_IF1_WEIGHTS` that names a file the process cannot see (an unbound directory in a container, a typo) counts as unset: upstream's loader
then downloads the 1.7 GB checkpoint into `$TORCH_HOME/hub/checkpoints/` instead of refusing — `bash run.sh check` prints `weights=absent` in that case.

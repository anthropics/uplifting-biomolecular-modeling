# Hub RMSNorm kernel — the tree's copy

`triton_layer_norm/` is the package directory `build/torch-universal/triton_layer_norm` of the Hugging Face hub kernel
`kernels-community/triton-layer-norm` at the revision pinned in `stock/PINS.json` (`hub_kernel.rev`) — the kernel stock E1 loads at
import (`E1/modeling.py`: `get_kernel("kernels-community/triton-layer-norm")`, then `rms_norm_fn` from `layer_norm.py`). `run.sh install`
(`python -m e1_opt.weights`) copies it into the cache layout the `kernels` package reads —
`<root>/models--kernels-community--triton-layer-norm/snapshots/<rev>/build/torch-universal/triton_layer_norm`, root = `KERNELS_CACHE`
when set, else `HF_HOME/hub` — and points `refs/main` at that revision, so the offline cache answers upstream's request for `main` with
these files; nothing is fetched for the kernel. Per-file digests: `stock/PINS.json` `hub_kernel.files_sha256` (`layer_norm.py` =
`hub_kernel.layer_norm_py_sha256`, the digest the run-time kernel gate checks).

Origin of the bytes: huggingface.co/kernels-community/triton-layer-norm, repository type `kernel`, commit
`f17bf36ff115829f6a08ed37a3925bb6878871ae` ("Migrated from kernels-community/triton-layer-norm"). The Hub moved the repository from
the `model` type, under which the pinned revision was recorded and which it no longer serves, to the `kernel` type; `layer_norm.py` at
that commit is byte-identical to the pinned digest. Licence: BSD-3-Clause (declared on the repository's card; `layer_norm.py` is
flash-attention's `flash_attn/ops/triton/layer_norm.py`, "Copyright (c) 2024, Tri Dao.") — see `LICENSE` in this directory.

# ESM C — optimization kit

An add-on that makes stock ESM C faster to load and to run — `esm` 3.4.0: `ESMC.from_pretrained` → `encode` → `logits`,
or the padded batched forward; models `esmc_300m`, `esmc_600m`, `esmc_6b`. You call ESM C exactly as before: the kit
installs beside `esm`, and one environment variable, `ESMC_OPT`, switches it on per process. Two modes:

- `off` — stock ESM C, exactly as released ('stock' below always means this unmodified upstream release). This is what
  runs with the variable unset.
- `exact` — identical outputs, faster.

There is no `fast` or `big` mode for this model.

What `exact` changes: `CHANGES.md`. Exact versions, the pinned software stack, variables and recipes: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock.

## Setup

Pick ONE way to get the pinned stack — stock ESM C 3.4.0 and everything it needs: Ubuntu 24.04, CUDA 13.0, Python 3.12,
torch 2.11.0+cu130 with upstream's `accel` extra (the 'Stack' section of `STOCK.md` lists every pin): **A — Docker**,
**B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA GPU whose host driver runs CUDA 13.0 (580 or newer);
- a weights directory, written `/weights` in the blocks (left of `:` on the mount lines it is your host directory).

Route C additionally needs:

- git, a C/C++ compiler and the CUDA 13.0 toolkit's `nvcc`;
- a libstdc++ with `CXXABI_1.3.15` (Ubuntu 24.04's; the Ubuntu 22.04 remedy is in STOCK.md's Stack section);
- Python headers — uv's Python brings its own.

Type the first block from the directory that holds `esmc/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image; then a shell in it
docker build -f esmc/environment/Dockerfile -t esmc-kit:dev .
docker run --rm -it --gpus all -w /kit -v /weights:/weights -v "$PWD":/work esmc-kit:dev bash   # /weights left of ":" = your host dir
# B — Apptainer / Singularity (the whole Setup for B): converts the image built in A (no daemon needed at run time); verbs run from the host
apptainer build esmc.sif esmc/environment/apptainer.def
kit() { apptainer run --nv --bind /weights:/weights --bind "$PWD":/work --env HF_HOME=/weights/esmc/hf esmc.sif "$@"; }   # kit = B's ./run.sh · /weights left of ":" = your host dir
kit install --weights /weights/esmc --variant 6b         # /weights/esmc: HF cache hf/, fetched if absent, else checked
kit check --variant 6b                                   # your scripts: the apptainer exec line under Run
#     B also writes in your bound home: ~/.cache/esmc_sdkfused (the CUDA extension, built once per user, about a minute) and Triton's cache.
# C — instead of A or B, on your own host (prerequisites above; no uv yet? STOCK.md §Stack line 2 installs it) — the whole route-C recipe; §Stack = these lines annotated (run one or the other)
uv venv --seed --managed-python --python 3.12 ~/esmc-env && . ~/esmc-env/bin/activate
pip install --no-deps -r esmc/environment/requirements.lock
```

Route B (Apptainer) is complete at this point; your own scripts then run through the `apptainer exec` line shown under
Run. Under **A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your activated
environment: after the route-C lines above you are still in the directory holding `esmc/`, so type the whole block; if
you followed STOCK.md's Stack section instead, you are inside `esmc/` with `install` already done — resume at `export`. Run:

```bash
[ -f run.sh ] || cd esmc                                 # no-op once inside · C via §Stack: resume at export
bash run.sh install --weights /weights/esmc --variant 6b    # kit, pin check, CUDA extension; weights fetched or checked
export HF_HOME=/weights/esmc/hf HF_HUB_OFFLINE=1         # where the weights are; offline pins loads to that snapshot
bash run.sh check --variant 6b                              # dry run: pins, GPU, mode; exit 0, loads nothing
```

What the blocks assume:

- **Weights.** `--weights DIR --variant V` is optional; without it `install` fetches and checks no weights. With it, the
  weights go into the Hugging Face cache `DIR/hf` (fetched if absent, else digest-checked), and `HF_HOME` then names
  that cache. `--variant` takes `300m|600m|6b` or the SDK model names.
- **Route B's cache files.** B also writes in your bound home: `~/.cache/esmc_sdkfused` (the CUDA extension, built once
  per user, about a minute) and Triton's cache.
- **What the pin check accepts.** `esm` 3.4.0 and `transformers` 4.57.6 only as the lock installs them (pip's git install
  at the pinned commit) or from the source archive named in `stock/PINS.json`; a wheel, an editable checkout or another
  commit is rejected with a message naming it (exit 3).

## Run

ESM C has no inference command line and the kit adds none: your own code runs unchanged and `ESMC_OPT` picks the mode
per process (with `HF_HOME` and `HF_HUB_OFFLINE=1` exported as in Setup). The kit's one command is the dry run,
`esmc-opt check --variant V [--mode M]`, the same as `bash run.sh check`.

- Under A, `cd /work` first — the host directory you started from; files written elsewhere go with the container.
- Under B, use the last line of the block.

```bash
python your_script.py                    # ESMC_OPT unset = off: stock; the kit stays out entirely
ESMC_OPT=exact python your_script.py     # exact: engages at `import esm`
apptainer exec --nv --bind /weights:/weights --bind "$PWD":/work --env HF_HOME=/weights/esmc/hf --env ESMC_OPT=exact esmc.sif python /work/your_script.py
```

`your_script.py` is any ESM C code, for example both call shapes on one public sequence (human ubiquitin):

```python
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
client = ESMC.from_pretrained("esmc_6b", device=torch.device("cuda"))                 # or esmc_300m / esmc_600m
out = client.logits(client.encode(ESMProtein(sequence=seq)), LogitsConfig(sequence=True, return_embeddings=True))
batch = client.tokenizer([seq] * 8, return_tensors="pt", padding=True).to("cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    hidden = client.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
```

**From code instead of the variable.** `import esmc_opt; esmc_opt.enable("exact")` before the first
`ESMC.from_pretrained`; `esmc_opt.status()` returns the report.

**Outputs.** Your code's tensors; the kit writes only compile caches (`~/.cache/esmc_sdkfused`, `TRITON_CACHE_DIR`).

**What a run prints** (stdout):

- Under `exact`, one line to check per model: `[esmc-opt] APPLIED model#1 variant=6b … levers_applied=pipe,fused levers_fallback=none …`
  (the kit calls its optimizations 'levers'), after an `[esmc-opt] ACTIVE mode=exact …` line at `import esm`.
- Under `off`, nothing.
- `run.sh check` prints `[esmc-opt] DRY-RUN mode=exact variant=6b …` and exits 0.
- If `exact` cannot engage, the kit names the reason and stops — `[esmc-opt] NOT ACTIVE: <reason>` with exit 3 at
  `import esm`, or `esmc_opt.ActivationError` from `ESMC.from_pretrained`. It never falls back to stock silently.

**First run.** The first `exact` process on a machine spends about 12 s more at model load (one-time compiles, cached);
compare from the second run on.

## Modes

- `off` — stock `ESMC.from_pretrained` and every call as released, with nothing of the kit applied.
- `exact` — device-side weight loading, an exact single-sequence tokenizer path and an eager fused forward (fused
  residual+LayerNorm and q/k-LayerNorm+rotary kernels; nothing captured or compiled per shape) on every size and call
  shape. Outputs identical to `off`.

## Notes

- **Cards.** Nothing in the kit reads the card name or memory size, so A100 (80 GB or 40 GB) and H200 run exactly as
  H100 does; `run.sh install` builds the CUDA extension for the GPU it sees (for compute capabilities 8.0 and 9.0 when
  none is visible). B200 (compute capability 10.0) is untested: `stock/PINS.json` names a cache key for it, but the
  pinned flash-attn / Transformer Engine wheels target compute capabilities 8.0–9.0.
- **A mode is a fixed set of optimizations** ('levers' on the kit's lines and in `CHANGES.md`), the same for every
  model built in the process; no switch drops one.
- **Drift versus refusal.** Library versions off the pins are named on the `ACTIVE` line (`drift=…`) and the mode still
  engages. Only `esm` outside the accepted install form, no visible GPU, or an optimization that cannot run on the
  machine stop it.
- **Exit codes of `run.sh`:** see STOCK.md's Variables section.
